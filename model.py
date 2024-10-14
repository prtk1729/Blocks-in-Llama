import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from typing import Optional

# decorator class just to store data -> no methods, no inheritance, therfore no super()
@dataclass
class ModelArgs:
    dim: int = 4096 # same as d_model in Vanilla Transformer
    n_layers: int = 32 # Nx in vanila Tr.
    n_heads: int = 32 # Llama has diff number of heads for KV and Q, here it is for Q
    n_kv_heads: Optional[int] = None # Number of heads for KV

    vocab_size: int = -1 # Will be set by the tokeniser

    norm_eps: float = 1e-5 # recall we use it in Norm (Layer Norm => x_ = (x - mu)/sqrt( std**2 + eps ) )
    multiple_of: int = 256 # acrhitecture design choice
    ffn_dim_multiplier: Optional[int] = None 
    
    max_batch_size: int = 32
    max_seq_len: int = 2048 # window size
    
    # device?
    device: str = None # will be set when we know where we are running it on



def precompute_m_theta_matrix(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # n_heads = args.n_heads # For the query, we have diff num of heads in Llama vs Vanilla Transf.
    # dim = args.dim
    assert head_dim % 2 == 0, "dim of head for query should be even as pre the paper"
    # 1 / 10000 ** (2(i - 1)/q_dim ), i = [1, 2, ..., dim//2], q_dim = head_dim = dim / n_heads
    # (B, seq_len, h_q, head_dim_q)
    theta_numerator = torch.arange( 0, head_dim, 2 ).float() # same as above [0, 2, 4, ..., head_dim of these]

    # Runs on the head_dim i.e q_dim, with every 2 consecutive dim_values make a vector in a vector-space of half the dim  
    # Recall (x1, x2) fig 1 in Paper
    # Shape: (head_dim / 2)
    theta_values = (1.0 / (theta ** (theta_numerator / head_dim) )).to(device)

    # runs on the seq_len, "m" rotations anti-cw preserves the position in the seq, that token occured
    # Shape: (seq_len)
    m_values = torch.arange(seq_len).to(device) # based on the positions

    # can think for each fixed position, there is a vector [0, 1, .., head_dim/2)
    # Shape: ( seq_len, head_dim/2 )
    m_theta_matrix = torch.outer( m_values, theta_values ).float()

    # convert each element in m_theta as e**(i * m_k_theta_t) for entry m_k_theta_t
    # Since, torch.polar to conert to e**i_theta, requires magnitude(r) and angles(theta)
    # to make cos( theta ) + isin( theta )
    mag = torch.ones_like( m_theta_matrix )
    polar_form_matrix = torch.polar( abs = mag, \
                angle = m_theta_matrix)
    return polar_form_matrix


def rotary_positional_embeddings(x: torch.Tensor, polar_form_matrix: torch.Tensor, device = str):
    '''
    Recall:
        - Here, we are getting the Embeddings after interacting with 
        - Weight matrices
        - And after splitting into heads

    Hence, 
    Shape of x: (B, seq_len, h, head_dim) i.e for Query -> h = n_heads_q, for Key -> h = n_kv_heads, 
    NOTE: head_dim is same for both
    '''
    # (B, seq_len, h, head_dim) -> (B, seq_len, h, head_dim/2)
    # 2 consecutive values will become single complex number
    # shape_tuple = *x.shape[:-1] # (B, seq_len, h)
    # last dim number of elements wants to make as 2 i.e old_dim splits into (old_dim/2, 2)

    # Recall fig 1: (x1, x2) -> (x1 + x2j)
    # x_complex's shape = (B, seq_len, h, head_dim) -> (B, seq_len, h, head_dim/2)
    x_complex = torch.view_as_complex( torch.reshape( *x.shape[:-1] ), -1, 2 )
    
    # But, polar_form_matrix has different shape
    # Shape of x_complex: (B, seq_len, h, dim/2)
    # Shape of polar_form_matrix: (seq_len, dim/2)
    # For matmul, we need to make 2 more dimensions at axis=0, axis=2 (Batch and head)
    # Shape of polar_form_matrix: (1, seq_len, 1, dim/2)
    polar_form_matrix = polar_form_matrix.unsqueeze(0).unsqueeze(2)

    # Now, recall Fig. 1 in paper, we need to rotate using rotation matrix
    x_rotated = x_complex * polar_form_matrix

    # Shape of x_rotated: (B, seq_len, h, dim/2)
    #  (B, seq_len, h, dim/2) ->  (B, seq_len, h, dim/2, 2) :: [ (x1 + y1_j) ] was innermost item becomes 2 [x1 y1]
    x_out = torch.view_as_real(x_rotated) # [( x1 + y1_j ), ( x2 + y2_j )] -> [ [x1 y1], [x2 y2] ]

    # We need to reshape so that last dimension is "dim"
    # (B, seq_len, h, dim/2, 2) -> (B, seq_len, h, dim)
    x_out = x_out.reshape( *x.shape )

    # typecast and move to device
    return x_out.as_type(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    '''
        If n_rep == 1 => MH-A usual
        Else: Grouped MH-A
    '''
    batch_size, seq_len, n_kv_heads, head_dim = x.shape 
    if n_rep == 1:
        return x

    # First we create a new dimension. But where?
    # (B, seq_len, n_kv_heads, head_dim ) -> (B, seq_len, n_kv_heads, n_rep, head_dim )
    # Can be thought of, for each head_id -> make n_repeat copies of it.
    # # (B, seq_len, n_kv_heads, head_dim) -> (B, seq_len, n_kv_heads, 1, head_dim)
    # x = x[:, :, :, None, :]
    # # (B, seq_len, n_kv_heads, 1, head_dim) -> (B, seq_len, n_kv_heads, n_rep, head_dim)
    # x = x.expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
    # # Merge for each head_id, so that dim for each head gets concat as n_rep * head_dim
    # # (B, seq_len, n_kv_heads, n_rep * head_dim)
    # x = x.reshape(batch_size, seq_len, n_kv_heads, n_rep * head_dim )
    # return x

    return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads, n_rep * head_dim)
            )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.n_heads_q = args.n_heads
        

        # Either GQA or MHA
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads != None else args.n_heads

        self.head_dim = args.dim // self.n_heads_q
        # Dimension of Key/Query/Value Heads are same i.e head_dim
        # Recall width was same in Grouped Viz
        # self.head_dim = self.q_dim = self.k_dim = self.v_dim 



        # Recall Multi-Query Attention:
        #    |     |     |    (n_v_heads)
        #    
        #    |     |     |    (n_k_heads)
        #   / \   / \   / \ 
        #  |   | |   | |   |  (n_heads_q)

        # How many times we repeat each key-head, value-head?
        # self.q_head_group_size = self.n_rep
        # print( "========> n_kv_heads", self.n_kv_heads )
        self.n_rep = self.n_heads_q // self.n_kv_heads


        # Recall MH-A learnables
        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim ,  bias = False)
        # Projects from dim -> n_kv_heads * self.head_dim
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim , bias = False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim , bias = False)

        # Recall: After concat all heads i.e n_heads * head_dim
        self.wo = nn.Linear(self.n_heads_q * self.head_dim, args.dim, bias = False)


        # Since, we will use this to store each head, i.e after splitting
        # cache_shape = tuple([args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim])
        # We create 2 separate caches for key and values, query? Simply last token in query
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, 
                start_pos: int, 
                polar_form_matrix: torch.Tensor, 
                ):
        '''
        Args:
            - x: Embedded token. Where?
            - start_pos: At position `start_pos` of the sequence
            - polar_form_matrix: 
                - This is non-learnable i.e deterministic
                - Simply an outer_product of all pairs of < theta, m >

            NOTE: The Seq_len == 1, here. Why?
            During Inference, we pass the token as position "start_pos" only
            Keys[0: start_pos), Values[0: start_pos]  are cached
            We require these, due to causal mask property to predict the next token
                - i.e, we need:-  
                    Keys[ 0: start_pos ), Values[ 0: start_pos )  {Retrieve this from cache}
                    Query[ start_pos ] {Apply RoPE and compute here}
            Recall: 
                - Viz, how we use these to predict the next token

            We write this code for inference only.
            We aren't training from scratch
        '''
        # Shape of x: (B, 1, dim) -> 1 Embedded token of dim = dim
        batch_size, seq_len, dim = x.shape 
        assert seq_len == 1, "During Inference, more than one token was passed"

        # 1st step of MHA / Self-Attention recall figure
        # (B, 1, dim) -> (B, 1, n_heads_q * head_dim )
        xq = self.wq(x)

        # (B, 1, dim) -> (B, 1, n_kv_heads * head_dim )
        xk = self.wk(x)

        # (B, 1, dim) -> (B, 1, n_kv_heads * head_dim )
        xv = self.wv(x)

        # Split them into heads
        xq = xq.reshape(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Recall self-attention mechanism in figure, but wait...
        # Did we use the positional Encodings?

        # Here, to each head we apply RoPE
        # But, on only important ones i.e other than values
        # (B, seq_len, n_heads_q, head_dim) -> (B, seq_len, n_heads_q, head_dim)
        # Why? rotary positional Embeddings conserves the shape after operation.
        xq = rotary_positional_embeddings(xq, polar_form_matrix, device = x.device)
        # (B, seq_len, n_kv_heads, head_dim) -> (B, seq_len, n_kv_heads, head_dim)
        xk = rotary_positional_embeddings(xk, polar_form_matrix, device = x.device)


        ## SO, we currently have the next tokens embeddings for query and key
        # Store them in cache for future predictions
        seq_len = 1 # since 1 token at inference
        self.cache_k[:batch_size, start_pos: start_pos+seq_len, :self.n_kv_heads, :self.head_dim] = xk
        self.cache_v[:batch_size, start_pos: start_pos+seq_len, :self.n_kv_heads, :self.head_dim] = xv

        # Now, we can get the required window in both key and value, for causal logic
        # to predict the next token
        # Recall ppt with growing K, V, But fixed 1 sequnce from Query
        keys = self.cache_k[:batch_size, 0: start_pos+seq_len, :self.n_kv_heads, :self.head_dim]
        values = self.cache_v[:batch_size, 0: start_pos+seq_len, :self.n_kv_heads, :self.head_dim]


        ## Multi-Grouped-Query
        ## Here, we have the required xv, xk, xq
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # MH-A as usual
        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / torch.sqrt(self.head_dim)
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # The hidden dim is basically the dim, 
        # the weight matrices project the x's dimension to (i.e from dim -> hidden_dim)

        # Calculate hdden_dim
        
        hidden_dim = 4 * args.dim
        hidden_dim = int((2 * hidden_dim) / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim *= args.ffn_dim_multiplier
        
        # Here, it needn't be a multiple of param "multiple_of"
        # But, we want to make the next multiple of "multiple_of"
        # recall: ceil(a, b) = (a + b - 1)/b
        # We want to ask what's the next multiple of "multiple_of"
        remainder, quotient = (hidden_dim % args.multiple_of), (hidden_dim // args.multiple_of)
        if remainder > 0:
            hidden_dim = (quotient * args.multiple_of) + args.multiple_of # crosses hidden_dim

        # recall    W*x, V*x -> swish(W*x)
        # W*x: Projection from dim -> hidden_dim 
        self.w1 = nn.Linear(args.dim, hidden_dim, bias = False) 
        # V*x: Projection from dim -> hidden_dim 
        self.w3 = nn.Linear(args.dim, hidden_dim, bias = False)
        # W2*x: Projection from hidden_dim -> dim
        self.w2 = nn.Linear(hidden_dim, args.dim, bias = False)

    def forward(self, x: torch.Tensor): 
        # (B, seq_len, dim) -> (B, seq_len, hidden_dim)
        x_V = self.w3(x)

        # (B, seq_len, dim) -> (B, seq_len, hidden_dim)
        x = self.w1(x)
        # (B, seq_len, dim) -> (B, seq_len, hidden_dim)
        x = F.silu(x) # swish(x) = x * sigmoid(beta*x)

        # (B, seq_len, hidden_dim) * (B, seq_len, hidden_dim) -> element-wise multi
        x = x * x_V
        return self.w2(x)




class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # init gamma learnables(nn.Parameter) which are scaling invar
        # self.gamma renamed as self.weight, when loading weights? 
        self.weight = nn.Parameter(torch.ones(dim)) # assoc with each feat

    def _norm(self, x: torch.Tensor):
        '''
            - Calculates the RMSNorm factor i.e everything except the "gamma"
            - "gamma": scaling invariant
        '''
        # Recall: a / RMS(a) = a * 1 / sqrt(X), X = (x1**2 + x2**2 + ...+ xN**2) / N
         # - RMS(a) = sqrt( (x1**2 + x2**2 + .. + xN**2) / N )
        # torch.rsqrt(x) = 1 / sqrt(x)
        x = x.pow(2)
        x = x.mean(dim=-1, keepdim=True) # Want: B=1, seq_len=1, [[[ 1 2 3 4 ... 2048 ]]] => [[[ Avg. ]]]
        x = x + self.eps # broadcast to avoid div by 0
        # (B, seq_len, dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        return x * torch.rsqrt(x)

    def forward(self, x: torch.Tensor):
        # (dim) * (B, seq_len, dim) -> (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)




class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        

        self.attention_norm = RMSNorm(self.dim, args.norm_eps)
        self.attention = SelfAttention(args)
        self.ffn_norm = RMSNorm(self.dim, args.norm_eps)
        self.feed_forward = FeedForward(args)

    def forward(self, \
                x: torch.Tensor,
                polar_form_matrix: torch.Tensor, 
                start_pos: int):
        # (B, seq_len, dim) -> (B, seq_len, dim)
        # x_normalised = self.attention_norm.forward(x) # don't make a new copy as we won't use x_norm later
        
        x_cat = x + self.attention.forward( 
                                   self.attention_norm.forward(x), \
                                   start_pos, 
                                   polar_form_matrix
                                   )
        x = self.ffn_norm(x_cat)
        x = x_cat + self.feed_forward.forward(x)
        return x

class Transformer(nn.Module):
    '''
        All the layers except the last i.e softmax
    '''
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab Size isn't set"

        self.args = args
        self.dim = args.dim
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.norm_eps = args.norm_eps

        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim) # for each token in vocab -> dim sized

        # Nx of these Encoder Blocks, here 32 repeats in Llama
        # Recall dotted part
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append( EncoderBlock(self.args) )

        # RMSNorm
        self.norm = RMSNorm(self.dim, self.norm_eps)
        # Linear Layer i.e projection to output
        self.output = nn.Linear(args.dim, args.vocab_size, bias = False)

        # Precompute just the m_theta matrix
        # Each element is m_i * theta_j
        # Recall diagram in the paper -> Figure 1 of the paper
        # If a token occurs at position m_i and for the angle the encoding makes
        # with positive X-axis be theta_j, it can be encoded by the angle m_i * theta_j

        self.m_theta = precompute_m_theta_matrix( \
                                                   head_dim= args.dim // args.n_heads, \
                                                   seq_len= args.max_seq_len * 2, # (inpt + prompt)  
                                                   device = args.device, 
                                                   theta = 10000.0     
                                                )


    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape

        assert seq_len == 1, "1 Token at a time, Hence, seq_len needs to be 1, We aren't training, rather simply inference this"

        # (B, seq_len)
        x = self.tok_embeddings(tokens)
        
        # These token Embeddings are simply sent to layers i.e N EncoderBlocks
        # Since, the dim is fixed for a particular variant i.e Here in 7b model, it is 4096
        # Only thing changes is seq_len: context-window, we are interested in a givenm slice only
        freqs_complex = self.m_theta[start_pos: start_pos+seq_len] # for now, single token imagine

        # Apply for the N_layers
        for layer in self.layers:
            x = layer(x, freqs_complex, start_pos)


        # out of the n layers i.e dotted part in fig
        x = self.norm(x)

        # Project inorder to apply softmax and get best predicted token
        out = self.output(x).float()
        return out
