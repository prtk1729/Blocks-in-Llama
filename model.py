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

    assert head_dim % 2 != 0, "dim of head for query should be even as pre the paper"
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



class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_q_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads

        # Recall Multi-Query Attention:
        #    |     |     |    (n_v_heads)
        #    
        #    |     |     |    (n_k_heads)
        #   / \   / \   / \ 
        #  |   | |   | |   |  (n_q_heads)

        # How many times we repeat each key-head, value-head?
        self.q_head_group_size = self.n_q_heads // self.n_kv_heads
        

class RMSNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_eps = args.norm_eps
        # init gamma learnables(nn.Parameter) which are scaling invar
        self.gamma = nn.Parameter(torch.ones(args.dim)) # assoc with each feat

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
        x = x + self.norm_eps # broadcast to avoid div by 0
        # (B, seq_len, dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        return x * torch.rsqrt(x)

    def forward(self, x: torch.Tensor):
        # (dim) * (B, seq_len, dim) -> (B, seq_len, dim)
        return self.gamma * self._norm(x.float()).type_as(x)





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
        self.eps = args.eps

        self.tok_emb = nn.Embedding(self.vocab_size, self.dim) # for each token in vocab -> dim sized

        # Nx of these Encoder Blocks, here 32 repeats in Llama
        # Recall dotted part
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append( EncoderBlock(self.args) )

        # RMSNorm
        self.norm = RMSNorm(args)
        # Linear Layer i.e projection to output
        self.output = nn.LinearLayer(args.dim, args.vocab_size)

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


    def forward(self, x):
        # x here can have multiple heads
        # (B, seq_len, h, q_dim)

        