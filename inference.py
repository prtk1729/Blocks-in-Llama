

from pathlib import Path
from model import *
from sentencepiece import SentencePieceProcessor
import time
from tqdm import tqdm
import json
from typing import Optional

import warnings
warnings.filterwarnings("ignore")


class LLaMa:

    def __init__(self, \
                model: Transformer, \
                tokenizer: SentencePieceProcessor, 
                model_args: ModelArgs):
        self.model = model
        self.model_args = model_args
        self.tokenizer = tokenizer

    # staticmethod -> So, that I can call this w/o instantiating from other places
    @staticmethod 
    def build( 
                checkpoints_dir: str, \
                tokenizer_path: str, \
                max_batch_size: int,
                max_seq_len: int, 
                load_model: bool,
                device: str              
              ):
        '''

            Verify if all the keys of dict are present -> strict = True
            Loads the model weights
            Also, the model_args keys are initialised by input from user
        '''
        prev_time = time.time() # to timestamp, for checking time-taken to load the model-weights

        if load_model:
            checkpoints = sorted( Path(checkpoints_dir).glob("*.pth") )
            # check if folder empty -> When training for first time
            assert len(checkpoints) > 0, "no checkpoints file to load from"
            chk_path = checkpoints[0]
            print(f"Loading the checkpoint file: {chk_path}")

            checkpoint = torch.load(chk_path, map_location=device) # dict with keys of weights, values as tensor-wts
            print(f"Time taken to load the checkpoint: {time.time() - prev_time:.2f} sec") # in sec
            prev_time = time.time()

        # params.json file
        with open( Path(checkpoints_dir) / "params.json", "r") as fp:
            params = json.loads(fp.read())

        # print( params )
        # print( "NOTE: vocab_size isn't set yet", params["vocab_size"] )

        # vocab_size info based on the tokenizer, we are using
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        vocab_size = tokenizer.vocab_size() # All tokenizers, after loading, can be inferrd abt the vocab

        # # init the model_args
        model_args = ModelArgs( max_batch_size=max_batch_size, \
                                max_seq_len=max_seq_len, 
                                device = device, \
                                **params
                                        )
        model_args.vocab_size = vocab_size


        # We may need to set the default dtypes, for precision vs inference time tradeoffs
        if device == "cuda":
            torch.set_default_tensor_type( torch.cuda.HalfTensor )
        else:
            torch.set_default_tensor_type( torch.cuda.BFloat16Tensor )

        #### Explicitly setting n_kv_heads
        model_args.n_kv_heads = 32

        # print( model_args.__dict__ )            

        # # Now, we can init Transformer
        model = Transformer( model_args ).to(device)
        print( "DEVICE", device )

        # Verify (Nice trick to know if any key of params is renamed)
        # load_state_dict
        if load_model:
            # polar_for_matrix can be ignored as we are computing it anyway
            # Moreover, these aren't learnable
            del checkpoint["rope.freqs"]
            model.load_state_dict(state_dict= checkpoint, \
                                  strict = True)
            
            print( f"Loaded state dict in {time.time() - prev_time:.2f} sec" )
            prev_time = time.time()

        return LLaMa(model, tokenizer, model_args)

    def _sample_top_p(self, probs: torch.Tensor, p: float):
        '''
            We want the minimal set of tokens, such that
            cumsum( prob(tokens) ) > p
        '''
        prob_score, original_token_pos = torch.sort(probs, dim = -1, descending=True)
        prob_cumsum = torch.cumsum(prob_score, dim=-1) # prefixSum

        # mask the remaining i.e 0 them out -> renormalise the rest [ i.e make rest the new distribution ]
        # mask the ones which are have cumsum > p
        # property -> excluding ith guy, i.e somewhere before this "i", cumsum[:i-1] > p
        # cumsum[i] - prob_score[i] > p
        # cumsum[i] - prob_score[i] > p -> This ith guy needs to be masked out
        mask_ids = (prob_cumsum - prob_score > p) # F F F .. T T T .. T
        prob_score[mask_ids] = 0.0 # 0 the ones which are T

        # renormalise for the ones which were left i.e that's the new distribution
        div_term = prob_score.sum(dim=-1, keepdim=True)
        prob_score.div_(div_term)

        # Now, we have resolved the issue with top-k sampling with this idea
        # Only required tokens will creep in.
        # sample from this new distribition
        # sample from this distribution s.t chance of picking a sample is its prob_score
        # Returns the sampled id in the prob_score--ordering -> id
        next_token_idx = torch.multinomial(prob_score, num_samples=1) 

        # original idx info is lost, as we had sorted them
        # Get the token's position in the vocab, based on this id called "next_token_idx"
        # Based on the prob_score_ordering which was sampled i.e "id", get the original_token_pos in vocab
        # original_token_pos[ id ] -> Where the token is present in vocab
        original_token_pos = torch.gather(input = original_token_pos, dim = -1, index = next_token_idx)
        return original_token_pos


    def set_(self, max_gen_len: int, max_batch_size: int):
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len - 1
        # Now max_gen_len is set
        assert max_gen_len <= self.model_args.max_seq_len - 1, f"Max generated len of response should be less than {self.model_args.max_seq_len - 1}"
        # Check whether num of prompts <= max_batch_size or not
        assert max_batch_size <= self.model_args.max_batch_size, f"Max Batch size should be less than {self.model_args.max_batch_size}"
        return max_gen_len, max_batch_size

    def populate_tokens_(self, \
                         prompt_tokens, \
                         max_batch_size, \
                         max_prompt_plus_gen_len):
        # 1st we populate the tenmsor with all padding tokens
        pad_id = self.tokenizer.pad_id()

        tokens = torch.full( size = (max_batch_size, max_prompt_plus_gen_len), \
                   fill_value= pad_id,
                    device = device,
                     dtype = torch.long )

        # Overwrite with actual prompt tokens
        for prompt_id, prompt_token in enumerate(prompt_tokens):
            # prompt_id == batch_id here
            # tokens[ prompt_id, : len(prompt_token) ] = prompt_token
            tokens[ prompt_id, : len(prompt_token) ] = torch.tensor(prompt_token, dtype=torch.long, device = device)

        return tokens
        
        
    def _predict_next_tokens(self, \
                            top_p, 
                            temperature,
                            tokens, 
                            eos_reached_flag, 
                            prompt_token_mask, 
                            max_prompt_plus_gen_len
                            ):
        
        # Iterating on the position in the sequences
        # Each token can be of max_len = max_prompt_plus_gen_len
        cur_iterator = tqdm( range(1, max_prompt_plus_gen_len ), desc = "Generating tokens" )

        for cur_pos in cur_iterator:
            with torch.no_grad():
                # Terminating condition
                eos_reached_flag |= ( ~prompt_token_mask[:, cur_pos] & ( tokens[:, cur_pos] == self.tokenizer.eos_id ) )
                if all(eos_reached_flag):
                    break

                # get the logits by one forward pass
                start_pos = cur_pos

                # Previous tokens can bve inferred from kv_cache [: cur_pos-2]
                logits = self.model( tokens = tokens[:, cur_pos-1: cur_pos], \
                        start_pos = start_pos ) # Recall we send one token at a time, during inference
                
                # Apply inference strategies
                if temperature > 0:
                    probs = torch.softmax( logits[:, -1] / temperature, dim = -1)
                    next_token_ids = self._sample_top_p( probs, top_p )
                else:
                    # last output layer shape: (dim, vocab_size) -> same shape of logits
                    next_token_ids = torch.argmax( logits[:, -1], dim = -1 ) # id in the vocab

                # Reshape (B, 1) -> (B, )
                next_token_ids = next_token_ids.reshape(-1) # flatten

                assert prompt_token_mask[:, cur_pos].shape == next_token_ids.shape, f"Shape mismatch with {prompt_token_mask}"
                assert tokens[:, cur_pos].shape == next_token_ids.shape, f"Shape mismatch with {tokens}"

                next_token_ids = torch.where( prompt_token_mask[:, cur_pos], tokens[:, cur_pos], next_token_ids ) # like a ternary cond ? from_1 : from_2

                # place at correct position, wait... 
                # If the cur_pos points to a prompt_token, is the prediction of any use?
                # No, we should use it as it is.... take care of this before overwriting
                tokens[:, cur_pos] = next_token_ids

        return tokens


    def text_completion(self, \
                         top_p: float,
                         prompts: list[str], 
                         temperature: float = 0.6, 
                         max_gen_len: Optional[int] = None, # Optional will always have default val
                         ):
        
        # Encode the tokens in the prompts
        # Each prompt is a str. encode each string to list of ints(token_ids)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        max_batch_size = len(prompt_tokens)

        max_gen_len, max_batch_size = self.set_(max_gen_len, max_batch_size)

        max_prompt_len = max(len(tokens) for tokens in prompt_tokens)
        assert max_prompt_len <= self.model_args.max_seq_len, f"Max Prompt len can't be more than {self.model_args.max_seq_len}"
        max_prompt_plus_gen_len = min( self.model_args.max_seq_len,  max_prompt_len + max_gen_len)

        # Populate tokens / Init tokens 
        tokens = self.populate_tokens_(
                                       prompt_tokens, 
                                       max_batch_size, 
                                       max_prompt_plus_gen_len
                                       )


        # For each batch_id / prompt_id, check whether eos is reached?
        eos_reached_flag = torch.tensor( [False] * max_batch_size, device = device )
        pad_id = self.tokenizer.pad_id()
        prompt_token_mask = (tokens != pad_id) # Whether a token is pad / prompt_token

        # For each prompt, we predict the next token, iteratively using 
            # Greedy and top_p sampling strategies
        tokens = self._predict_next_tokens(top_p, 
                                           temperature,
                                           tokens, 
                                           eos_reached_flag, 
                                           prompt_token_mask, 
                                           max_prompt_plus_gen_len)

        # Decode tokens
        out_tokens, out_text = [], []

        for prompt_id, prompt_token in enumerate(tokens.tolist()):
            
            if self.tokenizer.eos_id in prompt_token:
                eos_id = self.tokenizer.eos_id
                eos_index = prompt_token.index(eos_id) # index where this token_id is present
                # slice uptil that point -> rest are paddings
                prompt_token = prompt_token[: eos_index ]

            out_tokens.append(prompt_token)
            out_text.append(self.tokenizer.decode(prompt_token))
        return (out_tokens, out_text)


if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = True # set when we want to use the available gpu
    device = "cuda:0" if torch.cuda.is_available() and allow_cuda else "cpu"

    torch.cuda.empty_cache()

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",

        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Prateek Pani
        Decision: 
        """
    ]

    # model = LLaMa.build( 
    #             checkpoints_dir="llama2-7b",
    #             tokenizer_path="tokenizer.model",  \
    #             max_batch_size=3,
    #             max_seq_len=1024,
    #             load_model=True,
    #             device = device
    #            )

    
    model = LLaMa.build( 
                checkpoints_dir = "llama2-7b",
                tokenizer_path = "tokenizer.model",  \
                max_batch_size = len(prompts),
                max_seq_len = 1024,
                load_model = True,
                device = device
               )

    # Inference
    out_tokens, out_texts = (model.text_completion(prompts=prompts, max_gen_len=64, top_p=0.9, temperature=0.6))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)