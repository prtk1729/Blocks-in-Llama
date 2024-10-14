

from pathlib import Path
from model import *
from sentencepiece import SentencePieceProcessor
import time
import tqdm
import json

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



if __name__ == "__main__":
    allow_cuda = False # set when we want to use the available gpu
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    LLaMa.build( 
                checkpoints_dir="llama2-7b",
                tokenizer_path="tokenizer.model",  \
                max_batch_size=3,
                max_seq_len=1024,
                load_model=True,
                device = device
               )