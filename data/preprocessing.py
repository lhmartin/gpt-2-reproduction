import os
import tiktoken
import numpy as np

from datasets import load_dataset

def download_tokenize_and_shard_dataset(dataset_name : str,
                                        output_folder : str,
                                        remote_name : str):
    
    os.makedirs(output_folder, exist_ok=True)
    
    fw = load_dataset(dataset_name, name=remote_name, split='train')
    
    encoder = tiktoken.get_encoding('gpt2')
    eot = encoder._special_tokens['<|endoftext|>']
    
    def tokenize(document):
        
        tokens = [eot]
        tokens.extend(encoder.encode_ordinary(document['text']))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), 'Token dictionary contains invalid tokens'
        
        return tokens_np.astype(np.uint16)


    def write_datafile(filename, np_tokens):
        
        with open(filename, 'wb') as f:
            f.write(np_tokens.tobytes())
            
        
    def _loop_func():
        print()
        


if __name__ == '__main__':
    
    download_tokenize_and_shard_dataset('HuggingFaceFW/fineweb', './dataset_cache/', remote_name = 'sample-10BT')
    
