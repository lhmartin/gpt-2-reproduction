import torch
import tiktoken

class TokenDataset:
    def __init__(self, data_file : str, seq_len : int = 128):
        
        enc = tiktoken.get_encoding('gpt2')
        self.seq_len = seq_len
        
        with open(data_file, 'r') as f:
            text = f.read()
            
        self.tokens = torch.tensor(enc.encode(text))
        print(f'Loaded {len(self.tokens)} tokens')
        
        return
    
    def __len__(self):
        
        return len(self.tokens) // self.seq_len
    
    def __getitem__(self, idx : int):
        
        buf = self.tokens[idx * self.seq_len : idx * self.seq_len + self.seq_len + 1]
        
        x = buf[:-1]
        y = buf[1:]
        
        return x, y
    
if __name__ == "__main__":
    
    dataset = TokenDataset('data/input.txt')
    
    print(dataset[0])