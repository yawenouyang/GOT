from torch.utils.data import Dataset
import pandas as pd
import torch


class CLMDataset(Dataset):
    def __init__(self, fname, tokenizer, device):
        super().__init__()
        data = pd.read_csv(fname)
        text_out = list(data['utt'])
        text_in = [tokenizer.bos_token + txt for txt in text_out]
        token_in = tokenizer(text_in, return_tensors='pt', padding=True)
        token_out = tokenizer(text_out, return_tensors='pt', padding=True)
        self.X_in = token_in['input_ids'].to(device)
        self.X_out = token_out['input_ids'].to(device)
        self.mask = token_in['attention_mask'].to(device)
        self.len = [len(msk) for msk in tokenizer(text_out)['attention_mask']]
        self.y = torch.tensor(list(data['index'])).to(device)
        self.intent = list(data['intent'])
    
    def __getitem__(self, idx):
        return self.X_in[idx], self.X_out[idx], self.mask[idx], self.len[idx], self.y[idx], self.intent[idx]

    def __len__(self):
        return len(self.len)