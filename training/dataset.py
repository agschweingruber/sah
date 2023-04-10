import numpy as np
import torch
from torch.utils.data import Dataset


class ICUDataset(Dataset):
    
    def __init__(self, df, pat_ids, config, mode = "train", hours = 14*24):
        self.df = df
        self.pat_ids = pat_ids
        self.mode = mode
        self.hours = hours
        self.config = config
        
    def __len__(self):
        return len(self.pat_ids)
    
    def __getitem__(self, idx):
        pat_id = self.pat_ids[idx]
        sel = self.df[self.df["Fallnummer"] == pat_id].copy()
        
        X = sel[self.config.features].values
        temp = np.zeros((14*24, X.shape[1]))
        temp[:X.shape[0], :] = X
        X = temp
        
        if(self.mode == "train"):
            if(np.random.randint(100) > 50):
                X[np.random.randint(24, self.hours):, :] = 0
        else:
            X[self.hours:, :] = 0
            
        y = sel[self.config.target].values[0]

        return torch.tensor(X).float(), torch.tensor(y).long()
    