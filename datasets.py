import pandas as pd
import torch
from ast import literal_eval
import numpy as np
from torch.utils.data import Dataset

class HFLunarLander(Dataset):
    def __init__(self, df) -> None:
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        states1 = torch.tensor(np.array(self.df.iloc[index]['states']))
        actions1 = torch.tensor(np.array(self.df.iloc[index]['actions']))
        label1 = torch.tensor(self.df.iloc[index]['label'])

        return states1.to(torch.float32), actions1.to(torch.float32), label1.to(torch.float32)
