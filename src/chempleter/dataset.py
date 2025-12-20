import pandas as pd
import json
import torch
import selfies as sf
from torch.utils.data import Dataset

class ChempleterDataset(Dataset):
    def __init__(self, selfies_file,stoic_file):
        
        selfies_dataframe = pd.read_csv(selfies_file)
        self.data = selfies_dataframe["selfies"].to_list()
        with open(stoic_file) as f:
            self.selfies_to_integer = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        molecule = self.data[index]
        symbols_molecule = ["[START]"]+list(sf.split_selfies(molecule))+["[END]"]
        integer_molecule = [self.selfies_to_integer[symbol] for symbol in symbols_molecule]
        return torch.tensor(integer_molecule,dtype=torch.long)