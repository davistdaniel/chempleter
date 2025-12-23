import json
import torch
import selfies as sf
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class ChempleterDataset(Dataset):
    def __init__(self, selfies_file,stoi_file):
        super().__init__()
        selfies_dataframe = pd.read_csv(selfies_file)
        self.data = selfies_dataframe["selfies"].to_list()
        with open(stoi_file) as f:
            self.selfies_to_integer = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        molecule = self.data[index]
        symbols_molecule = ["[START]"]+list(sf.split_selfies(molecule))+["[END]"]
        integer_molecule = [self.selfies_to_integer[symbol] for symbol in symbols_molecule]
        return torch.tensor(integer_molecule,dtype=torch.long)

# def collate_fn(batch):
#     return pad_sequence(batch,batch_first=True)

def collate_fn(batch):

    tensor_lengths  = torch.tensor([len(x) for x in batch])
    tensor_lengths, sorted_idx = tensor_lengths.sort(descending=True)
    batch = [batch[i] for i in sorted_idx]
    
    padded_batch = pad_sequence(batch, batch_first=True,padding_value=0)

    return padded_batch, tensor_lengths

def get_dataloader(dataset,batch_size=64,shuffle=True,collate_fn=collate_fn):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn)