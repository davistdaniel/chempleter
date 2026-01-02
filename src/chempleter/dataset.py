import json
import torch
import logging
import random
import selfies as sf
import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class ChempleterDataset(Dataset):
    """
    PyTorch Dataset for SELFIES molecular representations.

    :param selfies_file: Path to CSV file containing SELFIES strings in a "selfies" column.
    :type selfies_file: str
    :param stoi_file: Path to JSON file mapping SELFIES symbols to integer tokens.
    :type stoi_file: str
    :returns: Integer tensor representation of tokenized molecule with dtype=torch.long.
    :rtype: torch.Tensor
    """

    def __init__(self, selfies_file, stoi_file):
        super().__init__()
        selfies_dataframe = pd.read_csv(selfies_file)
        self.data = selfies_dataframe["selfies"].to_list()
        with open(stoi_file) as f:
            self.selfies_to_integer = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        molecule = self.data[index]
        symbols_molecule = ["[START]"] + list(sf.split_selfies(molecule)) + ["[END]"]
        integer_molecule = [
            self.selfies_to_integer[symbol] for symbol in symbols_molecule
        ]
        return torch.tensor(integer_molecule, dtype=torch.long)


class ChempleterRandomisedSmilesDataset(Dataset):
    """
    PyTorch Dataset for SELFIES molecular representations.

    :param smiles_file: Path to CSV file containing SMILES strings in a "smiles" column.
    :type smiles_file: str
    :param stoi_file: Path to JSON file mapping SELFIES symbols to integer tokens.
    :type stoi_file: str
    :returns: Integer tensor representation of tokenized molecule with dtype=torch.long.
    :rtype: torch.Tensor
    """

    def __init__(self, smiles_file, stoi_file):
        super().__init__()
        smiles_dataframe = pd.read_csv(smiles_file)
        self.data = smiles_dataframe["smiles"].to_list()
        with open(stoi_file) as f:
            self.selfies_to_integer = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        molecule_in_smiles = self.data[index]

        # try randomisation
        molecule = Chem.MolFromSmiles(molecule_in_smiles)
        if molecule is not None:
            try:
                molecule_in_selfies = sf.encoder(
                    Chem.MolToSmiles(molecule, canonical=False, doRandom=True)
                )
            except Exception as e:
                logging.error(f"SELFIES encoding error for randomised SMILES: {e}")
        else:
            molecule_in_selfies = sf.encoder(molecule_in_smiles)

        symbols_molecule = (
            ["[START]"] + list(sf.split_selfies(molecule_in_selfies)) + ["[END]"]
        )
        integer_molecule = []

        # check if all symbols exist in stoi
        for symbol in symbols_molecule:
            if symbol not in self.selfies_to_integer:
                raise RuntimeError(
                    f"Molecule symbol not found in stoi. Add {symbol} in stoi with correct integer mapping."
                )
            else:
                integer_molecule.append(self.selfies_to_integer[symbol])

        return torch.tensor(integer_molecule, dtype=torch.long)


class ChempleterRandomisedBridgeDataset(Dataset):
    """
    PyTorch Dataset for SELFIES molecular representations.

    :param smiles_file: Path to CSV file containing SMILES strings in a "smiles" column.
    :type smiles_file: str
    :param stoi_file: Path to JSON file mapping SELFIES symbols to integer tokens.
    :type stoi_file: str
    :returns: Integer tensor representation of tokenized molecule with dtype=torch.long.
    :rtype: torch.Tensor
    """

    def __init__(self, smiles_file, stoi_file):
        super().__init__()
        smiles_dataframe = pd.read_csv(smiles_file)
        self.data = smiles_dataframe["smiles"].to_list()
        with open(stoi_file) as f:
            self.selfies_to_integer = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        molecule_in_smiles = self.data[index]

        # try randomisation
        molecule = Chem.MolFromSmiles(molecule_in_smiles)
        if molecule is not None:
            try:
                molecule_in_selfies = sf.encoder(
                    Chem.MolToSmiles(molecule, canonical=False, doRandom=True)
                )
            except Exception as e:
                molecule_in_selfies = sf.encoder(molecule_in_smiles)
                logging.error(f"SELFIES encoding error for randomised SMILES: {e}")
        else:
            molecule_in_selfies = sf.encoder(molecule_in_smiles)

        symbols = list(sf.split_selfies(molecule_in_selfies))

        # try fragmentation
        if len(symbols) > 10:
            len_frag1 = random.randint(1, len(symbols) - 8)
            len_bridge = random.randint(
                1, 10
            )  # bridge len ranges from 1 to 10.# this would be constraint later
            len_frag2 = min(len_frag1 + len_bridge, len(symbols) - 1)
            frag1 = symbols[:len_frag1]
            bridge = symbols[len_frag1:len_frag2]
            frag2 = symbols[len_frag2:]

        else:
            frag1 = symbols
            bridge = []
            frag2 = []

        symbols_molecule = (
            ["[START]"] + frag1 + ["[MASK]"] + frag2 + ["[BRIDGE]"] + bridge + ["[END]"]
        )
        integer_molecule = []

        # check if all symbols exist in stoi
        for symbol in symbols_molecule:
            if symbol not in self.selfies_to_integer:
                raise RuntimeError(
                    f"Molecule symbol not found in stoi. Add {symbol} in stoi with correct integer mapping."
                )
            else:
                integer_molecule.append(self.selfies_to_integer[symbol])

        return torch.tensor(integer_molecule, dtype=torch.long)


def collate_fn(batch):
    """
    Collate function for a PyTorch DataLoader.
    Sorts the incoming batch by sequence length in descending order, pads the sequences
    to the same length (batch_first=True, padding_value=0) using torch.nn.utils.rnn.pad_sequence,
    and returns the padded batch together with the sorted original lengths.
    :param batch: Iterable of 1D tensors representing variable-length sequences.
    :type batch: list[torch.Tensor]
    :returns: A tuple (padded_batch, tensor_lengths) where padded_batch is a 2D tensor
              of shape (batch_size, max_seq_len) containing padded sequences, and
              tensor_lengths is a 1D tensor of original sequence lengths sorted in
              descending order.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    tensor_lengths = torch.tensor([len(x) for x in batch])
    tensor_lengths, sorted_idx = tensor_lengths.sort(descending=True)
    batch = [batch[i] for i in sorted_idx]

    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)

    return padded_batch, tensor_lengths


def get_dataloader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn):
    """
    Create a PyTorch DataLoader.

    :param dataset: PyTorch Dataset instance.
    :type dataset: torch.utils.data.Dataset
    :param batch_size: Number of samples per batch.
    :type batch_size: int
    :param shuffle: Whether to shuffle the data each epoch.
    :type shuffle: bool
    :param collate_fn: Function to merge a list of samples to form a mini-batch.
    :type collate_fn: callable
    :return: Configured DataLoader.
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
