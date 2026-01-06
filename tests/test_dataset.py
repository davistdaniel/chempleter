# Tests for chempleter.dataset module


import pytest
import torch
import pandas as pd
from chempleter.dataset import (
    ChempleterDataset,
    ChempleterRandomisedSmilesDataset,
    ChempleterRandomisedBridgeDataset,
    collate_fn,
    get_dataloader,
)


class TestChempleterDataset:
    """Test for Chempleter dataset class"""

    def test_dataset_init(self,sample_selfies_csv,sample_stoi_file):
        """Test if length of dataset is loaded correctly"""
        ds = ChempleterDataset(selfies_file=sample_selfies_csv,stoi_file=sample_stoi_file)

        assert ds is not None
        assert len(ds) > 0 # see sample_selfies_csv in conftest.py

        for i in range(min(3, len(ds))):
            item = ds[i]
            assert isinstance(item, torch.Tensor)
            assert len(item) > 0

    def test_dataset_length(self,sample_selfies_csv,sample_stoi_file):
        """Test if length of loaded dataset is correct"""
        ds = ChempleterDataset(selfies_file=sample_selfies_csv,stoi_file=sample_stoi_file)
        df  = pd.read_csv(sample_selfies_csv)
        assert len(ds) == len(df)

    def test_dataset_get_item(self,sample_selfies_csv,sample_stoi_file):
        """Test dataset get item"""
        ds = ChempleterDataset(selfies_file=sample_selfies_csv,stoi_file=sample_stoi_file)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.dtype == torch.long
        assert len(item) > 0

    def test_dataset_get_item_structure(self,sample_selfies_csv,sample_stoi_file,sample_stoi):
        """Test dataset get item, if start and end tokens are at correct positions"""
        ds = ChempleterDataset(selfies_file=sample_selfies_csv,stoi_file=sample_stoi_file)
        stoi = sample_stoi
        item = ds[0]
        assert item[0] == stoi["[START]"]
        assert item[-1] == stoi["[END]"]

class TestChempleterRandomisedDataset:
    """Test for Chempleter dataset class"""

    def test_dataset_init(self,sample_smiles_csv,sample_stoi_file):
        """Test if length of dataset is loaded correctly"""
        ds = ChempleterRandomisedSmilesDataset(smiles_file=sample_smiles_csv,stoi_file=sample_stoi_file)

        assert ds is not None
        assert len(ds) > 0 # see sample_selfies_csv in conftest.py

        for i in range(min(3, len(ds))):
            item = ds[i]
            assert isinstance(item, torch.Tensor)
            assert len(item) > 0

    def test_dataset_length(self,sample_smiles_csv,sample_stoi_file):
        """Test if length of loaded dataset is correct"""
        ds = ChempleterRandomisedSmilesDataset(smiles_file=sample_smiles_csv,stoi_file=sample_stoi_file)
        df  = pd.read_csv(sample_smiles_csv)
        assert len(ds) == len(df)

    def test_dataset_get_item(self,sample_smiles_csv,sample_stoi_file):
        """Test dataset get item"""
        ds = ChempleterRandomisedSmilesDataset(smiles_file=sample_smiles_csv,stoi_file=sample_stoi_file)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.dtype == torch.long
        assert len(item) > 0

    def test_dataset_get_item_structure(self,sample_smiles_csv,sample_stoi_file,sample_stoi):
        """Test dataset get item, if start and end tokens are at correct positions"""
        ds = ChempleterRandomisedSmilesDataset(smiles_file=sample_smiles_csv,stoi_file=sample_stoi_file)
        stoi = sample_stoi
        item = ds[0]
        assert item[0] == stoi["[START]"]
        assert item[-1] == stoi["[END]"]

    def test_dataset_randomisation(self, sample_smiles_csv, sample_stoi_file,sample_itos):
        """Test that dataset randomises SMILES."""
        ds = ChempleterRandomisedSmilesDataset(smiles_file=sample_smiles_csv,stoi_file=sample_stoi_file)
        # Get same item multiple times - should potentially be different due to randomisation
        items = [ds[0] for _ in range(3)]
        assert len(set(items)) != 1

class TestChempleterBridgeDataset:
    """Test for Chempleter dataset class"""

    def test_dataset_init(self,sample_smiles_csv,sample_stoi_file):
        """Test if length of dataset is loaded correctly"""
        ds = ChempleterRandomisedBridgeDataset(smiles_file=sample_smiles_csv,stoi_file=sample_stoi_file)

        assert ds is not None
        assert len(ds) > 0 # see sample_selfies_csv in conftest.py

        for i in range(min(3, len(ds))):
            item = ds[i]
            assert isinstance(item, torch.Tensor)
            assert len(item) > 0

    def test_dataset_length(self,sample_smiles_csv,sample_stoi_file):
        """Test if length of loaded dataset is correct"""
        ds = ChempleterRandomisedBridgeDataset(smiles_file=sample_smiles_csv,stoi_file=sample_stoi_file)
        df  = pd.read_csv(sample_smiles_csv)
        assert len(ds) == len(df)

    def test_dataset_get_item(self,sample_smiles_csv,sample_stoi_file):
        """Test dataset get item"""
        ds = ChempleterRandomisedBridgeDataset(smiles_file=sample_smiles_csv,stoi_file=sample_stoi_file)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.dtype == torch.long
        assert len(item) > 0

    def test_dataset_get_item_structure(self,sample_smiles_csv,sample_stoi_file,sample_stoi):
        """Test dataset get item, if start and end tokens are at correct positions"""
        ds = ChempleterRandomisedBridgeDataset(smiles_file=sample_smiles_csv,stoi_file=sample_stoi_file)
        stoi = sample_stoi
        item = ds[0]
        assert item[0] == stoi["[START]"]
        assert item[-1] == stoi["[END]"]
        assert stoi["[MASK]"] in item
        assert stoi["[BRIDGE]"] in item

    def test_dataset_randomisation(self, sample_smiles_csv, sample_stoi_file,sample_itos):
        """Test that dataset randomises SMILES."""
        ds = ChempleterRandomisedBridgeDataset(smiles_file=sample_smiles_csv,stoi_file=sample_stoi_file)
        items = [ds[0] for _ in range(3)]
        assert len(set(items)) != 1 # each sequence must be unique in order, if randomisation was successful.

class TestCollateFn:

    def test_collate_fn_init(self):
        """Test collate function. init"""
        batch = [
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2]),
        ]
        padded_batch, tensor_lengths = collate_fn(batch)
        
        assert isinstance(padded_batch, torch.Tensor)
        assert isinstance(tensor_lengths, torch.Tensor)
        assert padded_batch.shape[0] == len(batch)
        assert len(tensor_lengths) == len(batch)
        assert padded_batch.shape[1] == max(len(x) for x in batch)

    def test_collate_fn_padding(self):
        """Test that collate function pads sequences correctly."""
        batch = [
            torch.tensor([1, 2]),
            torch.tensor([1, 2, 3, 4]),
        ]
        padded_batch, tensor_lengths = collate_fn(batch)
        
        # check padding value is 0, due to sorting the order would be sorted in descending
        # so sequence at 1 has to be checked
        assert padded_batch[1, 2].item() == 0
        assert padded_batch[1, 3].item() == 0

    def test_collate_fn_sorting(self):
        """Test that collate function sorts sequences by length in descending order."""
        batch = [
            torch.tensor([1, 2]),
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([1, 2, 3]),
        ]
        padded_batch, tensor_lengths = collate_fn(batch)
        
        # check that lengths are sorted in descending order
        assert tensor_lengths[0].item() == 5
        assert tensor_lengths[1].item() == 3
        assert tensor_lengths[2].item() == 2

class TestGetDataloader:
    """Test cases for get_dataloader function."""

    def test_get_dataloader(self, sample_selfies_csv, sample_stoi_file):
        """Test creating a dataloader."""
        ds = ChempleterDataset(selfies_file=sample_selfies_csv, stoi_file= sample_stoi_file)
        dataloader = get_dataloader(ds, batch_size=2, shuffle=True)
        
        assert dataloader is not None
        assert dataloader.batch_size == 2

    def test_get_dataloader_batch_iteration(self, sample_selfies_csv, sample_stoi_file):
        """Test iterating through dataloader batches."""
        ds = ChempleterDataset(selfies_file=sample_selfies_csv, stoi_file= sample_stoi_file)
        dataloader = get_dataloader(ds, batch_size=2, shuffle=False)
        
        batch_count = 0
        for batch_tuple in dataloader:
            batch, lengths = batch_tuple
            assert batch.shape[0] == dataloader.batch_size
            assert isinstance(batch, torch.Tensor)
            assert isinstance(lengths, torch.Tensor)
            batch_count += 1
            if batch_count >= 2: 
                break
        
        assert batch_count > 0
