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

    def test_dataset_init(self,sample_seflies_csv,sample_stoi_file):
        """Test if length of dataset is loaded correctly"""
        ds = ChempleterDataset(selfies_file=sample_seflies_csv,stoi_file=sample_stoi_file)

        assert ds is not None
        assert len(ds) > 0 # see sample_selfies_csv in conftest.py

        for i in range(min(3, len(ds))):
            item = ds[i]
            assert isinstance(item, torch.Tensor)
            assert len(item) > 0

    def test_dataset_length(self,sample_seflies_csv,sample_stoi_file):
        """Test if length of loaded dataset is correct"""
        ds = ChempleterDataset(selfies_file=sample_seflies_csv,stoi_file=sample_stoi_file)
        df  = pd.read_csv(sample_seflies_csv)
        assert len(ds) == len(df)

    def test_dataset_get_item(self,sample_seflies_csv,sample_stoi_file):
        """Test dataset get item"""
        ds = ChempleterDataset(selfies_file=sample_seflies_csv,stoi_file=sample_stoi_file)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.dtype == torch.long
        assert len(item) > 0

    def test_dataset_get_item_structure(self,sample_seflies_csv,sample_stoi_file,sample_stoi):
        """Test dataset get item, if start and end tokens are at correct positions"""
        ds = ChempleterDataset(selfies_file=sample_seflies_csv,stoi_file=sample_stoi_file)
        stoi = sample_stoi
        item = ds[0]
        assert item[0] == stoi["[START]"]
        assert item[-1] == stoi["[END]"]