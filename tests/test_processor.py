# Test for chemplter.processor

import pytest
import pandas as pd
import json
from chempleter.processor import generate_input_data, _selfies_encoder


class TestChempeleterProcessor():
    def test_selfies_encoder_valid(self):
        """
        Test for selfies encoder when a correct selfies string is given.
        """
        res, err = _selfies_encoder("CCO")
        assert isinstance(res, str)
        assert err == "No error"

    def test_selfies_encoder_invalid(self):
        """
        Test for selfies encoder when an icorrect selfies string is given.
        """
        res, err = _selfies_encoder("invalid_smiles")
        assert pd.isna(res)
        assert err != "No error"

    def test_generate_input_data_success(self,tmp_path):
        """
        Test for generate_input_data when tmp_path is valid.
        Checks whether the files are created correctly.
        Checks if special tokens are present in stoi
        """
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"smiles": ["CCO", "C", "CC"]})
        df.to_csv(csv_path, index=False)

        selfies_file, stoi_file, itos_file = generate_input_data(csv_path, working_dir=tmp_path)

        assert selfies_file.exists()
        assert stoi_file.exists()
        assert itos_file.exists()

        with open(stoi_file, "r") as f:    
            stoi = json.load(f)
            assert "[PAD]" in stoi
            assert "[START]" in stoi
            assert "[END]" in stoi
        with open(itos_file, "r") as f:
            itos = json.load(f)
            assert len(itos) == len(stoi)
            assert set(itos) == set(stoi.keys())

    def test_generate_input_data_missing_column(self,tmp_path):
        csv_path = tmp_path / "bad_test.csv"
        df = pd.DataFrame({"smile": ["CCO"]})
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Column `smiles` not found in the CSV file."):
            generate_input_data(csv_path, working_dir=tmp_path)

    def test_generate_input_data_file_not_found(self,tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_input_data(tmp_path / "nonexistent.csv", working_dir=tmp_path)

    def test_generate_input_data_invalid_working_dir(self,tmp_path):
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"smiles": ["C"]}).to_csv(csv_path, index=False)
        
        with pytest.raises(FileNotFoundError):
            generate_input_data(csv_path, working_dir=tmp_path / "tmp_folder")