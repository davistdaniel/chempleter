import json
import tempfile
import pytest
import torch
import pandas as pd
import selfies as sf
from pathlib import Path
from chempleter.model import ChempleterModel


@pytest.fixture
def sample_selfies_csv(tmp_path):
    "create a temp selfies csv"

    selfies_dict = {"selfies":["[C][C][O]",
            "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
            "[C][C][=Branch1][C][=O][O]",
            "[C][C][C][Branch1][C][C][Branch1][C][O][C][C][C][Ring1][Ring1][C]"]}
    
    seflies_df = pd.DataFrame(selfies_dict)
    csv_path = tmp_path / "selfies.csv"
    seflies_df.to_csv(csv_path,index=False)

    return csv_path

@pytest.fixture
def sample_stoi_file(tmp_path,sample_selfies_csv):
    "create a temp selfies csv"

    selfies_list = pd.read_csv(sample_selfies_csv)["selfies"].to_list()
    alphabet = sf.get_alphabet_from_selfies(selfies_list)
    # Branch1 must be added since randomisation might cause this token to appear.
    sample_stoi = {j:i for i,j in enumerate(["[PAD]", "[START]", "[END]", "[MASK]", "[BRIDGE]"]+list(sorted(alphabet))+["[Branch1]"])}

    stoi_file = tmp_path / "stoi.json"

    with open(stoi_file,"w") as f:
        json.dump(sample_stoi,f)

    return stoi_file

@pytest.fixture
def sample_stoi(sample_stoi_file):
    "create a temp selfies csv"

    with open(sample_stoi_file,"r") as f:
        sample_stoi = json.load(f)
    
    return sample_stoi

@pytest.fixture
def sample_itos(sample_stoi):
    "create a temp selfies csv"

    return list(sample_stoi.keys())

@pytest.fixture
def sample_itos_file(tmp_path,sample_itos):
    "create a temp selfies csv"

    itos_file = tmp_path / "itos.json"

    with open(itos_file,"w") as f:
        json.dump(sample_itos,f)

    return itos_file

@pytest.fixture
def sample_smiles_csv(tmp_path):
    "create a temp smiles csv"

    smiles_dict = {
        "smiles": [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "CCC(C)(O)C1CC1C",
        ]
    }
    
    smiles_df = pd.DataFrame(smiles_dict)
    csv_path = tmp_path / "smiles.csv"
    smiles_df.to_csv(csv_path,index=False)

    return csv_path

@pytest.fixture
def device():
    return "cpu"