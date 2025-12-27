import pytest
import pandas as pd
import json
import torch
from chempleter.processor import generate_input_data, _selfies_encoder
from chempleter.model import ChempleterModel
from chempleter.dataset import ChempleterDataset, collate_fn
from chempleter.descriptors import calculate_descriptors
from chempleter.inference import handle_prompt, handle_len, handle_sampling, output_molecule, extend, _get_default_data
from rdkit import Chem

@pytest.fixture
def mock_stoi():
    return {"[START]": 0, "[END]": 1, "[PAD]": 2, "[C]": 3, "[O]": 4}

@pytest.fixture
def mock_itos():
    return ["[START]", "[END]", "[PAD]", "[C]", "[O]"]

@pytest.fixture
def mock_files(tmp_path):
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"selfies": ["[C][C]", "[O]"]}).to_csv(csv_file, index=False)
    
    stoi_file = tmp_path / "stoi.json"
    stoi = {"[PAD]": 0, "[START]": 1, "[END]": 2, "[C]": 3, "[O]": 4}
    with open(stoi_file, "w") as f:
        json.dump(stoi, f)
        
    return csv_file, stoi_file

@pytest.fixture
def model_params():
    return {
        "vocab_size": 20,
        "embedding_dim": 16,
        "hidden_dim": 32,
        "num_layers": 2
    }

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
        
        :param tmp_path: Path to save files
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

class TestChempeleterModel():
    def test_model_initialization(self,model_params):
        """Test if model is initialized correctly"""
        model = ChempleterModel(**model_params)
        assert model.embedding.num_embeddings == model_params["vocab_size"]
        assert model.gru.hidden_size == model_params["hidden_dim"]
        assert model.fc.out_features == model_params["vocab_size"]

    def test_forward_pass_shape(self, model_params):
        """Test Forward pass shapes"""
        model = ChempleterModel(**model_params)
        batch_size = 4
        max_len = 10
        
        # prepare a padded batch
        # matrix of 4x10, random ints, range starts at 1
        x = torch.randint(low=1, high=model_params["vocab_size"], size=(batch_size, max_len))
        lengths = torch.tensor([10, 8, 5, 2])

        # pad zeros everywhere else
        x[0, lengths[0]:] = 0 
        x[1, lengths[1]:] = 0
        x[2, lengths[2]:] = 0
        x[3, lengths[3]:] = 0

        logits, hidden = model(x, lengths)

        assert logits.shape == (batch_size, max_len, model_params["vocab_size"])
        assert hidden.shape == (model_params["num_layers"], batch_size, model_params["hidden_dim"])

    def test_forward_pass_with_hidden(self,model_params):
        """
        Test forward pass with hidden
        """
        model = ChempleterModel(**model_params)
        batch_size = 2
        x = torch.randint(1, model_params["vocab_size"], (batch_size, 3))
        lengths = torch.tensor([3, 3])
        
        hidden_in = torch.zeros(model_params["num_layers"], batch_size, model_params["hidden_dim"])
        logits, hidden_out = model(x, lengths, hidden_in)
        
        assert not torch.equal(hidden_in, hidden_out)

    def test_enforce_sorted_error(self, model_params):
        """
        Test of the lengths are sorted in descending
        """
        model = ChempleterModel(**model_params)
        x = torch.randint(1, model_params["vocab_size"], (2, 10))
        unsorted_lengths = torch.tensor([5, 8])
        
        with pytest.raises(RuntimeError):
            model(x, unsorted_lengths)

class TestChempeleterDataset():
    def test_dataset_length(self,mock_files):
        """
        Test the length of output from ChempleterDataset
        """
        ds = ChempleterDataset(*mock_files)
        assert len(ds) == 2

    def test_dataset_getitem(self,mock_files):
        ds = ChempleterDataset(*mock_files)
        tensor = ds[0]
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
        assert tensor[0] == 1 # [START]
        assert tensor[-1] == 2 # [END]

    def test_collate_fn_sorting_and_padding(self):
        """
        Test the collat fn for dataloader
        """
        # create different lneght tensors
        t1 = torch.tensor([1, 3, 2]) 
        t2 = torch.tensor([1, 3, 3, 3, 2]) 
        t3 = torch.tensor([1, 4, 2]) 
        
        batch = [t1, t2, t3]
        padded, lengths = collate_fn(batch)
        
        assert torch.equal(lengths, torch.tensor([5, 3, 3]))
        assert padded.shape == (3, 5)
        assert padded[0, 4] == 2 # end of longest
        assert padded[1, 3] == 0 # padding in shorter seq
        assert padded[2, 3] == 0 # üadding in shorter seq

    def test_dataloader_integration(self,mock_files):
        ds = ChempleterDataset(*mock_files)
        dl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        
        batch_x, batch_len = next(iter(dl))
        
        assert batch_x.shape[0] == 2
        assert batch_len[0] >= batch_len[1]

class TestChempleterInference():

    def test_handle_prompt_smiles(self,mock_stoi):
        prompt = handle_prompt(smiles="CC", selfies=None, stoi=mock_stoi, alter_prompt=False)
        assert prompt[0] == "[START]"
        assert "[C]" in prompt

    def test_handle_prompt_none(self,mock_stoi):
        prompt = handle_prompt(smiles="", selfies=None, stoi=mock_stoi, alter_prompt=False)
        assert prompt == ["[START]"]

    def test_handle_prompt_invalid_selfies_token(self,mock_stoi):
        "Case where SELFIES not in stoi"
        with pytest.raises(ValueError):
            handle_prompt(smiles=None, selfies=["[INVALID]"], stoi=mock_stoi, alter_prompt=False)

    def test_handle_len_logic(self):
        prompt = ["[START]", "[C]", "[C]"] # len 3
        min_l, max_l = handle_len(prompt, min_len=None, max_len=10)
        assert min_l == 5 # prompt_len + 2
        assert max_l == 13 # 10 + 3

    def test_handle_len_adjustment(self):
        prompt = ["C"] * 10
        # min_len (5) < prompt_len (10)
        min_l, max_l = handle_len(prompt, min_len=5, max_len=10)
        assert min_l == 12

    def test_handle_sampling_greedy(self):
        logits = torch.tensor([-1.0, 5.0, 0.0]) # with greedy sampling, 5 should be selected.
        token_id = handle_sampling(logits, "greedy", temperature=1.0, k=10)
        assert token_id == 1

    def test_handle_sampling_temperature(self):
        logits = torch.ones(5)
        token_id = handle_sampling(logits, "temperature", temperature=0.1, k=10)
        assert 0 <= token_id < 5

    def test_output_molecule(self,mock_itos):
        gen_ids = [0, 3, 4, 1] # [START], [C], [O], [END]
        smiles, selfies = output_molecule(gen_ids, mock_itos)
        assert selfies == "[C][O]"
        assert smiles == "CO"

    def test_extend(self):

        model = ChempleterModel(vocab_size=2)
        m, smiles, selfies = extend(smiles="C", model=model, stoi_file=None, itos_file=None,next_atom_criteria="greedy",max_len=2)
        assert smiles == "C"
        assert selfies == "[C]"

    def test_get_default_data_no_files_given(self):

        stoi, itos, model = _get_default_data(model=None,stoi_file=None,itos_file=None)

        assert type(stoi) is dict
        assert type(itos) is list
        assert type(model) is ChempleterModel

    def test_get_default_data_files_given(self,tmp_path):

        stoi_file = tmp_path / "stoi.json"
        stoi = {"[PAD]": 0, "[START]": 1, "[END]": 2, "[C]": 3, "[O]": 4}
        with open(stoi_file, "w") as f:
            json.dump(stoi, f)

        itos_file = tmp_path / "itos.json"
        itos = ["[PAD]", "[START]", "[END]", "[C]", "[O]"]
        with open(itos_file, "w") as f:
            json.dump(itos, f)

        stoi, itos, model = _get_default_data(model=ChempleterModel(vocab_size=len(stoi)),stoi_file=stoi_file,itos_file=itos_file)

        assert type(stoi) is dict
        assert type(itos) is list
        assert type(model) is ChempleterModel

class TestChempleterDescriptors():
    def test_calculate_descriptors(self):
        exp_dict = {'MW': 78.11,
                    'LogP': 1.69,
                    'SA_Score': 1.0,
                    'QED': 0.443,
                    'Fsp3': 0.0,
                    'RotatableBonds': 0,
                    'RingCount': 1,
                    'TPSA': 0.0,
                    'RadicalElectrons': 0}
        
        m = Chem.MolFromSmiles("c1ccccc1")
        test_dict = calculate_descriptors(m)

        assert exp_dict == test_dict