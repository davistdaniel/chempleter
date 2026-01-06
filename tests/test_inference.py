# Tests for chempleter.inference module

import pytest
import torch
import json
import selfies as sf
from rdkit import Chem
from chempleter.model import ChempleterModel
from chempleter.inference import (
    handle_prompt,
    handle_len,
    handle_sampling,
    output_molecule,
    generation_loop,
    extend,
    evolve,
    bridge,
    _get_default_data
)


class TestHandlePrompt:
    """Test cases for handle_prompt function."""

    def test_handle_prompt_with_smiles(self, sample_stoi):
        """Test handling prompt with SMILES input."""
        smiles = "CCO"
        selfies_list = list(sf.split_selfies(sf.encoder(smiles)))
        prompt = handle_prompt(smiles="CCO", stoi=sample_stoi)
        assert isinstance(prompt, list)
        assert prompt == ["[START]"]+selfies_list

    def test_handle_prompt_with_empty_smiles(self, sample_stoi):
        """Test handling prompt with empty SMILES."""
        prompt = handle_prompt(smiles="", stoi=sample_stoi)
        assert prompt == ["[START]"]

    def test_handle_prompt_with_selfies(self, sample_stoi):
        """Test handling prompt with SELFIES tokens."""
        selfies_tokens = ["[C]", "[C]", "[O]"]
        prompt = handle_prompt(selfies=selfies_tokens, stoi=sample_stoi)
        assert prompt == ["[START]"] + selfies_tokens

    def test_handle_prompt_with_invalid_selfies_token(self, sample_stoi):
        """Test error handling for invalid SELFIES token."""
        invalid_tokens = ["[C]", "[INVALID_TOKEN]"]
        with pytest.raises(ValueError, match="Invalid SELFIES Token"):
            handle_prompt(selfies=invalid_tokens, stoi=sample_stoi)

    def test_handle_prompt_with_fragments(self, sample_stoi):
        """Test handling prompt with fragment SMILES."""
        frag1 = "CCO"
        frag2 = "c1ccccc1"
        result = handle_prompt(
            frag1_smiles=frag1, frag2_smiles=frag2, stoi=sample_stoi
        )
        prompt, frag1_symbols, frag2_symbols = result
        assert prompt[0] == "[START]"
        assert "[MASK]" in prompt
        assert "[BRIDGE]" in prompt


    def test_handle_prompt_invalid_smiles(self, sample_stoi):
        """Test error handling for invalid SMILES."""
        with pytest.raises(sf.EncoderError):
            handle_prompt(smiles="INVALID_SMILES", stoi=sample_stoi, alter_prompt=False)

class TestHandleLen:
    """Test cases for handle_len function."""

    def test_handle_len_with_none_min_len(self):
        """Test handling length with None min_len."""
        prompt = ["[START]", "[C]", "[C]", "[O]"]
        min_len, max_len = handle_len(prompt, min_len=None, max_len=10)
        assert min_len == len(prompt) + 2
        assert max_len == len(prompt) + 10

    def test_handle_len_with_small_min_len(self):
        """Test handling length with min_len smaller than prompt length."""
        prompt = ["[START]", "[C]", "[C]", "[C]", "[C]", "[O]"]
        min_len, max_len = handle_len(prompt, min_len=3, max_len=10)
        assert min_len == len(prompt) + 2
        assert max_len == len(prompt) + 10

    def test_handle_len_with_valid_min_len(self):
        """Test handling length with valid min_len."""
        prompt = ["[START]", "[C]", "[O]"]
        min_len, max_len = handle_len(prompt, min_len=10, max_len=20)
        assert min_len == 10
        assert max_len == len(prompt) + 20

    def test_handle_len_max_less_than_min(self):
        """Test handling length when max_len < min_len."""
        prompt = ["[START]", "[C]"]
        prompt_len = len(prompt)
        min_len, max_len = handle_len(prompt, min_len=10, max_len=5)
        assert min_len == 10
        assert max_len == len(prompt)+5+5

class TestHandleSampling:
    """Test cases for handle_sampling function."""

    def test_handle_sampling_greedy(self):
        """Test greedy sampling."""
        vocab_size = 10
        logits = torch.randn(vocab_size)
        next_id = handle_sampling(logits, "greedy", temperature=1.0, k=5)
        assert isinstance(next_id, int)
        assert 0 <= next_id < vocab_size
        assert next_id == torch.argmax(logits).item()

    def test_handle_sampling_temperature(self):
        """Test temperature sampling."""
        vocab_size = 10
        logits = torch.randn(vocab_size)
        next_id = handle_sampling(logits, "temperature", temperature=0.7, k=5)
        assert isinstance(next_id, int)
        assert 0 <= next_id < vocab_size

    def test_handle_sampling_top_k_temperature(self):
        """Test top-k temperature sampling."""
        vocab_size = 10
        logits = torch.randn(vocab_size)
        next_id = handle_sampling(logits, "top_k_temperature", temperature=0.7, k=5)
        assert isinstance(next_id, int)
        assert 0 <= next_id < vocab_size

    def test_handle_sampling_random(self):
        """Test random sampling strategy."""
        vocab_size = 10
        logits = torch.randn(vocab_size)
        next_id = handle_sampling(logits, "random", temperature=0.7, k=5)
        assert isinstance(next_id, int)
        assert 0 <= next_id < vocab_size

class TestOutputMolecule:
    """Test cases for output_molecule function."""

    def test_output_molecule_extend(self, sample_itos):
        """Test output molecule for extend generation type."""
        generated_ids = [1,9,7,2]
        print(sample_itos)
        smiles, selfies, ignored_factor = output_molecule(
            "extend", generated_ids, sample_itos
        )

        assert isinstance(smiles, str)
        assert isinstance(selfies, str)
        assert isinstance(ignored_factor, float)

        assert selfies == "[C][=O]"
        assert smiles == "C=O"
        assert ignored_factor == 0.0

    def test_output_molecule_bridge(self, sample_itos):
        """Test output molecule for bridge generation type."""
        generated_ids = [1, 9,3,4,10,2]
        frag1_symbols = ["[C]"]
        frag2_symbols = ["[O]"]
        smiles, selfies, ignored_factor = output_molecule(
            "bridge", generated_ids, sample_itos,
            frag1_symbols=frag1_symbols, frag2_symbols=frag2_symbols
        )
        assert isinstance(smiles, str)
        assert isinstance(selfies, str)
        assert isinstance(ignored_factor, float)

        assert selfies == "[C][C][O][O]" # should add frag1 and frag2 symbols at the start and end.
        assert smiles == "CCOO"
        assert ignored_factor == 0.0

class TestExtend:
    """Test cases for extend function."""

    @pytest.mark.inference
    def test_extend_with_smiles(self,device):
        """Test extend function with SMILES input."""
        mol, smiles, selfies = extend(smiles="CCO", max_len=10,device=device)
        assert mol is not None
        assert isinstance(smiles, str)
        assert isinstance(selfies, str)

    @pytest.mark.inference
    def test_extend_with_empty_smiles(self):
        """Test extend function with empty SMILES."""
        mol, smiles, selfies = extend(smiles="", max_len=10)
        assert mol is not None
        assert isinstance(smiles, str)
        assert isinstance(selfies, str)

    @pytest.mark.inference
    def test_extend_with_selfies(self):
        """Test extend function with SELFIES tokens."""
        selfies_tokens = list(sf.split_selfies(sf.encoder("CCO")))
        mol, smiles, selfies = extend(selfies=selfies_tokens, max_len=10)
        assert mol is not None
        assert isinstance(smiles, str)
        assert isinstance(selfies, str)

    @pytest.mark.inference
    def test_extend_different_sampling_strategies(self):
        """Test extend with different sampling strategies."""
        strategies = ["greedy", "temperature", "top_k_temperature","unknown"] # unknown should fallback to greedy
        for strategy in strategies:
            mol, smiles, selfies = extend(
                smiles="CCO",
                max_len=10,
                next_atom_criteria=strategy,
                temperature=0.7,
                k=10,
            )
            assert mol is not None
            assert isinstance(smiles, str)
            assert isinstance(selfies, str)

class TestGenerationLoop:
    """Test cases for generation_loop function."""

    def test_generation_loop_extend(self, sample_stoi, device):
        """Test generation loop for extend type."""
        prompt = ["[START]", "[C]", "[C]"]
        # Ensure model vocab size matches stoi
        model = ChempleterModel(vocab_size=max(sample_stoi.values())+1, embedding_dim=64, hidden_dim=128, num_layers=2)
        generated_ids = generation_loop(
            generation_type="extend",
            model=model,
            prompt=prompt,
            stoi=sample_stoi,
            min_len=5,
            max_len=10,
            next_atom_criteria="greedy",
            temperature=1.0,
            k=10,
            device=device,
        )
        assert isinstance(generated_ids, list)
        assert len(generated_ids) >= len(prompt)

    def test_generation_loop_bridge(self, sample_stoi, device):
        """Test generation loop for bridge type."""
        prompt = ["[START]", "[C]", "[MASK]", "[O]", "[BRIDGE]"]
        # Ensure model vocab size matches stoi
        model = ChempleterModel(vocab_size=max(sample_stoi.values())+1, embedding_dim=64, hidden_dim=128, num_layers=2)
        generated_ids = generation_loop(
            "bridge",
            model,
            prompt,
            sample_stoi,
            min_len=5,
            max_len=10,
            next_atom_criteria="greedy",
            temperature=1.0,
            k=10,
            device=device,
        )
        assert isinstance(generated_ids, list)

class TestEvolve:
    """Test cases for evolve function."""

    @pytest.mark.inference
    def test_evolve_with_smiles(self,device):
        """Test evolve function with SMILES input."""
        mols, selfies_list, smiles_list = evolve(
            smiles="CCO", max_len=5, n_evolve=2,device=device
        )
        assert isinstance(mols, list)
        assert isinstance(selfies_list, list)
        assert isinstance(smiles_list, list)
        assert len(mols) > 0

    @pytest.mark.inference
    def test_evolve_with_empty_smiles(self,device):
        """Test evolve function with empty SMILES."""
        mols, selfies_list, smiles_list = evolve(smiles="", max_len=5, n_evolve=2,device=device)
        assert isinstance(mols, list)
        assert isinstance(selfies_list, list)
        assert isinstance(smiles_list, list)
        assert len(mols) > 0


class TestBridge:
    """Test cases for bridge function."""

    @pytest.mark.inference
    def test_bridge_with_fragments(self,device):
        """Test bridge function with two fragments."""
        frag1 = "c1ccccc1"
        frag2 = "c1ccccc1"
        mol, smiles, selfies = bridge(frag1_smiles=frag1, frag2_smiles=frag2,device=device)
        assert mol is not None
        assert isinstance(smiles, str)
        assert isinstance(selfies, str)

class TestGetDefaultData:

    def test_get_default_data_no_files_given(self):

        stoi, itos, model = _get_default_data("extend",model=None,stoi_file=None,itos_file=None)

        assert type(stoi) is dict
        assert type(itos) is list
        assert type(model) is ChempleterModel

        stoi, itos, model = _get_default_data("bridge",model=None,stoi_file=None,itos_file=None)

        assert type(stoi) is dict
        assert type(itos) is list
        assert type(model) is ChempleterModel

    def test_get_default_data_files_given(self,sample_stoi,sample_stoi_file,sample_itos_file):
                
        with open(sample_stoi_file,"r") as f:
            stoi = json.load(f)

        stoi, itos, model = _get_default_data("extend",model=ChempleterModel(vocab_size=len(sample_stoi)),stoi_file=sample_stoi_file,itos_file=sample_itos_file)

        assert type(stoi) is dict
        assert type(itos) is list
        assert type(model) is ChempleterModel