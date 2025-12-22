# Chempleter

Chempleter is lightweight generative model which utlises a simple Gated Recurrent Unit (GRU) to predict syntactically valid extensions of a provided molecular fragment.
It accepts SMILES notation as input and enforces chemical syntax validity using SELFIES for the generated molecules.

## Project structure
* src/chempleter: Contains python modules relating to different functions.
* src/chempleter/processor.py: Contains fucntions for processing csv files containing SMILES data and generating training-related files.
* src/chempleter/dataset.py: ChempleterDataset class
* src/chempleter/model.py: ChempleterModel class
* src/chempleter/inference.py: Contains functions for inference
* src/chempleter/train.py: Contains functions for training
* src/chempleter/gui.py: Chempleter GUI built using NiceGUI


## Prerequisites
* Python ">=3.13"
* uv (optional but recommended)
* See pyproject.toml for dependencies

## Usage

### Install using uv

1. Clone this repo

    ``git clone https://github.com/davistdaniel/chempleter.git``

2. Install using uv

    In case of using GPU as accelerator and CUDA 12.8:

    ``uv sync --extra gpu128``

    If CPU is used:

    ``uv sync --extra cpu``

    For GUI, add it as an extra:
    
    ```bash
    # use --extra cpu instead for --extra gpu128 for CPU inference
    uv sync --extra gui --extra gpu128
    ```

## Usage
* To start the GUI:

    ``uv run src/chempleter/gui.py``

* Use as a python library:

    ```python
    from chempleter import extend
    generated_mol, generated_smiles, generated_selfies = extend(smiles="c1ccccc1")
    print(generated_smiles)
    >> C1=CC=CC=C1C2=CC=C(CN3C=NC4=CC=CC=C4C3=O)O2
    ```

    To draw the generated molecule :

    ```python
    from rdkit import Chem
    Chem.Draw.MolToImage(generated_mol)
    ```
