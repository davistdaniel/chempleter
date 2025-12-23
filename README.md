# Chempleter

Chempleter is lightweight generative model which utlises a simple Gated Recurrent Unit (GRU) to predict syntactically valid extensions of a provided molecular fragment.
It accepts SMILES notation as input and enforces chemical syntax validity using SELFIES for the generated molecules. 

* Why was Chempleter made?
    * Mainly for me to get into Pytorch. Also, I find it fun to generate random, possibly unsynthesisable molecules from a starting structure.

* What can Chempleter do?
    
    * Currently, Chempleter accepts an intial molecule/molecular fragment in SMILES format and generates a larger molecule with that intial structure included, while respecting chemical syntax. 
    
    * It can be used to generate a wide range of structural analogs which the share same core structure (by changing the sampling temperature) or decorate a core scaffold iteratively (by increasing generated token lengths)

    * In the future, it might be adapated to predict structures with a specific chemical property using a regressor to rank predictions and transition towards more "goal-directed" predictions.

<div align="center">
<img src="screenshots/demo.gif" alt="Demo Gif" width="400">
</div>

## Prerequisites
* Python ">=3.13"
* uv (optional but recommended)
* See pyproject.toml for dependencies.

## Get started

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
    # use --extra cpu instead of --extra gpu128 for CPU inference
    uv sync --extra gui --extra gpu128
    ```

### Usage
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

## Notes

### Framework

* Training data
    * QM9 and ZINC datasets. 379997 molecules were used for training in total.
* Running wihout a GPU
    * Chempleter uses a 2-layer GRU, it should run comfortably on a CPU.
* 

### Current model performance

Performance metrics were evaluated across 500 independent generations using a model checkpoint trained for 80 epochs with a batch size of 64.

| Metric     | Value | Description                                                                                                  |
|------------|-------|--------------------------------------------------------------------------------------------------------------|
| Validity   | 1.0   | Proportion of Generated SMILES which respect chemical syntax; tested using selfies decoder and RDkit parser. |
| Uniqueness | 0.96  | Proportion of Generated SMILES which were unique                                                             |
| Novelty    | 0.85  | Proportion of Generated SMILES which were different from the training datatset                               |


### Project structure
* src/chempleter: Contains python modules relating to different functions.
* src/chempleter/processor.py: Contains fucntions for processing csv files containing SMILES data and generating training-related files.
* src/chempleter/dataset.py: ChempleterDataset class
* src/chempleter/model.py: ChempleterModel class
* src/chempleter/inference.py: Contains functions for inference
* src/chempleter/train.py: Contains functions for training
* src/chempleter/gui.py: Chempleter GUI built using NiceGUI
* src/chempleter/data :  Contains trained model, vocabulary files

# License
MIT



