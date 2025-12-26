# Chempleter

Chempleter is lightweight generative model which utlises a simple Gated Recurrent Unit (GRU) to predict syntactically valid extensions of a provided molecular fragment.
It accepts SMILES notation as input and enforces chemical syntax validity using SELFIES for the generated molecules. 

<div align="center">
<img src="https://raw.githubusercontent.com/davistdaniel/chempleter/main/screenshots/demo.gif" alt="Demo Gif" width="400">
</div>


* Why was Chempleter made?
    * Mainly for me to get into Pytorch. Also, I find it fun to generate random, possibly unsynthesisable molecules from a starting structure.

* What can Chempleter do?
    
    * Currently, Chempleter accepts an intial molecule/molecular fragment in SMILES format and generates a larger molecule with that intial structure included, while respecting chemical syntax. 
    
    * It can be used to generate a wide range of structural analogs which the share same core structure (by changing the sampling temperature) or decorate a core scaffold iteratively (by increasing generated token lengths)

    * In the future, it might be adapated to predict structures with a specific chemical property using a regressor to rank predictions and transition towards more "goal-directed" predictions.


## Prerequisites
* Python ">=3.13"
* [uv](https://docs.astral.sh/uv/) (optional but recommended)
* See [pyproject.toml](pyproject.toml) for dependencies.

## Get started

#### Install using uv

1. Clone this repo

    ``git clone https://github.com/davistdaniel/chempleter.git``

2. Inside the project directory, exceute in a terminal:

    ``uv sync``

    By default, the CPU version of pytorch will be installed, in case of using GPU as accelerator and CUDA 12.8:

    ``uv sync --extra gpu128``

    Alternatively, you can install a PyTorch version compatible with your CUDA version by following the [Pytorch documentation](https://pytorch.org/get-started/locally/).

#### Install using pip

1. Clone this repo

    ``git clone https://github.com/davistdaniel/chempleter.git``

2. Inside the project directory, exceute in a terminal (a python virtual environment is optional but recommended):

    ``python -m pip install .``

    By default, the CPU version of pytorch will be installed. Alternatively, you can install a PyTorch version compatible with your CUDA version by following the [Pytorch documentation](https://pytorch.org/get-started/locally/).

### Usage

#### GUI
* To start the Chempleter GUI:
    
    ``chempleter-gui``

    or 

    ``uv run src/chempleter/gui.py``
    
    or 
    
    ``python src/chempleter/gui.py``

* Type in the SMILES notation for the starting structure or leave it empty to generate random molecules. Click on ``GENERATE`` button to generate a molecule.
* Options:
    * Temperature : Increasing the temperautre would result in more unusual molecules, while lowere values would generate more common structures.
    * Sampling : Most probable’ selects the molecule with the highest likelihood for the given starting structure, producing the same result on repeated generations. ‘Random’ generates a new molecule each time, while still including the input structure.


#### As a python library

* To use Chempleter as a python library:

    ```python
    from chempleter.inference import extend
    generated_mol, generated_smiles, generated_selfies = extend(smiles="c1ccccc1")
    print(generated_smiles)
    >> C1=CC=CC=C1C2=CC=C(CN3C=NC4=CC=CC=C4C3=O)O2
    ```

    To draw the generated molecule :

    ```python
    from rdkit import Chem
    Chem.Draw.MolToImage(generated_mol)
    ```
* For details on available parameters, refer to the ``extend`` (``chempleter.inference`` module) function’s docstring.”

### Current model performance

Performance metrics were evaluated across 500 independent generations using a model checkpoint trained for 80 epochs with a batch size of 64.

| Metric     | Value | Description                                                                                                  |
|------------|-------|--------------------------------------------------------------------------------------------------------------|
| Validity   | 1.0   | Proportion of Generated SMILES which respect chemical syntax; tested using selfies decoder and RDkit parser. |
| Uniqueness | 0.96  | Proportion of Generated SMILES which were unique                                                             |
| Novelty    | 0.85  | Proportion of Generated SMILES which were not present in the training datatset                             |


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

[MIT](https://github.com/davistdaniel/chempleter/tree/main?tab=MIT-1-ov-file#readme) License

Copyright (c) 2025 Davis Thomas Daniel

# Contributing

Any contribution, improvements, feature ideas or bug fixes are always welcome.

## Random Notes

* Training data
    * QM9 and ZINC datasets. 379997 molecules were used for training in total.
* Running wihout a GPU
    * Chempleter uses a 2-layer GRU, it should run comfortably on a CPU.






