# Chempleter

<div align="center">
<img src="https://raw.githubusercontent.com/davistdaniel/chempleter/refs/heads/main/docs/source/images/chempleter_logo.png" alt="Demo Gif" width="200">

<i>Molecular autocomplete</i>
</div>

<div align="center">

![PyPI - Status](https://img.shields.io/pypi/status/chempleter) ![PyPI - Version](https://img.shields.io/pypi/v/chempleter) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fdavistdaniel%2Fchempleter%2Frefs%2Fheads%2Fmain%2Fpyproject.toml) ![PyPI - License](https://img.shields.io/pypi/l/chempleter) ![GitHub last commit](https://img.shields.io/github/last-commit/davistdaniel/chempleter)

</div>


Chempleter is lightweight generative model which utlises a simple Gated Recurrent Unit (GRU) to predict syntactically valid extensions of a provided molecular fragment.
It accepts SMILES notation as input and enforces chemical syntax validity using SELFIES for the generated molecules. 

<div align="center">
<img src="https://raw.githubusercontent.com/davistdaniel/chempleter/refs/heads/main/docs/source/images/extend_demo.gif" alt="Demo Gif" width="400">
</div>


* What can Chempleter do?
    
    * Currently, Chempleter accepts an intial molecule/molecular fragment in SMILES format and generates a larger molecule with that intial structure included, while respecting chemical syntax. It also shows some interesting descriptors.
    
    * It can be used to generate a wide range of structural analogs which the share same core structure (by changing the sampling temperature) or decorate a core scaffold iteratively (by increasing generated token lengths)

    * It can be used to bridge two molecules/molecular fragments.

    * In the future, it might be adapated to predict structures with a specific chemical property using a regressor to rank predictions and transition towards more "goal-directed" predictions.


See [chempleter in action.](https://davistdaniel.github.io/chempleter/demo.html)


## Prerequisites
* Python ">=3.12"
* [uv](https://docs.astral.sh/uv/) (optional but recommended)

## Installation

See detailed [installation instructions](https://davistdaniel.github.io/chempleter/installation.html).

## Getting started

Visit [Chempleter's docs](https://davistdaniel.github.io/chempleter/).

## Quick start

- ### Run the GUI directly without installing (via uv):

    * On windows:

        ``uvx --from chempleter chempleter-gui.exe``
    * On linux/MacOS:
        
        ``uvx --from chempleter chempleter-gui``

    * To know more about using the GUI and various options, see [here](https://davistdaniel.github.io/chempleter/usage.html#use-the-gui).

    <div align="center">
    <h2> Or </h2>
    </div>

- ### Install using uv

    ``uv pip install chempleter``

- ### Use the GUI

    * To start the Chempleter GUI after installing, execute in a terminal:
        
        ``uv run chempleter-gui``

    * Type in the SMILES notation for the starting structure or leave it empty to generate random molecules. Click on ``GENERATE`` button to generate a molecule.

    * To know more about using the GUI and various options, see [here](https://davistdaniel.github.io/chempleter/usage.html#use-the-gui).


- ### Run GUI after installation

    ``uv run chempleter-gui``
    
 - ### Use as a python library

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
    * For details on available paramenters and inference functions, see [generating molecules](https://davistdaniel.github.io/chempleter/usage.html#generating-molecules).

### Model history and validation


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

Copyright (c) 2025-2026 Davis Thomas Daniel

# Contributing

Any contribution, improvements, feature ideas or bug fixes are always welcome.





