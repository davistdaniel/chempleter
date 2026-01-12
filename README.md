# Chempleter

<div align="center">
<img src="https://raw.githubusercontent.com/davistdaniel/chempleter/refs/heads/main/docs/source/images/chempleter_logo.png" alt="Demo Gif" width="200">

<i>Molecular autocomplete</i>
</div>

<div align="center">

[![PyPI - Status](https://img.shields.io/pypi/status/chempleter)](https://pypi.org/project/chempleter) [![PyPI - Version](https://img.shields.io/pypi/v/chempleter)](https://pypi.org/project/chempleter) [![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fdavistdaniel%2Fchempleter%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/davistdaniel/chempleter/blob/main/pyproject.toml) [![PyPI - License](https://img.shields.io/pypi/l/chempleter)](https://github.com/davistdaniel/chempleter/blob/main/LICENSE) [![GitHub last commit](https://img.shields.io/github/last-commit/davistdaniel/chempleter)](https://github.com/davistdaniel/chempleter/commits/main/) [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/davistdaniel/chempleter/test_chempleter.yml?label=Tests)](https://github.com/davistdaniel/chempleter/actions) [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/davistdaniel/chempleter/deploy_docs.yml?label=Docs)](https://github.com/davistdaniel/chempleter/actions)



</div>

Chempleter is a lightweight generative sequence model based on a multi-layer gated recurrent units (GRU) to predict syntactically valid extensions of a provided molecular fragment or bridge two molecules/molecular fragments. It operates on SELFIES token sequences, ensuring syntactically valid molecular generation and accepts SMILES notation as input. Due to its simple recurrent architecture and small vocabulary, the model runs efficiently on both CPUs and GPUs.

<div align="center">
<img src="https://raw.githubusercontent.com/davistdaniel/chempleter/refs/heads/main/docs/source/images/extend_demo.gif" alt="Demo Gif" width="400">
</div>


* What can Chempleter do?
    
    * Currently, Chempleter accepts an intial molecule/molecular fragment in SMILES format and generates a larger molecule with that intial structure included, while respecting chemical syntax. It also shows some interesting descriptors.
    
    * It can be used to generate a wide range of structural analogs which the share same core structure (by changing the sampling temperature) or decorate a core scaffold iteratively (by increasing generated token lengths)

    * It can be used to bridge two molecules/molecular fragments to explore linker chemistry.

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

These commands are valid for running Chempleter on CPU. For GPU, see [installation instructions](https://davistdaniel.github.io/chempleter/installation.html).

- ### Run the GUI directly without installing (via uv):

    * On windows:

        ``uvx --from "chempleter[cpu]" chempleter-gui.exe``
    * On linux/MacOS:
        
        ``uvx --from "chempleter[cpu]" chempleter-gui``

    * The very first start of the GUI on your device might be a bit slow. To know more about using the GUI and various options, see [here](https://davistdaniel.github.io/chempleter/usage.html#use-the-gui).

    <div align="center">
    <h2> Or </h2>
    </div>

- ### Install using uv

    ``uv pip install "chempleter[cpu]"``

- ### Run GUI after installation

    ``uv run chempleter-gui``

- ### Use the GUI

    * Type in the SMILES notation for the starting structure or leave it empty to generate random molecules. Click on ``GENERATE`` button to generate a molecule.

    * To know more about using the GUI and various other options, see [here](https://davistdaniel.github.io/chempleter/usage.html#use-the-gui).

    
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

See [model validation reports](https://davistdaniel.github.io/chempleter/validation.html).

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





