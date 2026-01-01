Getting started
=========================

Setting up chempleter on your device
-----------------------------------------

.. tab-set::

    .. tab-item:: Install on your device

        You can install Chempleter on your device using `uv <https://docs.astral.sh/uv/>`_ or pip. See :doc:`installation instructions <installation>`.

    .. tab-item:: Run Chempleter GUI without installing

        You can run Chempleter's GUI without installing via `uv <https://docs.astral.sh/uv/>`_:

        .. tab-set::

            .. tab-item:: On MacOS/Linux

                ``uvx --from chempleter chempleter-gui``

            .. tab-item:: On windows

                ``uvx --from chempleter chempleter-gui.exe``



Generating molecules
----------------------------------------

* Chempleter accepts a valid SMILES notation for a molecule/molecular fragment. If an initial input is not provided, chempleter generates a random molecule.
* There are two main interence functions:

    * ``extend``:

        Description: Takes a starting molecular structure (SMILES or SELFIES) and uses the GRU model to 
        append new atoms and functional groups until a complete molecule larger than the input molecule is formed.

        Behaviour: Includes a retry logic systes. If the model fails to add new atoms (i.e. returns the input), it can either "truncate" the prompt 
        (``alter_prompt``, false by default) or generate a new randomized SMILES string (``randomise_prompt``, true by default) to provide the model with new prompt based on the input prompt.

        An example with ``Extend`` for Benzene(c1ccccc1):

        .. image::
            images/extend_example.png
            :align: center
            
    * ``evolve``:

        Description: A wrapper function that calls extend multiple times in a chain. 
        It takes the output of one extension and uses it as the input for the next, effectively "evolving" a small fragment into 
        a complex structure over several steps.

        Behaviour: Automates the growth process over a set number of steps (``n_evolve``). If the molecule stops growing at any step in the chain, 
        the function halts to prevent redundant processing. Maintains a history of the evolution, returning a list of all intermediate molecules. 
        It is best to start with a small fragment.

        An example with ``Evolve`` for Benzene(c1ccccc1):

        .. image::
            images/evolve_example.png
            :align: center


Use the GUI
^^^^^^^^^^^^^^^^^^^^

* Type in the SMILES notation for the starting structure or leave it empty to generate random molecules. Click on ``GENERATE`` button to generate a molecule.

* GUI options:
    
    * **Temperature** : Increasing the temperature would result in more unusual molecules, while lower values would generate more common structures.

    * **Sampling** : Most probable selects the molecule with the highest likelihood for the given starting structure, producing the same result on repeated generations. Random generates a new molecule each time, while still including the input structure.

    * **Generation type** : Extend will ouput a generated molecule which is extended based on the input fragment, while Evolve will ouput multiple generated molecules each based on their previous molecular fragment.



Use as a Python library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chempleter can be used programmatically to extend or iteratively evolve molecules.

* To extend a molecule once, use ``chempleter.inference.extend``:

  .. code-block:: python

     from chempleter.inference import extend

     generated_mol, generated_smiles, generated_selfies = extend(
         smiles="c1ccccc1"
     )

     print(generated_smiles)

* To iteratively evolve a molecule, use ``chempleter.inference.evolve``:

  .. code-block:: python

     from chempleter.inference import evolve

     generated_mols, generated_selfies, generated_smiles = evolve(
         smiles="c1ccccc1",
         n_evolve=4
     )

* Options

    Both ``extend`` and ``evolve`` accept several optional arguments to control
    generation behaviour:

    - ``model``: Preloaded Chempleter model. If omitted, a default trained model is used.
    - ``stoi_file`` / ``itos_file``: Paths to token mapping files.
    - ``selfies``: Input SELFIES tokens (overrides ``smiles``).
    - ``smiles``: Input SMILES string to extend or evolve.
    - ``min_len``: Minimum final sequence length.
    - ``max_len``: Maximum number of generated tokens.
    - ``temperature``: Sampling temperature.
    - ``k``: Top-k sampling parameter.
    - ``next_atom_criteria``: Sampling strategy (``"greedy"``, ``"temperature"``, ``"top_k_temperature"``, or ``"random"``).
    - ``device``: Device to run inference on (e.g. ``"cpu"`` or CUDA device).
    - ``alter_prompt``: Allow prompt alteration if generation fails.

    Additional options for ``evolve``:

    - ``n_evolve``: Number of evolutionary extension steps.

    Both functions return RDKit molecule objects alongside the generated SMILES and SELFIES representations.






