Model history
=========================

All models were trained on a system equipped with:

* NVIDIA 4060Ti (16GB) GPU
* CUDA 12.8

The links below also contains some development notes about model development process along with the validation metrics.

.. toctree::
   :maxdepth: 1
   :caption: Model validation reports:

   validations/extend_v1
   validations/extend_v2
   validations/bridge_v1


Validation methodology
------------------------------------


For each model validation, depending on the type of model, ``chempleter.inference.extend`` or ``chempleter.inference.bridge`` are used.
Metrics and descriptors are calculated for ``N`` (500, by default) random samples. See ``chempleter.validate`` module for the source code.

Generation Quality Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following metrics assess the quality of the generated molecules:


.. list-table:: Generation Quality Metrics
    :header-rows: 1

    * - Metric
      - Description
    * - Uniqueness
      - Fraction of generated molecules that are unique. Computed as the number of distinct SMILES strings divided by the total number of generated molecules.
    * - Novelty
      - Fraction of generated molecules not present in the training dataset. Computed by comparing the set of generated SMILES to the reference SMILES dataset and calculating the proportion that are new.      
    * - SELFIES fidelity
      - Measures the encoding fidelity of the SELFIES representation. Computed as the average fraction of tokens in the generated SELFIES that are ignored when re-encoding the molecule back to SELFIES via from generated SMILES. Lower values indicate that a low fraction of generated tokens are ignored due to the decoding or the model is generating more valid molecules.


Descriptor statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following descriptors (as implemented in `RDKit <https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html>`_) and their distributions summarize the physicochemical and structural properties of the generated molecules:

.. list-table:: Descriptor Statistics
    :header-rows: 1

    * - Descriptor
      - Description
    * - MW
      - Molecular Weight. Average, min, and max over generated molecules.
    * - LogP
      - Octanol-water partition coefficient. Indicates lipophilicity.
    * - SA_Score
      - Synthetic Accessibility Score. Estimates ease of chemical synthesis (1 = easy, 10 = difficult).
    * - QED
      - Quantitative Estimate of Drug-likeness (0–1). Higher is more drug-like.
    * - Fsp3
      - Fraction of sp3 hybridized carbons.
    * - RotatableBonds
      - Number of rotatable bonds. Indicates molecular flexibility.
    * - RingCount
      - Number of rings present in the molecule.
    * - TPSA
      - Topological Polar Surface Area. Correlates with permeability and solubility.
    * - RadicalElectrons
      - Number of unpaired electrons (radicals) in the molecule.
