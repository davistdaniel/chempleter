.. chempleter documentation master file, created by
   sphinx-quickstart on Thu Jan  1 12:33:30 2026.

Welcome to Chempleter
===========================================

Chempleter is a lightweight generative model which utilises a simple Gated Recurrent Unit (GRU) to predict syntactically valid extensions of a provided molecular fragment or bridge two molecules/molecular fragments.
It accepts `SMILES <https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System>`_ notation as input and enforces chemical syntax validity using `SELFIES <https://selfies.readthedocs.io/en/latest/>`_ for the generated molecules. 
The library also has an optional graphical user interface to interact with the model.


.. image::
   images/extend_demo.gif
   :align: center
   :scale: 70 %

See :doc:`chempleter in action <demo>` or to get started, see :doc:`getting started <usage>`. Model validation reports can be viewed in :doc:`model history <validation>`.

What can chempleter do?
----------------------------

* Currently, Chempleter accepts an initial molecule/molecular fragment in SMILES format and generates a larger molecule with that intial structure included, while respecting chemical syntax. It also shows some interesting descriptors.
* It can be used to generate a wide range of structural analogs which the share same core structure (by changing the sampling temperature) or decorate a core scaffold iteratively (by increasing generated token lengths).
* It can be used to bridge two molecules/molecular fragments.
* In the future, it might be adapated to predict structures with a specific chemical property using a regressor to rank predictions and transition towards more "goal-directed" predictions.


Chempleter operates internally on SELFIES token sequences, which provide a robust representation of molecular structures with guaranteed syntactic validity. 
While Chempleter accepts molecular input and outputs generated molecules in standard SMILES notation, all training and inference is is performed in SELFIES space, 
ensuring that generated molecules conform to chemical syntax.

Due to its simple recurrent architecture and small vocabulary size, Chempleter is computationally efficient and runs comfortably on both CPUs and GPUs. 
However, this also has limitations. In particular, the GRU-based architecture may struggle to model long-range structural dependencies, and the model does not currently optimize for chemical properties, 
biological activity, or synthetic feasibility. Generated molecules are guaranteed to be syntactically valid but may require additional filtering or scoring to identify chemically meaningful candidates. 

Overall, Chempleter prioritizes an approach which is easier to understand, train and deploy making it well suited for rapid prototyping and educational purposes in molecular machine learning.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents:

   usage
   installation
   demo
   validation
   

