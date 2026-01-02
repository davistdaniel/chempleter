.. chempleter documentation master file, created by
   sphinx-quickstart on Thu Jan  1 12:33:30 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Chempleter
===========================================

Chempleter is a lightweight generative model which utilises a simple Gated Recurrent Unit (GRU) to predict syntactically valid extensions of a provided molecular fragment.
It accepts `SMILES <https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System>`_ notation as input and enforces chemical syntax validity using `SELFIES <https://selfies.readthedocs.io/en/latest/>`_ for the generated molecules. 
The library also has an optional graphical user interface to interact with the model.

.. image::
   images/chempleter_in_action.gif
   :align: center
   :scale: 60 %


What can chempleter do?
----------------------------

* Currently, Chempleter accepts an initial molecule/molecular fragment in SMILES format and generates a larger molecule with that intial structure included, while respecting chemical syntax. It also shows some interesting descriptors.
* It can be used to generate a wide range of structural analogs which the share same core structure (by changing the sampling temperature) or decorate a core scaffold iteratively (by increasing generated token lengths).
* It can be used to bridge two molecules/molecular fragments.
* In the future, it might be adapated to predict structures with a specific chemical property using a regressor to rank predictions and transition towards more "goal-directed" predictions.

To get started, see :doc:`getting started <usage>`.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents:

   usage
   installation
   

