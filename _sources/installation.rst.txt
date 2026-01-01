Installation
=========================

Prequisites
^^^^^^^^^^^^^^

* Python>=3.12
* `uv <https://docs.astral.sh/uv/>`_ (optional but recommended)

.. tab-set::
    :sync-group: installation

    .. tab-item:: Using uv
        :sync: key1

        ``uv pip install chempleter``

    .. tab-item:: Using pip
        :sync: key2

        ``python -m pip install chempleter``

    .. tab-item:: Install for development

        * Clone the chempleter repo:
            
            ``git clone https://github.com/davistdaniel/chempleter.git``
        
        * Inside the project directory, execute in a terminal:

            * For CPU:

                ``uv sync``
            
            * For GPU, CUDA 12.8:
                
                ``uv sync --extra gpu128``


.. note::
    
    By default, the CPU version of pytorch will be installed and used by Chempleter. 
    Alternatively, you can install a PyTorch version compatible with your CUDA version by following the `Pytorch documentation <https://pytorch.org/get-started/locally/>`_.


Starting Chempleter's GUI
-------------------------------

.. tab-set::
    :sync-group: installation

    .. tab-item:: Installed using uv
        :sync: key1

        ``uv run chempleter-gui``

    .. tab-item:: Installed using pip
        :sync: key2

        ``python -m chempleter.gui``


Importing chempleter as a library
--------------------------------------

``import chempleter``


Using chempleter
--------------------------------------
See :doc:`using chempleter <usage>`.