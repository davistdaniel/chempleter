Installation
=========================

Prequisites
^^^^^^^^^^^^^^

* Python>=3.12
* `uv <https://docs.astral.sh/uv/>`_ (optional but recommended)

.. tab-set::
   :sync-group: installation

   .. tab-item:: CPU

      .. tab-set::
         :sync-group: installation


         .. tab-item:: Using uv
            :sync: key1

            .. code-block:: bash

               uv pip install "chempleter[cpu]"

         .. tab-item:: Using pip
            :sync: key2

            .. code-block:: bash

               python -m pip install "chempleter[cpu]"

         .. tab-item:: Install for development

            * Clone the repo:

                .. code-block:: bash

                    git clone https://github.com/davistdaniel/chempleter.git

            * Inside the project directory:

                .. code-block:: bash

                    uv sync --dev --extra cpu --extra validation


   .. tab-item:: GPU and CUDA 12.8

      .. tab-set::
         :sync-group: installation

         .. tab-item:: Using uv
            :sync: key1

            .. code-block:: bash

               uv pip install "chempleter[gpu128]"

         .. tab-item:: Using pip
            :sync: key2

            .. code-block:: bash

               python -m pip install "chempleter[gpu128]"

         .. tab-item:: Install for development

            * Clone the repo:

                .. code-block:: bash

                    git clone https://github.com/davistdaniel/chempleter.git

            * Inside the project directory:


                .. code-block:: bash

                    uv sync --dev --extra gpu128 --extra validation


.. note::
    
    Alternatively, you can try using Chempleter with a PyTorch version compatible with your CUDA version by following the `Pytorch documentation <https://pytorch.org/get-started/locally/>`_.
    
.. note::

    To use the validation module of chempleter, you must install matplotlib and tqdm. In this case, you must install chempleter with:

    ``uv pip install "chempleter[cpu,validation]"``
    
    or for GPU, CUDA 12.8:

    ``uv pip install "chempleter[gpu128,validation]"``



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