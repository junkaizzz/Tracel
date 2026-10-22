Installation guide
==================

Main package
------------

``tracel`` requires Python 3.10 or later. We recommend a fresh conda environment:

.. code-block:: bash

    conda create -n tracel python=3.11
    conda activate tracel
    pip install git+https://github.com/junkaizzz/Tracel.git

or, from a local copy of the repository:

.. code-block:: bash

    git clone https://github.com/junkaizzz/Tracel.git
    pip install -e Tracel

This installs the Python package and the ``tracel`` command-line tool. Its dependencies are
`numpy`, `pandas`, `scipy`, `pyarrow`, `anndata`, `scanpy`, `jax`, `ott-jax` and `moscot`.

.. note::

    The results in the manuscript were produced on CPUs (32-bit JAX) with numpy 2.3.5, scipy 1.16.3,
    pandas 2.3.3, anndata 0.12.10, scanpy 1.11.5, jax/jaxlib 0.7.2, ott-jax 0.6.0 and moscot 0.5.0.
    Other versions, in particular of ``ott-jax`` and ``moscot``, may change solver outputs. To reproduce them
    exactly, pin these versions:

    .. code-block:: bash

        pip install "numpy==2.3.5" "scipy==1.16.3" "pandas==2.3.3" "anndata==0.12.10" "scanpy==1.11.5" \
                    "jax==0.7.2" "jaxlib==0.7.2" "ott-jax==0.6.0" "moscot==0.5.0"

Optional dependencies
---------------------

* **Slurm.** OT2 is solved independently in many spatial windows. For whole-slide datasets we run the windows
  as a Slurm job array (see :doc:`tutorials/cluster`). No Slurm-specific Python package is needed.
* **Image tools** such as ``tifffile`` are only needed for your own visualization of morphology images; they are
  not used by ``tracel``.
