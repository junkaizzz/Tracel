Extra: Running on a Slurm cluster
=================================

A whole Xenium slide has a few thousand OT2 windows (2,436 for Sample2). Each window is solved on its own, so the
work is split across a Slurm job array, eight windows per task. Finished windows are skipped, so the array can be
resubmitted after failures.

The repository contains a complete example in ``examples/sample2``:

.. code-block:: bash

    cd examples/sample2
    export WORK=/path/to/output TRACEL_ENV=/path/to/conda/env ACCOUNT=your_slurm_account
    export XENIUM_DIR=/path/to/xenium_sample2 SCRNA_DIR=/path/to/SCP1039
    bash run_sample2.sh      # builds the inputs, runs OT1, prepares OT2 windows
    # when both jobs have finished:
    bash submit_ot2.sh       # OT2 window array + merge (afterany dependency)

``NO_STRUCTURE=1 bash run_sample2.sh`` runs OT1 without the structure term.

One array task looks like this:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --cpus-per-task=2
    #SBATCH --mem=16G
    #SBATCH --time=24:00:00
    export OMP_NUM_THREADS=1 JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=false XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    start=$(( SLURM_ARRAY_TASK_ID * 8 )); end=$(( start + 7 ))
    tracel ot2-run --run-dir "$WORK/ot2" --soft "$WORK/ot1/ot1_soft_expression.h5ad" --windows "${start}-${end}"

Resources observed for Sample2 (about 15.6 million transcripts):

.. list-table::
   :header-rows: 1

   * - Step
     - Resources
     - Time
   * - OT1
     - 16 CPU cores, about 12 GB peak
     - about 1.5 h
   * - OT2, one task of 8 windows
     - 2 CPU cores, < 8 GB
     - median 1.5 h (each window about 12 min)
   * - OT2, all 2,436 windows
     - 40 concurrent tasks
     - about 11–12 h
   * - Merge
     - 4 CPU cores, about 11 GB
     - about 20 min

.. tip::

    JAX creates a thread pool sized to the visible CPU cores. On shared login nodes with a per-user thread limit,
    run only one ``tracel`` process at a time or restrict cores with ``taskset``.
