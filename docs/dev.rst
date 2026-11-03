Developer guide
===============

Main components
---------------

* ``tracel/ot1.py``: input normalization, joint expression embedding, structure embedding, type-balanced
  marginals, the two FGW solve paths (with and without structure) and the low-rank blend.
* ``tracel/ot2.py``: window construction, the distance cost with its hard cut, the per-window solver, window
  bookkeeping on disk, overlap reconciliation, the relative row-mass filter and count matrices.
* ``tracel/cli.py``: the ``tracel`` command.

On-disk layout of an OT2 run
----------------------------

.. code-block:: text

    run_dir/
      windows.csv  transcripts_min.parquet  cells_min.parquet  prepare_config.json
      worker_out/window_000000/assignments.parquet   # one row per (window, transcript)
                               summary.csv           # one row per gene-window problem (Sinkhorn diagnostics)
                               _SUCCESS.json         # written last; marks the window as finished
      threshold_0p1/                                 # merge_ot2 outputs

``assignments.parquet`` columns: ``window_id``, ``gene``, ``mol_id``, ``assigned_cell``, ``assign_conf``,
``raw_row_mass`` (unnormalized plan row sum), ``source_marginal`` (:math:`a_m`), ``assignment_mode`` and, for
weighted records, ``count``. Because the raw row mass is stored, the filter threshold can be changed by merging
again without solving OT2.

Supporting a new platform
-------------------------

OT2 needs only a transcript (or weighted record) table and cell centroids in micrometers. Map the platform's
columns with the arguments of :func:`tracel.prepare_ot2_inputs`; nothing else is platform specific. OT1 needs
cell-by-gene counts and labels as :class:`anndata.AnnData`.

Numerical notes
---------------

* JAX runs in 32-bit precision. Some Sinkhorn problems report a non-finite error trace while returning a finite
  plan; these are recorded in ``summary.csv`` and are not used to discard results.
* ``jax_clear_cache_every`` only bounds compilation-cache memory and does not change results.

Contributions are welcome!
--------------------------

Please open an issue or a pull request on GitHub.
