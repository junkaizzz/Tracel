Command-line interface
======================

Installing the package provides the ``tracel`` command with four subcommands. Every option of
:class:`tracel.OT1Params` and :class:`tracel.OT2Params` is available as a flag (underscores become dashes, e.g.
``--tau-a``); run ``tracel <subcommand> -h`` for the full list.

``tracel ot1``
--------------

.. code-block:: text

    tracel ot1 --spatial SPATIAL.h5ad --reference REFERENCE.h5ad
               --spatial-label KEY --reference-label KEY --out DIR
               [--no-structure] [--input-lognorm] [--alpha 0.9] [--rank 20] ...

Writes ``ot1_soft_expression.h5ad``, ``spatial_lognorm.h5ad``, ``ot1_coupling_factors.npz`` and
``ot1_summary.json``. ``--no-structure`` switches the structure term off.

``tracel ot2-prepare``
----------------------

.. code-block:: text

    tracel ot2-prepare --transcripts TRANSCRIPTS.parquet --cells CELLS.parquet --run-dir DIR
                       [--window-size 250] [--step 125] [--pad 60]
                       [--mol-id-col transcript_id] [--gene-col feature_name] [--x-col x_location] [--y-col y_location]
                       [--cell-id-col cell_id] [--cell-x-col x_centroid] [--cell-y-col y_centroid]
                       [--count-col COLUMN] [--min-qv 20]

``tracel ot2-run``
------------------

.. code-block:: text

    tracel ot2-run --run-dir DIR --soft ot1_soft_expression.h5ad [--windows 0-99] [--overwrite]
                   [--epsilon 0.05] [--hard-cut-um 12] ...

``--windows`` accepts ranges and lists (``0-99``, ``3,7,12``). Finished windows are skipped unless ``--overwrite``.

``tracel ot2-merge``
--------------------

.. code-block:: text

    tracel ot2-merge --run-dir DIR --soft ot1_soft_expression.h5ad [--threshold 0.1] [--out DIR]

Writes ``ot2_counts.h5ad``, ``ot2_lognorm.h5ad``, ``molecule_assignments.parquet``,
``rejected_assignments.parquet`` and ``ot2_summary.json`` (default ``DIR/threshold_0p1``).
