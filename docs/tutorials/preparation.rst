Stage 1: Data preparation
=========================

Tracel needs four inputs:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Input
     - Content
     - Used by
   * - Spatial cells
     - ``AnnData``, cells x genes, raw counts in ``X``, a cell-group label column in ``obs``
     - OT1
   * - scRNA-seq reference
     - ``AnnData``, cells x genes, raw counts in ``X``, a cell-group label column in ``obs``
     - OT1
   * - Transcript table
     - one row per transcript: id, gene, x, y (µm); optional UMI ``count`` for weighted records
     - OT2
   * - Cell table
     - one row per initial cell: id, centroid x, centroid y (same frame and units as transcripts)
     - OT2

The initial cells can come from the platform's default segmentation (e.g. Xenium Onboard Analysis), from
nucleus segmentation, or from any other method. Tracel keeps these cells and their centroids fixed.

Spatial cells
-------------

For Xenium, the initial cell-by-gene matrix is the 10x ``cell_feature_matrix.h5``. Labels can be any annotation
of the initial cells, for example published annotations or clusters of the initial matrix:

.. code-block:: python

    import pandas as pd
    import scanpy as sc

    xenium = "/path/to/xenium_outs"
    spatial = sc.read_10x_h5(f"{xenium}/cell_feature_matrix.h5")
    spatial.var_names_make_unique()
    labels = pd.read_csv("spatial_annotation.csv", index_col="cell_id")["celltype"]
    spatial.obs["celltype"] = labels.reindex(spatial.obs_names).fillna("Unknown").astype(str)
    spatial.write_h5ad("spatial_counts.h5ad")

scRNA-seq reference
-------------------

Any tissue-matched reference with cell-group labels works. Label names do **not** need to match the spatial
labels: the structure term of OT1 compares relationships among groups within each dataset, not the label
strings.

.. code-block:: python

    reference = sc.read_10x_mtx("/path/to/reference/expression", var_names="gene_symbols")
    meta = pd.read_csv("/path/to/reference/metadata.csv", index_col=0)
    reference = reference[reference.obs_names.isin(meta.index)].copy()
    reference.obs["celltype"] = meta.loc[reference.obs_names, "celltype"].astype(str)
    reference.write_h5ad("scrna_counts.h5ad")

Genes default to the intersection of the two ``var_names``. Counts are normalized within these genes
(``normalize_total(1e4)`` followed by ``log1p``); pass ``input_is_lognorm=True`` if your matrices are already
log-normalized.

Transcript and cell tables
--------------------------

:func:`tracel.prepare_ot2_inputs` reads the two tables, standardizes the columns and creates overlapping
250 µm windows (125 µm step). Column names default to the Xenium output:

.. code-block:: python

    import tracel

    tracel.prepare_ot2_inputs(
        f"{xenium}/transcripts.parquet",      # transcript_id, feature_name, x_location, y_location
        f"{xenium}/cells.parquet",            # cell_id, x_centroid, y_centroid
        "out/ot2",
    )

For CosMx or other formats, pass the column names explicitly (``mol_id_col``, ``gene_col``, ``x_col``,
``y_col``, ``cell_id_col``, ``cell_x_col``, ``cell_y_col``) and convert coordinates to micrometers first.
By default every transcript is used; ``min_qv=20`` drops low-quality Xenium transcripts if desired.

This writes ``transcripts_min.parquet``, ``cells_min.parquet``, ``windows.csv`` and ``prepare_config.json``
into ``out/ot2``.
