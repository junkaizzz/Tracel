Extra: Sequencing-based data (Visium HD)
========================================

Visium HD reports UMI counts in 2 µm × 2 µm bins rather than individual transcript positions. Tracel treats each
non-zero bin–gene pair as one **weighted record** located at the bin center, with weight equal to its UMI count.

Initial cells
-------------

Any set of initial cells with centroids works. In the manuscript, nuclei were detected on the H&E image with
StarDist, each 2 µm bin was given to its nearest nucleus within 12 µm, and the bins of each nucleus were summed
into the initial cell-by-gene matrix used by OT1.

Weighted records
----------------

Build a table with one row per bin–gene pair and pass the count column:

.. code-block:: python

    import numpy as np
    import tracel

    # records: DataFrame with bin center coordinates (µm), gene and UMI count
    records["record_id"] = np.arange(len(records))
    tracel.prepare_ot2_inputs(records, cells, "out/ot2",
                              mol_id_col="record_id", gene_col="gene", x_col="x_um", y_col="y_um",
                              count_col="umi", cell_id_col="cell_id", cell_x_col="x_um", cell_y_col="y_um")

With a ``count`` column:

* the source mass of a record is proportional to its UMI count;
* a gene–window problem with a single candidate cell assigns all records to that cell, and a problem with a single
  record assigns it by its expression-weighted distance posterior; these direct assignments have no row mass and
  are always kept by the filter;
* each assigned record contributes its full UMI count to one cell (a record is not split between cells).

Restricting OT2 to informative genes
------------------------------------

Whole-transcriptome Visium HD data have tens of thousands of genes. In the manuscript OT2 was run on the 2,000
genes whose expression changed most in OT1 (mean absolute difference between the soft and the initial
log-normalized profiles):

.. code-block:: python

    import numpy as np

    delta = np.abs(result.soft.X - result.spatial_lognorm.X).mean(axis=0)
    genes = result.soft.var_names[np.argsort(-np.asarray(delta).ravel())[:2000]]
    result.soft[:, genes].copy().write_h5ad("out/ot1/ot1_soft_top2000.h5ad")
    records = records[records.gene.isin(genes)]

Then run OT2 with ``ot1_soft_top2000.h5ad`` as the soft profile.
