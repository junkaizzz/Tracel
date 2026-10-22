Stage 2: Cell alignment (OT1)
=============================

OT1 aligns the spatial cells with the scRNA-seq reference and returns a soft expression profile for every
spatial cell.

.. code-block:: python

    import anndata as ad
    import tracel

    spatial = ad.read_h5ad("spatial_counts.h5ad")
    reference = ad.read_h5ad("scrna_counts.h5ad")

    result = tracel.run_ot1(
        spatial, reference,
        spatial_label_key="celltype", reference_label_key="celltype",
        params=tracel.OT1Params(),          # paper defaults
    )
    result.write("out/ot1")

or from the command line:

.. code-block:: bash

    tracel ot1 --spatial spatial_counts.h5ad --reference scrna_counts.h5ad \
               --spatial-label celltype --reference-label celltype --out out/ot1

What OT1 computes
-----------------

* **Expression cost.** Both log-normalized matrices are concatenated; each gene is scaled (clipped at 10) and
  a joint 50-component PCA is fitted. The cost between a spatial cell and a reference cell is their squared
  Euclidean distance in this PCA space.
* **Structure cost.** Each cell-group label becomes a one-hot vector with entry :math:`1/\sqrt{2}`, so two cells
  of the same group have distance 0 and cells of different groups have distance 1, within each dataset.
* **Marginals.** Every observed group receives the same total mass within its dataset; the mass is split
  uniformly among the group's cells.
* **Solver.** Low-rank (rank 20) unbalanced fused Gromov–Wasserstein with :math:`\alpha = 0.9`,
  :math:`\varepsilon = 0`, :math:`\tau_a = 0.8`, :math:`\tau_b = 0.9` and mean cost scaling. The plan is kept as
  factors :math:`P = Q\,\mathrm{diag}(g)^{-1} R^\top` and is never materialized.
* **Soft profile.** With row mass :math:`\rho_i = \sum_j P_{ij}` and prescribed mass :math:`a_i`,

  .. math::

      \lambda_i = \mathrm{clip}(\rho_i / a_i, 0, 1), \qquad
      \tilde{x}_i = \lambda_i \, \frac{\sum_j P_{ij} x^{R}_j}{\rho_i} + (1 - \lambda_i)\, x^{S}_i ,

  so cells that the reference explains well take mostly reference expression, and the rest keeps more of its
  own expression.

Switching the structure term off
--------------------------------

.. code-block:: python

    result = tracel.run_ot1(spatial, reference, "celltype", "celltype",
                            tracel.OT1Params(use_structure=False))

(command line: ``--no-structure``). Every structural distance is set to zero; ``alpha``, the type-balanced
marginals and the blend are unchanged, so cell-group labels then enter only through the marginals. Because a
zero geometry cannot be mean-scaled, the expression PCA coordinates are divided by the square root of their mean
cross-dataset squared distance and the solver uses ``scale_cost=1``, which is the same normalization of the
expression cost.

Outputs
-------

``result.write("out/ot1")`` writes

* ``ot1_soft_expression.h5ad``: spatial cells x genes, soft log-normalized expression; ``obs`` gains
  ``ot1_row_mass``, ``ot1_source_marginal`` and ``ot1_blend_fraction``;
* ``spatial_lognorm.h5ad``: the log-normalized spatial input on the same axes;
* ``ot1_coupling_factors.npz``: ``q``, ``r``, ``g``, marginals and cell/gene ids;
* ``ot1_summary.json``: settings, sizes, solver convergence flag and blend statistics.

``ot1_soft_expression.h5ad`` is the input of OT2.

Main parameters
---------------

.. list-table::
   :header-rows: 1

   * - ``OT1Params``
     - default
     - meaning
   * - ``use_structure``
     - ``True``
     - include the cell-group structure term
   * - ``alpha``
     - 0.9
     - weight of the structure term; :math:`1-\alpha` weights expression
   * - ``epsilon``
     - 0.0
     - entropic regularization
   * - ``rank``
     - 20
     - rank of the coupling
   * - ``tau_a``, ``tau_b``
     - 0.8, 0.9
     - marginal relaxation of spatial / reference cells (1 = balanced)
   * - ``n_pcs``, ``scale_max_value``
     - 50, 10
     - expression embedding
   * - ``batch_size``, ``seed``
     - 1024, 0
     - solver batch size, random seed

A whole Xenium slide (about 140,000 spatial and 100,000 reference cells) took about 1.5 h on 16 CPU cores with a
peak memory of about 12 GB.
