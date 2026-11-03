Stage 3: Transcript alignment (OT2)
===================================

OT2 reassigns every transcript to a nearby cell. Windows are independent, so they can be solved in one process
(small data) or across a job array (whole slides, see :doc:`cluster`).

.. code-block:: python

    import tracel

    soft = "out/ot1/ot1_soft_expression.h5ad"
    tracel.run_ot2_windows("out/ot2", soft)                 # all windows; resumable
    summary = tracel.merge_ot2("out/ot2", soft, threshold=0.1)
    print(summary["retention_fraction"])

Command line:

.. code-block:: bash

    tracel ot2-run   --run-dir out/ot2 --soft out/ot1/ot1_soft_expression.h5ad            # all windows
    tracel ot2-run   --run-dir out/ot2 --soft out/ot1/ot1_soft_expression.h5ad --windows 0-99
    tracel ot2-merge --run-dir out/ot2 --soft out/ot1/ot1_soft_expression.h5ad --threshold 0.1

What OT2 computes
-----------------

For each gene :math:`g` in each window :math:`W`:

1. **Candidates.** The union of cells within 12 µm of any transcript of :math:`g` in :math:`W`.
2. **Cost.** For distance :math:`d` (µm) between a transcript and a candidate cell centroid,

   .. math::

       C = \left(\frac{d}{s}\right)^2 + 10^6\,\mathbf{1}(d > \delta),

   with distance scale :math:`s = 15` µm and cut-off :math:`\delta = 12` µm. In the entropically regularized
   plan, the attraction of a cell for a transcript decays as :math:`\exp\!\bigl(-d^2/(s^2\varepsilon)\bigr)`, and
   pairs farther than :math:`\delta` apart are excluded. (``OT2Params`` also has an optional term
   :math:`\alpha\bigl[\exp\bigl((d-s)_+/\sigma\bigr) - 1\bigr]` for distances beyond :math:`s`; with
   :math:`\delta \le s` it is always zero.)

3. **Marginals.** Transcripts receive equal mass (or mass proportional to their UMI count for weighted
   records). Cells receive mass proportional to their OT1 soft expression of :math:`g`.
4. **Transport.** Unbalanced Sinkhorn with :math:`\varepsilon = 0.05`, :math:`\tau_a = 0.8`,
   :math:`\tau_b = 0.9`, up to 2,000 iterations and threshold :math:`10^{-3}`.
5. **Assignment.** Each transcript goes to the cell with the largest entry in its row-normalized plan row; that
   entry is the assignment confidence. The unnormalized row mass :math:`\sum_j T_{mj}` is stored.

Merging and the relative row-mass filter
----------------------------------------

Windows overlap, so a transcript can be assigned several times. :func:`tracel.merge_ot2` keeps the assignment
with the highest confidence and then keeps it only if

.. math::

    r_m = \frac{\sum_j T_{mj}}{a_m} \ge 0.1,

where :math:`a_m` is the transcript's prescribed source mass in the winning window. Otherwise the transcript is
left unassigned; it is not replaced by another window's assignment. Transcripts with no cell within 12 µm end
up with zero row mass and are therefore always left unassigned.

Outputs
-------

``out/ot2/threshold_0p1/`` contains

* ``ot2_counts.h5ad``: integer cell-by-gene counts on the OT1 cell and gene axes;
* ``ot2_lognorm.h5ad``: ``normalize_total(1e4)`` + ``log1p`` of the counts (raw counts in ``layers["counts"]``);
* ``molecule_assignments.parquet``: retained assignments (``mol_id``, ``gene``, ``assigned_cell``,
  ``assign_conf``, ``raw_row_mass``, ``relative_row_mass``, ...);
* ``rejected_assignments.parquet``: assignments removed by the filter (``proposed_cell``);
* ``ot2_summary.json``: input mass, retained and rejected mass, retention fraction and Sinkhorn diagnostics.

Per-window results are in ``out/ot2/worker_out/window_XXXXXX/`` (``assignments.parquet``, ``summary.csv`` with one
row per gene–window problem, ``_SUCCESS.json``). Sinkhorn diagnostics are informational; no result is discarded
because of them.

Trying another threshold does not require solving OT2 again:

.. code-block:: bash

    tracel ot2-merge --run-dir out/ot2 --soft out/ot1/ot1_soft_expression.h5ad --threshold 0.05

Main parameters
---------------

.. list-table::
   :header-rows: 1

   * - ``OT2Params``
     - default
     - meaning
   * - ``epsilon``
     - 0.05
     - Sinkhorn entropic regularization
   * - ``tau_a``, ``tau_b``
     - 0.8, 0.9
     - marginal relaxation of transcripts / cells
   * - ``max_iters``, ``thresh``
     - 2000, 1e-3
     - Sinkhorn stopping rule
   * - ``r0``, ``p_in``
     - 15, 2
     - distance scale :math:`s` (µm) and exponent of the cost
   * - ``sigma``, ``alpha``
     - 5, 10
     - optional term beyond ``r0``; zero when ``hard_cut_um`` :math:`\le` ``r0``
   * - ``hard_cut_um``, ``big_penalty``
     - 12, 1e6
     - candidate radius and penalty beyond it
   * - ``threshold`` (merge)
     - 0.1
     - minimum relative row mass
