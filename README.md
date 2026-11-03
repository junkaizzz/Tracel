# Tracel

Reference-guided transcript-to-cell assignment for imaging- and sequencing-based spatial transcriptomics with
two-stage optimal transport (OT).

* **OT1 — cell alignment.** Low-rank, unbalanced fused Gromov–Wasserstein (FGW) transport aligns the initial
  spatial cells with a tissue-matched scRNA-seq reference and estimates a soft expression profile for every
  spatial cell. A switch turns the cell-group **structure term** on (default) or off.
* **OT2 — transcript alignment.** For every gene in overlapping windows, unbalanced Sinkhorn transport moves
  transcripts to cells within 12 µm, using the OT1 profile as the cell-side marginal and a distance cost. Each
  transcript goes to its highest-probability cell; after overlapping windows are reconciled, an assignment is
  kept only if its **relative row mass** (transported mass / prescribed source mass) is at least 0.1.

The initial cells and their coordinates are kept fixed; Tracel changes which cell each transcript is counted in
and returns an integer cell-by-gene matrix.

## Installation

```bash
pip install -e /path/to/Tracel          # Python >= 3.10
```

Results in the paper were produced with numpy 2.3.5, scipy 1.16.3, pandas 2.3.3, anndata 0.12.10, scanpy 1.11.5,
jax/jaxlib 0.7.2, ott-jax 0.6.0 and moscot 0.5.0 (CPU, float32 JAX). Other versions may change solver outputs.

## Quick start (Python)

```python
import anndata as ad
import tracel

spatial = ad.read_h5ad("spatial_counts.h5ad")      # cells x genes, raw counts, obs["celltype"]
reference = ad.read_h5ad("scrna_counts.h5ad")      # cells x genes, raw counts, obs["celltype"]

# OT1 (use_structure=False switches the structure term off)
res = tracel.run_ot1(spatial, reference, "celltype", "celltype", tracel.OT1Params(use_structure=True))
res.write("out/ot1")                               # out/ot1/ot1_soft_expression.h5ad, ...

# OT2
tracel.prepare_ot2_inputs("transcripts.parquet", "cells.parquet", "out/ot2")   # Xenium column names by default
tracel.run_ot2_windows("out/ot2", "out/ot1/ot1_soft_expression.h5ad")          # or a subset of window ids
summary = tracel.merge_ot2("out/ot2", "out/ot1/ot1_soft_expression.h5ad", threshold=0.1)
```

## Command line

```bash
tracel ot1 --spatial spatial_counts.h5ad --reference scrna_counts.h5ad \
           --spatial-label celltype --reference-label celltype --out out/ot1 [--no-structure]
tracel ot2-prepare --transcripts transcripts.parquet --cells cells.parquet --run-dir out/ot2
tracel ot2-run   --run-dir out/ot2 --soft out/ot1/ot1_soft_expression.h5ad --windows 0-99   # one array task
tracel ot2-merge --run-dir out/ot2 --soft out/ot1/ot1_soft_expression.h5ad --threshold 0.1
```

`ot2-run` is resumable (finished windows are skipped) and windows are independent, so large datasets are split
across a Slurm array; see `examples/sample2/`.

## OT1

**Inputs.** Spatial and reference AnnData with raw counts in `X` (or log-normalized with `input_is_lognorm=True`)
and a cell-group label column in each. Label names do not need to match between the two datasets.
Genes default to the sorted intersection of both `var_names`; counts are normalized within those genes
(`normalize_total(1e4)` + `log1p`).

**Model.** Expression cost: squared Euclidean distance in a joint 50-PC PCA of both datasets after per-gene
scaling clipped at 10. Structure cost: within-dataset one-hot label distances (0 same group, 1 otherwise).
Marginals: every observed label gets the same total mass within its dataset. Low-rank (rank 20) unbalanced FGW
with `alpha = 0.9`, `epsilon = 0`, `tau_a = 0.8`, `tau_b = 0.9`, mean cost scaling.
The plan is kept as factors `P = Q diag(1/g) Rᵀ`; the dense matrix is never built.

**Soft profile.** `lambda_i = clip(rowmass_i / a_i, 0, 1)` and
`soft_i = lambda_i · barycenter_i + (1 − lambda_i) · x_i`, in log-normalized gene space.

**Structure switch.** `OT1Params(use_structure=False)` (CLI `--no-structure`) sets every structural distance to
zero and keeps everything else, including `alpha` and the label-based type-balanced marginals. The expression
coordinates are pre-divided by the square root of their mean cross-dataset squared distance and the solver uses
`scale_cost=1`, which equals mean cost scaling for the expression term.

| `OT1Params` | default | meaning |
|---|---|---|
| `use_structure` | `True` | include the cell-group structure term |
| `alpha` | 0.9 | FGW weight of the structure term |
| `epsilon` | 0.0 | entropic regularization |
| `rank` | 20 | rank of the coupling |
| `tau_a`, `tau_b` | 0.8, 0.9 | marginal relaxation (1 = balanced) |
| `batch_size` | 1024 | solver batch size |
| `n_pcs`, `scale_max_value` | 50, 10 | expression embedding |
| `seed` | 0 | PCA / initializer seed |

**Outputs** (`OT1Result.write`): `ot1_soft_expression.h5ad` (obs: `ot1_row_mass`, `ot1_source_marginal`,
`ot1_blend_fraction`), `spatial_lognorm.h5ad`, `ot1_coupling_factors.npz`, `ot1_summary.json`.

## OT2

**Inputs** (`prepare_ot2_inputs`). A transcript table (id, gene, x, y in µm; Xenium names by default) and a cell
table (id, centroid x/y in the same frame). Optional `count_col` marks weighted records such as Visium HD bin–gene
UMIs; optional `min_qv` drops low-quality transcripts (not applied by default). Windows: 250 µm, step 125 µm.

**Per gene and window.** Candidates are the union of cells within 12 µm of any transcript of that gene. Cost:
`(d/15)^2 + 10^6·1(d > 12)` (the optional `sigma`/`alpha` term acts only beyond `r0` = 15 µm, so with the 12 µm
cut it is always zero). Source mass uniform (or proportional to `count`);
target mass proportional to the gene's OT1 profile over the candidates. Unbalanced Sinkhorn: `epsilon = 0.05`,
`tau_a = 0.8`, `tau_b = 0.9`, up to 2,000 iterations, threshold `1e-3`. Each transcript goes to the argmax of its
row-normalized plan row; its raw (unnormalized) row mass is stored.

**Merge** (`merge_ot2`). Per transcript, the window with the highest row-normalized confidence wins. The winner is
kept if `row_mass / a_m ≥ threshold` (default 0.1); otherwise it is left unassigned and is not replaced by another
window. Rows with zero row mass (typically transcripts with no cell within 12 µm) are always rejected.

Unweighted gene–window problems with fewer than two transcripts or two candidate cells are skipped. Weighted records
follow the original Visium HD worker: one-candidate and one-record problems are assigned directly (no row mass) and
are always kept.

| `OT2Params` | default | meaning |
|---|---|---|
| `epsilon` | 0.05 | Sinkhorn regularization |
| `tau_a`, `tau_b` | 0.8, 0.9 | marginal relaxation |
| `max_iters`, `thresh` | 2000, 1e-3 | Sinkhorn stopping |
| `r0`, `p_in` | 15, 2 | distance scale (µm) and exponent of the cost |
| `sigma`, `alpha` | 5, 10 | optional term beyond `r0`; zero when `hard_cut_um` ≤ `r0` |
| `hard_cut_um`, `big_penalty` | 12, 1e6 | candidate radius and hard cut |

**Outputs** (`run_dir/threshold_0p1/` by default): `ot2_counts.h5ad` (integer counts), `ot2_lognorm.h5ad`
(`normalize_total(1e4)` + `log1p`, counts in `layers["counts"]`), `molecule_assignments.parquet`,
`rejected_assignments.parquet`, `ot2_summary.json`. Sinkhorn diagnostics are recorded per gene–window problem
(`worker_out/window_*/summary.csv`); they are informational and do not gate results.

## Reproducing the Sample2 analysis

See `examples/sample2/README.md`.

## Documentation

The documentation (Read the Docs theme, same layout as https://scglue.readthedocs.io) lives in `docs/`.

Build locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html      # open docs/_build/html/index.html
```

Publish on Read the Docs: push this repository to GitHub, sign in at https://readthedocs.org with GitHub,
choose "Import a Project" and select the repository. The build is configured by `.readthedocs.yaml`; heavy
dependencies are mocked in `docs/conf.py`, so the documentation builds without installing JAX or moscot.
