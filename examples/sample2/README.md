# Example: Xenium Sample2 (breast) with the SCP1039 reference

```bash
cd examples/sample2
export WORK=/path/to/output TRACEL_ENV=/path/to/conda/env ACCOUNT=your_slurm_account
export XENIUM_DIR=/path/to/xenium_sample2 SCRNA_DIR=/path/to/SCP1039
bash run_sample2.sh            # 1) inputs + OT1 job, 2) ot2-prepare job
# after both jobs finish:
bash submit_ot2.sh             # 3) OT2 window array (2,436 windows), 4) merge with threshold 0.1
```

* `NO_STRUCTURE=1 bash run_sample2.sh` runs OT1 without the structure term.
* Resources used for the paper: OT1 16 CPUs, ~1.5 h, ~12 GB peak; OT2 2 CPUs and < 8 GB per task of 8 windows,
  ~11 h for all windows at 40 concurrent tasks; merge ~20 min.
* `submit_ot2.sh` can be rerun after failures: finished windows are skipped.

`make_inputs.py` builds `spatial_counts.h5ad` from the 10x `cell_feature_matrix.h5` (Q20 transcripts per 10x cell)
with the author annotations (`Cell_Barcode_Type_Matrices.xlsx`, sheet "Xenium Fig 6 (unsupervised)"), and
`scrna_counts.h5ad` from SCP1039 joined to `Whole_miniatlas_meta.csv` (`celltype_minor`). OT2 uses every
transcript in `transcripts.parquet` (no qv filter), as in the paper. The paper's run kept the spatial cells in a
different order (sorted by distance from the slide center); cell order can change low-rank solver results slightly.
