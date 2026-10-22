"""Build Tracel inputs for Xenium Sample2 (breast) with the SCP1039 scRNA-seq reference.

Writes spatial_counts.h5ad and scrna_counts.h5ad (raw counts + cell-group labels) into OUT.
Set XENIUM_DIR (Xenium output folder) and SCRNA_DIR (SCP1039 folder with expression/ and metadata/).
"""
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import scanpy as sc

XENIUM = Path(os.environ["XENIUM_DIR"])
SCRNA = Path(os.environ["SCRNA_DIR"])
LABELS = Path(os.environ.get("LABELS_XLSX", XENIUM / "Cell_Barcode_Type_Matrices.xlsx"))
SHEET = os.environ.get("LABELS_SHEET", "Xenium Fig 6 (unsupervised)")
OUT = Path(os.environ.get("OUT", "inputs"))


def clean_barcode(value):
    hits = re.findall(r"[ACGT]+", str(value).upper())
    return max(hits, key=len) if hits else ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Spatial: 10x cell-feature matrix (Q20 transcripts per 10x cell), author labels, centroids.
    sp = sc.read_10x_h5(XENIUM / "cell_feature_matrix.h5")
    sp.var_names_make_unique()
    sp.obs_names = sp.obs_names.astype(str)
    labels = pd.read_excel(LABELS, sheet_name=SHEET)[["Barcode", "Cluster"]].dropna().astype(str)
    sp.obs["celltype"] = pd.Categorical(sp.obs_names.map(dict(zip(labels.Barcode, labels.Cluster))).fillna("Unknown"))
    cells = pd.read_parquet(XENIUM / "cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"])
    cells["cell_id"] = cells.cell_id.astype(str)
    xy = cells.drop_duplicates("cell_id").set_index("cell_id").reindex(sp.obs_names)
    sp.obs["x_centroid"], sp.obs["y_centroid"] = xy.x_centroid.to_numpy(np.float32), xy.y_centroid.to_numpy(np.float32)
    sp.obsm["spatial"] = xy[["x_centroid", "y_centroid"]].to_numpy(np.float32)
    sp.write_h5ad(OUT / "spatial_counts.h5ad")
    # Reference: SCP1039 counts joined to metadata by cleaned barcode; unmatched cells dropped.
    ref = sc.read_10x_mtx(SCRNA / "expression", var_names="gene_symbols", cache=False)
    ref.var_names_make_unique()
    ref.obs_names = ref.obs_names.astype(str)
    meta = pd.read_csv(SCRNA / "metadata/Whole_miniatlas_meta.csv", low_memory=False)
    meta = meta[meta.NAME.astype(str) != "TYPE"].copy()
    meta["barcode"] = meta.NAME.astype(str).map(clean_barcode)
    meta = meta[meta.barcode != ""].drop_duplicates("barcode").set_index("barcode")
    key = pd.Index(ref.obs_names).map(clean_barcode)
    matched = key.isin(meta.index)
    ref = ref[np.asarray(matched)].copy()
    ref.obs["celltype"] = meta.loc[key[matched], "celltype_minor"].astype(str).to_numpy()
    ref.write_h5ad(OUT / "scrna_counts.h5ad")
    print(f"spatial {sp.shape}, reference {ref.shape} -> {OUT}")


if __name__ == "__main__":
    main()
