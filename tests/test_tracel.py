"""Fast unit and end-to-end tests on synthetic data (real OTT solvers, tiny problems)."""
import json
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd

import tracel
from tracel import ot1, ot2


def toy_spatial(seed=0, n_cells=60, genes=("g0", "g1", "g2", "g3", "g4", "g5")):
    rng = np.random.default_rng(seed)
    labels = np.array(["A", "B", "C"])[np.arange(n_cells) % 3]
    profile = {"A": [9, 1, 1, 1, 2, 1], "B": [1, 9, 1, 2, 1, 1], "C": [1, 1, 9, 1, 1, 2]}
    counts = np.vstack([rng.poisson(profile[l]) for l in labels]).astype(np.float32)
    a = ad.AnnData(X=counts, obs=pd.DataFrame({"celltype": labels}, index=[f"c{i}" for i in range(n_cells)]),
                   var=pd.DataFrame(index=list(genes)))
    a.obs["x"] = (np.arange(n_cells) % 10) * 10.0 + 5
    a.obs["y"] = (np.arange(n_cells) // 10) * 10.0 + 5
    return a


class OT1Tests(unittest.TestCase):
    def test_marginal_and_structure(self):
        m = ot1.type_balanced_marginal(["A", "A", "B"])
        np.testing.assert_allclose(m, [0.25, 0.25, 0.5])
        z = ot1.structure_embedding(["A", "B"], ["A", "B"])
        self.assertAlmostEqual(float(np.sum((z[0] - z[1]) ** 2)), 1.0, places=6)

    def test_lowrank_blend_matches_dense(self):
        rng = np.random.default_rng(1)
        q, r, g = rng.random((5, 2)), rng.random((4, 2)), rng.random(2) + 0.5
        x_sp, x_rf = rng.random((5, 3)).astype(np.float32), rng.random((4, 3)).astype(np.float32)
        a = np.full(5, 0.2, dtype=np.float32)
        soft, mass, blend = ot1.lowrank_blend(q, r, g, x_sp, x_rf, a)
        P = q @ np.diag(1 / g) @ r.T
        np.testing.assert_allclose(mass, P.sum(1), rtol=1e-5)
        lam = np.clip(P.sum(1) / a, 0, 1)
        np.testing.assert_allclose(soft, lam[:, None] * (P @ x_rf) / P.sum(1)[:, None] + (1 - lam[:, None]) * x_sp, rtol=1e-4)

    def test_run_ot1_both_switches(self):
        sp, rf = toy_spatial(0), toy_spatial(1, 45)
        out = {}
        for flag in (True, False):
            res = tracel.run_ot1(sp, rf, "celltype", "celltype", tracel.OT1Params(use_structure=flag, rank=3, n_pcs=4))
            self.assertEqual(res.soft.shape, sp.shape)
            self.assertTrue(np.isfinite(res.soft.X).all())
            self.assertTrue(np.all((res.blend_fraction >= 0) & (res.blend_fraction <= 1)))
            out[flag] = res
            with tempfile.TemporaryDirectory() as tmp:
                res.write(tmp)
                self.assertTrue((Path(tmp) / "ot1_soft_expression.h5ad").exists())
        self.assertTrue(out[True].summary["params"]["use_structure"])
        self.assertFalse(out[False].summary["params"]["use_structure"])


class OT2Tests(unittest.TestCase):
    def test_windows_cover_bbox(self):
        w = tracel.make_windows(0, 0, 600, 300, 250, 125, 60)
        self.assertEqual(w.x1.max(), 600)
        self.assertEqual(w.y1.max(), 300)

    def test_threshold_and_winner(self):
        f = pd.DataFrame(dict(mol_id=[1, 1, 2, 3, 4], window_id=[0, 1, 0, 0, 0], assign_conf=[0.4, 0.8, 0.9, 0.9, 1.0],
                              raw_row_mass=[0.2, 0.0, 0.2, 0.01, np.nan], source_marginal=[0.25] * 5,
                              assignment_mode=[ot2.OT_MODE] * 4 + [ot2.DIRECT_MODES[0]]))
        w = ot2.select_winners(f)
        self.assertEqual(int(w.set_index("mol_id").loc[1, "window_id"]), 1)
        keep, ratio, reason = ot2.threshold_decisions(w, 0.1)
        self.assertEqual(dict(zip(w.mol_id, keep)), {1: False, 2: True, 3: False, 4: True})

    def test_end_to_end(self):
        sp = toy_spatial(0)
        rng = np.random.default_rng(3)
        rows = []
        for i, (cx, cy) in enumerate(zip(sp.obs.x, sp.obs.y)):
            for g, k in zip(sp.var_names, np.asarray(sp.X[i]).astype(int)):
                for _ in range(k):
                    rows.append((g, cx + rng.normal(0, 2), cy + rng.normal(0, 2)))
        tx = pd.DataFrame(rows, columns=["feature_name", "x_location", "y_location"])
        tx["transcript_id"] = np.arange(len(tx))
        cells = pd.DataFrame({"cell_id": sp.obs_names, "x_centroid": sp.obs.x, "y_centroid": sp.obs.y})
        soft = sp.copy()
        soft.X = np.log1p(soft.X)
        with tempfile.TemporaryDirectory() as tmp:
            tracel.prepare_ot2_inputs(tx, cells, tmp, window_size=60, step=30, pad=10)
            tracel.run_ot2_windows(tmp, soft)
            summary = tracel.merge_ot2(tmp, soft, threshold=0.1)
            counts = ad.read_h5ad(Path(tmp) / "threshold_0p1" / "ot2_counts.h5ad")
            self.assertEqual(counts.shape, sp.shape)
            self.assertEqual(int(counts.X.sum()), summary["retained_mass"])
            self.assertEqual(summary["retained_records"] + summary["rejected_records"], summary["assigned_records"])
            self.assertGreater(summary["retention_fraction"], 0.5)
            again = tracel.run_ot2_windows(tmp, soft)       # resumable: nothing left to do
            self.assertEqual(again, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
