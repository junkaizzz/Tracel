"""OT2: reassign transcripts to nearby cells with gene-wise unbalanced optimal transport.

For every gene in every overlapping window, transcripts (sources) are transported to the union of cells
within ``hard_cut_um`` of any of those transcripts (targets). The target marginal is the gene's OT1 soft
expression over the candidate cells; the cost is a hinge-exponential function of distance with a large
penalty beyond ``hard_cut_um``. Each transcript goes to the cell with the largest entry in its row of the
transport plan. The unnormalized row mass is recorded so that, after overlapping windows are reconciled
(highest row-normalized confidence wins), an assignment is kept only if ``row_mass / a_m >= threshold``.

Unweighted records (one row per transcript; Xenium, CosMx) follow the original worker exactly: gene-window
problems with fewer than two transcripts or two candidate cells are skipped. Weighted records (a ``count``
column, e.g. Visium HD bin-gene UMIs) use count-proportional source mass and assign one-candidate and
one-record problems directly, as in the original weighted worker; those direct assignments carry no row
mass and are always kept.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from scipy import sparse
from scipy.spatial import cKDTree

__all__ = ["OT2Params", "make_windows", "prepare_ot2_inputs", "solve_window", "run_ot2_windows", "merge_ot2"]

EPS_SAFE = 1e-12
OT_MODE = "ott_unbalanced_sinkhorn"
DIRECT_MODES = ("direct_only_candidate_cell", "direct_single_source_expression_distance_posterior")
MODES = (OT_MODE,) + DIRECT_MODES


@dataclass
class OT2Params:
    """OT2 settings. Defaults are the values used in the manuscript.

    The cost between a transcript and a candidate cell at distance ``d`` (um) is
    ``(d / r0) ** p_in + big_penalty * (d > hard_cut_um)``, i.e. ``(d/15)^2 + 1e6 * (d > 12)`` with the defaults.
    An optional term ``alpha * (exp(max(d - r0, 0) / sigma) - 1)`` is added for ``d > r0``; it is zero whenever
    ``hard_cut_um <= r0``, as with the defaults.

    Parameters
    ----------
    tau_a, tau_b
        Marginal relaxation of the transcripts and of the cells (1 = balanced).
    epsilon
        Sinkhorn entropic regularization.
    max_iters, thresh
        Sinkhorn iteration limit and convergence threshold.
    r0, p_in
        Distance scale (um) and exponent of the cost.
    sigma, alpha
        Length scale (um) and weight of the optional term beyond ``r0`` (zero when ``hard_cut_um <= r0``).
    hard_cut_um
        Candidate radius and hard-cut distance (um).
    big_penalty
        Cost added beyond ``hard_cut_um``.
    jax_clear_cache_every
        Clear JAX compilation caches every N solved genes. Bounds memory; does not change results.
    """

    tau_a: float = 0.8
    tau_b: float = 0.9
    epsilon: float = 0.05
    max_iters: int = 2000
    thresh: float = 1e-3
    r0: float = 15.0
    p_in: float = 2.0
    sigma: float = 5.0
    alpha: float = 10.0
    hard_cut_um: float = 12.0
    big_penalty: float = 1e6
    jax_clear_cache_every: int = 10


# ----------------------------------------------------------------------------------------------------------
# Inputs and windows
# ----------------------------------------------------------------------------------------------------------
def make_windows(xmin, ymin, xmax, ymax, window_size=250.0, step=125.0, pad=60.0) -> pd.DataFrame:
    """Overlapping square windows covering the bounding box (the last window is aligned to the far edge)."""
    def starts(vmin, vmax):
        if vmax - vmin <= window_size:
            return np.array([vmin], dtype=np.float64)
        arr = np.arange(vmin, vmax - window_size + 1e-9, step, dtype=np.float64)
        last = vmax - window_size
        if len(arr) == 0 or abs(arr[-1] - last) > 1e-6:
            arr = np.append(arr, last)
        return arr

    rows, wid = [], 0
    for x0 in starts(xmin, xmax):
        for y0 in starts(ymin, ymax):
            x1, y1 = x0 + window_size, y0 + window_size
            rows.append(dict(window_id=wid, x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                             xp0=float(x0 - pad), yp0=float(y0 - pad), xp1=float(x1 + pad), yp1=float(y1 + pad)))
            wid += 1
    return pd.DataFrame(rows)


def _read(table):
    return pd.read_parquet(table) if isinstance(table, (str, os.PathLike)) else table.copy()


def prepare_ot2_inputs(transcripts, cells, run_dir, *, window_size=250.0, step=125.0, pad=60.0,
                       mol_id_col="transcript_id", gene_col="feature_name", x_col="x_location",
                       y_col="y_location", count_col=None, cell_id_col="cell_id", cell_x_col="x_centroid",
                       cell_y_col="y_centroid", min_qv=None, qv_col="qv"):
    """Write ``transcripts_min.parquet``, ``cells_min.parquet`` and ``windows.csv`` into ``run_dir``.

    ``transcripts`` / ``cells`` are DataFrames or parquet paths; defaults are Xenium column names.
    ``count_col`` marks weighted records (e.g. Visium HD bin-gene UMI counts). ``min_qv`` optionally drops
    transcripts with ``qv < min_qv`` (the paper used no filter for Sample2/Janesick2023 and 20 for Xenium lung).
    Cell coordinates must be in the same units and frame as transcript coordinates (micrometers).
    """
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    tx = _read(transcripts)
    if min_qv is not None:
        tx = tx[tx[qv_col] >= float(min_qv)]
    keep = [mol_id_col, gene_col, x_col, y_col] + ([count_col] if count_col else [])
    tx = tx[keep].rename(columns={mol_id_col: "mol_id", gene_col: "gene", x_col: "x", y_col: "y"})
    if count_col:
        tx = tx.rename(columns={count_col: "count"})
    tx["gene"] = tx["gene"].astype(str)
    tx["x"] = pd.to_numeric(tx["x"], errors="coerce")
    tx["y"] = pd.to_numeric(tx["y"], errors="coerce")
    tx = tx.dropna(subset=["mol_id", "gene", "x", "y"])
    tx["mol_id"] = pd.to_numeric(tx["mol_id"], errors="raise").astype(np.int64)
    if tx["mol_id"].duplicated().any():
        raise ValueError("mol_id must be unique")
    if count_col:
        counts = tx["count"].to_numpy()
        if (counts <= 0).any() or not np.all(counts == np.floor(counts)):
            raise ValueError("Weighted records need positive integer counts")
        tx["count"] = tx["count"].astype(np.int32)
    cl = _read(cells)[[cell_id_col, cell_x_col, cell_y_col]]
    cl = cl.rename(columns={cell_id_col: "cell_id", cell_x_col: "x", cell_y_col: "y"})
    cl["cell_id"] = cl["cell_id"].astype(str)
    cl["x"] = pd.to_numeric(cl["x"], errors="coerce")
    cl["y"] = pd.to_numeric(cl["y"], errors="coerce")
    cl = cl.dropna(subset=["cell_id", "x", "y"])
    bbox = dict(xmin=float(min(tx.x.min(), cl.x.min())), ymin=float(min(tx.y.min(), cl.y.min())),
                xmax=float(max(tx.x.max(), cl.x.max())), ymax=float(max(tx.y.max(), cl.y.max())))
    windows = make_windows(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"], window_size, step, pad)
    tx.to_parquet(run / "transcripts_min.parquet", index=False)
    cl.to_parquet(run / "cells_min.parquet", index=False)
    windows.to_csv(run / "windows.csv", index=False)
    config = dict(window_size_um=window_size, step_um=step, pad_um=pad, bbox=bbox, n_windows=int(len(windows)),
                  n_transcripts=int(len(tx)), n_cells=int(len(cl)), weighted=bool(count_col), min_qv=min_qv)
    (run / "prepare_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


# ----------------------------------------------------------------------------------------------------------
# Cost and per-window solve
# ----------------------------------------------------------------------------------------------------------
def _cost_class():
    import jax
    import jax.numpy as jnp
    from ott.geometry import costs

    @jax.tree_util.register_pytree_node_class
    class HardCutHingeExpDistanceCost(costs.CostFn):
        """(d/r0)^p + alpha*(exp(max(d-r0,0)/sigma)-1), plus big_penalty when d > hard_cut_um."""

        def __init__(self, r0, sigma, alpha, p, hard_cut_um, big_penalty):
            super().__init__()
            self.r0, self.sigma, self.alpha, self.p = (jnp.asarray(v) for v in (r0, sigma, alpha, p))
            self.hard_cut_um, self.big_penalty = jnp.asarray(hard_cut_um), jnp.asarray(big_penalty)

        def __call__(self, x, y):
            d = jnp.linalg.norm(x - y)
            normal = (d / self.r0) ** self.p + self.alpha * jnp.expm1(jnp.maximum(d - self.r0, 0.0) / self.sigma)
            return jnp.where(d <= self.hard_cut_um, normal, self.big_penalty + normal)

        def tree_flatten(self):
            return (self.r0, self.sigma, self.alpha, self.p, self.hard_cut_um, self.big_penalty), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(*children)

    return HardCutHingeExpDistanceCost


def _diagnostics(out, max_iters, threshold):
    import jax

    converged = getattr(out, "converged", None)
    n_iters = getattr(out, "n_iters", None)
    n_iters = -1 if n_iters is None else int(np.asarray(jax.device_get(n_iters)).item())
    errors = getattr(out, "errors", None)
    last = minimum = np.nan
    used_n = 0
    if errors is not None:
        err = np.asarray(jax.device_get(errors), dtype=np.float64).ravel()
        used = err[err != -1]
        finite = used[np.isfinite(used) & (used >= 0)]
        used_n = int(used.size)
        if finite.size:
            last, minimum = float(finite[-1]), float(finite.min())
    return dict(sinkhorn_converged=False if converged is None else bool(np.asarray(jax.device_get(converged)).item()),
                sinkhorn_n_iters=n_iters, sinkhorn_reached_max_iters=bool(n_iters >= int(max_iters)),
                sinkhorn_last_error=last, sinkhorn_min_error=minimum, sinkhorn_error_entries_used=used_n,
                sinkhorn_threshold=float(threshold), sinkhorn_max_iters=int(max_iters))


def _direct_diagnostics(p):
    return dict(sinkhorn_converged=True, sinkhorn_n_iters=0, sinkhorn_reached_max_iters=False,
                sinkhorn_last_error=0.0, sinkhorn_min_error=0.0, sinkhorn_error_entries_used=0,
                sinkhorn_threshold=float(p.thresh), sinkhorn_max_iters=int(p.max_iters))


class _SoftMarginals:
    """Per-gene cell-side marginals from the OT1 soft expression: max(x,0)+1e-12, normalized over all cells."""

    def __init__(self, soft: ad.AnnData):
        self.soft = soft
        self.genes = pd.Index(soft.var_names.astype(str))
        self.cache = {}

    def __call__(self, gene):
        if gene not in self.cache:
            column = self.soft.X[:, int(self.genes.get_loc(gene))]
            column = column.toarray().ravel() if hasattr(column, "toarray") else np.asarray(column).ravel()
            column = np.maximum(column.astype(np.float64), 0.0) + EPS_SAFE
            total = float(column.sum())
            self.cache[gene] = (np.full(self.soft.n_obs, 1.0 / self.soft.n_obs, dtype=np.float32) if total <= 0
                                else (column / total).astype(np.float32))
        return self.cache[gene]


def _direct_singleton(mol_xy, cell_xy, b, p):
    distances = np.linalg.norm(cell_xy - mol_xy[0], axis=1)
    cost = (distances / p.r0) ** p.p_in + p.alpha * np.expm1(np.maximum(distances - p.r0, 0.0) / p.sigma)
    score = np.log(np.maximum(b.astype(np.float64), EPS_SAFE)) - cost / max(p.epsilon, EPS_SAFE)
    score -= score.max()
    prob = np.exp(score)
    prob /= prob.sum()
    best = np.asarray([int(np.argmax(prob))], dtype=np.int64)
    return best, np.asarray([float(prob[best[0]])], dtype=np.float32)


def solve_window(window_id, tx_w, cells, marginals, params: OT2Params | None = None, weighted=False):
    """Solve every gene-window problem in one window.

    ``tx_w``: transcripts inside the window (columns mol_id, gene, x, y[, count]) in file order.
    ``cells``: dict with ``xy`` (float32 array), ``ids`` (str array) and ``tree`` (cKDTree), in OT1 soft order.
    ``marginals``: callable gene -> per-cell OT1 marginal (see :class:`_SoftMarginals`).
    Returns (assignments DataFrame, list of per-gene diagnostics).
    """
    import jax
    import jax.numpy as jnp
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    p = params or OT2Params()
    cost_class = _cost_class()
    genes_here = [g for g in tx_w["gene"].astype(str).unique().tolist() if g in marginals.genes]
    frames, summary, solved = [], [], 0
    for gene in genes_here:
        sub = tx_w.loc[tx_w["gene"].astype(str) == gene, ["mol_id", "x", "y"] + (["count"] if weighted else [])]
        mol_ids = sub["mol_id"].to_numpy(dtype=np.int64)
        mol_xy = sub[["x", "y"]].to_numpy(dtype=np.float32)
        m = mol_xy.shape[0]
        if m == 0 or (not weighted and m < 2):
            continue
        neighbors = cells["tree"].query_ball_point(mol_xy, r=float(p.hard_cut_um))
        flat = [np.asarray(lst, dtype=np.int64) for lst in neighbors if len(lst) > 0]
        if not flat:
            continue
        candidates = np.unique(np.concatenate(flat))
        if not weighted and candidates.size < 2:
            continue
        cell_xy, cell_ids, n = cells["xy"][candidates], cells["ids"][candidates], len(candidates)
        if weighted:
            mol_counts = sub["count"].to_numpy(dtype=np.int32)
            a = mol_counts.astype(np.float64)
            a = (a / a.sum()).astype(np.float32)
        else:
            a = np.full((m,), 1.0 / float(m), dtype=np.float32)
        b = marginals(gene)[candidates].astype(np.float64) + EPS_SAFE
        b_sum = float(b.sum())
        if b_sum <= 0:
            continue
        b = (b / b_sum).astype(np.float32)
        raw_row_mass = np.full(m, np.nan, dtype=np.float32)
        if weighted and n == 1:
            best, conf = np.zeros(m, dtype=np.int64), np.ones(m, dtype=np.float32)
            mode, conv, p_sum = DIRECT_MODES[0], _direct_diagnostics(p), float(a.sum())
        elif weighted and m == 1:
            best, conf = _direct_singleton(mol_xy, cell_xy, b, p)
            mode, conv, p_sum = DIRECT_MODES[1], _direct_diagnostics(p), float(a.sum())
        else:
            cost_fn = cost_class(p.r0, p.sigma, p.alpha, p.p_in, p.hard_cut_um, p.big_penalty)
            geom = pointcloud.PointCloud(x=jnp.asarray(mol_xy), y=jnp.asarray(cell_xy), cost_fn=cost_fn,
                                         epsilon=float(p.epsilon))
            prob = linear_problem.LinearProblem(geom, a=jnp.asarray(a), b=jnp.asarray(b),
                                                tau_a=float(p.tau_a), tau_b=float(p.tau_b))
            out = sinkhorn.Sinkhorn(threshold=float(p.thresh), max_iterations=int(p.max_iters))(prob)
            conv = _diagnostics(out, p.max_iters, p.thresh)
            transport = np.asarray(out.matrix, dtype=np.float32)
            row_sum = transport.sum(axis=1, keepdims=True)
            raw_row_mass = row_sum[:, 0].copy()          # before row normalization; used by the filter
            row_sum[row_sum == 0] = 1.0
            normalized = transport / row_sum
            best = normalized.argmax(axis=1)
            conf = normalized[np.arange(m), best]
            mode, p_sum = OT_MODE, float(transport.sum())
            del out, geom, prob, transport, normalized
        frame = dict(window_id=window_id, gene=gene, mol_id=mol_ids, assigned_cell=cell_ids[best].astype(str),
                     assign_conf=conf.astype(np.float32))
        if weighted:
            frame["count"] = mol_counts
        frame.update(raw_row_mass=raw_row_mass, source_marginal=a, assignment_mode=mode)
        frames.append(pd.DataFrame(frame))
        summary.append(dict(window_id=window_id, gene=gene, M=int(m), N=int(n), assignment_mode=mode,
                            mean_conf=float(np.mean(conf)), median_conf=float(np.median(conf)), P_sum=p_sum, **conv))
        solved += 1
        if p.jax_clear_cache_every > 0 and solved % p.jax_clear_cache_every == 0:
            jax.clear_caches()          # bounds compilation-cache memory; does not change results
            gc.collect()
    assignments = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        {k: pd.Series(dtype=t) for k, t in dict(window_id="int64", gene="object", mol_id="int64", assigned_cell="object",
                                                assign_conf="float32", raw_row_mass="float32", source_marginal="float32",
                                                assignment_mode="object").items()})
    return assignments, summary


# ----------------------------------------------------------------------------------------------------------
# Running windows
# ----------------------------------------------------------------------------------------------------------
def _load_soft(soft):
    a = ad.read_h5ad(soft) if isinstance(soft, (str, os.PathLike)) else soft
    a.obs_names = a.obs_names.astype(str)
    return a


def _cells_for(soft, run):
    cells = pd.read_parquet(run / "cells_min.parquet")
    cells["cell_id"] = cells["cell_id"].astype(str)
    xy = cells.set_index("cell_id")[["x", "y"]]
    keep = soft.obs_names.intersection(xy.index)
    if len(keep) == 0:
        raise ValueError("No OT1 soft cells have coordinates in cells_min.parquet")
    soft = soft[keep].copy()
    cell_xy = xy.loc[soft.obs_names].to_numpy(dtype=np.float32)
    return soft, dict(xy=cell_xy, ids=soft.obs_names.to_numpy(), tree=cKDTree(cell_xy))


def run_ot2_windows(run_dir, soft, window_ids=None, params: OT2Params | None = None, overwrite=False):
    """Solve the given windows (default: all) and write ``worker_out/window_XXXXXX/``.

    Completed windows (``_SUCCESS.json`` present) are skipped unless ``overwrite``. Each window writes
    ``assignments.parquet`` (possibly empty), ``summary.csv`` and ``_SUCCESS.json``. Windows are
    independent, so a Slurm array can split ``window_ids`` across tasks.
    """
    p = params or OT2Params()
    run = Path(run_dir)
    windows = pd.read_csv(run / "windows.csv")
    known = windows.window_id.astype(int).tolist()
    ids = known if window_ids is None else [int(w) for w in window_ids if int(w) in set(known)]
    soft, cells = _cells_for(_load_soft(soft), run)
    marginals = _SoftMarginals(soft)
    dataset = pads.dataset(run / "transcripts_min.parquet", format="parquet")
    weighted = "count" in dataset.schema.names
    columns = ["mol_id", "gene", "x", "y"] + (["count"] if weighted else [])
    done = []
    for wid in ids:
        wdir = run / "worker_out" / f"window_{wid:06d}"
        if (wdir / "_SUCCESS.json").exists() and not overwrite:
            continue
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "_SUCCESS.json").unlink(missing_ok=True)
        row = windows.loc[windows.window_id == wid].iloc[0]
        x0, y0, x1, y1 = float(row.x0), float(row.y0), float(row.x1), float(row.y1)
        start = time.time()
        # Half-open window bbox, rows in file order (use_threads=False keeps the order deterministic).
        tx_w = dataset.to_table(columns=columns, filter=(pads.field("x") >= x0) & (pads.field("x") < x1)
                                & (pads.field("y") >= y0) & (pads.field("y") < y1), use_threads=False).to_pandas()
        assignments, summary = solve_window(wid, tx_w, cells, marginals, p, weighted)
        assignments.to_parquet(wdir / "assignments.parquet", index=False)
        pd.DataFrame(summary).to_csv(wdir / "summary.csv", index=False)
        meta = dict(window_id=wid, bbox=dict(x0=x0, y0=y0, x1=x1, y1=y1), n_records=int(len(tx_w)),
                    n_assigned_rows=int(len(assignments)), n_gene_problems=len(summary), weighted=weighted,
                    params=asdict(p), seconds=time.time() - start,
                    note="completion is not a Sinkhorn convergence certificate; see summary.csv")
        (wdir / "_SUCCESS.json").write_text(json.dumps(meta, indent=2) + "\n")
        done.append(wid)
        import jax
        jax.clear_caches()
        gc.collect()
    return done


# ----------------------------------------------------------------------------------------------------------
# Merge, relative row-mass filter, count matrix
# ----------------------------------------------------------------------------------------------------------
def select_winners(frame: pd.DataFrame, weighted=False) -> pd.DataFrame:
    """One record per mol_id: highest assign_conf; ties keep the earlier window file (weighted: lower window_id)."""
    keys = ["mol_id", "assign_conf"] + (["window_id"] if weighted else [])
    order = [True, False] + ([True] if weighted else [])
    return frame.sort_values(keys, ascending=order).drop_duplicates("mol_id", keep="first").copy()


def threshold_decisions(frame: pd.DataFrame, threshold=0.1):
    """Keep OT assignments with raw_row_mass / source_marginal >= threshold; direct assignments are kept."""
    if not 0 < float(threshold) < float("inf"):
        raise ValueError("threshold must be finite and positive")
    modes = frame.assignment_mode.astype(str).to_numpy()
    if not np.isin(modes, MODES).all():
        raise ValueError("Unknown assignment_mode")
    is_ot = modes == OT_MODE
    mass = frame.raw_row_mass.to_numpy(dtype=np.float64)
    a = frame.source_marginal.to_numpy(dtype=np.float64)
    valid = is_ot & np.isfinite(mass) & (mass >= 0) & np.isfinite(a) & (a > 0)
    ratio = np.full(len(frame), np.nan)
    np.divide(mass, a, out=ratio, where=valid)
    keep = (~is_ot) | (valid & (ratio >= threshold))
    reason = np.full(len(frame), "kept", dtype=object)
    reason[~is_ot] = "kept_direct_assignment_no_ot_mass"
    reason[valid & (ratio < threshold)] = "below_relative_mass_threshold"
    reason[is_ot & ~valid] = "invalid_row_mass_or_source_marginal"
    return keep, ratio, reason


def _counts(frame, cells, genes):
    rows = cells.get_indexer(frame.assigned_cell.astype(str))
    cols = genes.get_indexer(frame.gene.astype(str))
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("Assignment outside the cell/gene axes")
    weight = frame["count"].to_numpy(dtype=np.int64) if "count" in frame else np.ones(len(frame), dtype=np.int64)
    return sparse.csr_matrix((weight, (rows, cols)), shape=(len(cells), len(genes)), dtype=np.int64)


def merge_ot2(run_dir, soft, out_dir=None, threshold=0.1):
    """Merge all windows, keep the overlap winner, apply the relative row-mass filter, build count matrices.

    Writes into ``out_dir`` (default ``run_dir/threshold_<t>``): ``ot2_counts.h5ad`` (integer counts),
    ``ot2_lognorm.h5ad`` (normalize_total 1e4 + log1p; counts in ``layers['counts']``),
    ``molecule_assignments.parquet``, ``rejected_assignments.parquet`` and ``ot2_summary.json``.
    A rejected winner is never replaced by another window's assignment of the same transcript.
    """
    import scanpy as sc

    run = Path(run_dir)
    out = Path(out_dir) if out_dir else run / ("threshold_" + format(float(threshold), ".12g").replace(".", "p"))
    out.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(run / "windows.csv").window_id.astype(int).tolist()
    missing = [w for w in windows if not (run / "worker_out" / f"window_{w:06d}" / "_SUCCESS.json").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} windows are not finished (first: {missing[:10]}); run them before merging")
    soft = _load_soft(soft)
    cells, genes = pd.Index(soft.obs_names.astype(str)), pd.Index(soft.var_names.astype(str))
    weighted = "count" in pq.read_schema(run / "transcripts_min.parquet").names
    frames, summaries, total_rows = [], [], 0
    for w in windows:
        wdir = run / "worker_out" / f"window_{w:06d}"
        frame = pd.read_parquet(wdir / "assignments.parquet")
        total_rows += len(frame)
        frame["assigned_cell"] = pd.Categorical(frame.assigned_cell.astype(str), categories=cells)
        frame["gene"] = pd.Categorical(frame.gene.astype(str), categories=genes)
        frames.append(frame.loc[frame.assigned_cell.notna() & frame.gene.notna()])
        try:
            summaries.append(pd.read_csv(wdir / "summary.csv"))
        except pd.errors.EmptyDataError:
            pass
    winners = select_winners(pd.concat(frames, ignore_index=True), weighted)
    del frames
    keep, ratio, reason = threshold_decisions(winners, threshold)
    winners["relative_row_mass"] = ratio
    winners["decision"] = pd.Categorical(reason)
    kept, rejected = winners.loc[keep], winners.loc[~keep].rename(columns={"assigned_cell": "proposed_cell"})
    rejected["assigned_cell"] = "UNASSIGNED"
    kept.to_parquet(out / "molecule_assignments.parquet", index=False)
    rejected.to_parquet(out / "rejected_assignments.parquet", index=False)
    X = _counts(kept, cells, genes)
    counts = ad.AnnData(X=X, obs=soft.obs.copy(), var=soft.var.copy())
    counts.uns["tracel_ot2"] = dict(relative_mass_threshold=float(threshold))
    counts.write_h5ad(out / "ot2_counts.h5ad", compression="gzip")
    lognorm = counts.copy()
    lognorm.layers["counts"] = lognorm.X.copy()
    sc.pp.normalize_total(lognorm, target_sum=1e4)
    sc.pp.log1p(lognorm)
    lognorm.write_h5ad(out / "ot2_lognorm.h5ad")
    pool = pq.read_table(run / "transcripts_min.parquet", columns=["gene"] + (["count"] if weighted else [])).to_pandas()
    in_axes = pool.gene.astype(str).isin(genes)
    input_mass = int(pool.loc[in_axes, "count"].sum()) if weighted else int(in_axes.sum())
    weights = winners["count"].to_numpy(np.int64) if weighted else np.ones(len(winners), dtype=np.int64)
    diag = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    ot_rows = diag[diag.assignment_mode == OT_MODE] if len(diag) else diag
    summary = dict(
        relative_mass_threshold=float(threshold), weighted=weighted, n_windows=len(windows), total_window_rows=total_rows,
        input_records_in_gene_axis=int(in_axes.sum()), input_mass_in_gene_axis=input_mass,
        assigned_records=int(len(winners)), retained_records=int(keep.sum()), rejected_records=int((~keep).sum()),
        rejected_zero_row_mass=int(((~keep) & (winners.raw_row_mass.to_numpy() == 0)).sum()),
        retained_mass=int(X.sum()), rejected_mass=int(weights[~keep].sum()),
        retention_fraction=float(X.sum()) / input_mass if input_mass else None,
        kept_direct_assignments=int((reason == "kept_direct_assignment_no_ot_mass").sum()),
        gene_window_problems=int(len(diag)),
        sinkhorn_nonfinite_last_error=int((~np.isfinite(ot_rows.sinkhorn_last_error)).sum()) if len(ot_rows) else 0,
        sinkhorn_reached_max_iters=int(ot_rows.sinkhorn_reached_max_iters.astype(str).str.lower().eq("true").sum()) if len(ot_rows) else 0,
        note="Sinkhorn diagnostics are informational; no convergence gate is applied.")
    (out / "ot2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
