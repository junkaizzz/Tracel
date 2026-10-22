"""OT1: align spatial cells with a scRNA-seq reference and estimate soft cell expression profiles.

The alignment is a low-rank, unbalanced fused Gromov-Wasserstein (FGW) problem solved with moscot/OTT:

* expression cost: squared Euclidean distance in a joint PCA of the log-normalized spatial and
  reference cells (each gene scaled and clipped before PCA);
* structure cost (``use_structure=True``): within-domain one-hot cell-group distances
  (0 for the same group, 1 otherwise);
* marginals: every observed cell group receives the same total mass within its own domain.

The transport plan is kept as low-rank factors ``P = Q diag(1/g) R^T`` and never materialized.
Each spatial cell's soft profile blends its reference barycenter with its own expression:
``lambda_i = clip(rowmass_i / a_i, 0, 1)``, ``soft_i = lambda_i * barycenter_i + (1 - lambda_i) * x_i``.

``use_structure=False`` sets every structural distance to zero while keeping the solver, ``alpha``,
the type-balanced marginals and the blend unchanged (the cell-group labels then only enter through the
marginals). Because a zero geometry cannot be mean-scaled, the expression PCA coordinates are divided by
the square root of their mean cross-domain squared distance and the solver uses ``scale_cost=1``; this is
mathematically the same expression-cost normalization as ``scale_cost="mean"``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.metadata
import json
from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

__all__ = ["OT1Params", "OT1Result", "log_normalize", "type_balanced_marginal", "structure_embedding",
           "expression_embedding", "run_ot1"]


@dataclass
class OT1Params:
    """OT1 settings. Defaults are the values used in the manuscript.

    Parameters
    ----------
    use_structure
        Include the cell-group structure term. ``False`` sets every structural distance to zero and keeps
        ``alpha``, the type-balanced marginals and the blend unchanged.
    alpha
        FGW weight of the structure term; ``1 - alpha`` weights the expression term.
    epsilon
        Entropic regularization passed to the solver.
    rank
        Rank of the low-rank coupling.
    tau_a, tau_b
        Marginal relaxation of the spatial and reference cells (1 = balanced).
    batch_size
        Solver batch size for cost evaluation.
    n_pcs
        Principal components of the joint expression embedding.
    scale_max_value
        Clip value after per-gene scaling, before PCA.
    seed
        PCA random state (and low-rank initializer seed when ``use_structure=False``).
    """

    use_structure: bool = True
    alpha: float = 0.9
    epsilon: float = 0.0
    rank: int = 20
    tau_a: float = 0.8
    tau_b: float = 0.9
    batch_size: int = 1024
    n_pcs: int = 50
    scale_max_value: float = 10.0
    seed: int = 0


@dataclass
class OT1Result:
    """Outputs of :func:`run_ot1`.

    Attributes
    ----------
    soft
        Spatial cells x genes, soft log-normalized expression. ``obs`` gains ``ot1_row_mass``,
        ``ot1_source_marginal`` and ``ot1_blend_fraction``. This is the input of OT2.
    spatial_lognorm
        The log-normalized spatial input on the same axes.
    q, r, g
        Low-rank factors of the coupling ``P = q diag(1/g) r^T``.
    row_mass
        Transported mass of each spatial cell, ``P.sum(axis=1)``.
    source_marginal, target_marginal
        Type-balanced marginals of the spatial and reference cells.
    blend_fraction
        ``clip(row_mass / source_marginal, 0, 1)``: weight of the reference barycenter in the soft profile.
    reference_obs_names
        Reference cell ids in factor ``r`` order.
    summary
        Settings, sizes, solver convergence flag and blend statistics.
    """

    soft: ad.AnnData
    spatial_lognorm: ad.AnnData
    q: np.ndarray
    r: np.ndarray
    g: np.ndarray
    row_mass: np.ndarray
    source_marginal: np.ndarray
    target_marginal: np.ndarray
    blend_fraction: np.ndarray
    reference_obs_names: np.ndarray
    summary: dict = field(default_factory=dict)

    def write(self, out_dir):
        """Write ``ot1_soft_expression.h5ad``, ``spatial_lognorm.h5ad``, factors and a JSON summary."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.soft.write_h5ad(out / "ot1_soft_expression.h5ad")
        self.spatial_lognorm.write_h5ad(out / "spatial_lognorm.h5ad")
        np.savez_compressed(out / "ot1_coupling_factors.npz", q=self.q, r=self.r, g=self.g, row_mass=self.row_mass,
                            source_marginal=self.source_marginal, target_marginal=self.target_marginal,
                            blend_fraction=self.blend_fraction,
                            spatial_cell_ids=self.soft.obs_names.to_numpy(dtype=str),
                            reference_cell_ids=self.reference_obs_names.astype(str),
                            gene_names=self.soft.var_names.to_numpy(dtype=str))
        (out / "ot1_summary.json").write_text(json.dumps(self.summary, indent=2, allow_nan=True) + "\n")
        return out


def _dense32(x):
    return x.toarray().astype(np.float32) if sparse.issparse(x) else np.asarray(x, dtype=np.float32)


def log_normalize(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    """``normalize_total(target_sum)`` + ``log1p`` on a copy; returns dense float32 ``X``."""
    a = adata.copy()
    a.X = a.X.astype(np.float32) if sparse.issparse(a.X) else np.asarray(a.X, dtype=np.float32)
    sc.pp.normalize_total(a, target_sum=target_sum)
    sc.pp.log1p(a)
    a.X = _dense32(a.X)
    return a


def type_balanced_marginal(labels) -> np.ndarray:
    """Each observed label gets total mass ``1 / n_labels``, split uniformly among its cells."""
    labels = np.asarray(labels).astype(str)
    counts = pd.Series(labels).value_counts().sort_index()
    n_types = float(len(counts))
    m = np.array([(1.0 / n_types) / float(counts[x]) for x in labels], dtype=np.float64)
    return m / m.sum()


def structure_embedding(labels, all_types) -> np.ndarray:
    """One-hot rows with entry ``1/sqrt(2)``: squared distance 0 within a label and 1 between labels."""
    index = {t: i for i, t in enumerate(all_types)}
    z = np.zeros((len(labels), len(all_types)), dtype=np.float32)
    z[np.arange(len(labels)), [index[str(x)] for x in labels]] = 1.0 / np.sqrt(2.0)
    return z


def expression_embedding(x_spatial, x_reference, n_pcs=50, scale_max_value=10.0, seed=0):
    """Joint per-gene scaling (clipped) and PCA of the two log-normalized matrices."""
    joint = ad.AnnData(np.vstack([x_spatial, x_reference]).astype(np.float32))
    sc.pp.scale(joint, max_value=scale_max_value)
    n_comps = int(min(n_pcs, max(1, min(joint.n_obs - 1, joint.n_vars - 1))))
    sc.pp.pca(joint, n_comps=n_comps, random_state=seed)
    z = np.asarray(joint.obsm["X_pca"], dtype=np.float32)
    if not np.isfinite(z).all():
        raise ValueError("Non-finite expression PCA")
    n = len(x_spatial)
    return z[:n], z[n:], float(np.sum(joint.uns["pca"]["variance_ratio"]))


def _mean_cross_sq_distance(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = np.square(x).sum(axis=1).mean() + np.square(y).sum(axis=1).mean() - 2 * np.dot(x.mean(axis=0), y.mean(axis=0))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Mean cross-domain expression cost must be positive and finite")
    return float(mean)


def _factors(problem):
    out = problem[("spatial", "scrna")].solution._output
    if not all(hasattr(out, k) for k in ("q", "r", "g")):
        raise RuntimeError("Expected low-rank factors (rank > 0); refusing a dense coupling")
    return out


def _solve_with_structure(z_sp, z_ref, s_sp, s_ref, a, b, p: OT1Params):
    from moscot.problems.generic import FGWProblem

    n, m = len(z_sp), len(z_ref)
    adata = ad.AnnData(np.zeros((n + m, 1), dtype=np.float32))
    adata.obs["domain"] = pd.Categorical(["spatial"] * n + ["scrna"] * m, categories=["spatial", "scrna"])
    adata.obs["marginal"] = np.concatenate([a, b])
    adata.obsm["X_expr"] = np.vstack([z_sp, z_ref]).astype(np.float32)
    adata.obsm["X_struct"] = np.vstack([s_sp, s_ref]).astype(np.float32)
    problem = FGWProblem(adata).prepare(
        key="domain", policy="explicit", subset=[("spatial", "scrna")],
        joint_attr="X_expr", x_attr="X_struct", y_attr="X_struct",
        cost={"xy": "sq_euclidean", "x": "sq_euclidean", "y": "sq_euclidean"}, a="marginal", b="marginal")
    problem = problem.solve(alpha=p.alpha, epsilon=p.epsilon, tau_a=p.tau_a, tau_b=p.tau_b, rank=p.rank,
                            batch_size=p.batch_size, scale_cost="mean")
    return _factors(problem), None


def _solve_without_structure(z_sp, z_ref, a, b, p: OT1Params):
    import jax
    from moscot.problems.generic import FGWProblem

    n, m = len(z_sp), len(z_ref)
    mean_cost = _mean_cross_sq_distance(z_sp, z_ref)
    adata = ad.AnnData(np.zeros((n + m, 1), dtype=np.float32))
    adata.obs["domain"] = pd.Categorical(["spatial"] * n + ["scrna"] * m)
    adata.obs["marginal"] = np.concatenate([a, b])
    adata.obsm["X_expr"] = (np.vstack([z_sp, z_ref]) / np.sqrt(mean_cost)).astype(np.float32)
    adata.obsm["X_zero_structure"] = np.zeros((n + m, 1), dtype=np.float32)
    problem = FGWProblem(adata).prepare(
        key="domain", policy="explicit", subset=[("spatial", "scrna")],
        joint_attr="X_expr", x_attr="X_zero_structure", y_attr="X_zero_structure",
        cost={"xy": "sq_euclidean", "x": "sq_euclidean", "y": "sq_euclidean"}, a="marginal", b="marginal")
    problem = problem.solve(alpha=p.alpha, epsilon=p.epsilon, tau_a=p.tau_a, tau_b=p.tau_b, rank=p.rank,
                            batch_size=p.batch_size, threshold=1e-3, min_iterations=10000, max_iterations=100000,
                            inner_iterations=2000, gamma=10.0, gamma_rescale=True, scale_cost=1.0,
                            initializer_kwargs={"rng": jax.random.PRNGKey(p.seed)})
    out = _factors(problem)
    for geom in (out.ot_prob.geom_xx, out.ot_prob.geom_yy):
        values = np.asarray(geom.apply_cost(np.ones((geom.shape[1], 1), dtype=np.float32), axis=1))
        if not np.isfinite(values).all() or np.max(np.abs(values)) != 0:
            raise ValueError("Structural cost is not identically zero")
    return out, mean_cost


def lowrank_blend(q, r, g, x_spatial, x_reference, source_marginal):
    """Row mass, reference barycenters and the unbalanced blend from low-rank factors (no dense P)."""
    q, r, g = (np.asarray(v, dtype=np.float32) for v in (q, r, g))
    if np.any(g <= 0) or np.any(source_marginal <= 0):
        raise ValueError("Non-positive factor g or source marginal")
    inv_g = (1.0 / g.astype(np.float64)).astype(np.float32)
    row_mass = (q @ (r.sum(axis=0).astype(np.float32) * inv_g)).astype(np.float32)
    safe = row_mass.copy()
    safe[safe == 0] = 1.0
    projected = (r.T @ x_reference).astype(np.float32) * inv_g[:, None]
    barycenter = (q @ projected).astype(np.float32) / safe[:, None]
    blend = np.clip(row_mass / source_marginal.astype(np.float32), 0, 1).astype(np.float32)
    soft = (blend[:, None] * barycenter + (1 - blend[:, None]) * x_spatial).astype(np.float32)
    if not np.isfinite(soft).all():
        raise ValueError("Non-finite soft expression")
    return soft, row_mass, blend


def _versions():
    out = {}
    for name in ("numpy", "scipy", "pandas", "anndata", "scanpy", "jax", "jaxlib", "ott-jax", "moscot"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_ot1(spatial: ad.AnnData, reference: ad.AnnData, spatial_label_key: str, reference_label_key: str,
            params: OT1Params | None = None, genes=None, input_is_lognorm: bool = False) -> OT1Result:
    """Align spatial cells with a scRNA-seq reference and estimate soft expression profiles (OT1).

    Parameters
    ----------
    spatial
        Spatial cells x genes. ``X`` holds raw counts (or log-normalized values if ``input_is_lognorm``).
        Cell order is preserved in the output.
    reference
        Reference cells x genes, in the same format as ``spatial``.
    spatial_label_key
        ``obs`` column of ``spatial`` with cell-group labels.
    reference_label_key
        ``obs`` column of ``reference`` with cell-group labels. Names need not match the spatial labels.
    params
        :class:`OT1Params`; ``params.use_structure`` switches the structure term on or off.
    genes
        Genes to use. Default: the sorted intersection of both ``var_names``. Counts are normalized within
        these genes only.
    input_is_lognorm
        ``X`` is already ``normalize_total(1e4)`` + ``log1p``.

    Returns
    -------
    OT1Result
        Soft profiles, low-rank factors, marginals and a summary; ``result.write(dir)`` saves them.
    """
    p = params or OT1Params()
    started = time.monotonic()
    for name, a_, key in (("spatial", spatial, spatial_label_key), ("reference", reference, reference_label_key)):
        if key not in a_.obs:
            raise KeyError(f"{name}: obs[{key!r}] not found")
        if a_.obs[key].isna().any():
            raise ValueError(f"{name}: obs[{key!r}] has missing labels")
        if not a_.obs_names.is_unique or not a_.var_names.is_unique:
            raise ValueError(f"{name}: obs_names and var_names must be unique")
    shared = (np.intersect1d(spatial.var_names.astype(str), reference.var_names.astype(str))
              if genes is None else np.asarray(list(genes), dtype=str))
    if len(shared) < 2:
        raise ValueError("Fewer than two shared genes")
    sp = spatial[:, shared].copy()
    rf = reference[:, shared].copy()
    if input_is_lognorm:
        sp.X, rf.X = _dense32(sp.X), _dense32(rf.X)
    else:
        sp, rf = log_normalize(sp), log_normalize(rf)
    for name, x in (("spatial", sp.X), ("reference", rf.X)):
        if not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError(f"{name}: invalid log-normalized expression")
    sp_labels = sp.obs[spatial_label_key].astype(str).to_numpy()
    rf_labels = rf.obs[reference_label_key].astype(str).to_numpy()
    a = type_balanced_marginal(sp_labels)
    b = type_balanced_marginal(rf_labels)
    z_sp, z_rf, explained = expression_embedding(sp.X, rf.X, p.n_pcs, p.scale_max_value, p.seed)
    solve_start = time.monotonic()
    if p.use_structure:
        all_types = sorted(set(sp_labels) | set(rf_labels))
        out, mean_cost = _solve_with_structure(z_sp, z_rf, structure_embedding(sp_labels, all_types),
                                               structure_embedding(rf_labels, all_types), a, b, p)
    else:
        out, mean_cost = _solve_without_structure(z_sp, z_rf, a, b, p)
    solve_seconds = time.monotonic() - solve_start
    q, r, g = (np.asarray(getattr(out, k), dtype=np.float32) for k in ("q", "r", "g"))
    source = a.astype(np.float32)
    soft_x, row_mass, blend = lowrank_blend(q, r, g, sp.X, rf.X, source)
    soft = sp.copy()
    soft.X = soft_x
    soft.obs["ot1_row_mass"] = row_mass
    soft.obs["ot1_source_marginal"] = source
    soft.obs["ot1_blend_fraction"] = blend
    errors = np.asarray(out.errors).ravel()
    summary = dict(
        params=asdict(p), n_spatial=int(sp.n_obs), n_reference=int(rf.n_obs), n_genes=int(len(shared)),
        spatial_label_key=spatial_label_key, reference_label_key=reference_label_key,
        n_spatial_labels=int(len(set(sp_labels))), n_reference_labels=int(len(set(rf_labels))),
        expression_pca_variance_ratio_sum=explained,
        expression_cost_mean_before_scaling=mean_cost,
        solver_converged=bool(out.converged), solver_iterations=int(out.n_iters),
        solver_last_finite_error=float(errors[np.isfinite(errors) & (errors != -1)][-1]) if np.any(np.isfinite(errors) & (errors != -1)) else None,
        solve_seconds=solve_seconds, total_seconds=time.monotonic() - started,
        coupling_mass=float(row_mass.sum()), blend_min=float(blend.min()),
        blend_median=float(np.median(blend)), blend_max=float(blend.max()),
        coupling_representation="q diag(1/g) r^T (low rank; never materialized)", versions=_versions())
    return OT1Result(soft=soft, spatial_lognorm=sp, q=q, r=r, g=g, row_mass=row_mass, source_marginal=source,
                     target_marginal=b.astype(np.float32), blend_fraction=blend,
                     reference_obs_names=rf.obs_names.to_numpy(dtype=str), summary=summary)
