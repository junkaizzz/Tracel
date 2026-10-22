"""Command line interface: ``tracel ot1 | ot2-prepare | ot2-run | ot2-merge``."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json


def _add_dataclass_args(parser, cls, skip=()):
    for f in fields(cls):
        if f.name in skip:
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            continue
        parser.add_argument(flag, type=type(f.default), default=f.default)


def _params(cls, args, **extra):
    values = {f.name: getattr(args, f.name) for f in fields(cls) if hasattr(args, f.name)}
    values.update(extra)
    return cls(**values)


def _window_ids(text):
    if text is None:
        return None
    ids = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return ids


def main(argv=None):
    from .ot1 import OT1Params
    from .ot2 import OT2Params

    parser = argparse.ArgumentParser(prog="tracel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("ot1", help="Align spatial cells with a scRNA-seq reference (FGW) -> soft expression")
    p1.add_argument("--spatial", required=True, help="Spatial cells x genes h5ad (raw counts unless --input-lognorm)")
    p1.add_argument("--reference", required=True, help="Reference cells x genes h5ad (raw counts unless --input-lognorm)")
    p1.add_argument("--spatial-label", required=True, help="obs column with spatial cell-group labels")
    p1.add_argument("--reference-label", required=True, help="obs column with reference cell-group labels")
    p1.add_argument("--out", required=True, help="Output directory")
    p1.add_argument("--no-structure", action="store_true", help="Set every structural distance to zero")
    p1.add_argument("--input-lognorm", action="store_true", help="X is already normalize_total(1e4)+log1p")
    _add_dataclass_args(p1, OT1Params, skip=("use_structure",))

    p2 = sub.add_parser("ot2-prepare", help="Write minimal transcript/cell tables and overlapping windows")
    p2.add_argument("--transcripts", required=True)
    p2.add_argument("--cells", required=True)
    p2.add_argument("--run-dir", required=True)
    for name, default in (("window-size", 250.0), ("step", 125.0), ("pad", 60.0)):
        p2.add_argument(f"--{name}", type=float, default=default)
    for name, default in (("mol-id-col", "transcript_id"), ("gene-col", "feature_name"), ("x-col", "x_location"),
                          ("y-col", "y_location"), ("cell-id-col", "cell_id"), ("cell-x-col", "x_centroid"),
                          ("cell-y-col", "y_centroid"), ("qv-col", "qv")):
        p2.add_argument(f"--{name}", default=default)
    p2.add_argument("--count-col", default=None, help="Weighted records, e.g. Visium HD bin-gene UMI counts")
    p2.add_argument("--min-qv", type=float, default=None, help="Optional: drop transcripts with qv below this")

    p3 = sub.add_parser("ot2-run", help="Solve OT2 windows (resumable; split ranges across a Slurm array)")
    p3.add_argument("--run-dir", required=True)
    p3.add_argument("--soft", required=True, help="ot1_soft_expression.h5ad")
    p3.add_argument("--windows", default=None, help="e.g. 0-99 or 3,7,12 (default: all)")
    p3.add_argument("--overwrite", action="store_true")
    _add_dataclass_args(p3, OT2Params)

    p4 = sub.add_parser("ot2-merge", help="Reconcile windows, apply the relative row-mass filter, build counts")
    p4.add_argument("--run-dir", required=True)
    p4.add_argument("--soft", required=True)
    p4.add_argument("--out", default=None)
    p4.add_argument("--threshold", type=float, default=0.1, help="Keep assignments with row_mass / a_m >= threshold")

    args = parser.parse_args(argv)
    if args.command == "ot1":
        import anndata as ad
        from .ot1 import run_ot1

        params = _params(OT1Params, args, use_structure=not args.no_structure)
        result = run_ot1(ad.read_h5ad(args.spatial), ad.read_h5ad(args.reference), args.spatial_label,
                         args.reference_label, params, input_is_lognorm=args.input_lognorm)
        result.write(args.out)
        print(json.dumps({k: v for k, v in result.summary.items() if k != "versions"}, indent=2))
    elif args.command == "ot2-prepare":
        from .ot2 import prepare_ot2_inputs

        config = prepare_ot2_inputs(args.transcripts, args.cells, args.run_dir, window_size=args.window_size,
                                    step=args.step, pad=args.pad, mol_id_col=args.mol_id_col, gene_col=args.gene_col,
                                    x_col=args.x_col, y_col=args.y_col, count_col=args.count_col,
                                    cell_id_col=args.cell_id_col, cell_x_col=args.cell_x_col,
                                    cell_y_col=args.cell_y_col, min_qv=args.min_qv, qv_col=args.qv_col)
        print(json.dumps(config, indent=2))
    elif args.command == "ot2-run":
        from .ot2 import run_ot2_windows

        done = run_ot2_windows(args.run_dir, args.soft, _window_ids(args.windows), _params(OT2Params, args),
                               overwrite=args.overwrite)
        print(f"Solved {len(done)} windows")
    else:
        from .ot2 import merge_ot2

        print(json.dumps(merge_ot2(args.run_dir, args.soft, args.out, args.threshold), indent=2))


if __name__ == "__main__":
    main()
