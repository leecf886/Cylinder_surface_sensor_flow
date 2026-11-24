"""Command-line interface and argument parser for ``cylinder_flow``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

from cylinder_flow.config import PathConfig, get_path_config


def _add_path_override_args(parser: argparse.ArgumentParser, *, is_subparser: bool = False) -> None:
    """Register path/config override arguments without subparser default clobbering."""
    prefix = "sub_" if is_subparser else ""
    parser.add_argument(
        "--data-dir",
        dest=f"{prefix}data_dir",
        type=str,
        default=None,
        help="Root directory for dataset (overrides CYLINDER_DATA_ROOT and configs/paths.json)",
    )
    parser.add_argument(
        "--processed-dir",
        dest=f"{prefix}processed_dir",
        type=str,
        default=None,
        help="Directory for processed HDF5 files",
    )
    parser.add_argument(
        "--runs-dir",
        dest=f"{prefix}runs_dir",
        type=str,
        default=None,
        help="Root directory for run outputs and checkpoints",
    )
    parser.add_argument(
        "--config",
        dest=f"{prefix}config",
        type=str,
        default=None,
        help="Path to JSON configuration or protocol file",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser with all workflow and pipeline subcommands."""
    parser = argparse.ArgumentParser(
        prog="cylinder_flow",
        description="Cylinder Surface Sensor Flow Pipeline",
    )
    _add_path_override_args(parser, is_subparser=False)

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Work package workflows
    p_wpa = subparsers.add_parser("wp-a", help="Run Work Package A: Sensor placement & evaluation")
    _add_path_override_args(p_wpa, is_subparser=True)
    p_wpa.add_argument("--stages", type=str, default="preflight")
    p_wpa.add_argument("--output-dir", type=str, default=None)
    p_wpa.add_argument("--dry-run", action="store_true")
    p_wpa.add_argument("--allow-test", action="store_true")
    p_wpa.add_argument("--overwrite", action="store_true")
    p_wpa.add_argument("--no-png", action="store_true")
    p_wpa.add_argument("--strict-aggregate", action="store_true")

    p_wpb = subparsers.add_parser("wp-b", help="Run Work Package B: Dynamic encoder & architecture ablations")
    _add_path_override_args(p_wpb, is_subparser=True)
    p_wpb.add_argument("--stages", type=str, default="preflight")
    p_wpb.add_argument("--mode", choices=["dynamic", "arch_ablation"], default=None)
    p_wpb.add_argument("--output-dir", type=str, default=None)
    p_wpb.add_argument("--dry-run", action="store_true")
    p_wpb.add_argument("--allow-test", action="store_true")
    p_wpb.add_argument("--formal", action="store_true")

    p_wpc = subparsers.add_parser("wp-c", help="Run Work Package C: Noise robustness evaluation")
    _add_path_override_args(p_wpc, is_subparser=True)
    p_wpc.add_argument("--stages", type=str, default="preflight")
    p_wpc.add_argument("--output-dir", type=str, default=None)
    p_wpc.add_argument("--output-root", type=str, default=None)
    p_wpc.add_argument("--dry-run", action="store_true")
    p_wpc.add_argument("--resume", action="store_true")
    p_wpc.add_argument("--allow-test", action="store_true")
    p_wpc.add_argument("--no-png", action="store_true")

    p_wpd = subparsers.add_parser("wp-d", help="Run Work Package D: Full flow field reconstruction")
    _add_path_override_args(p_wpd, is_subparser=True)
    p_wpd.add_argument("--stages", type=str, default="preflight")
    p_wpd.add_argument("--output-dir", type=str, default=None)
    p_wpd.add_argument("--dry-run", action="store_true")
    p_wpd.add_argument("--allow-test", action="store_true")
    p_wpd.add_argument("--formal", action="store_true")
    p_wpd.add_argument("--no-png", action="store_true")

    # Core pipeline stages
    p_pre = subparsers.add_parser("preprocess", help="Show HDF5 preprocessing contract notice")
    _add_path_override_args(p_pre, is_subparser=True)

    p_te = subparsers.add_parser("train-encoder", help="Train sensor encoder (MLP / DeepSets)")
    _add_path_override_args(p_te, is_subparser=True)
    p_te.add_argument("--encoder", choices=["mlp", "deepsets"], default="mlp")
    p_te.add_argument("--fold", type=str, default="default")
    p_te.add_argument("--sensor-indices", type=str, default=None)
    p_te.add_argument("--output-dir", type=str, default=None)
    p_te.add_argument("--epochs", type=int, default=10)
    p_te.add_argument("--batch-size", type=int, default=256)
    p_te.add_argument("--lr", type=float, default=1e-3)
    p_te.add_argument("--seed", type=int, default=42)
    p_te.add_argument("--device", type=str, default="cpu")
    p_te.add_argument("--dry-run", action="store_true")

    p_tp = subparsers.add_parser("train-ppinn", help="Train parametric PINN")
    _add_path_override_args(p_tp, is_subparser=True)
    p_tp.add_argument("--fold", type=str, default="default")
    p_tp.add_argument("--output-dir", type=str, default=None)
    p_tp.add_argument("--epochs", type=int, default=10)
    p_tp.add_argument("--batch-size", type=int, default=1024)
    p_tp.add_argument("--lr", type=float, default=1e-3)
    p_tp.add_argument("--lambda-phys", type=float, default=1.0)
    p_tp.add_argument("--seed", type=int, default=42)
    p_tp.add_argument("--device", type=str, default="cpu")
    p_tp.add_argument("--dry-run", action="store_true")

    p_ev = subparsers.add_parser("eval", help="Evaluate end-to-end pipeline")
    _add_path_override_args(p_ev, is_subparser=True)
    p_ev.add_argument("--mode", choices=["oracle", "closed_loop", "pod"], default="oracle")
    p_ev.add_argument("--fold", type=str, default="default")
    p_ev.add_argument("--mesh-file", type=str, default="mesh.h5")
    p_ev.add_argument("--ppinn-ckpt", type=str, default=None)
    p_ev.add_argument("--encoder-ckpt", type=str, default=None)
    p_ev.add_argument("--pod-ckpt", type=str, default=None)
    p_ev.add_argument("--re-list", type=str, default=None)
    p_ev.add_argument("--output-dir", type=str, default=None)
    p_ev.add_argument("--device", type=str, default="cpu")
    p_ev.add_argument("--chunk-size", type=int, default=16384)
    p_ev.add_argument("--allow-test", action="store_true")
    p_ev.add_argument("--analysis-lock", type=Path, help="Approved analysis lock for held-out test Re")
    p_ev.add_argument("--dry-run", action="store_true")

    return parser


def _merge_path_overrides(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    return {
        "config": getattr(args, "sub_config", None) or getattr(args, "config", None),
        "data_dir": getattr(args, "sub_data_dir", None) or getattr(args, "data_dir", None),
        "processed_dir": (
            getattr(args, "sub_processed_dir", None) or getattr(args, "processed_dir", None)
        ),
        "runs_dir": getattr(args, "sub_runs_dir", None) or getattr(args, "runs_dir", None),
    }


def _parse_int_csv(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or not str(raw).strip():
        return None
    return [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]


def _dispatch_workflow(args: argparse.Namespace, overrides: Dict[str, Optional[str]]) -> Dict[str, Any]:
    subcmd = args.subcommand
    stages = [s.strip() for s in str(args.stages).split(",") if s.strip()]

    if subcmd == "wp-a":
        from cylinder_flow.workflows.wp_a import load_wp_a_config, run_wp_a

        cfg = load_wp_a_config(
            overrides["config"],
            data_dir=overrides["data_dir"],
            processed_dir=overrides["processed_dir"],
            runs_dir=overrides["runs_dir"],
        )
        if args.output_dir:
            cfg.output_root = Path(args.output_dir).resolve()
        return run_wp_a(
            cfg,
            stages=stages,
            allow_test=bool(args.allow_test),
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
            no_png=bool(args.no_png),
            strict_aggregate=bool(args.strict_aggregate),
        )

    if subcmd == "wp-b":
        from cylinder_flow.workflows.wp_b import load_wp_b_config, run_wp_b

        cfg = load_wp_b_config(
            overrides["config"],
            data_dir=overrides["data_dir"],
            processed_dir=overrides["processed_dir"],
            runs_dir=overrides["runs_dir"],
        )
        if args.output_dir:
            cfg.output_root = Path(args.output_dir).resolve()
        return run_wp_b(
            cfg,
            stages=stages,
            mode=args.mode,
            allow_test=bool(args.allow_test),
            dry_run=bool(args.dry_run),
            formal=bool(args.formal),
        )

    if subcmd == "wp-c":
        from cylinder_flow.workflows.wp_c import load_wp_c_config, run_wp_c

        out_root = args.output_dir or args.output_root
        cfg = load_wp_c_config(
            overrides["config"],
            data_dir=overrides["data_dir"],
            processed_dir=overrides["processed_dir"],
            runs_dir=overrides["runs_dir"],
            output_root=out_root,
        )
        if args.no_png:
            cfg.no_png = True
        return run_wp_c(
            cfg,
            stages=stages,
            allow_test=bool(args.allow_test),
            dry_run=bool(args.dry_run),
            resume=bool(args.resume),
        )

    if subcmd == "wp-d":
        from cylinder_flow.workflows.wp_d import load_wp_d_config, run_wp_d

        cfg = load_wp_d_config(
            overrides["config"],
            data_dir=overrides["data_dir"],
            processed_dir=overrides["processed_dir"],
            runs_dir=overrides["runs_dir"],
        )
        if args.output_dir:
            cfg.output_root = Path(args.output_dir).resolve()
        return run_wp_d(
            cfg,
            stages=stages,
            allow_test=bool(args.allow_test),
            dry_run=bool(args.dry_run),
            formal=bool(args.formal),
            no_png=bool(args.no_png),
        )

    raise ValueError(f"Unsupported workflow subcommand: {subcmd!r}")


def _dispatch_eval(args: argparse.Namespace, path_cfg: PathConfig) -> Dict[str, Any]:
    from cylinder_flow.artifacts.checkpoints import (
        load_ablation_checkpoint,
        load_deepsets_checkpoint,
        load_legacy_mlp_checkpoint,
        load_legacy_ppinn_checkpoint,
    )
    from cylinder_flow.artifacts.manifests import validate_re_split_locked_test_gate
    from cylinder_flow.artifacts.results import write_metrics_long
    from cylinder_flow.data.h5 import load_mesh, load_re_cases
    from cylinder_flow.data.splits import select_re_fold
    from cylinder_flow.evaluation.runner import (
        evaluate_cases_to_long_rows,
        evaluate_closed_loop_pipeline,
        evaluate_pod_baseline,
        evaluate_ppinn_oracle,
    )
    from cylinder_flow.models.pod import SurfacePOD

    split = select_re_fold(args.fold, strict=False)
    re_ids = _parse_int_csv(args.re_list) or (split.test_re if args.allow_test else split.val_re)
    checkpoints = {"pod": args.pod_ckpt} if args.mode == "pod" else {
        "ppinn": args.ppinn_ckpt,
        "selected_ppinn": args.ppinn_ckpt,
        "encoder": args.encoder_ckpt if args.mode == "closed_loop" else None,
    }
    gate = validate_re_split_locked_test_gate(
        re_ids,
        re_split=split,
        allow_test=args.allow_test,
        lock_path=args.analysis_lock,
        artifact_paths={role: path for role, path in checkpoints.items() if path is not None},
    )
    locked_re = set(gate["locked_test_re_requested"])
    is_test = bool(locked_re)
    if is_test and set(re_ids) != locked_re:
        raise ValueError("Evaluate test and non-test Reynolds numbers in separate runs")
    split_label = "test" if is_test else "val"

    out_dir = Path(args.output_dir).resolve() if args.output_dir else path_cfg.runs_dir / "eval"
    if args.dry_run:
        return {
            "status": "dry_run",
            "subcommand": "eval",
            "mode": args.mode,
            "fold": split.fold_id,
            "re_ids": list(re_ids),
            "allow_test": bool(args.allow_test),
            "split": split_label,
            "processed_dir": str(path_cfg.processed_dir),
            "output_dir": str(out_dir),
        }

    required = {
        "oracle": {"--ppinn-ckpt": args.ppinn_ckpt},
        "closed_loop": {"--ppinn-ckpt": args.ppinn_ckpt, "--encoder-ckpt": args.encoder_ckpt},
        "pod": {"--pod-ckpt": args.pod_ckpt},
    }
    missing = [flag for flag, path in required[args.mode].items() if not path]
    if missing:
        raise ValueError(f"eval --mode {args.mode} requires {', '.join(missing)}")

    mesh = load_mesh(path_cfg.processed_dir / args.mesh_file)
    cases = load_re_cases(
        path_cfg.processed_dir,
        re_ids,
        mesh=mesh,
        load_field=True,
        test_guard_split=None if is_test else split,
    )

    if args.mode == "oracle":
        ppinn = load_legacy_ppinn_checkpoint(args.ppinn_ckpt, map_location=args.device)
        res = evaluate_ppinn_oracle(
            ppinn,
            cases,
            mesh,
            allow_test=is_test,
            re_split=split,
            chunk_size=int(args.chunk_size),
            device=args.device,
        )
    elif args.mode == "closed_loop":
        ppinn = load_legacy_ppinn_checkpoint(args.ppinn_ckpt, map_location=args.device)
        enc_path = Path(args.encoder_ckpt)
        encoder: Any
        try:
            encoder = load_deepsets_checkpoint(enc_path)
        except Exception:
            try:
                encoder = load_ablation_checkpoint(enc_path)
            except Exception:
                encoder = load_legacy_mlp_checkpoint(enc_path)
        res = evaluate_closed_loop_pipeline(
            encoder,
            ppinn,
            cases,
            mesh,
            allow_test=is_test,
            re_split=split,
            chunk_size=int(args.chunk_size),
            device=args.device,
        )
    else:
        pod_model = SurfacePOD.load(args.pod_ckpt)
        res = evaluate_pod_baseline(
            pod_model,
            cases,
            mesh=mesh,
            allow_test=is_test,
            re_split=split,
        )

    long_rows = evaluate_cases_to_long_rows(
        res,
        run_id=f"cli_eval_{args.mode}",
        stage="eval_pipeline",
        re_fold=split.fold_id,
        split=split_label,
    )
    metrics_path = write_metrics_long(long_rows, out_dir / "metrics_long.csv")
    return {
        "status": "ok",
        "mode": args.mode,
        "re_ids": res.re_ids,
        "overall": res.overall,
        "metrics_long_csv": str(metrics_path),
    }


def _dispatch_train_encoder(args: argparse.Namespace, path_cfg: PathConfig) -> Dict[str, Any]:
    from cylinder_flow.data.splits import select_re_fold

    split = select_re_fold(args.fold, strict=False)
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else path_cfg.runs_dir / f"train_encoder_{args.encoder}"
    )
    if args.dry_run:
        return {
            "status": "dry_run",
            "subcommand": "train-encoder",
            "encoder": args.encoder,
            "fold": split.fold_id,
            "train_re": list(split.train_re),
            "val_re": list(split.val_re),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "processed_dir": str(path_cfg.processed_dir),
            "output_dir": str(out_dir),
        }

    from cylinder_flow.data.h5 import load_mesh, load_re_cases
    from cylinder_flow.data.sampling import create_dynamic_sensor_data_loader
    from cylinder_flow.models.deepsets import positions_from_theta
    from cylinder_flow.training.encoder import (
        DeepSetsTrainingConfig,
        EncoderTrainingConfig,
        train_sensor_deepsets,
        train_sensor_mlp,
    )
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    mesh = load_mesh(path_cfg.processed_dir / "mesh.h5")
    train_cases = load_re_cases(path_cfg.processed_dir, split.train_re, mesh=mesh, test_guard_split=split)
    val_cases = load_re_cases(path_cfg.processed_dir, split.val_re, mesh=mesh, test_guard_split=split)

    if args.encoder == "deepsets":
        train_loader = create_dynamic_sensor_data_loader(
            train_cases,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        )
        cfg = DeepSetsTrainingConfig(
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
            device=args.device,
        )
        _, history = train_sensor_deepsets(
            train_loader,
            val_loader=val_cases,
            config=cfg,
            candidate_positions=positions_from_theta(mesh.sensor_theta),
            re_split=split,
            output_dir=out_dir,
        )
    else:
        indices = _parse_int_csv(args.sensor_indices) or list(range(0, mesh.ncand, 5))
        x_tr = torch.cat(
            [torch.as_tensor(c.get_sensors(indices, flatten=True), dtype=torch.float32) for c in train_cases.values()],
            dim=0,
        )
        y_tr = torch.cat(
            [
                torch.stack(
                    [
                        torch.full((c.nt,), float(c.re)),
                        torch.as_tensor(c.phase_sin, dtype=torch.float32),
                        torch.as_tensor(c.phase_cos, dtype=torch.float32),
                    ],
                    dim=-1,
                )
                for c in train_cases.values()
            ],
            dim=0,
        )
        x_va = torch.cat(
            [torch.as_tensor(c.get_sensors(indices, flatten=True), dtype=torch.float32) for c in val_cases.values()],
            dim=0,
        )
        y_va = torch.cat(
            [
                torch.stack(
                    [
                        torch.full((c.nt,), float(c.re)),
                        torch.as_tensor(c.phase_sin, dtype=torch.float32),
                        torch.as_tensor(c.phase_cos, dtype=torch.float32),
                    ],
                    dim=-1,
                )
                for c in val_cases.values()
            ],
            dim=0,
        )
        train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=int(args.batch_size), shuffle=True)
        val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=int(args.batch_size), shuffle=False)
        cfg_mlp = EncoderTrainingConfig(
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
            device=args.device,
        )
        _, history = train_sensor_mlp(
            train_loader,
            val_loader=val_loader,
            config=cfg_mlp,
            sensor_indices=indices,
            re_split=split,
            output_dir=out_dir,
        )
    return {
        "status": "ok",
        "encoder": args.encoder,
        "output_dir": str(out_dir),
        "best_epoch": int(getattr(history, "best_epoch", 0)),
    }


def _dispatch_train_ppinn(args: argparse.Namespace, path_cfg: PathConfig) -> Dict[str, Any]:
    from cylinder_flow.data.splits import select_re_fold

    split = select_re_fold(args.fold, strict=False)
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else path_cfg.runs_dir / "train_ppinn"
    )
    if args.dry_run:
        return {
            "status": "dry_run",
            "subcommand": "train-ppinn",
            "fold": split.fold_id,
            "train_re": list(split.train_re),
            "val_re": list(split.val_re),
            "epochs": int(args.epochs),
            "lambda_phys": float(args.lambda_phys),
            "seed": int(args.seed),
            "processed_dir": str(path_cfg.processed_dir),
            "output_dir": str(out_dir),
        }

    from cylinder_flow.data.h5 import load_mesh, load_re_cases
    from cylinder_flow.training.ppinn import PPINNTrainingConfig, train_ppinn

    mesh = load_mesh(path_cfg.processed_dir / "mesh.h5")
    train_cases = list(
        load_re_cases(
            path_cfg.processed_dir, split.train_re, mesh=mesh, load_field=True, test_guard_split=split
        ).values()
    )
    val_cases = list(
        load_re_cases(
            path_cfg.processed_dir, split.val_re, mesh=mesh, load_field=True, test_guard_split=split
        ).values()
    )
    cfg = PPINNTrainingConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lambda_phys=float(args.lambda_phys),
        seed=int(args.seed),
        device=args.device,
    )
    _, history = train_ppinn(train_cases, val_cases, mesh, config=cfg, output_dir=out_dir)
    return {
        "status": "ok",
        "output_dir": str(out_dir),
        "best_epoch": int(getattr(history, "best_epoch", 0)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for ``python -m cylinder_flow``."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.subcommand is None:
        parser.print_help()
        return 0

    overrides = _merge_path_overrides(args)
    if args.subcommand in {"wp-a", "wp-b", "wp-c", "wp-d"}:
        result = _dispatch_workflow(args, overrides)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    path_cfg = get_path_config(
        config_path=overrides["config"],
        data_dir=overrides["data_dir"],
        processed_dir=overrides["processed_dir"],
        runs_dir=overrides["runs_dir"],
    )

    if args.subcommand == "preprocess":
        notice = {
            "status": "external_step_notice",
            "subcommand": "preprocess",
            "message": (
                "Raw ANSYS Fluent-to-HDF5 conversion is an external preprocessing step "
                "not bundled in cylinder_flow. Prepare mesh.h5 and ReXXX.h5 matching "
                "docs/data_format.md before running training or evaluation."
            ),
            "data_root": str(path_cfg.data_root),
            "processed_dir": str(path_cfg.processed_dir),
        }
        print(json.dumps(notice, ensure_ascii=False, indent=2))
        return 0

    if args.subcommand == "eval":
        result = _dispatch_eval(args, path_cfg)
    elif args.subcommand == "train-encoder":
        result = _dispatch_train_encoder(args, path_cfg)
    elif args.subcommand == "train-ppinn":
        result = _dispatch_train_ppinn(args, path_cfg)
    else:  # pragma: no cover - guarded by argparse choices
        parser.error(f"Unknown subcommand: {args.subcommand}")
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
