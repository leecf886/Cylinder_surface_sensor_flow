"""Command-line interface and argument parser for cylinder_flow."""
import argparse
import sys
from cylinder_flow.config import get_path_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cylinder_flow",
        description="Cylinder Surface Sensor Flow Pipeline",
    )
    # Global / path configuration options
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root directory for dataset (overrides CYLINDER_DATA_ROOT and configs/paths.json)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Directory for processed HDF5 files",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Root directory for run outputs and checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom JSON configuration file",
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Workflow commands
    subparsers.add_parser("wp-a", help="Run Work Package A: Sensor placement & evaluation")
    subparsers.add_parser("wp-b", help="Run Work Package B: Dynamic encoder & architecture ablations")
    subparsers.add_parser("wp-c", help="Run Work Package C: Noise robustness evaluation")
    subparsers.add_parser("wp-d", help="Run Work Package D: Full flow field reconstruction")

    # Pipeline stages
    subparsers.add_parser("preprocess", help="Preprocess Fluent data to H5")
    subparsers.add_parser("train-encoder", help="Train sensor encoder (MLP / DeepSets)")
    subparsers.add_parser("train-ppinn", help="Train parametric PINN")
    subparsers.add_parser("eval", help="Evaluate end-to-end pipeline")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    cfg = get_path_config(
        config_path=args.config,
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        runs_dir=args.runs_dir,
    )
    print(f"Executing subcommand: {args.subcommand}")
    print(f"Resolved data_root: {cfg.data_root}")
    print(f"Resolved runs_dir: {cfg.runs_dir}")


if __name__ == "__main__":
    main()
