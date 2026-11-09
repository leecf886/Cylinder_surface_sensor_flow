"""Command-line interface and argument parser for cylinder_flow."""
import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cylinder_flow",
        description="Cylinder Surface Sensor Flow Pipeline",
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
    print(f"Executing subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
