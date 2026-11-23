"""Smoke tests for workflows.__init__ exports, CLI subcommands, and PS1 flag parity."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
from typing import List

import pytest

import cylinder_flow.workflows as workflows
from cylinder_flow.cli import build_parser, main as cli_main

SUBCOMMANDS = (
    "wp-a",
    "wp-b",
    "wp-c",
    "wp-d",
    "preprocess",
    "train-encoder",
    "train-ppinn",
    "eval",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def test_workflows_package_exports() -> None:
    expected = (
        "WPAConfig",
        "WP_A_STAGES",
        "load_wp_a_config",
        "select_ppinn_by_validation",
        "run_wp_a",
        "WPBConfig",
        "WP_B_STAGES",
        "load_wp_b_config",
        "generate_wp_b_layout_holdout",
        "run_wp_b",
        "WPCConfig",
        "WP_C_STAGES",
        "load_wp_c_config",
        "WPCProgressStore",
        "evaluate_wp_c_unit",
        "run_wp_c",
        "WPDConfig",
        "WP_D_STAGES",
        "load_wp_d_config",
        "evaluate_wp_d_reconstruction",
        "build_reconstruction_report",
        "run_wp_d",
    )
    for name in expected:
        assert name in workflows.__all__
        assert hasattr(workflows, name)


def test_cli_help_and_all_subcommands_via_subprocess() -> None:
    root = _repo_root()
    top = subprocess.run(
        [sys.executable, "-m", "cylinder_flow", "--help"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert top.returncode == 0, top.stderr
    for sub in SUBCOMMANDS:
        assert sub in top.stdout
        sub_proc = subprocess.run(
            [sys.executable, "-m", "cylinder_flow", sub, "--help"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert sub_proc.returncode == 0, f"{sub} --help failed: {sub_proc.stderr}"


@pytest.mark.parametrize(
    ("subcmd", "cfg_rel", "extra"),
    [
        ("wp-a", "configs/experiments/wp_a/protocol.example.json", []),
        ("wp-b", "configs/experiments/wp_b/protocol.example.json", ["--mode", "dynamic"]),
        ("wp-b", "configs/experiments/wp_b/arch_ablation_protocol.example.json", ["--mode", "arch_ablation"]),
        ("wp-c", "configs/experiments/wp_c/protocol.example.json", []),
        ("wp-d", "configs/experiments/wp_d/protocol.example.json", []),
    ],
)
def test_workflow_subcommands_dry_run_main_and_subprocess(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    subcmd: str,
    cfg_rel: str,
    extra: List[str],
) -> None:
    root = _repo_root()
    cfg_path = root / cfg_rel
    out_dir = tmp_path / f"out_{subcmd}_{len(extra)}"

    argv = [
        subcmd,
        "--config",
        str(cfg_path),
        "--stages",
        "preflight",
        "--output-dir",
        str(out_dir),
        "--dry-run",
        *extra,
    ]
    rc = cli_main(argv)
    assert rc == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["dry_run"] is True

    # Subprocess verification for python -m cylinder_flow
    proc = subprocess.run(
        [sys.executable, "-m", "cylinder_flow", *argv],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    sub_payload = json.loads(proc.stdout)
    assert sub_payload["dry_run"] is True


def test_ps1_wrappers_flags_accepted_by_parser(tmp_path: Path) -> None:
    root = _repo_root()
    parser = build_parser()

    # Inspect all 4 scripts and verify every forwarded --flag is accepted by argparse
    ps1_files = {
        "wp-a": root / "scripts" / "windows" / "run_wp_a.ps1",
        "wp-b": root / "scripts" / "windows" / "run_wp_b.ps1",
        "wp-c": root / "scripts" / "windows" / "run_wp_c.ps1",
        "wp-d": root / "scripts" / "windows" / "run_wp_d.ps1",
    }
    for subcmd, ps1_path in ps1_files.items():
        text = ps1_path.read_text(encoding="utf-8")
        flags = sorted(set(re.findall(r"--[a-z0-9-]+", text)))
        # Build representative argv containing every explicitly forwarded flag in that wrapper
        argv: List[str] = []
        if subcmd == "wp-b":
            # run_wp_b.ps1 forwards path overrides before wp-b and workflow flags after
            argv.extend(
                [
                    "--config",
                    str(root / "configs" / "experiments" / "wp_b" / "protocol.example.json"),
                    "--data-dir",
                    str(tmp_path / "data"),
                    "--processed-dir",
                    str(tmp_path / "proc"),
                    "--runs-dir",
                    str(tmp_path / "runs"),
                    "wp-b",
                    "--stages",
                    "preflight",
                    "--mode",
                    "dynamic",
                    "--dry-run",
                    "--allow-test",
                ]
            )
        elif subcmd == "wp-c":
            argv.extend(
                [
                    "wp-c",
                    "--config",
                    str(root / "configs" / "experiments" / "wp_c" / "protocol.example.json"),
                    "--stages",
                    "preflight",
                    "--output-dir",
                    str(tmp_path / "out_c"),
                    "--dry-run",
                    "--resume",
                    "--allow-test",
                ]
            )
        else:
            argv.append(subcmd)
            for fl in flags:
                if fl in {"--dry-run", "--allow-test", "--no-png", "--overwrite", "--formal"}:
                    argv.append(fl)
                elif fl == "--config":
                    argv.extend([fl, str(root / "configs" / "experiments" / subcmd.replace("-", "_") / "protocol.example.json")])
                elif fl == "--stages":
                    argv.extend([fl, "preflight"])
                else:
                    argv.extend([fl, str(tmp_path / "x")])
        parsed = parser.parse_args(argv)
        assert parsed.subcommand == subcmd


def test_pipeline_subcommands_dry_run_precedence_and_guards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    custom_data = tmp_path / "custom_data"
    custom_runs = tmp_path / "custom_runs"

    # Global --data-dir before subcommand is NOT clobbered by subcommand parser
    assert cli_main(["--data-dir", str(custom_data), "preprocess"]) == 0
    pre_out = json.loads(capsys.readouterr().out)
    assert pre_out["status"] == "external_step_notice"
    assert Path(pre_out["data_root"]) == custom_data.resolve()

    # eval --dry-run and locked-test Re guard
    assert (
        cli_main(
            [
                "eval",
                "--mode",
                "oracle",
                "--re-list",
                "450,800",
                "--data-dir",
                str(custom_data),
                "--runs-dir",
                str(custom_runs),
                "--dry-run",
            ]
        )
        == 0
    )
    eval_out = json.loads(capsys.readouterr().out)
    assert eval_out["status"] == "dry_run"
    assert eval_out["re_ids"] == [450, 800]
    assert eval_out["allow_test"] is False

    with pytest.raises(PermissionError):
        cli_main(["eval", "--mode", "oracle", "--re-list", "250", "--dry-run"])

    # train-encoder --dry-run (both mlp & deepsets)
    for enc in ("mlp", "deepsets"):
        assert (
            cli_main(
                [
                    "train-encoder",
                    "--encoder",
                    enc,
                    "--epochs",
                    "3",
                    "--seed",
                    "7",
                    "--output-dir",
                    str(tmp_path / f"enc_{enc}"),
                    "--dry-run",
                ]
            )
            == 0
        )
        te_out = json.loads(capsys.readouterr().out)
        assert te_out["status"] == "dry_run"
        assert te_out["encoder"] == enc
        assert te_out["epochs"] == 3

    # train-ppinn --dry-run
    assert (
        cli_main(
            [
                "train-ppinn",
                "--epochs",
                "4",
                "--lambda-phys",
                "0.5",
                "--seed",
                "9",
                "--output-dir",
                str(tmp_path / "ppinn"),
                "--dry-run",
            ]
        )
        == 0
    )
    tp_out = json.loads(capsys.readouterr().out)
    assert tp_out["status"] == "dry_run"
    assert tp_out["epochs"] == 4
    assert tp_out["lambda_phys"] == pytest.approx(0.5)
