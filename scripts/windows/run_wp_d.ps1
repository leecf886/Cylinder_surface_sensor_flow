<#
.SYNOPSIS
  Thin Windows PowerShell wrapper for Work Package D (Full-Field Reconstruction & Report).
.DESCRIPTION
  Forwards arguments to `python -m cylinder_flow wp-d` using a configurable
  -PythonExe (defaulting to `python`). Defaults to the safe `preflight` stage.
#>
[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [string]$Config = "configs/experiments/wp_d/protocol.example.json",
    [string]$Stages = "preflight",
    [string]$DataDir = "",
    [string]$ProcessedDir = "",
    [string]$RunsDir = "",
    [switch]$DryRun,
    [switch]$AllowTest,
    [switch]$NoPng,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$CliArgs = @("-m", "cylinder_flow", "wp-d")
if ($Config) {
    $CliArgs += @("--config", $Config)
}
if ($Stages) {
    $CliArgs += @("--stages", $Stages)
}
if ($DataDir) {
    $CliArgs += @("--data-dir", $DataDir)
}
if ($ProcessedDir) {
    $CliArgs += @("--processed-dir", $ProcessedDir)
}
if ($RunsDir) {
    $CliArgs += @("--runs-dir", $RunsDir)
}
if ($DryRun) {
    $CliArgs += "--dry-run"
}
if ($AllowTest) {
    $CliArgs += "--allow-test"
}
if ($NoPng) {
    $CliArgs += "--no-png"
}
if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $CliArgs += $ExtraArgs
}

Push-Location $RepoRoot
try {
    & $PythonExe @CliArgs
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $exitCode
