<#
.SYNOPSIS
  Thin Windows PowerShell wrapper for Work Package A (Scheme A).
.DESCRIPTION
  Forwards arguments to `python -m cylinder_flow wp-a` using a configurable
  -PythonExe (defaulting to `python`). Defaults to the safe `preflight` stage.
#>
[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [string]$Config = "configs/experiments/wp_a/protocol.example.json",
    [string]$Stages = "preflight",
    [string]$DataDir = "",
    [string]$ProcessedDir = "",
    [string]$RunsDir = "",
    [switch]$DryRun,
    [switch]$AllowTest,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$CliArgs = @("-m", "cylinder_flow", "wp-a")
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
