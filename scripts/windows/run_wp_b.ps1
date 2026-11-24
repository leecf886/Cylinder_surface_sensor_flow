<#
.SYNOPSIS
  Thin Windows PowerShell wrapper for Work Package B (dynamic encoder & architecture ablations).
.DESCRIPTION
  Forwards arguments to `python -m cylinder_flow wp-b`. Defaults to the safe
  `preflight` stage with configurable `-PythonExe` (no hardcoded drive paths).
#>
[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [string]$ConfigPath = ".\configs\experiments\wp_b\protocol.example.json",
    [string]$DataDir = "",
    [string]$ProcessedDir = "",
    [string]$RunsDir = "",
    [string]$Stages = "preflight",
    [ValidateSet("dynamic", "arch_ablation")]
    [string]$Mode = "dynamic",
    [switch]$DryRun,
    [switch]$AllowTest,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$CliArgs = @("-m", "cylinder_flow")
if ($ConfigPath) {
    $CliArgs += @("--config", $ConfigPath)
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
$CliArgs += @("wp-b", "--stages", $Stages)
if ($PSBoundParameters.ContainsKey("Mode")) {
    $CliArgs += @("--mode", $Mode)
}
if ($DryRun.IsPresent) {
    $CliArgs += "--dry-run"
}
if ($AllowTest.IsPresent) {
    $CliArgs += "--allow-test"
}
if ($ExtraArgs) {
    $CliArgs += $ExtraArgs
}

Write-Host "WP-B wrapper: stages=$Stages dry_run=$($DryRun.IsPresent) allow_test=$($AllowTest.IsPresent)"
Write-Host ("Command: " + $PythonExe + " " + ($CliArgs -join " "))

& $PythonExe @CliArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
