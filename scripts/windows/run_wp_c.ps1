<#
.SYNOPSIS
  Thin Windows wrapper for the Work Package C (WP-C) robustness CLI entry point.
.DESCRIPTION
  Forwards arguments to ``python -m cylinder_flow wp-c`` using a configurable
  ``-PythonExe`` parameter (default: ``python``) with zero hardcoded drive paths.
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$PythonExe = "python",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

$ArgList = @("-m", "cylinder_flow", "wp-c")
if ($ForwardArgs) {
    $ArgList += $ForwardArgs
}

Push-Location $RepoRoot
try {
    & $PythonExe @ArgList
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($null -ne $exitCode -and $exitCode -ne 0) {
    exit $exitCode
}
