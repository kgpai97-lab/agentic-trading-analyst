<#
.SYNOPSIS
  Nightly forward-test driver for Windows Task Scheduler.

  Runs all three portfolios for today's weekday (one PortfolioGraph rebalance
  each) and appends to outputs/forward/<portfolio>/<weekday>/. Writes a
  timestamped log per run under outputs/forward/_logs/ so unattended runs are
  auditable. Uses the project venv's python directly (no activation needed).

  This script is what the scheduled task invokes. It is intentionally tiny and
  self-contained: set location -> run -> log -> exit code.

.NOTES
  Schedule it (run once, in an elevated PowerShell) with the schtasks command in
  scripts/register_forward_task.txt. The task runs as the current user so it
  inherits the .env (Ollama key). The machine must be awake at fire time.
#>

$ErrorActionPreference = "Stop"

# Project root = the parent of this script's folder.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "venv python not found at $Python"
    exit 1
}

# Per-run timestamped log.
$LogDir = Join-Path $Root "outputs\forward\_logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Log = Join-Path $LogDir "forward_$Stamp.log"

"=== forward run started $(Get-Date -Format o) ===" | Out-File -FilePath $Log -Encoding utf8

# Run BOTH variants (prod 0.5 + full-coverage 1.0) for today's weekday across all
# three portfolios. Force UTF-8 so any unicode in model output doesn't crash on
# the default cp1252 console.
$env:PYTHONIOENCODING = "utf-8"
# Pipe stdout to Tee (terminal + log). Stderr is redirected separately to the
# same log via Start-Process so that yfinance/library warnings on stderr don't
# become NativeCommandErrors that abort the run under $ErrorActionPreference=Stop.
$proc = Start-Process -FilePath $Python -ArgumentList "forward_run.py","--variant","both" `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput "$Log.stdout.tmp" `
    -RedirectStandardError  "$Log.stderr.tmp"
$proc.WaitForExit()
$code = $proc.ExitCode
# Merge stdout + stderr into the log (UTF-8) and echo to terminal.
if (Test-Path "$Log.stdout.tmp") {
    $txt = Get-Content "$Log.stdout.tmp" -Encoding utf8 -Raw
    $txt | Out-File -FilePath $Log -Encoding utf8 -Append
    Write-Host $txt
    Remove-Item "$Log.stdout.tmp"
}
if (Test-Path "$Log.stderr.tmp") {
    $txt = Get-Content "$Log.stderr.tmp" -Encoding utf8 -Raw
    $txt | Out-File -FilePath $Log -Encoding utf8 -Append
    Write-Host $txt
    Remove-Item "$Log.stderr.tmp"
}

"=== forward run finished $(Get-Date -Format o) exit=$code ===" | Out-File -FilePath $Log -Encoding utf8 -Append
exit $code
