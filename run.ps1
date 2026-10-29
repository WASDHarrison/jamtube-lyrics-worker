# Runs a worker on Windows, setting it up the first time (a Python environment in .venv, with
# the GPU build when there's an NVIDIA GPU). Anything after it goes to worker.py:
#
#   powershell -ExecutionPolicy Bypass -File run.ps1 --server https://lyrics.example.com --key <WORKER_KEY>
#
# (or set JAMTUBE_LYRICS_URL and WORKER_KEY).
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path .venv\Scripts\python.exe)) {
  # A Python the vocal removal works with (3.10 to 3.13), newest first, through the py launcher.
  $version = '3.13', '3.12', '3.11', '3.10' | Where-Object { & py "-$_" -c 'pass' 2>$null; $LASTEXITCODE -eq 0 } | Select-Object -First 1
  if (-not $version) {
    Write-Error 'Needs Python 3.10 to 3.13: winget install Python.Python.3.12'
  }
  & py "-$version" -m venv .venv
}

$kind = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { 'gpu' } else { 'cpu' }
$installed = if (Test-Path .venv\installed) { Get-Content .venv\installed } else { '' }
if ($installed -ne $kind -or (Get-Item "requirements-$kind.txt").LastWriteTime -gt (Get-Item .venv\installed -ErrorAction SilentlyContinue).LastWriteTime) {
  Write-Host "Installing what it needs ($kind), the first time only..."
  & .venv\Scripts\python.exe -m pip install --quiet --upgrade pip
  & .venv\Scripts\python.exe -m pip install --quiet -r "requirements-$kind.txt"
  Set-Content .venv\installed $kind
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Write-Error 'Needs ffmpeg: winget install Gyan.FFmpeg (then open a new window)'
}

& .venv\Scripts\python.exe worker.py @args
exit $LASTEXITCODE
