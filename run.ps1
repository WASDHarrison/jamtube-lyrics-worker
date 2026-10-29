# Starts a worker on Windows, setting everything up the first time, with nothing to install first.
# Straight from the server, which fills in its address and key (in PowerShell):
#
#   irm https://lyrics.example.com/workers/start/<WORKER_KEY> | iex
#
# or from a copy of this folder:
#
#   powershell -ExecutionPolicy Bypass -File run.ps1 --server https://lyrics.example.com --key <WORKER_KEY>
#
# Anything after it goes to worker.py. It all goes in one folder: this one, or from the server
# %LOCALAPPDATA%\jamtube-lyrics-worker (JAMTUBE_WORKER_DIR). uv, the Python it brings, what the
# worker needs and the model. Deleting the folder removes the worker.

# Run as a block, so nothing of it stays in the window it was started from.
& {
  # The server fills these in when it hands this script out.
  $server = ''
  $key = ''

  $ErrorActionPreference = 'Stop'
  $ProgressPreference = 'SilentlyContinue' # downloads are far slower drawing their progress
  $pythonVersion = if ($env:PYTHON_VERSION) { $env:PYTHON_VERSION } else { '3.12' } # 3.10 to 3.13
  $uvInstaller = if ($env:UV_INSTALLER) { $env:UV_INSTALLER } else { 'https://astral.sh/uv/install.ps1' }

  $workerArgs = $args
  if ($server) {
    $dir = if ($env:JAMTUBE_WORKER_DIR) { $env:JAMTUBE_WORKER_DIR } else { Join-Path $env:LOCALAPPDATA 'jamtube-lyrics-worker' }
    New-Item -ItemType Directory -Force $dir | Out-Null
    Write-Host "Getting the worker from $server"
    foreach ($file in 'worker.py', 'requirements-cpu.txt', 'requirements-gpu.txt') {
      Invoke-WebRequest -UseBasicParsing -Headers @{ Authorization = "Bearer $key" } "$server/workers/files/$file" -OutFile (Join-Path $dir $file)
    }
    $workerArgs = @('--server', $server, '--key', $key) + $args
  } else {
    $dir = $PSScriptRoot
  }

  # uv: it brings the right Python and installs what the worker needs, all in this folder. Its
  # settings are only for this, and taken away after.
  $settings = @{ UV_CACHE_DIR = "$dir\.uv\cache"; UV_PYTHON_INSTALL_DIR = "$dir\.uv\python"; UV_INSTALL_DIR = "$dir\.uv\bin"; UV_NO_MODIFY_PATH = '1' }
  $before = @{}
  foreach ($name in $settings.Keys) {
    $before[$name] = [Environment]::GetEnvironmentVariable($name)
    [Environment]::SetEnvironmentVariable($name, $settings[$name])
  }
  Push-Location $dir
  try {
    $uv = "$dir\.uv\bin\uv.exe"
    if (-not (Test-Path $uv)) {
      Write-Host 'Getting uv (it brings Python)'
      powershell -NoProfile -ExecutionPolicy Bypass -Command "irm '$uvInstaller' | iex" | Out-Null
      if (-not (Test-Path $uv)) { throw "Couldn't get uv from $uvInstaller" }
    }
    if (-not (Test-Path .venv\Scripts\python.exe)) {
      & $uv venv --quiet --python $pythonVersion .venv
      if ($LASTEXITCODE) { throw "Couldn't set up Python $pythonVersion" }
    }

    # The GPU build with an NVIDIA GPU. .venv\installed is what was installed last.
    $kind = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { 'gpu' } else { 'cpu' }
    $installed = if (Test-Path .venv\installed) { Get-Content -Raw .venv\installed } else { '' }
    if ((Get-Content -Raw "requirements-$kind.txt") -ne $installed) {
      Write-Host "Installing what it needs ($kind): the first time only, a few minutes"
      & $uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r "requirements-$kind.txt"
      if ($LASTEXITCODE) { throw "Couldn't install what it needs" }
      Copy-Item "requirements-$kind.txt" .venv\installed
    }

    & .venv\Scripts\python.exe worker.py @workerArgs
  } finally {
    Pop-Location
    foreach ($name in $before.Keys) { [Environment]::SetEnvironmentVariable($name, $before[$name]) }
  }
} @args
