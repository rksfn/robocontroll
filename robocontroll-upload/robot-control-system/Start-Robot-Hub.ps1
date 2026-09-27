$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required.' }
    & $pythonPath -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
$process = Start-Process -FilePath $pythonPath -ArgumentList '-m','robot_hub.server' -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 4
if ($process.HasExited) { throw 'Robot Hub could not start. Run the server command manually to see the error.' }
Start-Process 'http://127.0.0.1:8765/'
Write-Host 'Robot Hub is running at http://127.0.0.1:8765/'
Write-Host 'Close this window or press Ctrl-C to stop the service.'
try { Wait-Process -Id $process.Id } finally { if (-not $process.HasExited) { Stop-Process -Id $process.Id } }
