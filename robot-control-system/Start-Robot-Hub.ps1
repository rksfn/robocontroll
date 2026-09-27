$ErrorActionPreference = 'Stop'
# Stop Windows powering down USB ports (a common cause of the arm's COM port dropping).
try {
    powercfg /SETACVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0 | Out-Null
    powercfg /SETDCVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0 | Out-Null
    powercfg /SETACTIVE SCHEME_CURRENT | Out-Null
    Write-Host 'USB selective suspend disabled (keeps the arm connection alive).'
} catch { Write-Host 'Could not change USB power setting (not critical).' }
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
