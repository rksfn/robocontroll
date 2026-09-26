param([Parameter(Mandatory=$true)][string]$Goal)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& '.\.venv\Scripts\python.exe' qwen_agent.py --goal $Goal
