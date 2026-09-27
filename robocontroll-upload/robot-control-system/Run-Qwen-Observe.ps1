param([Parameter(Mandatory=$true)][string]$Goal, [switch]$Execute)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($Execute) { & '.\.venv\Scripts\python.exe' qwen_agent.py --goal $Goal --execute }
else { & '.\.venv\Scripts\python.exe' qwen_agent.py --goal $Goal }
