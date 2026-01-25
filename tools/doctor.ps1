Write-Host "[doctor] py_compile..."
python -m py_compile .\main.py
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[doctor] report..."
python .\main.py report
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[doctor] ledger_check..."
python .\tools\ledger_check.py
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[doctor] OK"
