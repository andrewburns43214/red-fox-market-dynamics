# Inspect the ledger_rows population area
Get-Content .\main.py | Select-String -Pattern 'ledger_rows' -Context 5,10
