# Self-healing bulk-download loop, launched detached via Start-Process so it survives
# beyond any single tool-call. Re-runs bulk-fetch (which resumes via the incremental
# manifest, skipping completed files) until data/raw crosses the target, then stops.
Set-Location C:\Users\murar\pdac-circuit
$target = 108000000000   # ~100.6 GB
$log = "$env:TEMP\bulk.log"
while ($true) {
  $sz = (Get-ChildItem -Recurse -File data\raw -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
  if ($sz -ge $target) { "GOAL_DONE raw=$([math]::Round($sz/1GB,1))GB" | Out-File -Append $log; break }
  & .\.venv\Scripts\python.exe -m pdac_circuit.pipeline.cli bulk-fetch --target-gb 110 --workers 4 *>> $log
  Start-Sleep -Seconds 3
}
