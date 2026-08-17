# Orchestrate Eco's full fractal regeneration, oldest->newest.
# Phase 1: all weeklies (Eco life from 2026-03-03). Phase 2: monthlies. Phase 3: quarterly.
# Each phase waits for completion before the next so telescopic stacking has its sources.
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Admin\Documents\EcoDB"
$key = (Get-Content ".env" | Where-Object { $_ -match "^ECODB_API_KEY=" }) -replace "^ECODB_API_KEY=",""
$jwt = (Invoke-RestMethod -Uri "http://localhost:8080/auth/token" -Method Post -Body (@{api_key=$key.Trim()}|ConvertTo-Json) -ContentType "application/json").access_token
$h = @{ Authorization = "Bearer $jwt" }

function Trigger($level, $ps, $pe) {
    $u = "http://localhost:8080/api/v1/cells/trigger/consolidation?agent_identifier=Eco&level=$level&period_start=$ps&period_end=$pe"
    try { Invoke-RestMethod -Uri $u -Method Post -Headers $h | Out-Null } catch { Write-Output "trigger-fail $level $ps : $($_.Exception.Message)" }
}

function WaitConsolidation($label) {
    while ($true) {
        $running = docker exec ecodb-postgres psql -U ecodb -d ecodb -t -A -c "SELECT count(*) FROM cell_runs cr JOIN agents a ON a.id=cr.agent_id WHERE a.identifier='Eco' AND cr.cell_type='consolidation' AND cr.status='running'"
        if ([int]$running -eq 0) { break }
        Start-Sleep 6
    }
    Write-Output "PHASE_DONE $label"
}

# Phase 1 — weekly 7-day windows from 2026-03-03 to current
$start = [datetime]"2026-03-03"
$today = Get-Date
$weeks = @()
$cur = $start
while ($cur -le $today) {
    $end = $cur.AddDays(6)
    if ($end -gt $today) { $end = $today }
    $weeks += ,@($cur.ToString("yyyy-MM-dd"), $end.ToString("yyyy-MM-dd"))
    $cur = $cur.AddDays(7)
}
Write-Output "WEEKS_TOTAL $($weeks.Count)"
foreach ($w in $weeks) { Trigger "weekly" $w[0] $w[1]; Start-Sleep 2 }
WaitConsolidation "weekly"

# Phase 2 — monthly (complete months Eco lived: Mar, Apr, May)
$months = @(
    @("2026-03-01","2026-03-31"),
    @("2026-04-01","2026-04-30"),
    @("2026-05-01","2026-05-31")
)
foreach ($m in $months) { Trigger "monthly" $m[0] $m[1]; Start-Sleep 2 }
WaitConsolidation "monthly"

# Phase 3 — quarterly (the trimester spanning her monthlies)
Trigger "quarterly" "2026-03-01" "2026-05-31"
WaitConsolidation "quarterly"

# Summary
$summary = docker exec ecodb-postgres psql -U ecodb -d ecodb -t -A -c "SELECT level||':'||count(*) FROM memory_clusters mc JOIN agents a ON a.id=mc.agent_id WHERE a.identifier='Eco' GROUP BY level ORDER BY level"
Write-Output "ECO_FRACTAL_COMPLETE $summary"
