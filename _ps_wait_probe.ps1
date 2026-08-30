# probe: how does Select-String -Quiet behave on a missing file inside while()?
$missing = 'E:\DATA\vscode\FluxPhased\_definitely_missing_chain.log'
$exists  = 'E:\DATA\vscode\FluxPhased\experiments\array_face_s7\learning_repair\taes_chain.log'

$r1 = Select-String -Path $missing -Pattern 'X' -Quiet -ErrorAction SilentlyContinue
"missing ->Quiet returns: [$r1]  -not -> [$(-not $r1)]"
$r2 = Test-Path $missing
"TestPath missing: [$r2]"

# the pattern used in the failed chain (no -ErrorAction):
try { $r3 = Select-String -Path $missing -Pattern 'X' -Quiet } catch { "THREW: $($_.Exception.Message)" }
"no-EA result: [$r3]  -not -> [$(-not $r3)]"

# hardened form evaluation:
$hard = -not ((Test-Path $missing) -and (Select-String -Path $missing -Pattern 'X' -Quiet))
"hardened while-condition on missing: [$hard]  (true = would sleep, correct)"
$hard2 = -not ((Test-Path $exists) -and (Select-String -Path $exists -Pattern 'ALL TAES RUNS DONE' -Quiet))
"hardened while-condition on real log without marker: [$hard2]  (true = would sleep, correct)"
