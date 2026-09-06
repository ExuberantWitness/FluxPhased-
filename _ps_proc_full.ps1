Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -or $_.Name -eq 'powershell.exe' } | ForEach-Object {
  [PSCustomObject]@{PID=$_.ProcessId; Name=$_.Name; Command=$_.CommandLine}
} | ConvertTo-Json -Depth 3
