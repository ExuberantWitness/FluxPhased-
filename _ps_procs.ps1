Get-CimInstance Win32_Process | Where-Object { $_.Name -like '*python*' -or $_.Name -like '*pwsh*' -or $_.Name -like '*powershell*' } | ForEach-Object {
  $cmd = $_.CommandLine
  if ($cmd.Length -gt 150) { $cmd = $cmd.Substring(0, 150) }
  '{0}  {1}  {2}' -f $_.ProcessId, $_.Name, $cmd
}
