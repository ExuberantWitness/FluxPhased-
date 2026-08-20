$base = "E:\DATA\vscode\FluxPhased\experiments\array_face_s6\learning_repair\s6_selfplay_output_seed20260729"
while ($true) {
  if (Test-Path "$base\selfplay_latest.pt") {
    Copy-Item "$base\selfplay_latest.pt" "$base\selfplay_backup.pt" -Force
  }
  Start-Sleep -Seconds 300
}
