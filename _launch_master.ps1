Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','E:\DATA\vscode\FluxPhased\_run_master_chain.ps1' -WindowStyle Hidden -RedirectStandardOutput 'E:\DATA\vscode\FluxPhased\_master_chain_stdout.log' -RedirectStandardError 'E:\DATA\vscode\FluxPhased\_master_chain_stderr.log'
'detached launch issued'
