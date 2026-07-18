[gpu] Using device: cuda
[gpu] GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
[gpu] VRAM: 101.9 GB
Loading RL ckpt: checkpoints/blind/wp3_20260718_090802/iter_final.pt
  iter=100

======================================================================
  Condition: low_interference (channel_mode=orthogonal)
======================================================================
  [RL=team0] batch done, total eps=16/50
  [RL=team0] batch done, total eps=32/50
  [RL=team0] batch done, total eps=48/50
  [RL=team0] batch done, total eps=50/50
  [RL=team1] batch done, total eps=16/50
  [RL=team1] batch done, total eps=32/50
  [RL=team1] batch done, total eps=48/50
  [RL=team1] batch done, total eps=50/50

  RL kills:    0.031 ± 0.174  (n=128)
  BC kills:    0.938 ± 0.242
  Δ = -0.906  (Welch t=-34.26, p=0.000)
  RL survival: 0.815
  RL expos:    1.247
  RL trace_P:  284.952

======================================================================
  Condition: high_interference (channel_mode=same_channel)
======================================================================
  [RL=team0] batch done, total eps=16/50
  [RL=team0] batch done, total eps=32/50
  [RL=team0] batch done, total eps=48/50
  [RL=team0] batch done, total eps=50/50
  [RL=team1] batch done, total eps=16/50
  [RL=team1] batch done, total eps=32/50
  [RL=team1] batch done, total eps=48/50
  [RL=team1] batch done, total eps=50/50

  RL kills:    0.039 ± 0.194  (n=128)
  BC kills:    0.211 ± 0.408
  Δ = -0.172  (Welch t=-4.29, p=0.000)
  RL survival: 0.939
  RL expos:    1.200
  RL trace_P:  326.627

======================================================================
Smoke cross-play done in 1.8 min
======================================================================

Verdict per WP-3 spec:
  low_interference: Δ_kills=-0.906  → FAIL
  high_interference: Δ_kills=-0.172  → MARGINAL/FAIL (诚实记录)

Report → experiments/twoteam/wp3_smoke_crossplay_report.md
