# WP-D — Executable Agent Instruction Package

**Purpose**: One-stop runbook for each work package in PAPER_PLAN_EAAI.md. Each WP has:
1. **Config file** (ready to run, no edits needed)
2. **Command line** (copy-paste to PRO 6000)
3. **PASS/FAIL validator** (scripted, not subjective)
4. **GPU budget** + wall-clock estimate
5. **Output artifacts** (logs, checkpoints, figures)

All paths are relative to repo root `/home/ubuntu/CODE/FluxPhased-/`.
Conda env: `/home/ubuntu/miniconda3/envs/fluxphased/bin/python`.

---

## WP1 — FluxLeague Convergence Gate (止损闸门)

**Status**: RUNNING (started 2026-06-21, PID in logs/wp1_gate_seed42.log)

| Item | Value |
|---|---|
| Config | [configs/wp1_gate.yaml](configs/wp1_gate.yaml) |
| Validator | [scripts/wp1_validate.py](scripts/wp1_validate.py) |
| Log | `logs/wp1_gate_seed42.log` |
| Checkpoints | `checkpoints/wp1_gate_seed42/` |
| Budget | ~60-150 GPU-h (30 iters × ~2-5h/iter) |

### Run command

```bash
nohup /home/ubuntu/miniconda3/envs/fluxphased/bin/python -m training.train \
  --config configs/wp1_gate.yaml \
  > logs/wp1_gate_seed42.log 2>&1 &
```

### PASS criteria (all 4 must hold)

```bash
/home/ubuntu/miniconda3/envs/fluxphased/bin/python scripts/wp1_validate.py logs/wp1_gate_seed42.log
# exit 0 = PASS, exit 1 = FAIL
```

| # | Criterion | Threshold |
|---|---|---|
| 1 | Final kill_radius | ≤0.5m ideal, ≤1m minimum |
| 2 | Final iter 0.50 ratio | <10% |
| 3 | Final NashConv | <0.05 OR monotone decreasing |
| 4 | kill_radius monotonic | no rebound |

### Decision logic

- **PASS** → this run becomes WP2 cell-A seed-0; proceed to WP2.
- **FAIL with kr>5m** → STOP. Pivot paper to C1-engineering + train_laser PSRO-lite (honest negative result).
- **FAIL only on 0.50 ratio** → rerun with `n_eval_games=100` (variance issue, not fundamental).

---

## WP2 — External Baselines (命门对比)

**Status**: BLOCKED on WP1 PASS. Implement adapters first (1-2 weeks engineering).

| Baseline | Config | Adapter | Est. GPU-h |
|---|---|---|---|
| IPPO | `configs/wp2_ippo.yaml` | set `league.team_critic_enabled=False`, no PFSP | ~40 × 5 seeds |
| MAPPO | `configs/wp2_mappo.yaml` | current config + `league=false` (no league) | ~40 × 5 seeds |
| QMIX | `configs/wp2_qmix.yaml` | **NEW adapter**: PyMARL wrapper | ~40 × 5 seeds |
| Classical MPC | `configs/wp2_classical.yaml` | **NEW**: greedy beam scheduler + Kalman + threshold fire | ~5 × 3 seeds |

### QMIX adapter (TODO)

```python
# training/baselines/qmix_adapter.py  (TO WRITE)
# - Wraps MFARVecEnv as PettingZoo-compatible
# - Imports PyMARL QMIX from pip
# - Converts obs/action spaces
# - Runs 30 iters of QMIX training
```

### Classical MPC controller (TODO)

```python
# training/baselines/classical_mpc.py  (TO WRITE)
# - Uses env's Kalman fused estimate directly (no RL)
# - Greedy: each radar beam-steers to fused target position
# - Fires when illumination_progress derivative > threshold
# - Same eval protocol as RL policies
```

### PASS criteria (main paper claim)

Cell A (FluxLeague full) must beat ALL 4 baselines on:
1. **Sample efficiency**: GPU-h to reach kr=1m (Welch's t, p<0.05, 5 seeds)
2. **Held-out exploitability**: win rate vs unseen opponent >0.7
3. **Final kr**: ≤0.5m, statistically indistinguishable from or better than best baseline

---

## WP3 — Engineering Realism Anchors

### WP3.1 — CRLB Physical Anchor (DONE)

| Item | Value |
|---|---|
| Script | [scripts/wp3_crlb_anchor.py](scripts/wp3_crlb_anchor.py) |
| Outputs | [figures/wp3_crlb_vs_baseline.pdf](figures/wp3_crlb_vs_baseline.pdf), [figures/wp3_crlb_ellipse.pdf](figures/wp3_crlb_ellipse.pdf), [logs/wp3_crlb_summary.txt](logs/wp3_crlb_summary.txt) |
| GPU cost | 0 (analytical) |

**Key result**: STATIC CRLB knee at baseline ≥1.19km; C1's 5km requirement gives 4× margin. Tracked mode (N=120) gives sub-0.2m everywhere, but only if initial anchor is good.

### WP3.2 — Damage Injection (TODO)

| Damage | Config key | Sweep range |
|---|---|---|
| Clutter | `env.clutter_model` | {none, weibull, k-dist} × {−10, 0, 10} dB |
| Multipath | `env.multipath_model` | {none, 2-ray, rayleigh} |
| Sensing bias | `sensing_bias_m` | {0, 1, 5} m |
| Control delay | `env.control_delay_steps` | {0, 1, 3} |
| Beam slew limit | `env.max_slew_rate_deg_per_s` | {∞, 60, 30} |
| Comm rate (ISAC) | `sensing.comm_rate_bps` | {∞, 1k, 100} |

**Run command (per damage cell)**:

```bash
/home/ubuntu/miniconda3/envs/fluxphased/bin/python -m training.train \
  --config configs/wp1_gate.yaml \
  --override env.clutter_model=weibull \
  --override env.clutter_snr_db=-10 \
  > logs/wp3_clutter_weibull_neg10.log 2>&1
```

(Then `scripts/wp3_robustness_aggregate.py` — TODO — collates into robustness table.)

### WP3.3 — Real Radar Data Validation (stretch goal)

- **Dataset**: RadarScenes (public, ~12 GB)
- **What to validate**: detection statistics + range-doppler preprocessing
- **Not validated**: full closed loop (stays simulation)
- **Cost**: 2-3 days engineering, no GPU

---

## WP4 — Generalization Studies

Train one policy on nominal, test on variants.

| Axis | Train | Test variants |
|---|---|---|
| Target dynamics | static | {static, const-vel, maneuvering} |
| Geometry | fixed 5km baseline | {1km, 3km, 7km, random} |
| Radar count | 4 (2 per team) | {2, 3, 6} |
| Team count | 2 | {2, 3} |
| EW condition | none | {jamming, exposure race} |

**Command (eval-only on trained WP1 checkpoint)**:

```bash
/home/ubuntu/miniconda3/envs/fluxphased/bin/python -m training.train \
  --resume checkpoints/wp1_gate_seed42/league_state.pt \
  --override env.target_dynamics=maneuvering \
  --override training.psro_iterations=0 \
  --override league.n_eval_games=50
```

**Output**: `figures/wp4_generalization_heatmap.pdf` (train × test matrix, OOD degradation %).

---

## WP5 — Internal Ablations

All configs inherit from [configs/wp1_gate.yaml](configs/wp1_gate.yaml) with one field overridden.

| Cell | Ablation | Config override | Expected effect |
|---|---|---|---|
| D | No-TC-DAMS | `league.meta_solver=nash` (already) → `league.tcdams_lambda=0` | Lower diversity |
| E | No-curriculum | `training.kill_radius_init=0.2`, `training.kill_rate_threshold=inf` | Slower / no convergence |
| F | No-CTDE | `league.team_critic_enabled=false` | Worse team coordination |
| G | No-exploiters | `league.mutation.enabled=false` | Less diverse pool |
| **C1-ablation** | **No-deployment-baseline** | `env.min_radar_baseline_m=0` | **Catastrophic — reproduces 0.5-bug** |
| **C1-sensing** | **Sensing ladder** | `sensing_noise.mode=fused/static/tracked` | Quantify each mode's contribution |

**Commands (one per ablation)**:

```bash
for cell in D:league.tcdams_lambda=0.0 \
            E:training.kill_radius_init=0.2 \
            F:league.team_critic_enabled=False \
            G:league.mutation.enabled=false \
            C1no:env.min_radar_baseline_m=0.0; do
    name="${cell%%:*}"
    override="${cell#*:}"
    nohup /home/ubuntu/miniconda3/envs/fluxphased/bin/python -m training.train \
      --config configs/wp1_gate.yaml \
      --override "$override" \
      > logs/wp5_${name}.log 2>&1 &
done
```

**PASS**: report `degradation% = (metric_wp1 - metric_ablation) / metric_wp1 × 100` per cell, with Cohen's d.

---

## WP6 — Statistical Discipline (no runs, protocol only)

Specified in [PAPER_PLAN_EAAI.md §6](PAPER_PLAN_EAAI.md). Key requirements:

- ≥5 seeds per main-comparison cell (old 3-seed plan was too thin)
- Report mean ± 95% CI (bootstrap, 1e4 resamples)
- Pairwise significance: Welch's t (or Mann-Whitney for non-normal)
- Effect size: Cohen's d
- Multiple comparison: Holm-Bonferroni correction
- Learning curves: mean + CI shaded band
- Sample efficiency: GPU-h to threshold AND AUC

**Validator** (TODO): `scripts/wp6_stats.py` ingests multi-seed logs and produces publication-ready tables.

---

## Execution Order (Recommended)

```
Day 1-3:    WP1 (running)     [60-150 GPU-h]
Day 2-3:    WP3.1 (DONE)      [0 GPU-h]      ← already complete
Day 4-7:    WP-D agent instructions + WP2 adapters (engineering, no GPU)
Day 8-10:   WP3.2 damage injection configs (CPU engineering)
Day 11-14:  WP5 ablations (5 cells × 3 seeds × ~50 GPU-h = ~750 GPU-h)
            [WP1 must PASS first]

After WP1+WP5:
Day 15-35:  WP2 main comparison (5 baselines × 5 seeds × ~40 GPU-h = ~1000 GPU-h)
Day 20-30:  WP4 generalization (eval-only, ~400 GPU-h)
Day 25-35:  WP3.2 damage scans (~300 GPU-h)

Total: ~2500-3500 GPU-h, ~10 weeks on PRO 6000 (or 4-5 weeks with cluster).
```

---

## Per-WP PASS Summary (one-line)

| WP | PASS criterion (one line) |
|---|---|
| WP1 | FluxLeague reaches kr≤0.5m with <10% residual 0.5 in 30 iters |
| WP2 | FluxLeague beats IPPO/MAPPO/QMIX/Classical on sample efficiency (p<0.05) |
| WP3.1 | achieved RMSE / CRLB ≤1.3× (DONE: CRLB derived, awaiting achieved-RMSE from WP1) |
| WP3.2 | ≥70% performance retention under each damage type |
| WP4 | <30% degradation on OOD test conditions |
| WP5 | Each ablation loses ≥10% on at least one metric (else that component is useless) |
| WP6 | All claims have CI + significance test attached |
