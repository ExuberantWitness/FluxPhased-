# Phase 1.5 MAPPO Baseline — Experiment Report

> **One-line summary**: Under matched env / reward / curriculum / seed, MAPPO (team critic + uniform opponent sampling) **outperforms** PfspFix (per-agent critic + PFSP f_hard priority sampling) on every headline metric — challenging the original "PFSP league is the core contribution" framing.

> **Status**: Complete. PASS on all 5 pre-registered gates.
> **Commit**: `5c24f0d` (push `911c5ef..5c24f0d` to `origin/main`, 2026-07-01)
> **Wall-clock**: 3.82h MAPPO + 3.37h PfspFix baseline ≈ 7.2h GPU-h total

---

## 1. Motivation & Hypothesis

### 1.1 Why this experiment

The EAAI Q1 paper draft positions the **AlphaStar-style PFSP league** as the central methodological contribution, with IPPO and MAPPO relegated to "baselines we beat". A reviewer at the desk-reject stage flagged this claim as **unsupported** because no baseline numbers existed.

Phase 1.5 closes that gap with the minimum viable comparison: a single seed of MAPPO under matched conditions. IPPO and a 5-seed statistical study are deferred to Phase 3.

### 1.2 Pre-registered hypothesis

> Under identical env / reward / curriculum / seed, **PfspFix ≥ MAPPO** on `cum_red` and `eval_kill_rate` at iter 20. If MAPPO matches or exceeds PfspFix, the league framework's value must be re-examined before Phase 2 (3-role exploiter) work begins.

This was written into `/home/ubuntu/.claude/plans/snuggly-exploring-parrot.md` §1.5.4 before any MAPPO code was run.

### 1.3 Pre-registered gate thresholds

| Metric | PASS | FAIL | Rationale |
|---|---|---|---|
| `cum_red` @ iter 20 | ≥ 0.75 | < 0.60 | Self-play win share; 0.88-0.90 validated on 4090 |
| `eval_kill_rate` @ iter 20 | ≥ 0.50 | < 0.30 | Deterministic eval, 12 episodes vs current pool |
| `aim_res` (m) | ≤ 0.10 | > 0.30 | Sub-meter precision is the kill-enabling signal |
| `adv_std` (PPO health) | < 50 | ≥ 100 | Above 100 = value function diverging |
| `cmd_pl` (no collapse) | \|x\| > 1e-4 | \|x\| < 1e-5 | policy_loss → 0 = PPO not learning |

---

## 2. Experimental Setup

### 2.1 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (98 GB VRAM) |
| Driver | 595.71.05 |
| CUDA | 13.2 |
| OS | Linux 6.17.0-35-generic (x86_64), glibc 2.39 |

### 2.2 Software

| Package | Version |
|---|---|
| python | 3.10.20 |
| torch | 2.12.0+cu132 |
| numpy | 1.24.4 |
| cudnn | 92000 (set to `deterministic=True` via `set_global_seed`) |
| warp | 1.10.1 |

### 2.3 Task configuration (identical for both runs)

| Field | Value | Note |
|---|---|---|
| Grid | 25 × 25 cells, 20000 × 20000 m map | S-300/S-400 footprint scale |
| Agents | 4 radars × 2 teams = 8 total | red = training, blue = opponent pool |
| Episode | 500 pulses max = 50 ms simulated | proven length for kr 0.2m convergence |
| Sensing | tracked mode, q=0.02, Kalman burn-in 120 | realistic S-300 crossrange noise |
| Deployment baseline | 5000 m | key to anchor stability (per `ANCHOR_DIAGNOSIS.md`) |
| Kill radius (initial) | 50.0 m | curriculum start |
| Kill radius (final) | 0.5 m | curriculum floor (avoids fine-kr degradation zone) |
| Pulse rate | PRF 10000 Hz, 4 pulses/CPI, 5 pulses/control | |
| Bandwidth | 200 MHz | range resolution σ=0.05 m |

### 2.4 Training hyperparameters (identical for both runs)

| Section | Field | Value |
|---|---|---|
| **PPO shared** | γ | 0.999 |
| | GAE λ | 0.99 |
| | n_epochs | 3 |
| | batch_size | 256 |
| | buffer_size | 2048 (cmd + radar each) |
| | value_coef | 1.0 |
| | vf_clip_range | 10.0 |
| | max_grad_norm | 0.5 |
| **Commander** | lr | 3.0e-4 |
| | clip_range | 0.2 |
| | entropy_coef | 0.01 |
| **Radar** | lr | 1.0e-4 |
| | clip_range | 0.1 |
| | entropy_coef | 0.02 |
| **League** | population_cap | 30 |
| | episodes_per_training | 10 |
| | league_snapshot_every | 3 |
| | n_eval_games | 10 |
| **Reward shaping** | kill_bonus | 100 |
| | death_penalty | -10 |
| | illum_reward_weight | 50 |
| | beam_accuracy_weight | 5 |
| | (full list in `experiments/*/config.yaml`) | |

### 2.5 Algorithmic differences (the only thing that changed)

| Component | PfspFix (Phase 1) | MAPPO (Phase 1.5) |
|---|---|---|
| Critic architecture | per-agent value head (76-dim obs) | **team critic** (CTDE, 104-dim team state) |
| Opponent sampling | **PFSP f_hard**: `w_i = (winrate_i + 0.1)^2` | **uniform**: `pfsp_p = 0` |
| Pool win-rate update | EMA η=0.25 after each eval | identical |
| Snapshot cadence | every 3 iters | identical |
| `use_mappo` config flag | `false` | **`true`** |
| `league.pfsp_p` config flag | (default = 2) | **`0`** |
| Checkpoint directory | `checkpoints/laser_pro6000/` | `checkpoints/laser_mappo/` (隔离防覆盖) |

**Everything else is bit-exact identical**: same env, same reward, same curriculum, same seed (42), same BC weight schedule (5.0 → 2.0 over 24 iters), same log_std decay (-1.0 init, -4.0 floor, 0.40 decay).

### 2.6 Seed handling (bit-exact reproducibility)

```python
# training/train_laser.py:24-36 (commit 911c5ef)
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Verified bit-exact by re-running PfspFix and diffing per-iter outputs: algorithmic values (kills, kr, cmd_pl, adv_std, aim_res) **identical through iter 13**, only wall-clock seconds differ (jitter ±1%).

Cost: ~10-20% GPU throughput (cudnn deterministic mode).

### 2.7 Evaluation protocol

- **Deterministic policy** (no exploration noise) for the red commander.
- 12 episodes per PSRO iter, vs the most-recently-snapshotted opponent in the pool.
- Metrics recorded: `eval_kills`, `eval_kill_rate`, `cum_red/blue/draw` (running pool-wide), `wr_opp` (EMA win-rate vs chosen opponent).

---

## 3. Results

### 3.1 Headline comparison (iter 20, single seed)

| Metric | PfspFix | MAPPO | Δ | Gate |
|---|---|---|---|---|
| `cum_red` (self-play) | 0.810 | **0.970** | +0.160 | PASS ✅ |
| `eval_kill_rate` | 0.667 | **0.875** | +0.208 | PASS ✅ |
| `aim_res` (m) | 0.034 | **0.032** | -0.002 | PASS ✅ |
| `adv_std` (last) | 9.49 | 18.71 | +9.22 | PASS (both < 50) ✅ |
| `cmd_pl` (last) | -0.0044 | -0.0103 | larger \|x\| | PASS (no collapse) ✅ |
| `cum_blue` @ iter 20 | 0.190 | **0.000** | -0.190 | (no gate; lower better) |

**Overall verdict**: PASS ✅ on all 5 pre-registered gates.

### 3.2 Per-iteration trajectory (MAPPO)

| iter | kr_train (m) | kr_eval_next | eval_kr | cum_red | aim_res (m) | adv_std | cmd_pl |
|---|---|---|---|---|---|---|---|
| 1 | 50.00 | 35.00 | 0.500 | 0.920 | 0.412 | 19.52 | -0.00182 |
| 2 | 35.00 | 35.00 | 0.333 | 0.790 | 0.427 | 15.16 | -0.00599 |
| 3 | 35.00 | 24.50 | 0.542 | 0.860 | 0.297 | 11.98 | -0.00282 |
| 4 | 24.50 | 17.15 | 0.708 | 0.830 | 0.201 | 8.29 | -0.00304 |
| 5 | 17.15 | 12.00 | 0.542 | 0.870 | 0.140 | 8.14 | -0.00606 |
| 6 | 12.00 | 8.40 | 0.500 | 0.890 | 0.094 | 12.85 | -0.00386 |
| 7 | 8.40 | 5.88 | 0.542 | 0.900 | 0.068 | 12.46 | -0.00609 |
| 8 | 5.88 | 4.12 | 0.792 | 0.920 | 0.049 | 7.49 | -0.00414 |
| 9 | 4.12 | 2.88 | 0.667 | 0.930 | 0.037 | 9.37 | -0.00900 |
| 10 | 2.88 | 2.02 | 0.792 | 0.930 | 0.034 | 9.15 | -0.00777 |
| 11 | 2.02 | 1.41 | 0.500 | 0.940 | 0.038 | 10.80 | -0.00978 |
| 12 | 1.41 | 0.99 | **1.000** | 0.940 | 0.035 | 8.52 | -0.00826 |
| 13 | 0.99 | 0.69 | **1.000** | 0.950 | 0.033 | 11.61 | +0.00209 |
| 14 | 0.69 | 0.50 | 0.667 | 0.950 | 0.034 | 6.20 | -0.01098 |
| 15 | 0.50 | 0.50 | 0.500 | 0.960 | 0.032 | 9.16 | -0.00666 |
| 16 | 0.50 | 0.50 | 0.667 | 0.960 | 0.032 | 9.07 | -0.00596 |
| 17 | 0.50 | 0.50 | 0.625 | 0.960 | 0.030 | 8.39 | -0.00648 |
| 18 | 0.50 | 0.50 | 0.958 | 0.960 | 0.030 | 10.35 | -0.00353 |
| 19 | 0.50 | 0.50 | 0.875 | 0.960 | 0.033 | 6.23 | -0.00559 |
| 20 | 0.50 | 0.50 | 0.875 | **0.970** | 0.032 | 18.71 | -0.01029 |

> Per-iter curves for PfspFix + MAPPO overlays are in [experiments/phase1.5_mappo_seed42/figures/](experiments/phase1.5_mappo_seed42/figures/).

### 3.3 Three-phase behavior

| Phase | iter | avg eval_kr | avg cum_red | avg aim_res (m) | What's happening |
|---|---|---|---|---|---|
| **Exploration** | 1-5 | 0.525 | 0.854 | 0.295 | Policy learns to aim under loose kr |
| **Annealing** | 6-13 | 0.724 | 0.925 | 0.049 | kr tightens 12m→0.7m; aim enters sub-meter |
| **Tight-kr plateau** | 14-20 | 0.738 | 0.960 | 0.032 | 0.5m floor; pool diversifies, blue→0 |

### 3.4 PPO health indicators

- **`adv_std`**: range 6.20–19.52, mean ≈ 10. No explosion (>50), no collapse (<1e-3). Healthy.
- **`cmd_pl`**: range -0.011 to +0.002, mean ≈ -0.006. **Never collapsed to 0** — this is the failure mode that killed R0-R5 (commit history 0623-0625). Avoided here.
- **NaN / crashes**: 0 across all 20 iters.
- **`log_std`**: decayed from -1.0 → -4.0 (floor reached at iter 9, stayed there).

### 3.5 Bit-exact backward compatibility

After adding the `pfsp_p` config knob to `training/train_laser.py`, the default (`pfsp_p=2`) was verified to produce **bit-identical** output to the original `** 2` literal:

- Re-ran PfspFix (laser_25x25_pro6000_stable.yaml) → `logs/phase1_pfsp_recheck.log`
- Diff vs `logs/phase1_seed42_run1.log` (the seed=42 reference)
- **Result**: iter 1-13 algorithmic values character-identical; only `time=NNNs` differs (jitter ±1% from GPU scheduling).
- Critical implementation detail: `int()` cast on `pfsp_p` preserves the numpy integer-power code path (`** 2` ≠ `** 2.0` at the lowest bits).

---

## 4. Discussion

### 4.1 Why did MAPPO win? (Three candidate explanations)

| Explanation | Evidence | Plausibility |
|---|---|---|
| **(a) Task too small for league to help** | 4 radars × 2 teams, pool only reaches 7 by iter 20; PFSP prioritization has little to prioritize | **High** |
| **(b) PfspFix implementation immature** | cum_red 0.81 is below the 0.88-0.90 4090-validated range; suggests one or more bugs | **Medium** |
| **(c) Team critic genuinely better for this task** | CTDE sees full team state (104-dim), can coordinate 4 radars better than 4 independent value heads | **Medium** |

(a) and (c) are not mutually exclusive. Phase 2 (3-role exploiter) would test (a) by adding *structural* diversity that simple self-play can't generate. Phase 3 (≥5 seeds) would test (b) by separating signal from seed variance.

### 4.2 Implications for paper narrative

The original EAAI draft positions PFSP league as the headline contribution. **This single result does not support that framing** at the current scale. Three pivots are available:

1. **Honest-finding framing**: "At 4-radar scale, MAPPO matches or exceeds league; the league framework's value emerges at larger scale / 3-role exploiters" (requires Phase 2 + scaling results to substantiate).
2. **Engineering-insight framing**: Reframe the contribution as the **recipe** (reward shaping + curriculum + anchor fix + bit-exact reproducibility), with league vs MAPPO as a methodological ablation.
3. **Methodological-comparison framing**: Position the paper as a careful empirical study of multi-agent RL approaches on the radar precise-kill task (less novel, lower EAAI fit but defensible).

The user has not yet chosen a pivot. This decision gates Phase 2.

### 4.3 Threats to validity

| Threat | Severity | Mitigation in this work | Open |
|---|---|---|---|
| **Single seed** | High | Pre-registered gate thresholds; bit-exact verified | Need ≥5 seeds in Phase 3 |
| **Pool too small** | High | Capped at 30 (only reached 7 by iter 20) | Try larger pool / longer training |
| **No IPPO** | Medium | Deferred to Phase 1.5b | Run IPPO baseline (`use_mappo=false, pfsp_p=0`) |
| **Optimal hyperparams unknown** | Medium | Used PfspFix-verified values for both; MAPPO may benefit more from tuning | Hyperparam sweep out of scope |
| **Eval vs current pool only** | Low | This is the standard AlphaStar protocol | Could add held-out evaluation vs final pool |
| **2 teams, 4 radars — small scale** | Medium | Acknowledge in paper | Scale to 6+ radars in Phase 3 |

---

## 5. Reproduction

### 5.1 Quick start (5 minutes)

```bash
# Clone and checkout the exact commit
git clone https://github.com/ExuberantWitness/FluxPhased-.git
cd FluxPhased-
git checkout 5c24f0d  # Phase 1.5 MAPPO baseline + APP archive

# Activate environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fluxphased

# Verify environment matches the report
python -c "import torch; print(torch.__version__)"  # expect 2.12.0+cu132

# Inspect the artifacts
cat experiments/AGENTS.md                       # top-level index
cat experiments/phase1.5_mappo_seed42/AGENTS.md  # MAPPO run summary
cat experiments/phase1.5_mappo_seed42/comparison.md  # auto-generated PASS/FAIL table
```

### 5.2 Reproduce the MAPPO run from scratch (~3.8h on RTX PRO 6000)

```bash
# 1. Launch training (unbuffered stdout, faulthandler enabled)
bash scripts/run_train.sh configs/laser_25x25_mappo.yaml logs/phase1.5_mappo.log

# 2. Watch progress
tail -f logs/phase1.5_mappo.log

# 3. (Optional) Stall check if log stops updating > 15min
/home/ubuntu/miniconda3/envs/fluxphased/bin/py-spy dump --pid $(cat /tmp/train_laser.pid)
#    py-spy is NON-invasive; NEVER use kill -USR1 (terminates by default).

# 4. After training completes (~3.8h), generate APP artifacts
RUN_DIR=experiments/phase1.5_mappo_seed42
cp logs/phase1.5_mappo.log $RUN_DIR/train.log

python scripts/experiments/parse_log_to_metrics.py \
    --log $RUN_DIR/train.log --out $RUN_DIR/metrics.json --run-id phase1.5_mappo_seed42

python scripts/experiments/make_figures.py \
    --run MAPPO:$RUN_DIR/metrics.json --out-dir $RUN_DIR/figures

python scripts/experiments/compare_runs.py \
    --baseline experiments/phase1_pfsp_seed42/metrics.json \
    --candidate $RUN_DIR/metrics.json \
    --baseline-name PfspFix --candidate-name MAPPO \
    --out $RUN_DIR/comparison.md

python scripts/experiments/write_metadata.py \
    --run-dir $RUN_DIR --config configs/laser_25x25_mappo.yaml --seed 42 \
    --start "2026-07-01T11:42:00+08:00" --end "$(date -Iseconds)" \
    --reproduce "bash scripts/run_train.sh configs/laser_25x25_mappo.yaml logs/phase1.5_mappo.log"

python scripts/experiments/write_agents_md.py --run-dir $RUN_DIR \
    --baseline-dir experiments/phase1_pfsp_seed42
```

### 5.3 One-command finalizer (handles all of 5.2 step 4 automatically)

```bash
bash scripts/experiments/monitor_run.sh \
    experiments/phase1.5_mappo_seed42 \
    logs/phase1.5_mappo.log \
    configs/laser_25x25_mappo.yaml
# Exit 0  = still running, refreshed
# Exit 99 = training done, all artifacts up to date, ready to commit
```

### 5.4 Reproduce PfspFix baseline (~3.4h)

```bash
bash scripts/run_train.sh configs/laser_25x25_pro6000_stable.yaml logs/phase1_seed42_run1.log
# Then run monitor_run.sh with experiments/phase1_pfsp_seed42/ as RUN_DIR.
```

### 5.5 Add a new comparison run (e.g., IPPO, or a second seed)

```bash
# 1. Create a config that differs only in the dimensions you're studying
cp configs/laser_25x25_mappo.yaml configs/laser_25x25_ippo.yaml
# Edit: use_mappo: false, pfsp_p: 0, checkpoint_dir: checkpoints/laser_ippo

# 2. Create a new run directory from the same template
mkdir -p experiments/phase1.5_ippo_seed42
cp configs/laser_25x25_ippo.yaml experiments/phase1.5_ippo_seed42/config.yaml

# 3. Run + finalize (same as §5.2 / §5.3)
bash scripts/run_train.sh configs/laser_25x25_ippo.yaml logs/phase1.5_ippo.log
bash scripts/experiments/monitor_run.sh experiments/phase1.5_ippo_seed42 \
    logs/phase1.5_ippo.log configs/laser_25x25_ippo.yaml

# 4. Regenerate the top-level comparison index
python scripts/experiments/write_agents_md.py --index
```

### 5.6 Reproduce figures only (no training, ~10s)

```bash
# Overlay both runs on every curve
python scripts/experiments/make_figures.py \
    --run PfspFix:experiments/phase1_pfsp_seed42/metrics.json \
    --run MAPPO:experiments/phase1.5_mappo_seed42/metrics.json \
    --out-dir experiments/comparison_figures
```

---

## 6. APP Artifact Inventory

The `experiments/` directory follows the [Agentic Publication Protocol](https://arxiv.org/abs/2606.27386) (Lu & Qi) — each run is self-contained, machine-readable, and reproducible.

```
experiments/
├── AGENTS.md                                  # Top-level index, 60-second read
│
├── phase1_pfsp_seed42/                        # PfspFix baseline (commit 911c5ef)
│   ├── AGENTS.md                              # Per-run summary + reproduction
│   ├── config.yaml                            # Frozen config snapshot
│   ├── metadata.json                          # seed, commit SHA, GPU, wall-clock
│   ├── metrics.json                           # Per-iter structured metrics
│   ├── train.log                              # Raw stdout
│   ├── reproduce.sh                           # One-command re-run
│   └── figures/
│       ├── kr_curve.png                       # kr trajectory (log scale)
│       ├── cum_red_curve.png                  # red vs blue win share
│       ├── aim_res_curve.png                  # aim residual (precision)
│       ├── adv_std_curve.png                  # PPO health (collapse watch)
│       ├── cmd_pl_curve.png                   # policy_loss (collapse watch)
│       └── eval_kill_rate_curve.png           # deterministic eval performance
│
└── phase1.5_mappo_seed42/                     # MAPPO baseline (commit 5c24f0d)
    ├── AGENTS.md
    ├── config.yaml
    ├── metadata.json
    ├── metrics.json
    ├── train.log
    ├── reproduce.sh
    ├── comparison.md                          # AUTO-GENERATED PASS/FAIL table vs PfspFix
    └── figures/                               # MAPPO + PfspFix overlay
        └── (same 6 PNGs)

scripts/experiments/                           # Reusable tooling
├── parse_log_to_metrics.py                    # log → metrics.json
├── make_figures.py                            # metrics → PNG curves
├── compare_runs.py                            # two metrics → comparison.md
├── write_metadata.py                          # seed/commit/host → metadata.json
├── write_agents_md.py                         # per-run + index AGENTS.md
├── write_reproduce_sh.py                      # one-command reproduce.sh
└── monitor_run.sh                             # live monitor + finalizer
```

### What each file gives a future agent

- **`AGENTS.md`**: 60-second orientation — what was this, did it pass, how to reproduce.
- **`config.yaml`** (snapshot, not symlink): survives edits to the live yaml in `configs/`.
- **`metadata.json`**: machine-readable record of seed, commit, branch, dirty flag, GPU, wall-clock, exact reproduce command.
- **`metrics.json`**: per-iter dict, parseable without log scraping.
- **`comparison.md`**: side-by-side headline numbers + per-iter delta, with PASS/MARGINAL/FAIL verdict.
- **`reproduce.sh`**: single shell command that re-runs the experiment, then regenerates all artifacts.

---

## 7. Next steps

Decision required before proceeding:

1. **Run IPPO baseline** (`use_mappo=false, pfsp_p=0`) to complete the league/IPPO/MAPPO 3-way comparison. ~3.5h.
2. **Run ≥3 seeds of MAPPO and PfspFix** to estimate seed variance before claiming MAPPO > PfspFix. ~21h.
3. **Start Phase 2** (3-role AlphaStar league). Plan §2. The risk is that if MAPPO > league at 4-radar scale, the league framework needs scale or role diversity to demonstrate value.
4. **Investigate PfspFix underperformance** — cum_red 0.81 is below the 0.88-0.90 validated range; suggests bug or suboptimal config.

Recommended sequence: (1) + (4) in parallel (single-day each), then (2) overnight, then decide on (3).

---

## 8. References

- **Agentic Publication Protocol**: Lu & Qi, arXiv:2606.27386, 2026. https://arxiv.org/abs/2606.27386
- **AlphaStar PFSP / league design**: Vinyals et al., Nature 2019. (Adapted here as `pfsp_p=2` f_hard sampling + frozen opponent pool.)
- **MAPPO**: Yu et al., NeurIPS 2022. (CTDE team critic; `TeamCritic` class in `training/ppo/actor_critic.py:1075`.)
- **Phase 1 PfspFix baseline**: `commit 911c5ef`, documented in `experiments/phase1_pfsp_seed42/AGENTS.md`.
- **Pre-registered plan**: `/home/ubuntu/.claude/plans/snuggly-exploring-parrot.md` §1.5 (written 2026-07-01 before any MAPPO code was run).

---

## 9. Changelog

| Date | Change |
|---|---|
| 2026-07-01 | Initial Phase 1.5 report (commit `5c24f0d`). Single seed, MAPPO PASS on 5/5 gates. |
