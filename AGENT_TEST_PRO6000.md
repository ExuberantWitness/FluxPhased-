# 🤖 AGENT TEST GUIDE — PRO 6000 (Blackwell 98GB)

**For an agent running ON the PRO 6000 machine.** Pull this repo, set up the env,
run the config, verify PASS criteria. Everything here is on branch `evo/laser-fix`.

---

## TL;DR

```bash
# env already set up? then just:
python -m training.train_laser --config configs/laser_25x25_pro6000.yaml 2>&1 | tee logs/pro6000_test.log
```
**PASS** = `kr` descends `50m → ~0.20m`, eval `aim=0m`, `cum red ≥ 0.80 / blue = 0.00`.

---

## 0. What this tests

Multi-agent **PSRO-lite league** (red=training vs blue=frozen opponent pool) for
laser precise-kill. Config `configs/laser_25x25_pro6000.yaml` = the **verified p14
baseline** (reached kr=0.20m on a 4090, num_envs=2) **scaled up for 98GB**
(`num_envs=12`, bigger buffers/pool, league on). The precision-critical params
(5km radar baseline + Kalman-tracked sensing + 6m residual aim + kill_radius
curriculum) are **unchanged from the verified run** — only scale knobs differ.

## 1. Environment setup (one-time)

Full details in [DEPLOY_PRO6000.md](DEPLOY_PRO6000.md). Minimum:
```bash
conda create -n fluxphased python=3.10 -y && conda activate fluxphased
pip install torch --index-url https://download.pytorch.org/whl/cu130   # CUDA 13, Blackwell
pip install warp-lang numpy pyyaml
# VERIFY sm_120 is supported (REQUIRED for Blackwell — else CUDA kernel errors):
python -c "import torch; assert 'sm_120' in torch.cuda.get_arch_list(), 'torch lacks sm_120!'; print('sm_120 OK', torch.cuda.get_device_name(0))"
```

## 2. Run

```bash
cd <repo root>
mkdir -p logs checkpoints
python -m training.train_laser --config configs/laser_25x25_pro6000.yaml 2>&1 | tee logs/pro6000_test.log
```
Runtime: ~24 PSRO iters. On a 4090 each iter was ~100–250s; Blackwell should be
faster. Expect roughly 30–60 min total.

## 3. PASS / FAIL criteria

Parse `logs/pro6000_test.log` for `[Eval @ iter N]` lines. **PASS if ALL hold:**

| Check | PASS condition | Where |
|---|---|---|
| Precision | `kr_next` reaches **≤ 0.24m** (ideally 0.20m) by the final iters | `[Eval]` `kr_next=` |
| Aim | `eval_min_aim_dist` = **0m** (sub-0.5m) for most mid/late iters | `[Eval]` `eval_min_aim_dist=` |
| Curriculum | `kr` descends monotonically 50→~0.2m (small back-offs OK) | `[PSRO]`/`[Eval]` `kr=` |
| Red dominance | `cum red ≥ 0.80` and **`cum blue = 0.00`** (red never loses) | `[Eval]` `cum red=.. blue=..` |
| No crash | reaches `Training complete.`; no `Traceback`/OOM | log tail |

**Verified reference trajectory** (from the 4090 league run — the PRO 6000 run
should match or beat this, with less late jitter thanks to num_envs=12):
```
iter:    1    3    5    7    9   11   13   15   17   19   20
kr:    50  24.5  12  5.88 2.88 1.41 0.69 0.34 0.41 0.20 0.20 m
red:  0.00 0.67 0.80 0.86 0.89 0.91 0.92 0.93 0.85 0.87 0.88   (blue=0.00 throughout)
```

## 4. VRAM tuning (use the 98GB)

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
```
Rough model: ~3GB fixed + ~6GB per env. `num_envs=12` ≈ 75GB. If headroom remains,
edit `configs/laser_25x25_pro6000.yaml` `env.num_envs` up to **16**. If OOM, drop to 8.
More envs = more parallel samples = better gradients (reduces the 0.20/0.24m end jitter).

## 5. Gotchas (READ)

1. **`sm_120` MUST be in `torch.cuda.get_arch_list()`** — else Blackwell kernels fail.
2. **`checkpoint_dir` must NOT be `/tmp`.** This config already uses
   `checkpoints/laser_pro6000` (persistent). Other configs (`p*`, `ew_*`, `local`)
   point at `/tmp/laser_run/...` which **filled the disk to 100% and crashed
   torch.save** on the 4090 — if you run those, override the dir first.
3. Episodes are short (500 pulses) by design — fast iteration, reaches 0.2m. The
   full 60s missile-combat scale is `configs/laser_25x25_config.yaml` (max_steps=600000).

## 6. Next: EW frontier (after baseline PASSes)

**`configs/laser_25x25_pro6000_ew.yaml`** — integrated-EW frontier, already scaled
for 98GB with a persistent `checkpoint_dir` (no /tmp). Smoke-validated; run like the baseline:
```bash
python -m training.train_laser --config configs/laser_25x25_pro6000_ew.yaml 2>&1 | tee logs/pro6000_ew.log
```
Adds: jamming degrades enemy localization (`jam_gain`), a kill-fast/survive race, and
the emission-exposure tradeoff (home-on-jam beacon → jamming must be **timed, not max**).
Expect `kr → ~0.2m` by ~iter17, then `jam>0` emerges in the tight-kr regime. 30 iters.
⚠️ **Frontier — not yet verified to convergence** (this is the run that previously crashed
on the /tmp disk-full bug, now with safe checkpointing).

> The original `laser_25x25_ew_race.yaml` / `laser_25x25_ew_exposure.yaml` still point
> their `checkpoint_dir` at `/tmp/laser_run/...` — prefer the ready-made `pro6000_ew`
> config above, or override their dir off /tmp first.

Report back: the full `[Eval]` trajectory + final `kr` + `cum red/blue/draw` + peak VRAM.
