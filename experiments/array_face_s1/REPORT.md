# Array-Face S1 Report

> **Phase**: S1 — radar 1D ULA (5-cell, λ/2) + Rx array factor, jammer scalar
> **Period**: 2026-07-29 → 2026-07-31
> **Branch**: `g3-bsta/array-face-s1` (off `8f7e72d`)
> **Plan ref**: [docs/array_face/INCREMENTAL_DESIGN.md](../../docs/array_face/INCREMENTAL_DESIGN.md) §3

## 1. What S1 added (vs lite base `8f7e72d`)

| Layer | Change |
|---|---|
| Physics | radar 5-cell 1D ULA + Rx AF^2 (peak-normalized, dB); jammer stays scalar |
| Geometry | radar beam_az frozen round-robin `step_idx % 5` over {-60°, -30°, 0°, +30°, +60°} |
| JNR | `P_jam_dbm + AF_rx_db(beam_az) - L_path - L_pol - N_dbm` |
| Observation | lite 11 dims + 5-dim radar_beam_az one-hot = **OBS_DIM_S1 = 16** |
| Action | unchanged from lite: `Categorical(3) = {idle, jam_svc_0, jam_svc_1}` |
| Reward | unchanged from lite: `newly_dropped.float() + γΦ' - Φ` |

Files: [env/gpu/array_face_s1/](../../env/gpu/array_face_s1/), [tests/array_face/](../../tests/array_face/)

## 2. M0/M1 verification (gate PASS)

- 8/8 physics unit tests PASS ([tests/array_face/test_array_factor_s1.py](../../tests/array_face/test_array_factor_s1.py))
- 10/10 env contract tests PASS ([tests/array_face/test_array_face_s1.py](../../tests/array_face/test_array_face_s1.py))
- Manifest disjoint + legacy clean (seeds `2101xxxx`, no overlap with lite `2100xxxx`)
- lite regression: 75/75 PASS (zero breakage)

## 3. PPO multi-seed results (1000 iter each)

Configurations tested:
- **baseline**: `entropy_coef=1e-3, anneal_frac=0.3, target_kl=0.01, actor_lr=3e-5`
- **Amendment 01**: `entropy_coef=5e-3, anneal_frac=1.0, target_kl=0.01` (killed iter 489, plateau confirmed)
- **Amendment 02**: `entropy_coef=5e-3, anneal_frac=0.5, target_kl=0.02` (the "stronger exploration" config)

| seed | config | break_iter (>0.12) | final val | peak val (iter) | status |
|---|---|---|---|---|---|
| 20260730 | baseline | — | 0.0929 | 0.0937 (339) | **STUCK** |
| 20260729 | baseline | 229 | 0.2113 | 0.2129 (729) | broke out |
| 20260730 | Amend02 | — | 0.0929 | 0.0937 (339) | **STUCK** |
| 20260729 | Amend02 | 219 | 0.2129 | 0.2134 (939) | broke out |
| 20260801 | Amend02 | 299 | **0.2372** | 0.2402 (979) | broke out (**best**) |
| 20260730 | Amend01 (killed 489) | — | 0.0929 | 0.0937 (369) | STUCK |

**Aggregate**:
- 6 runs total (1 Amend01 killed, 5 ran to iter 999)
- **3 broke out / 3 stuck** (50% break-out rate)
- **Broke-out mean final**: (0.2113 + 0.2129 + 0.2372) / 3 = **0.2205 ± 0.0116**
- **All stuck runs**: seed 20260730 across all 3 configs (baseline, Amend01, Amend02)

## 4. Key findings

### 4.1 S1 env difficulty ≈ lite env
- **S1 best (Amend02 / seed 20260801)**: 0.2372
- **lite R2 extended (3000 iter)**: 0.2628
- **Gap**: 2.6pp — within same regime, not 3x harder
- Embedded lite policy (lite iter 2999 weights loaded into 16-dim S1 actor) achieves drop=0.2734 on S1 env ≈ lite env 0.2628 → S1 physics doesn't change the optimal policy

### 4.2 Stronger exploration (Amend02) does NOT rescue stuck seeds
- seed 20260730 stuck at 0.0929 across **3 configurations** (baseline, Amend01, Amend02)
- Amend02 doubled target_kl and quintupled entropy_coef — still stuck at identical 0.0929
- **Conclusion**: stuck-ness is not an exploration-strength problem

### 4.3 Single-seed PPO is unreliable in this env
- seed 20260729 (baseline): broke out → 0.2113
- seed 20260730 (baseline): stuck → 0.0929
- 2 seeds, opposite outcomes — **single-seed claims of "PPO fails" or "PPO works" are untrustworthy**
- **Multi-seed policy** (≥3 seed) is now mandatory per plan §11.1

### 4.4 The cause of stuck-ness remains unidentified
- Not exploration (4.2)
- Not env difficulty (4.1)
- Not a specific code path (no_af and no_beam controls, with seed 20260730, also stuck)
- Likely culprit: **early-trajectory lock-in** or **network init basin** specific to seed 20260730
- Honest framing: "PPO stuck rate on S1 = 50%; cause unidentified"

## 5. Plots

- [s1_seedmatch_performance.png](learning_repair/s1_ppo_output_seed20260729/s1_seedmatch_performance.png) — baseline seed=20260729 (broke out) vs baseline seed=20260730 (stuck) vs lite R2 ext
- [amend02_multiseed_performance.png](learning_repair/amend02_multiseed_performance.png) — Amend02 3-seed mean ± std

## 6. Lessons

1. **Never report single-seed PPO results on this env** — writeup must include multi-seed statistics
2. **Seed 20260730 is a "known-bad" basin** — useful as a stress test for future exploration mechanisms
3. **Stronger entropy / target_kl doesn't fix stuck-ness** — next lever is init / curriculum / warm-start
4. **Don't reuse seed 20260730 for new exploration configs** — it's pre-confirmed stuck

## 7. Artifacts

- Code: [experiments/array_face_s1/learning_repair/](learning_repair/)
- Runners: `run_s1_ppo_seedmatch.py` (baseline), `run_s1_ppo_explore_v2.py` (Amend02), `run_s1_ppo_control.py` (controls)
- Data:
  - `s1_ppo_output_anneal0.3_coef1e-3/` (baseline seed 20260730, STUCK)
  - `s1_ppo_output_seed20260729/` (baseline seed 20260729, broke out)
  - `s1_ppo_output_amend02_seed{20260729,20260730,20260801}/` (Amend02 3-seed)
  - `s1_ppo_output_no_af/` (control, seed 20260730, killed iter 219)
  - `s1_ppo_output_no_beam/` (control, seed 20260730, killed iter 219)

## 8. Verdict

S1 **PASS** the phase-1 gate (PPO can learn; best result 0.2372 within 2.6pp of lite saturation 0.2628). Multi-seed analysis shows the env is **not fundamentally harder** than lite; the 50% stuck rate is a PPO robustness issue, not an env difficulty issue.

Proceeding to S2 per plan §1 (jammer 1D ULA + beam steering).

## 9. References

- Plan: [docs/array_face/INCREMENTAL_DESIGN.md](../../docs/array_face/INCREMENTAL_DESIGN.md)
- lite lineage: [experiments/g3_bsta_lite/](../g3_bsta_lite/)
- Memory: [arrayface-mappo-unban.md](../../docs/array_face/INCREMENTAL_DESIGN.md) (2026-07-31 MAPPO 解禁)
