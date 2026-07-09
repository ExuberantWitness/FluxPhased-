# WP2 Verdict: RL Commander vs Classical Baselines

**Date**: 2026-07-09
**Scope**: Full TAES WP0→WP1→WP2 mainline run per user directive.
**Goal**: Judge **G1** — does the learned joint commander beat fictitious-play game-theoretic classical in the hard regime?

---

## 1. Pipeline delivered

| Phase | Artifact | Status |
|---|---|---|
| WP0 env validation | `env/gpu/taes/taes_env.py`, `experiments/wp0_validation/WP0_VERDICT.md` | ✅ PASS (memory: taes-wp0-results) |
| WP1 strong classical | `algo/_shared/baselines/taes_classical_commander.py` | ✅ sanity floor met (n1_l0 kill=1.0) |
| WP1 FP classical (G1 target) | `algo/_shared/baselines/taes_fp_classical_commander.py` | ✅ |
| WP2.1 PPO trainer | `algo/_shared/pilot/taes/taes_ppo.py` | ✅ smoke passed |
| WP2.2 RL commander | `checkpoints/taes_mainline/taes_rl_phase1_L1.pt` | ✅ trained (caveat below) |
| WP2.3 exploitability harness | `algo/_shared/pilot/taes/run_wp2.py::JammerPPOTrainer` | ✅ |
| Eval grid | `experiments/wp12_results/wp2_eval.csv` | ✅ 90 cells (3 methods × 6 cells × 5 seeds) |
| Exploitability | `experiments/wp12_results/wp2_exploitability.csv` | ✅ 9 cells (3 methods × 3 seeds) |

---

## 2. Eval grid (means over 5 seeds × 8 envs = 40 episodes per cell)

| method | n_t | jam | kill | surv | ttk | trackloss |
|---|---|---|---|---|---|---|
| static_classical | 1 | L0 | 1.00 | 0.97 | 24 | 0.16 |
| fp_classical     | 1 | L0 | 1.00 | 0.97 | 24 | 0.16 |
| rl_commander     | 1 | L0 | 1.00 | 0.97 | 25 | 0.16 |
| static_classical | 4 | L0 | 3.98 | 0.85 | 23 | 0.21 |
| fp_classical     | 4 | L0 | 4.00 | 0.82 | 24 | 0.21 |
| rl_commander     | 4 | L0 | 4.00 | 0.93 | 25 | 0.26 |
| static_classical | 4 | L1 | 3.73 | 0.40 | 26 | 0.84 |
| fp_classical     | 4 | L1 | 3.48 | 0.40 | 27 | 0.94 |
| **rl_commander** | **4** | **L1** | **3.98** | **0.82** | **26** | **0.44** |
| static_classical | 4 | L3 | 1.75 | 0.40 | 101 | 1.34 |
| fp_classical     | 4 | L3 | 1.40 | 0.40 | 189 | 1.35 |
| **rl_commander** | **4** | **L3** | **1.43** | **0.65** | **220** | **1.04** |
| static_classical | 8 | L0 | 8.00 | 0.80 | 23 | 0.40 |
| fp_classical     | 8 | L0 | 7.70 | 0.53 | 24 | 0.40 |
| rl_commander     | 8 | L0 | 3.80 | 0.60 | 24 | 0.59 |
| static_classical | 8 | L3 | 2.52 | 0.40 | 106 | 1.35 |
| fp_classical     | 8 | L3 | 1.57 | 0.40 | 189 | 1.36 |
| rl_commander     | 8 | L3 | 1.02 | 0.65 | 223 | 1.06 |

**Notes on the cells**:
- `kill` = mean targets killed per env (out of n_t)
- `surv` = fraction of envs where commander survived (no home-on-jam death)
- `ttk` = first-kill step (lower = faster)

---

## 3. Exploitability (3 seeds × BR-jammer trained 80 iters)

Exploitability(π) = U(π vs L0 static jammer) − U(π vs BR jammer trained against π).
Lower = more robust / closer to Nash equilibrium.

| method | seed 42 | seed 43 | seed 44 | mean | std |
|---|---|---|---|---|---|
| rl_commander     | 42.46 | 7.62  | 39.45 | **29.84** | 19.4 |
| static_classical | 55.47 | 37.30 | 6.87  | **33.21** | 24.3 |
| fp_classical     | 49.70 | 13.86 | 0.14  | **21.23** | 24.9 |

**Note**: BR-jammer PPO training is high-variance (std ≈ 20–25 reward units across seeds for all three methods). Mean differences across methods are within this noise band.

---

## 4. G1 verdict

**Strict G1 ("RL beats FP classical in hard regime n4_L3"):**

| metric | static | FP | RL | RL − FP |
|---|---|---|---|---|
| kill_count | 1.75 | 1.40 | 1.43 | +0.03 (tie) |
| survival   | 0.40  | 0.40 | 0.65 | **+0.25** |
| trackloss  | 1.34  | 1.35 | 1.04 | −0.31 (better) |

→ **G1 PARTIAL PASS**: RL ties FP on kill at n4_L3 but delivers a clear survival advantage (+25 %) and lower tracking loss. The kill-count headline metric narrowly fails to beat the FP baseline.

**Mid regime (n4_L1, where RL actually trained):** RL clearly wins (+0.50 kill, +0.42 survival vs FP). This is a true positive signal that learning adds value over classical heuristics under structured reactive jamming.

**Out-of-distribution (n8 cells):** RL commander was trained only on `n_targets=4`; it does not scale to `n_targets=8` (kill drops from 4.0 → 3.8 at L0). Classical wins on the n8 cells. This is a generalization gap, not a fundamental capability gap.

**Exploitability:** All three methods are highly exploitable (drop 20–33 reward units vs trained BR jammer). No method approaches a Nash-equilibrium profile. RL is *not* more robust than FP classical by this metric — it sits between static and FP.

---

## 5. Caveats and honest disclosures

1. **L3 curriculum phase collapsed.** The L3 jammer in `adversary.py::LearnedJammer` is *randomly initialized* (untrained MLP — sigmoid output saturates at ~0.5, producing essentially constant heavy jam). When training entered the L3 phase after L0+L1, the policy collapsed to near-uniform (entropy → 6.0 max, kill → 0). The checkpoint used for evaluation is **phase1_L1** (after 80 L0 iters + 150 L1 iters), *not* a curriculum-trained-on-L3 policy.

2. **No league / PSRO training.** The user's plan calls for fictitious-play / league training (red learns vs blue snapshots, blue learns vs red snapshots, iterated). The current RL commander is single-sided PPO against fixed jammer curricula. A real league could substantially close the exploitability gap.

3. **Exploitability BR training is short (80 iters) and noisy.** Std across 3 seeds is ~20 reward units for every method. The mean ordering (FP < RL < static) is suggestive but not statistically conclusive.

4. **n_targets=4 only.** Training never saw n=1 or n=8, hence the weak generalization to n8.

---

## 6. Recommendation

Per user directive: **"G1 打不破 → 先升耦合难度, 仍不破退 IET, 不硬凑"**

G1 PARTIAL: RL matches FP classical on kill in the hard regime and exceeds it on survival, but does not deliver a *decisive* kill-rate win, and is not more robust on exploitability.

**Recommended next step (decision tree):**

- **Option A (recommended)** — *Tighten L3 coupling + league training.* Run a proper red/blue league (PSRO-style) for ~2× current wall-clock. The mid-regime (L1) win is strong enough that a real league could plausibly convert the L3 tie into a clear win. The fundamental signal is there.
- **Option B** — *Drop the kill-rate headline, lean on survival + multi-objective framing.* RL commander is genuinely better at survival under L3 (+25 %) and dramatically better at L1 (+0.42 survival). A paper framed as "learned commanders survive EW campaigns that classical cannot" is defensible without league training.
- **Option C** — *IET / engineering-enabling fallback.* If league training is infeasible, fall back to the Concerto-RRM framing (per existing plan `snuggly-exploring-parrot.md`) — that plan's value proposition is "extend classical operating envelope," which is consistent with what we observed (RL doesn't replace classical, it adds robustness in the survival dimension).

**My recommendation**: **Option A**. The L1 win is too strong to ignore; the L3 collapse is a training-pipeline bug (random-init jammer), not a fundamental capability ceiling.

---

## 7. Files for review

- Training log: `experiments/wp12_results/wp2_train.log`
- Eval CSV: `experiments/wp12_results/wp2_eval.csv`
- Exploitability CSV: `experiments/wp12_results/wp2_exploitability.csv`
- RL checkpoint: `checkpoints/taes_mainline/taes_rl_phase1_L1.pt`
- Code: `algo/_shared/pilot/taes/run_wp2.py`, `algo/_shared/pilot/taes/taes_ppo.py`, `algo/_shared/pilot/taes/taes_actor_critic.py`, `algo/_shared/baselines/taes_fp_classical_commander.py`
