# IDEA_REPORT v2 — Task-Coverage Diversity-Aware Meta-Solver for Radar EW League

**Date**: 2026-05-17
**Direction**: 在已实现的 FluxLeague (PSRO + 3-role league + 分阶段课程) 之上设计一个**新颖**算法扩展
**Config target**: 25×25 array, P=4 pulses, FFT bins=64, num_envs=1
**Status**: Builds on commit `01b9e5d` (FluxLeague) and `309218d` (validation). Existing IDEA_REPORT.md (2026-05-14) is the *implementation blueprint*; this v2 is the *next-step research idea*.

---

## 1. Why a New Idea Is Needed

FluxLeague is operational: 3 roles per team, Nash meta-solver via LP, PFSP opponent sampling, 4-phase curriculum, dual-arm PPO. The Explore audit (2026-05-17) ranks every league/PPO file as **DONE/PROD** and flags these gaps:

1. **Single-env training bottleneck** — engineering.
2. **No opponent curriculum / Elo band** — moderately novel.
3. **No intrinsic motivation** — applicable but fragile on 13k-dim action.
4. **Meta-Nash gives no diversity guarantee** — *concentrates mass on policies that maximize win rate, not strategic coverage*. This is the most paper-worthy gap.
5. **Hyperparameter sensitivity** — engineering.
6. **No policy distillation** — orthogonal.

The non-transitive structure of FluxPhased — detect→jam→recon→detect — is *task-axis-aligned*: each policy can be characterized by how it allocates its 625 elements across {detect, jam, comm, recon}. A 4-simplex of pure strategies. **Nash on the payoff matrix does NOT see this axis**; it can collapse onto two locally dominant policies even when a third axis is structurally important and only currently under-exploited.

This is exactly the failure mode AlphaStar's `League Exploiter` role was designed to patch. But the patch is *reactive*: an exploiter has to *discover* the blindspot through training. We can do better by making the meta-solver *intrinsically aware* of the task-allocation axis.

---

## 2. Proposed Idea: TC-DAMS

**Task-Coverage Diversity-Aware Meta-Solver (TC-DAMS)** — a drop-in replacement for `training/self_play/meta_solver.py` that augments the Nash LP with an explicit diversity term over the *task-allocation simplex*.

### 2.1 The 4-dim Strategic Fingerprint

For every policy `π_i` in the population, define its **task fingerprint** as the long-run average fraction of elements assigned to each of the 4 tasks across an evaluation episode:

```
f(π_i) ∈ Δ^3,   f_t(π_i) = E[ (1/N) Σ_n 1{ task(n, π_i) = t } ]   for t ∈ {detect, jam, recon, comm}
```

This is *free to compute* — it's already a byproduct of any evaluation game in `vec_mfar_env.step()`. The fingerprint lives on the 3-simplex (4 components summing to 1).

### 2.2 The TC-DAMS Objective

Standard Nash on payoff matrix `U` returns mixture `σ` minimizing exploitability:
```
σ_Nash = arg max_σ min_τ σᵀ U τ
```

TC-DAMS adds a **coverage entropy** term over the marginal task fingerprint induced by the mixture:

```
F(σ) = Σ_i σ_i · f(π_i)   ∈ Δ^3                     # mixture-induced task marginal
H(F) = -Σ_t F_t · log F_t                            # Shannon entropy over 4 tasks
σ_TCD = arg max_σ [ min_τ σᵀ U τ  +  λ · H(F(σ)) ]   # λ ≥ 0, the diversity coefficient
```

When `λ=0` this reduces exactly to Nash; as `λ↑`, the meta-solver biases toward mixtures that maintain coverage over all four task axes. The objective is **concave in σ** (Nash value is concave; entropy of a linear function is concave), so it remains a convex program — solvable with the same LP infrastructure plus one entropy-regularization Frank-Wolfe iteration.

### 2.3 Why This Is Novel and Specific

1. **Existing diversity methods operate in policy-space distance** (PSD-PSRO, DPP-PSRO, BD&RD) using behavioral embeddings that need to be learned. **TC-DAMS uses a domain-grounded fingerprint** that is free, interpretable, and aligned with the actual non-transitive structure of radar EW (the rock-paper-scissors *is* a rock-paper-scissors on the task axes).
2. **It's a meta-solver swap, not a new training loop** — sits at a single function in `meta_solver.py`. Easy to ablate λ ∈ {0, 0.1, 0.3, 1.0} cleanly.
3. **Predictions are concrete**: TC-DAMS should reduce time-to-detect a non-transitive blindspot (measured by League-Exploiter win-rate trajectory), and should produce a mixture that is more robust under task-axis adversarial perturbation (swap one team's task prior). Both are measurable in this codebase.

### 2.4 Companion Mechanism: Elo-Band PFSP

Orthogonal to TC-DAMS, add an **Elo-band scheduler** on top of existing PFSP sampling:
- Maintain Elo per policy (updated from payoff matrix entries).
- For each training step, sample opponents in a band `[Elo_self − Δ, Elo_self + 2Δ]` weighted by PFSP within that band.
- Anneal `Δ` from wide (early, exploration) to narrow (late, hard-exploit).

This is well-precedented (AlphaStar used a softer variant) but **not implemented in the current `opponent_pool.py`**. It's the right ablation companion to TC-DAMS because both touch *opponent distribution*, and we can factorially ablate (TC-DAMS yes/no × Elo-band yes/no) for a clean 2×2 study.

---

## 3. Hypothesis & Falsifiable Predictions

| H | Prediction | Measurement |
|---|------------|-------------|
| **H1** | TC-DAMS (λ=0.3) yields task-fingerprint entropy at least 25% higher than Nash baseline | `H(F(σ))` after each PSRO iteration, averaged over last 5 iters |
| **H2** | TC-DAMS reduces exploitability (Nash-conv) by ≥10% at the same PSRO budget | Nash-conv per iter, measured via best-response training on held-out checkpoints |
| **H3** | TC-DAMS main agent wins ≥55% vs Nash main agent in head-to-head over 200 games | Direct H2H eval |
| **H4** | Elo-band PFSP reduces win-rate variance across PSRO iterations by ≥30% | Var of main-vs-random win rate, smoothed |
| **H5** | TC-DAMS + Elo-band > Nash baseline on cross-scenario generalization (held-out missile geometries) | Win rate on 3 unseen geometries |

H3 is the headline. H1 is the mechanism. H2 is the theoretical justification. H4 is the companion ablation. H5 is the practical claim.

---

## 4. Experimental Design

### 4.1 Configurations

All runs: **25×25 array, P=4 pulses, FFT bins=64, num_envs=1**. Single GPU (RTX 2060 or equivalent).

| Run | Meta-Solver | Opponent Sampling | PSRO Iters | Phases |
|-----|-------------|-------------------|------------|--------|
| **R0** | Nash (baseline, existing) | PFSP (existing) | 20 | A→B→C→D |
| **R1** | TC-DAMS, λ=0.3 | PFSP | 20 | A→B→C→D |
| **R2** | Nash | Elo-band PFSP | 20 | A→B→C→D |
| **R3** | TC-DAMS, λ=0.3 | Elo-band PFSP | 20 | A→B→C→D |
| **R1λ** | TC-DAMS, λ ∈ {0.1, 1.0} | PFSP | 20 each | A→B→C→D (sensitivity sweep, 2 runs) |

Total: **6 runs**. Each ≈ 8h at the user's reported budget → ~48 GPU-hours fits the "8-40h, full multi-phase" answer (we'll prioritize R0, R1, R3 first, then R2, then sweeps if time permits).

### 4.2 Seeds & Variance

3 seeds per critical comparison (R0, R1, R3). Report mean ± std. Document variance honestly in the report — radar EW games have high variance, single-seed wins prove nothing.

### 4.3 Metrics (all logged per PSRO iter)

1. **Win rate vs random** (rolling, 100 games)
2. **Win rate vs naive self-play** (frozen baseline)
3. **Nash-conv** (exploitability proxy, via 200-step best-response training on held-out copy)
4. **Task-fingerprint entropy** `H(F(σ))`
5. **Effective population size** (entropy of meta-mixture, exp(H(σ)))
6. **Elo of main agent** (vs static pool)
7. **Cross-scenario win rate** on 3 held-out missile-geometry scenarios (final-eval only)

### 4.4 Falsification Conditions

- If H1 fails (TC-DAMS does not increase entropy of mixture-induced fingerprint), the implementation is wrong → debug.
- If H1 holds but H3 fails (more diverse but no head-to-head win), report as **negative result** — diversity does not translate to strength in this game.
- If R0 main agent doesn't beat random > 80%, the *baseline* is broken → fix infra before claiming anything about TC-DAMS.

---

## 5. Risk & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 8-hour budget per run exceeded at 25×25 | Medium | High | Cap PSRO at 10 iters if Phase A+B+C runs slow; report partial results |
| TC-DAMS LP becomes non-convex (numerical) | Low | Medium | Frank-Wolfe with small λ ≤ 1.0; fall back to Nash on solver failure |
| Single env (num_envs=1) makes payoff eval too slow (K² games sequential) | High | High | Reduce `payoff_eval_games` to 20; subsample opponent pairs |
| Task fingerprint is constant across policies (all converge to same allocation) | Medium | Critical | This would *prove* TC-DAMS is needed (low diversity!); use as primary motivation in narrative |
| Existing FluxLeague has latent bug under 25×25 (vs 5×5 default tested) | Medium | High | Run R0 baseline first as smoke test before launching ablations |

---

## 6. Implementation Plan (Stage 2)

1. **Phase 2.1** — Read existing `meta_solver.py`, `payoff_matrix.py`, `opponent_pool.py`, `flux_league.py`. Confirm 1-function swap surface.
2. **Phase 2.2** — Implement `TaskCoverageMetaSolver` in `training/self_play/tc_dams_solver.py`. Use Frank-Wolfe with `λ` config-exposed.
3. **Phase 2.3** — Add task-fingerprint logger to `vec_mfar_env.step()` (return mean per-task allocation alongside rewards).
4. **Phase 2.4** — Implement `EloBandSampler` wrapper on `OpponentPool.sample()`.
5. **Phase 2.5** — Wire both into `flux_league.py` behind config flags so all 4 cells of the 2×2 ablation share the same loop.
6. **Phase 2.6** — Add `configs/league_tcdams.yaml` with sweep parameters.
7. **Phase 2.7** — Smoke test at 5×5 (existing) to verify nothing regressed; then 25×25 R0 smoke; then full deploy.

---

## 7. Relation to Prior Idea Report (2026-05-14)

The original `IDEA_REPORT.md` proposed FluxLeague and was **fully implemented**. This v2 is strictly *additive* — it does not redesign or contradict prior work. It addresses the open research question naturally posed by what was built: *"Now that we have a working league, how do we make the meta-game itself aware of the structure of radar EW?"*

---

## 8. Bottom Line

**Headline claim if successful**: A 50-line LP modification (TC-DAMS) raises FluxPhased main-agent strength by ≥5% and exploitability robustness by ≥10%, by exploiting the task-allocation simplex as a free, domain-grounded diversity axis.

**Headline insight if negative**: Domain-grounded diversity is *necessary but not sufficient* for non-transitive games — even when the rock-paper-scissors axis is exactly known, behavioral within-axis diversity may dominate.

Either result is publishable and useful.
