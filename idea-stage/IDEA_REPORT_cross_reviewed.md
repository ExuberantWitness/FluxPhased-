# Stage 1 Cross-Reviewed IDEA_REPORT (Gate 1 input)

**Run ID**: `wp1_gate_seed42_kill_diagnosis_v2`
**Date**: 2026-06-21
**Pipeline stage**: idea-discovery (post-cross-review)
**Direction**: Find the real bottleneck preventing FluxLeague main policies from learning kills.

---

## What changed since v1

v1 hypothesized H1 (RX beam_az hardcode = kill bottleneck). v2 cross-review **refuted H1** with concrete code evidence. This document is the corrected synthesis.

## Three findings from independent agents

### 1. Literature survey (Zhu et al. 2025 closest prior art)
- **Zhu et al. (2025) Nature Sci Rep** — PPO+HAS-RL for MIMO radar CRLB, also hardcodes `G = K` (boresight) in their FIM. This is the direct ablation target.
- **Godrich et al. (2010) IEEE TAES** — Ignoring RX beam pattern degrades CRLB by 3-8 dB.
- **Capuano et al. (2025) arXiv:2503.00499** — SAC for laser pulse shaping, closest dwell-to-kill analogue.
- File: [LITERATURE_SURVEY_beam_fix.md](LITERATURE_SURVEY_beam_fix.md)

### 2. Novelty check (LOW standalone)
- Technical: textbook radar signal processing (Skolnik, RadarSimPy already does this).
- Research: "we model RX beam correctly" is engineering hygiene, not a contribution.
- **Recommended framing**: Ship fix as substrate, ablate kill-rate delta. Don't list as standalone contribution.
- File: [NOVELTY_CHECK_beam_fix.md](NOVELTY_CHECK_beam_fix.md)

### 3. Adversarial review (H1 REFUTED)
- [vec_drone.py:271-274](../radar_sim/gpu/vec_drone.py#L271): `obs[:, t, off] = radar_pos[:, enemy_idx[0], 0] / half_x` — **ground-truth enemy pos in commander obs**.
- [vec_battlefield.py:193-196](../radar_sim/gpu/vec_battlefield.py#L193): Kill uses `self.drone.laser_aim` vs `actual_pos` directly — **no RX in kill chain**.
- The `beam_az=zeros` hardcode only affects radar-latent obs slots [4:68], which are redundant.
- **Conclusion**: Fixing beam_az won't change kill rate.
- File: [ADVERSARIAL_REVIEW.md](ADVERSARIAL_REVIEW.md)

## New hypothesis space

Since H1 is dead, the kill-learning bottleneck must be in one of:

### H4: Aim head learning failure
- hybrid_fire=True zeros aim head → aim = enemy_anchor (perfect init)
- But aim head must learn to KEEP residual small AND follow moving targets
- Test: oracle-aim (replace aim with ground-truth enemy pos) → if kill rate jumps, aim head is broken

### H5: Fire head commitment failure
- Fire head is Bernoulli, ~50% at init
- Kill requires 4 consecutive control steps of fire_on=True
- PPO hasn't learned to commit fire when aim is good
- Test: oracle-fire (force fire_on=True always) → if kill rate jumps, fire head is broken

### H6: PSRO dynamics (exploiter-main asymmetry)
- Exploiters train against FROZEN past version of main (static target)
- Mains train against CURRENT exploiter (moving target)
- This naturally explains why exploiters learn faster
- Test: oracle-both (perfect aim+fire) → if kill rate ~100%, kill chain works; mains just need more training time

### H7: Privileged information leak (deeper)
- Ground-truth enemy pos in cmd obs creates a sim-to-real gap
- Policy never needs to learn sensing pipeline
- Could be the actual paper contribution (information-theoretic analysis)
- Test: ablation comparing privileged vs Kalman-fused enemy pos

## Recommended next experiment: Oracle sweep

**Cost**: 10-20 min (no training, just inference on existing checkpoints)
**Value**: Definitively isolates the kill-learning bottleneck

### Experimental design

Load `main_team0_gen1.pt` + `main_team1_gen1.pt`. Run 20 episodes × 4 conditions:

| Condition | Override | Tests |
|---|---|---|
| C0: baseline | none | current behavior |
| C1: oracle-aim | patch `commander_action[..., 1:3]` to ground-truth enemy pos | H4 |
| C2: oracle-fire | patch `commander_action[..., 0]` to 1.0 | H5 |
| C3: oracle-both | both | H6 (should ≈ 100% if kill chain works) |

### Predictions and interpretations

| Outcome | Interpretation | Next step |
|---|---|---|
| C3 ≈ 100%, C1 ≈ 100%, C2 low | Aim head is fine, fire commitment is the bottleneck | Tune fire head entropy, reward shaping for sustained fire |
| C3 ≈ 100%, C2 ≈ 100%, C1 low | Fire head is fine, aim head can't track | Investigate aim head residual scale, Kalman tracker convergence |
| C3 ≈ 100%, C1+C2 both ≈ 100% | Kill chain works, policy just needs more training | Continue training (H2) |
| C3 < 50% | Kill chain is fundamentally broken | Debug env physics (kill_radius, illumination_time, aim coordinate frame) |

## Paper contribution re-framing (EAAI target)

Original plan: "We fixed RX beam hardcode" — DEAD (novelty too low, doesn't move kill rate).

**New plan** (ranked by novelty):

### Option α: Multi-static radar RL benchmark with information-theoretic ablations
- Contribute: full env + 5 damage cells + 6 OOD axes + dwell-to-kill task
- Ablations: privileged vs Kalman-fused enemy pos (H7), boresight vs learnable RX beam (Zhu et al. comparison), exploiters vs mains dynamics
- Novelty: MEDIUM-HIGH (no comparable multi-static radar RL benchmark exists)

### Option β: Diagnostic methodology for slow-learning RL policies
- Contribute: oracle sweep technique (oracle aim / fire / both) for isolating bottlenecks in continuous-action RL
- Apply to: laser dwell-to-kill task, possibly other domains
- Novelty: MEDIUM (methodology papers are harder to land but publishable)

### Option γ (RECOMMENDED): Combine α + β
- Frame paper around the multi-static radar RL benchmark
- Use oracle sweep as the diagnostic that motivated the design choices
- Ablate information leak + RX beam + PSRO dynamics
- This is the strongest framing for EAAI Q1

## Decision needed (Gate 1)

Pick the experiment path:

**Option A**: Run oracle sweep now (10-20 min), then decide based on results.
- Pro: cheap, definitive, informs all downstream choices
- Con: doesn't directly advance training

**Option B**: Continue current training (24h+ for 30 iters), evaluate at iter 5 milestone.
- Pro: preserves investment
- Con: if bottleneck is aim/fire head, more training won't help (need to fix architecture)

**Option C**: Hybrid - run oracle sweep NOW (10 min), use results to decide whether to keep current training or pivot.
- Pro: best of both
- Con: requires a quick context switch

**Recommendation**: Option C. The oracle sweep is cheap and removes guesswork.

## What we still ship regardless of oracle sweep outcome

The `beam_az=zeros` hardcode is a real bug (even if not the kill bottleneck):
- Affects radar-latent obs quality (relevant for sim-to-real)
- Should be fixed as engineering hygiene
- Ablation "boresight vs learnable RX beam" still useful for paper (vs Zhu et al.)

But fix it AFTER the oracle sweep, so we can attribute any improvement cleanly.

---

## Cross-review status

| Agent | Verdict | Confidence |
|---|---|---|
| Literature | MEDIUM novelty, Zhu et al. is direct prior art | HIGH |
| Novelty | LOW standalone, ship as substrate | HIGH |
| Adversarial | H1 REFUTED, ground-truth leak bypasses RX | HIGH (code-verified) |

**Net**: Original H1 dead. New hypotheses H4-H7 ranked by testability. Oracle sweep is the cheapest next step.
