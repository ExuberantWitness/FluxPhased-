Reviewer independence: same-family, adversarial filesystem review; provisional pending cross-family/human rerun.

## Overall verdict

**FAIL / BLOCK CLAIMS**

Reason codes: `ARTIFACTS_ABSENT`, `BASELINE_SUBSTITUTION`, `WRONG_STATISTICAL_NULL`, `SINGLE_TRAIN_SEED`, `SIMULATION_ONLY`, `STRUCTURAL_OVERCLAIM`.

The narrowest defensible statement is: “The narrative reports that two seed-0 PPO checkpoints failed a simulator gate.” Even that statement is not independently reproducible because none of the cited MFR code, scripts, checkpoints, curves, raw seed values, or summary JSON exists at the supplied paths or in the public commit.

## Checklist

| Check | Status | Evidence and impact |
|---|---|---|
| A. Ground-truth / metric provenance | **FAIL** | Neither reachable tree contains `drop_ratio`, MFR code, numerator/denominator logic, raw counts, or seed arrays. Report only names the metric at [pasted-text.txt:16](/home/exuber/.codex/attachments/1121ecad-3ed7-4ea3-accb-61bf80e4d7fd/pasted-text.txt:16) and gives aggregate values at lines 72–78. Eight “seeds” at line 70 cannot be traced. |
| B. Score normalization | **FAIL (auditability)** | No explicit self-normalization is visible in the prose, but missing metric code means it cannot be ruled out. The post-hoc `1/sqrt(1+JNR)` plus `[0.1,1]` clamp at lines 29–37 modifies the simulator after an initially flat result; no physical calibration is provided. Saturation admitted at line 130 makes `drop_ratio` non-identifying. |
| C. Result existence | **FAIL** | All 13 expected targets are absent: four `/tmp` scripts, four implementation files, `g2a_summary.json`, two checkpoints, and two curves. The local checkout is clean at `fa485ad4`; the public snapshot is `80769974`; neither tree contains MFR/G2a paths. Current remote heads also contain no later MFR branch. Checkpoints would be ignored by [`.gitignore`:8](/home/exuber/CODE/CORE/pythonProject1/FLUXPH/FluxPhased-/.gitignore:8), and no hashes/manifests are supplied. |
| D. Dead-code / reachability | **FAIL** | There is no reachable function to audit. Repository-wide search in both trees finds no `drop_ratio`, `tgt_jnr`, `JAM_POLICY_NOISE`, `progress_factor`, G2a, or MFR implementation. Thus the quoted fragments at report lines 31–35 and 119–123 cannot be tied to executed code. |
| E. Scope / seeds | **FAIL** | Artifact names explicitly use `s0` at lines 188–190. v1 and v2 are two hyperparameter variants from one training seed, not independent training replicates. The eight reported seeds are apparently evaluation RNG seeds for fixed checkpoints. They may estimate rollout variability, not training/algorithm variability. |
| F. Evaluation type | **PASS: classified** | **`simulation_only`**; `drop_ratio` is a synthetic simulator outcome, with no real-world ground truth, external calibration, or human evaluation. |

## Baseline and gate errors

**FAIL — wrong comparator.** The report itself declares `noise = 0.520` the strongest scripted baseline at lines 72–78, but both paired tests use `blink = 0.450` at lines 80–83.

Correct comparison from the reported rounded means:

| Model | Learning − actual best (`noise`) | Shortfall from required +5 pp |
|---|---:|---:|
| v1 | −2.9 pp | 7.9 pp |
| v2 | −4.5 pp | 9.5 pp |

The reported “short by 0.9/2.4 pp” is only relative to blink and materially understates failure against the gate’s actual comparator.

The public snapshot also contains additional adaptive scripted jammer code, e.g. [`adaptive_spectrum_jammer.py`:75](/tmp/fluxphased-m7-audit.lewl9x/algo/_shared/baselines/adaptive_spectrum_jammer.py:75), although its compatibility with the missing MFR environment cannot be assessed. Thus “best scripted” over the available repository is not demonstrated either.

## Paired t-test audit

**FAIL for gate validity; WARN on arithmetic plausibility.**

The reported `t=16.4` for Δ=.041 and `t=13.3` for Δ=.025 are numerically compatible with a paired test of **H₀: Δ=0**:

- v1 two-sided p ≈ `7.64e-7`
- v2 two-sided p ≈ `3.18e-6`

So `<0.0001` is arithmetically plausible, but it tests the wrong scientific null and the wrong baseline.

The superiority gate requires **H₀: Δ≤0.05**. Using the reported rounded values:

- v1 vs blink: `t_margin≈−3.6`, one-sided p for Δ>0.05 ≈ `0.9956`
- v2 vs blink: `t_margin≈−13.3`, p ≈ `0.999998`

Approximate 95% CIs inferred from the reported t-statistics are `[.0351,.0469]` and `[.0206,.0294]`; even against blink, both exclude the required +.05 in the wrong direction.

Raw paired values, seed IDs, difference distributions, exact p-values, handling of 32 vector environments × 4 episodes, and action-RNG pairing are absent. Implied paired correlations are unusually high (`~.957` and `~.985`) but cannot be checked. With only eight seed aggregates, normality and independence are unassessed.

## Structural-impossibility claim

**FAIL.** Five observed policies cannot establish a universal statement about “any algorithm or training amount” at lines 141–147.

Specific problems:

- “Always-on is structurally optimal” requires executable code plus a proof that increasing jammer activity monotonically increases every relevant transition/output and has no cueing, countermeasure, targeting, congestion, or delayed adverse effects. None is available.
- The prose alternates between “near-optimal” and “no policy can exceed it”; those are not equivalent.
- Equality after rounding is not equivalence. v2 sampled is `0.519`, not `0.520`, at lines 109–115.
- Near-zero logits and ~0.5 Bernoulli actions are equally consistent with an untrained/random policy. There is no initialization/random-50%-duty baseline, checkpoint progression test, or equivalence test showing PPO learned anything.
- Claiming that two binary jammers provide a “large-number average” is not a structural argument.
- Evaluation semantics change post hoc: the gate uses greedy behavior, while the exculpatory conclusion relies on sampled behavior. Deployment semantics were not preregistered or preserved.
- Adding only `jam_rew -= α·active_frac` at lines 155–159 cannot lower an always-on script’s **raw drop_ratio**. It only changes learner reward unless the comparator receives the same hard budget/cost and the gate metric is redefined.

## Internal contradictions

- **Arithmetic falsehood:** line 85 says `0.491 < 0.489`; actually `0.491 > 0.489`.
- “Both greedy evals are 0.484” at line 64 conflicts without explanation with `0.491/0.475` at lines 77–78 and `0.486/0.471` at lines 109–113.
- “Precisely matches noise” is based on three-decimal rounded means and is false for v2 (`.519` versus `.520`).
- The environment was modified after observing a flat metric specifically to create policy separation at lines 25–47; without held-out calibration this supports only a result about that tuned simulator.

## Claim impact

| Claim | Impact |
|---|---|
| G2’a failed for these checkpoints | Directionally consistent with the prose, but **unverified** |
| PPO learned a noise-equivalent strategy | **Unsupported**; random/untrained parity not excluded |
| Learning cannot beat noise by 5 pp | **Unsupported universal claim** |
| Noise is the actual best scripted comparator | Report says so, but the test did not compare against it |
| Reported p-values validate the gate | **False**; wrong baseline and wrong null |
| Negative result is publication-ready | **No**; simulation-only, single training seed, no artifacts/provenance |

## Required actions

1. Restore and commit the entire MFR implementation and all four evaluators under version control; never cite `/tmp` as the sole source.
2. Publish raw per-episode/per-seed counts, exact seed IDs, `g2a_summary.json`, curve files, checkpoint SHA-256 hashes, commit/config/dependency metadata, and reproduction commands.
3. Define `drop_ratio` numerator, denominator, reset semantics, aggregation order, and every transformation; add a call-path/instrumentation test.
4. Pre-register stochastic versus greedy deployment semantics and evaluate the same semantics throughout.
5. Compare against `max(scripted)` on paired exogenous seeds. Test the actual margin with `H₀: Δ≤.05` or require the one-sided CI lower bound to exceed `.05`.
6. Use independent training seeds, not eight evaluations of one `s0` checkpoint; use a hierarchical analysis separating training-seed, environment-seed, episode, and policy-action RNG.
7. Add untrained PPO, random Bernoulli, and fixed-duty sweeps from 0–100%; use TOST/equivalence bounds before claiming parity.
8. For a structural claim, provide a formal monotonicity argument plus exhaustive duty/target/action sweeps and tests for countermeasure/cue side effects. Otherwise narrow the claim to the tested PPO runs.
9. Apply energy constraints symmetrically as a hard physical budget or redefine the utility for every comparator; a learner-only reward penalty does not reduce scripted raw drop.
10. Obtain an independent cross-family or human rerun before treating the verdict as more than provisional.