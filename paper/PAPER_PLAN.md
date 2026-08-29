# Paper Plan — FluxPhased TAES Manuscript

**Working title:** Scaling the Attack Breaks Defense Containment in Task-Level Radar–Jammer Self-Play

**Target venue:** IEEE Transactions on Aerospace and Electronic Systems (TAES)

**Paper type:** Empirical, physics-grounded multi-agent reinforcement-learning study

**Core contribution:** In a task-level, deadline-constrained multifunction-radar game, adding a second jammer changes the converged allocation of adversarial power: defense containment falls from 63.7% ± 1.0% in the valid two-seed S6 baseline (1 jammer vs 2 radars) to 23.0% ± 1.1% in S7 (2 vs 2), while radar idle-jammer loss remains low and equilibrium stability is preserved. A co-located-jammer control attributes most of the change to attacker count within the separated S7 radar geometry and a smaller component to cross-fire geometry. A lightweight opponent-class mixture reduces singleton leverage in a dose-responsive way.

**Scope boundary:** Results are simulation-based, use two services and five azimuth sectors, one validated S7 regime (baseline SNR 12 dB, P_jam 0.1 W), fixed 63-token team budget, and one R5 seed per mixture condition. Claims are about this benchmark family and task-level mission loss, not deployed radar performance or universal MARL laws.

## Claims–Evidence Matrix

| ID | Claim | Type | Evidence | Status |
|---|---|---|---|---|
| C1 | Task-level self-play reaches a stable, reproducible equilibrium when trained to 3000 iterations. | Descriptive | S7 seeds 20260801/02/03; 2000–3000 plateau; full-protocol evaluations. | Supported |
| C2 | Doubling attackers reduces floor-adjusted defense neutralization from 63.7% ± 1.0% (two valid S6 seeds) to 23.0% ± 1.1%. | Benchmark-level/associational | S6 two valid 12-dB seeds; S7 three converged seeds; team activation controlled; S6/S7 radar geometry differs. | Supported with scope boundary |
| C3 | Attacker count is the primary driver; cross-fire geometry adds a secondary penalty. | Mechanistic | Co-located +60/+60 ablation: 28.4% neutralization vs 24.2% cross-fire. | Supported for tested geometry |
| C4 | Pair-trained radar defenses trade singleton robustness for pairwise defense. | Descriptive/associational | j1_only 0.2086 ± 0.0745 across S7 seeds; seed-01 continuation 0.12→0.24. | Supported with seed heterogeneity |
| C5 | Opponent-class mixing improves cross-class robustness at the ratio level and saturates near 50–75% singleton exposure. | Associational/intervention | R5-lite mix 0/0.25/0.5/0.75; j1/jvs 0.502→0.262; eta 20.2→26.1→24.6. | Supported as lite, one seed/condition; gradient-budget confound retained |

## Story spine

Because a task-level radar must satisfy deadlines rather than maximize instantaneous SINR, adversarial power is filtered through scheduling and detection. We ask whether a defense advantage learned against one jammer survives attacker scaling. We find a stable but less containable 2v2 equilibrium, identify attacker count as the primary mechanism with cross-fire as a secondary amplifier, establish a singleton vulnerability boundary, and show that lightweight opponent-class mixing can reduce the vulnerability.

## Structure and evidence

1. **Introduction** — task-level operational question; gap between link-level EW games and non-adversarial radar scheduling; contributions C1–C5; hero result 63.7%→23.0%.
2. **Related Work** — EW game learning; multifunction-radar resource management; self-play robustness, role emergence, and league methods. Position the work as task-level adversarial co-adaptation.
3. **Benchmark and Method** — mission state, physics, observation/action spaces, MAPPO/CTDE, evaluation views, neutralization metric, contestability gate, convergence protocol.
4. **Results** — exactly five claim-led subsections:
   - Stable task-level equilibria require a converged training horizon (C1).
   - A second attacker reduces defense containment by roughly two-thirds (C2).
   - Attacker count dominates the effect and cross-fire adds a secondary penalty (C3).
   - Pair specialization creates a singleton vulnerability (C4).
   - Opponent-class mixing reduces singleton leverage and improves ratio-level containment (C5).
5. **Discussion and Limitations** — implications for threat-count stress testing, training against opponent distributions, benchmark limitations, and next work.
6. **Conclusion and Reproducibility** — concise close and artifact pointers.

## Main figures

1. Benchmark anatomy and pre-training contestability oracle: mission-bearing detector, pairwise link geometry, and predicted profiles for S6/S7/control.
2. Full seed-01 3000-iteration trajectory: h2h, jam_vs_sweep, j1_only, radar floor; stage boundaries and plateau band.
3. Hero result: S6 versus S7 floor-adjusted neutralization, with per-seed points and uncertainty.
4. Mechanism decomposition: S6, co-located jammer control, cross-fire S7; neutralization and h2h side by side.
5. Equilibrium behavior: radar azimuth specialization, jammer budget use, and pairwise JNR contribution matrix.
6. R5 dose-response: singleton relative leverage j1/jvs and neutralization eta against mixture fraction.

## Main tables

- T1: environment and physics constants.
- T2: S6/S7 converged per-seed and aggregate results.
- T3: co-located versus cross-fire mechanism control.
- T4: R5 mixture dose-response, with the gradient-budget limitation explicitly stated.

## Numerical conventions

- Use `drop ratio` for mission-level failures; higher means more successful denial by the jammer side.
- Use `radar floor` for learned-radar drop against idle jammers.
- Use `eta` only with its explicit floor-adjusted formula.
- Use mean ± standard deviation over three action seeds for each checkpoint; use across-training-seed mean ± standard deviation only when explicitly labeled.
- Do not mix the S6 seed 20260729 (baseline SNR 22 dB) into the SNR-12 comparison.
- Treat 1000-iteration numbers as budget-sensitivity controls, not converged headline results.

## Required honesty boxes

- S7 uses one geometry family and one primary SNR regime.
- Co-located ablation has one training seed.
- R5-lite has one seed per mixture condition and skips jammer updates on singleton iterations, so absolute drops are not comparable across mixture fractions; ratio metrics are the primary readout.
- The benchmark is simulation-only and does not establish fielded radar performance.

## Writing order

1. Freeze data and claim matrix.
2. Draft Introduction, Related Work, Benchmark/Method.
3. Draft Results around Figures 2–6 and Tables 2–4.
4. Generate publication figures and captions.
5. Build verified bibliography and run citation audit.
6. Compile PDF, run claim audit and reviewer-style audit.
7. Prepare rebuttal template and resubmission package only after a review identifies issues.
