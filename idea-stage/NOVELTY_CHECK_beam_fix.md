# Novelty Check: RX Beam Azimuth/Elevation Fix

## Technical novelty: LOW

Modeling a phased-array receive beam pattern (gain roll-off as a function of angle between target and boresight) is textbook radar signal processing — it appears in standard references (Skolnik, Richards) and in essentially every credible open-source radar simulator (RadarSimPy, FMCW-MIMO-Radar-Simulation, MIT phased-array-radar). Independent TX/RX beam steering in a fully digital or subarray receiver is also well established in the MIMO/bistatic literature (e.g., the IET *Journal of Radar* memory-based DRL paper explicitly considers transmit-receive beamforming for distributed phased-MIMO radar). The fix is "obviously correct once stated": a Gaussian RX beam loss term applied on top of the existing TX steering is the natural dual of what the channel already does for TX. It is a real design choice only in the sense that the simulator previously did not expose it as a learnable action.

## Research contribution novelty: LOW (as a standalone claim)

"We model RX beam gain correctly in our RL env" is engineering hygiene, not a contribution. Reviewers at EAAI would reject it as a standalone novelty because (a) the technique is standard, (b) the only question that matters scientifically is whether the policy *learns to use* the extra action — and that depends on the rest of the env, not on this fix. If presented as a contribution it would attract the exact criticism "this is just a bug fix."

## Closest prior art

- **Memory-based DRL for Cognitive Radar (IET J. Radar, 2024, rsn2.12469)** — explicitly models transmit-receive beamforming for distributed phased-MIMO radar with DRL; the closest conceptual match for independent TX/RX steering under RL.
- **Vincent, *RL for Multi-Function Radar Resource Management* (PhD thesis, 2023)** — covers beam pattern design, beam spacing, and beamwidth in an RL surveillance loop.
- **radarsimx/radarsimpy (GitHub)** — open-source Python/C++ radar simulator that correctly applies per-receiver array patterns; the canonical reference implementation for "this is how you do RX gain."
- **DeepMIMO / FMCW-MIMO-Radar-Simulation (GitHub)** — phased-array MIMO simulators with separate TX/RX array models; demonstrate the standard engineering pattern.
- **Charan, *Signal attenuation enables scalable decentralized MARL* (arXiv 2505.11461)** — multi-agent RL with physically-grounded signal attenuation; methodologically adjacent for the kill-rate framing.

No GitHub radar RL env found in search (gym-hybrid, awesome-rl-envs, space-gym, etc.) ships an RX-beam-as-action with a learned policy; most treat the antenna pattern as a fixed observation-side filter rather than an action-side gain. So while the *technique* is standard, the *combination* (RX beam az/el as RL actions in a multi-static combat env) is uncommon in open-source code.

## Recommended framing for paper

Do **not** list this fix as a contribution. Instead:

1. **Mention it once in the System Model section** as part of the channel description ("the receive array applies a Gaussian beam loss parameterized by the policy's az/el actions, scaled to ±60°/±45°"), with a citation to Skolnik and to RadarSimPy as standard practice. This pre-empts the "your env is unrealistic" reviewer.
2. **Use it as the substrate for a real contribution**: an **ablation** comparing (a) RX-beam fixed at boresight vs (b) RX-beam as a learnable action. Report the delta in kill rate, P_detect, and convergence speed. If the delta is large (>5–10% kill rate), *that* is the contribution, framed as "RX beam steering is a necessary action-space dimension for combat-relevant multi-static RL — and we show policy degradation when it is omitted." If the delta is small, drop the ablation and let the fix be invisible engineering.
3. **Frame the paper's actual novelty around** the kill-learning diagnosis / curriculum / MARL coordination, not around this channel detail.

## Verdict

**Standalone contribution: NO.** This is a correctness fix to the channel model. It belongs in the System Model / Implementation section with one ablation row in the results table. The paper's claim should rest on whatever the fix *enables the policy to learn* (kill-rate improvement, coordination behavior), not on the fix itself. Ship the fix immediately — it removes a real bug where the policy's RX actions were no-ops — but do not let it carry paper real estate.
