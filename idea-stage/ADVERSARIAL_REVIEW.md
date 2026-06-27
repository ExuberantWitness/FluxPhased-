# Adversarial Review of H1 (RX beam_az bottleneck)

## Summary verdict

- **H1 strength: WEAK** (nearly refuted by code reading)
- **H2 strength: MODERATE** (consistent with sparse-reward early training, but does not explain exploiter vs. main asymmetry)
- **H3 strength: STRONG** (the channel is structurally monostatic-equivalent; furthermore the kill chain bypasses RX entirely)

The headline finding: **the kill chain does not pass through RX IQ at all.**
`vec_battlefield.py:183-208` computes kills from `(commander_aim - actual_enemy_pos).norm < kill_radius AND fire_on AND illum >= 2ms`.
`commander_aim` is decoded directly from `commander_actions[..., 1:3]` (`vec_drone.py:193-194`), and `get_commander_obs`
(`vec_drone.py:273-282`) hands the policy **ground-truth enemy positions** in obs slots `[68:72]`. The radar RX path
(affected by the `beam_az=zeros` hardcode at `vec_mfar_env.py:326-327`) only feeds `radar_latents[:, :, 0:32]`, which
is *redundant* with the ground-truth channels already in obs. Threading beam_az through RX therefore cannot, by itself,
move the kill rate at all — the policy already has perfect target state.

## Counter-arguments to H1

1. **The hardcode at `vec_mfar_env.py:326-327` is not "the RX side."** It is the single channel-model invocation that
   produces both TX-propagation and RX-propagation gain (the model is monostatic: `2*tx_gain_db` at `vec_channel.py:183`).
   The TX-side *beamforming* is baked into `tx_signal` upstream by `episode.py:227-256` from `elem_actions[..., 4:6]`,
   entirely independent of this hardcode. So the policy's TX beam steering already works; the hardcode only governs a
   *cosmetic* RX-side scalar that scales the per-element voltage gain.

2. **The reward does not depend on RX SNR.** `reward.py:179-199` rewards `(commander_aim - enemy_pos).norm < kill_radius`
   using intended aim. There is no detection threshold, no track covariance, no Kalman sigma gating the kill. RX IQ
   affects only the radar-latent observation channel, which is auxiliary.

3. **The commander already sees ground truth.** `get_commander_observation` places normalized enemy x/y at obs
   indices `[68:72]` (`vec_drone.py:273-282`). The 32-d radar latents in slots `[4:68]` are extra signal, not the
   primary aim source. A policy can solve the aim sub-problem without ever reading the latent.

4. **Fire-head is independently a 50% bottleneck.** `commander_actions[..., 0] > 0.5` gates kills. The action head
   initializes near 0, so early training fire probability is ~0.5 even with perfect aim. This alone caps kill rate
   around 50% *if aim were perfect*, and ~25% *if aim hits 50% of the time*. Observed ~3-10% kill rate is more
   sensitive to aim head and fire head than to RX.

5. **Kill observation at step 15 is consistent with H2, not just H1.** A 2ms illumination requirement equals 4 control
   steps. Once a policy commits fire+aim correctly, a kill at step ~15 (out of probably a 30-60 step episode) is
   exactly what "correct greedy behaviour occasionally emerges under self-play" looks like.

## Physics calculation

Map 25 km × 25 km. Minimum radar baseline 5 km. Take two friendly radars at (−2.5 km, −10 km) and (+2.5 km, −10 km)
and a target at the centre (0, 0). For a 25×25 (625-element) array, `bw_az ≈ 0.886/(25·0.5) ≈ 4.25°` (matches
`bw_az_deg=4.06` default). `array_directivity_db ≈ 44 dB`.

- Radar 0 → target: range ≈ 10.3 km, world-frame az ≈ atan2(0−(−10000), 0−(−2500)) = atan2(10000, 2500) ≈ 76°.
- Radar 1 → target: range ≈ 10.3 km, world-frame az ≈ atan2(10000, −2500) ≈ 104°.
- `array_rotation = 0` (env resets it to zero, `vec_mfar_env.py:273`, and vehicle action only slowly rotates it).
- With the hardcode `beam_az = 0`, the off-boresight error is 76° for radar 0 and 104° for radar 1.
- Gaussian loss formula: `loss_db = -3·(d_az/bw)²`. For radar 0: `-3·(76/4.06)² ≈ -1050 dB`. For radar 1: `-3·(104/4.06)² ≈ -1970 dB`.
- This makes `tx_gain_db` absurdly negative, which after `2*tx_gain_db` drives `rx_power_dbm` to ~-2000 dBm. After
  `clamp(min=0)` on the inside of `sqrt`, `gain_linear → 0`. RX IQ from targets is effectively **zero** under the
  hardcode for any target not within ±4° of array boresight.

This is genuinely catastrophic **for the radar-latent observation**. The TX path is unaffected because TX
beamforming is done upstream. So:
- **H3 is essentially correct**: the channel model is monostatic-equivalent. If you "fix" beam_az on RX, you still
  have `2*tx_gain_db`, which double-counts any RX steering. To do this correctly you need `G_tx + G_rx` with separate
  steering terms, plus monostatic-baseline assumptions only when TX and RX arrays are co-located.
- **However**, this matters only for the latent observation, not the kill chain.

## Kill rate math

Current observed ~3.3% (4 kills / ~120 episodes) decomposes as:

```
P(kill) = P(fire on) × P(aim within kr) × P(illum ≥ 2ms) × P(target alive)
```

With `kill_radius = 100 m` (init) over a 25 km × 25 km map, the aim hit-probability for a random aim is roughly
`π·100² / (25000²) ≈ 5×10⁻⁵` if aim were uniform — but the policy isn't uniform; it has ground-truth enemy position
in obs. A *minimal* policy that copies enemy-x/y from obs[68:70] into aim should hit ~100% of the time. The fact that
it doesn't strongly suggests the aim head hasn't learned to read those channels yet, not that RX SNR is the bottleneck.

H1's "10% → >50%" prediction is **not supported** because the kill chain has no RX SNR term. If the aim/fire heads
learned perfectly tomorrow with the hardcode in place, kill rate would jump to >50% *without any beam_az fix*. The
hardcode is at most a second-order effect on the radar-latent observation.

## Alternative explanations for observed pattern

1. **Exploiters train against frozen, weaker opponents.** League exploiters typically play against a *past* main
   snapshot. The past snapshot fires/aims poorly, giving the exploiter easy targets → 3 of 4 kills appearing in
   exploiters is the expected pattern under H2, independent of any RX issue.
2. **Main policy trains against the *current* exploiter** (or itself), producing a hard adversarial curriculum.
   Sparse kills against a competent opponent are exactly what we'd expect; "all-timeout" is the locally optimal
   non-commital strategy when unsure.
3. **2 ms illumination gate.** With `pulses_per_control=5` and `PRI=200µs`, each control step is 1 ms of dwell. A
   kill requires ~4 consecutive control steps with aim held within `kill_radius` *and* fire held on. Random jitter
   in aim/fire across steps will frequently reset the illumination timer even when the policy is "roughly correct."
4. **Reward shape.** `reward.py` gives non-saturating potential on intended aim *every step*, so the policy can earn
   steady reward without ever committing fire — pushing it toward cautious aim-only behaviour. This is a known
   reward-shaping failure mode and is fully decoupled from RX.
5. **Curriculum gating.** `kill_radius` anneals toward 0.2 m only when `kill_rate ≥ 0.7`. The policy never crosses
   the gate, so it never sees the harder regime — but also never gets a *stronger* signal that "you must be precise."
   This self-reinforcing low-kill equilibrium is a curriculum-stall, not an RX problem.

## What evidence would DISCONFIRM H1

H1 predicts: after threading RX `beam_az` through, kill rate on the **main** policy jumps from ~10% to >50% within
iter 0, with no other changes. Disconfirming observations:

- **Aim-channel ablation.** Zero out `radar_latents` in `commander_obs` (set slots `[4:68]` to 0). If kill rate is
  unchanged, RX was never the bottleneck → H1 false. (Highly likely given ground-truth obs.)
- **TX-only fix.** Apply the `beam_az` fix to the TX-side `tx_signal` assembly (already done at `episode.py:239`)
  and *leave RX hardcoded*. If kill rate climbs anyway, RX was irrelevant → H1 false.
- **Perfect-aim oracle.** Replace `commander_aim` with true enemy position. If kill rate jumps above 50% with the
  RX hardcode still in place, RX was never the bottleneck → H1 false.
- **Fire-only oracle.** Force `fire_on=True` always, leave aim to the policy. If kill rate stays low, fire head
  was not the bottleneck; if it jumps, fire head *was* the bottleneck → either way, not RX.
- **Iter-0 trajectory.** If by iter 1-2 (with hardcode unchanged) kill rate climbs above 30%, H2 is supported and
  H1 was unnecessary.

## Recommended decision

**Do NOT block on the H1 fix. Pursue it only as a low-priority correctness cleanup, paired with the H3 refactor.**

Rationale:

1. The kill chain bypasses RX (`vec_battlefield.py:183-208`), so H1 cannot deliver its headline prediction.
2. H3 dominates H1: a proper `G_tx + G_rx` refactor is needed before RX SNR is even physically meaningful. A
   thread-through patch on top of the `2*tx_gain_db` formulation will double-count RX steering and could *degrade*
   the latent observation.
3. The actual bottleneck candidates, in priority order, are: (a) aim head not learning to read obs[68:72],
   (b) fire head stuck near 0.5, (c) illumination-gate reset due to step-to-step jitter, (d) reward shape
   rewarding aim-only behaviour, (e) league curriculum asymmetry (exploiters see weak opponents).

**Right experiment to run first (cheap, 1-2 GPU-hours):**

- Oracle sweep on a small frozen main policy: (i) oracle aim (true enemy pos), (ii) oracle fire (always on),
  (iii) oracle both. Measure kill rate in each condition with the RX hardcode *untouched*.
- Whichever oracle produces the largest jump identifies the real bottleneck.
- Expectation: oracle aim + oracle fire → kill rate near 100% with the hardcode still present, refuting H1.
- If none of the oracles moves kill rate above ~30%, *then* suspect an env/kill-chain bug and investigate RX/SNR.

**If, after the oracle sweep, RX still looks implicated:** combine the beam_az thread-through with the H3
`G_tx+G_rx` refactor in a single PR, validate against a known-SNR test target (`radar_sim/evaluation/`), and only
then re-train. Do not ship the thread-through alone — it is mathematically wrong under the current monostatic
formulation and risks silently training the policy on a broken latent.
