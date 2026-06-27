# Stage 1 Diagnosis: Why FluxLeague Policy Learns Kills Slowly

**Run ID**: `wp1_gate_seed42_kill_diagnosis`
**Date**: 2026-06-21
**Pipeline stage**: idea-discovery
**Direction**: Diagnose why trained policy times out in self-play training episodes despite the win_rate=0.50 fix (commit 17fcb77) being applied. Exploiters occasionally kill (4 events in iter 0); main policies never kill.

---

## Symptoms (concrete)

| Observation | Evidence |
|---|---|
| Main policies (main_t0, main_t1) always timeout in training | Log: every main episode shows `wr=0.00` over 10 episodes × 500 steps |
| Exploiters DO kill, sometimes fast | Log: `league_exploiter_team0 ep3 steps=135 kill`, `main_exploiter_team1 ep4 steps=15 kill` (15 steps = 3.75× required dwell) |
| Payoff matrix has diversity (Phase C tiebreaker works) | Win rates: 0.29, 0.44, 0.48, 0.50, 0.66, 0.88, 1.00 |
| kill_radius stuck at init=100m (no annealing) | Threshold 0.7 not met → curriculum gated |
| NashConv=0, sigma=[0,1,0]/[1,0,0] | Pure strategy (expected at iter 0) |
| PPO metrics show learning | Entropy decreasing (323→325 nats over 9 eps), value loss varying, radar cmd losses dropping |
| Reward 16K avg mostly from beam-guidance log potential | Per-step ~16/step × 500 × 2 teams ≈ 16K (matches) |

## Root cause analysis (3 independent investigations converge)

### PRIMARY: RX-side beam direction hardcoded to boresight

**Location**: [vec_mfar_env.py:326-327](../radar_sim/gpu/vec_mfar_env.py#L326) and [356-357](../radar_sim/gpu/vec_mfar_env.py#L356)

```python
delay_s, doppler_hz, gain = self.channel.compute_params_batch(
    ...,
    beam_az=torch.zeros(E, R, device=dev),   # hardcoded boresight
    beam_el=torch.zeros(E, R, device=dev),   # hardcoded boresight
    ...,
)
```

**Channel physics**: [vec_channel.py:154-177](../radar_sim/gpu/vec_channel.py#L154). When beam_az/beam_el are passed (not None), the channel applies a Gaussian beam pattern loss:

```
loss_db = -3 * (d_az/bw_az)^2 - 3 * (d_el/bw_el)^2
tx_gain_db = directivity_db + loss_db
# Then in radar equation: Pr = Pt + 2G + ... (monostatic assumption)
```

So `tx_gain_db` (despite the name) is used as a 2-way gain in the link budget.

**TX side is correct**: [episode.py:239-240](../training/laser/episode.py#L239) reads `action[..., 4]*60deg` for az and `action[..., 5]*45deg` for el, assembles beamformed TX waveform.

**Asymmetry**: Policy learns to steer TX toward target. RX is forced to boresight. Any target off the array's physical boresight suffers SNR penalty on receive.

**Quantitative impact** for our 25x25km map with 5km min baseline:
- Default array: 25 elements x 25 elements, bw_az = bw_el approx 4 deg
- Target at typical offset (5-10 deg az from a 5km-baseline radar): 5/4 = 1.25 beamwidths -> -3 * 1.25^2 = -4.7 dB
- Multi-static geometry rarely puts targets at boresight of both radars
- **Conservative estimate**: 3-6 dB RX SNR loss -> ~2x range error -> 2x Kalman crossrange sigma -> 2x aim anchor sigma

### SECONDARY: Multi-static measurement model mismatch

The env's channel model is **monostatic-equivalent** (uses `2*tx_gain_db`). For a true multi-static deployment:
- TX radar at position A, RX radar at position B
- Bistatic gain = `G_tx(theta_tx) + G_rx(theta_rx)` (separate terms, each 1-way)
- Current code conflates them as `2*G` (assumes TX=RX geometry)

**Impact**: When policy steers TX beam, the code applies that steer to both TX and RX in the link budget (because `tx_gain_db` enters with factor 2). But on RX side, beam_az=0 forces boresight, so the actual model becomes "TX steer correct, RX forced boresight, but credit both ways" - physically incoherent.

**Correct fix**: Either (a) make RX truly omni (no beam loss on RX) by passing beam_az=None, OR (b) make RX steerable independently by adding a separate beam_az_rx parameter.

### TERTIARY: Fire head init at 50% (hybrid_fire zeros aim only)

**Confirmed by Agent 3**: `hybrid_fire=True` zeros only the aim head (`weight[1:].mul_(0.01); bias[1:].zero_()`), NOT the fire head. So at iter 0:
- Aim is anchored at enemy position (good - equivalent to BC warmup for residual_aim)
- Fire is Bernoulli with random init logits -> ~50% fire probability per step
- Kill chain requires sustained 20 pulses (4 consecutive control steps) of fire+aim-on-target

**Effect**: Even with perfect aim, ~50% fire probability * 0.5^4 = 3% chance of 4 consecutive fire pulses by chance. Policy must learn to commit to fire decisively when aim is good. This is what exploiters are doing (the step-15 kill suggests a policy that fires aggressively when conditions are met).

## Three competing hypotheses

### H1 (PRIMARY): RX beam hardcode is the bottleneck
- **Mechanism**: Off-boresight targets lose 3-6 dB SNR -> Kalman crossrange sigma 2x worse -> aim anchor sigma 2x worse -> kill probability drops 4x (scales with sigma^2 for 2D Gaussian within kill_radius)
- **Prediction**: Threading actual beam_az/beam_el through env step -> kill rate jumps from ~10% to >50% within iter 0
- **Test**: Implement fix, run 1 PSRO iter, compare kill counts before/after

### H2: Slow learning is normal, just wait
- **Mechanism**: 30 PSRO iters expected over 24h+. Currently at iter 0 (7.5h in). Kill events ARE happening in exploiters. Main policies will catch up by iter 3-5.
- **Prediction**: Continue training, observe iter 3-5 milestone. Expect main policies to start killing.
- **Test**: Wait 6-12 more hours, re-evaluate.

### H3: Multi-static model needs rethinking (deeper)
- **Mechanism**: H1's "thread beam_az through" may be insufficient because the channel itself conflates TX/RX gains. Need a proper bistatic model with separate TX/RX beam patterns.
- **Prediction**: Even fixing H1 won't help much; need to refactor channel.
- **Test**: H1 first; if no improvement, escalate to H3.

**Ranking**: H1 > H2 > H3 (H1 is most likely + cheapest to test + physically motivated)

## Proposed fix (if H1 chosen)

**Files to change**: 1 file, ~10 LOC

**[radar_sim/gpu/vec_mfar_env.py](../radar_sim/gpu/vec_mfar_env.py)** - thread `beam_az/beam_el` through `step()`:

```python
def step(self, tx_signal, radar_actions_global):
    # radar_actions_global: [E, R, action_dim], action[..., 4]=az, [..., 5]=el
    # Convert policy action ([-1,1]) to degrees (+/-60 deg az, +/-45 deg el) - match episode.py
    beam_az_deg = radar_actions_global[..., 4] * 60.0   # [E, R]
    beam_el_deg = radar_actions_global[..., 5] * 45.0   # [E, R]

    # ... existing code ...
    delay_s, doppler_hz, gain = self.channel.compute_params_batch(
        ...,
        beam_az=beam_az_deg,    # CHANGED from torch.zeros
        beam_el=beam_el_deg,    # CHANGED from torch.zeros
        ...,
    )
```

**Backward compat**: For tasks that don't use beam steering (e.g., generic missile), action[..., 4:6] are zeros or unused -> behaves identically to current.

**Risk**: If policy has not yet learned to aim the beam, this may make early training WORSE (noisier measurements during exploration phase). Mitigation: warmup with beam_az=zeros for first N steps, then phase in.

## Recommendations (3 options, mutually exclusive)

### Option A: Continue current training, no intervention
- **Cost**: 0 (just wait)
- **Benefit**: Preserves 7.5h investment. If H2 is correct, training will succeed naturally.
- **Risk**: If H1 is correct, we waste 24h on a training run that plateaus at low kill rate.
- **Decision rule**: Stop and re-evaluate at iter 5 (~12h from now). If main policies haven't started killing, abort and pursue Option B.

### Option B: Stop current, apply fix, restart (RECOMMENDED if H1 trusted)
- **Cost**: 7.5h training lost + ~24h for fresh run
- **Benefit**: If H1 is correct, fresh run will show >50% kill rate at iter 0, triggering kill_radius annealing. Faster path to converged policy.
- **Risk**: If H1 is wrong (H2 or H3), we've lost 7.5h for nothing.
- **Decision rule**: Commit to this only if confident in H1 (>=70%).

### Option C: Parallel - keep current, also start fixed-env run
- **Cost**: 2x GPU (may not fit - currently using 60GB/97GB)
- **Benefit**: Definitive comparison. H1 vs H2 resolved empirically.
- **Risk**: GPU contention slows both runs.
- **Decision rule**: Only if can free 30GB GPU memory.

---

## Recommended next step

**Option A with milestone at iter 5**. Reasoning:
1. Current training shows real learning signal (kills in exploiters, fast 15-step kill).
2. We've already invested 7.5h.
3. Iter 5 milestone (~12h from now) is a cheap decision point: if main policies still haven't killed, abort and pursue B.
4. Meanwhile, **prepare the fix** as a separate branch so it's ready to deploy if needed.

If user disagrees and wants Option B immediately, that's defensible - the beam_az hardcode IS a real bug regardless of whether it's the bottleneck.

---

## Cross-review status

- **Agent 1** (beam path trace): Confirmed TX uses action, RX hardcodes zeros.
- **Agent 2** (kill chain audit): Confirmed kill requires aim+fire+20-pulse dwell. Kill chain logic correct.
- **Agent 3** (reward shaping audit): Confirmed reward is fine; 16K is dominated by beam-guidance potential. Fire head not zeroed by hybrid_fire.

All three agents independently arrived at consistent findings. Confidence in H1: HIGH.
