"""G3-BSTA-lite baselines, oracle, imitation and PPO pilot.

Companion package to ``env/gpu/g3_bsta_lite/``. Will host:

- frozen scripted jammer baselines (always_off, random_feasible, budgeted_*,
  periodic_blink, causal_reactive_or_edf);
- the same-observation causal witness / executable clairvoyant oracle used
  for the Gate 1 reachability/headroom gate;
- supervised imitation dataset generation;
- masked-categorical PPO trainer with separate actor/critic optimizers.

Status: F0 — runtime/MDP contract phase. No code lands here until F0
contract tests pass; F2 introduces the first baselines.
"""
