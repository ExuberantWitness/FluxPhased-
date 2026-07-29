"""G3-BSTA-lite clean implementation line.

New namespace for the budgeted service-target allocation MFR interference
benchmark. This package is authored from the G3-BSTA fast-work specification
(``docs/g3-bsta-lite/DEBUG_CONTRACT.md``) against the verified repository
dependencies in ``env/gpu/twoteam/``.

Status: F0 — runtime/MDP contract phase. Modules implementing the causal
budgeted environment, baselines, oracle, imitation and masked PPO land in
later phases per the F0..F6 gate order in the fast-work plan.

Provenance note: this clean line is independent of the quarantined
``mfr-orphans-20260728T094154Z`` package. No orphan bytes are imported or
executed here. The orphan package exists only as static defect evidence
under ``evidence/`` on the forensic branches.
"""
