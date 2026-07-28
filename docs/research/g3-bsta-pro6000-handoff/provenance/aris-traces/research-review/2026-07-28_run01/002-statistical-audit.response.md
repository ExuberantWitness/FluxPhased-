# Independent Statistical Review

**Run**: `fluxphased-g2a-redesign-20260728`  
**Reviewer**: `/root/stats_reviewer`  
**Model / effort**: `gpt-5.6-sol / xhigh`  
**Independence**: `same-family`  
**Acceptance status**: `provisional`  
**Verdict**: `MAJOR REVISION`

## Confirmatory estimand

Let `B` be all baseline finalists selected and frozen on baseline-calibration data. For training seed `i` and script `b`:

\[
d_{i,b}=\operatorname{mean}_{s,a,e}
\left(D^{L}_{i,s,a,e}-D^{b}_{s,a,e}\right).
\]

The redesigned gate passes only if:

\[
\min_{b\in B}
\left[
\bar d_b-t_{0.95,K-1}\frac{s_b}{\sqrt K}
\right] > 0.05.
\]

This is an intersection-union gate. A per-scenario post-hoc maximum over scripts is only an oracle-switching sensitivity and is not the primary comparator.

For each `b`, report the one-sample t statistic and t-tail p-value for `H0: E[d] <= .05`; do not label this an exact p-value.

## Split requirements

Keep mutually exclusive:

1. physics fit;
2. physics validation;
3. baseline tuning;
4. planner development;
5. untouched headroom confirmation;
6. PPO checkpoint validation;
7. locked final test.

No budget, physics fit, planner, script, checkpoint, evaluation mode or sample size may be selected using the locked test.

## Reachability and headroom

For current-environment infeasibility, use an exact/admissible pathwise upper bound:

\[
g_s=U_s-B_{b^*,s},\qquad
\mathrm{STOP}\iff\mathrm{UCB}_{95}(E[g_s])<.05.
\]

An approximate planner can prove reachability but cannot prove infeasibility.

On untouched headroom-confirmation scenarios, require the paired lower bound against every frozen script to exceed `0.075`; every preregistered sensitivity cell must exceed `0.05`.

## Independence and randomness

- the training seed is the primary independent unit for the locked conditional claim;
- every training seed runs the complete scenario × action-replicate × episode grid;
- scenario/action/episode repeats do not inflate t-test degrees of freedom;
- exogenous randomness is keyed by `(namespace, scenario, episode, time, entity, event_type)`;
- action RNG is separate;
- a sequential global RNG is not acceptable because policy-dependent event counts desynchronize CRN.

The primary t inference is conditional on the frozen scenario suite. A scenario-population claim additionally needs crossed random-effects or two-way cluster sensitivity and baseline rollout uncertainty propagation.

## Sample size and sensitivity

- two short seeds are smoke tests only;
- estimate final-budget variance from external evidence or at least `6–8` full-config/full-budget variance pilots;
- preregister `mu_alt > .05`, `delta_star = mu_alt - .05`, a conservative SD upper bound, power `>= .8` and `N_max`;
- solve sample size with the noncentral-t distribution;
- `K >= 8` is an engineering floor, not proof of adequate power;
- at small K, use seed-level t-LCB as primary; sign-flip is a sensitivity, not a substitute;
- preregister the complete finite energy × detector-envelope × scenario-shift grid.