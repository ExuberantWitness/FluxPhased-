---
protocol: APP
protocol_version: "1.0"
paper_title: "FluxLeague: AlphaStar-style Self-Play for Phased Array Laser Defense"
authors: ["anonymous"]
domain: rl
algorithm: coma
commit_pinned: TBD
seed: 42
license: TBD
---

# COMA Baseline (Phase 1.5 — counterfactual credit assignment)

## 1. Identity
- **Algorithm**: COMA (Counterfactual Multi-Agent Policy Gradients, Foerster et al. 2018)
- **Variant**: COMA-S (sample-based approximation; see `algo/_shared/ppo/coma_advantage.py`)
- **Role**: Phase 1.5 candidate — answers EAAI reviewer question "why MAPPO/V-critic instead of COMA/Q-critic?"
- **Commit pinned**: TBD (filled post-sanity, see `code/.git_commit`)
- **Key difference vs MAPPO**: replaces team V critic with centralized Q critic + counterfactual advantage; counterfactual baseline computed by sampling K action draws per agent and averaging Q

## 2. Summary
Same env / reward / curriculum as MAPPO/IPPO/PfspFix, but uses a **centralized Q critic**
that conditions on the full joint action vector (1222 dims: 2 commanders + 4 radars in
compact sub-array form). Per-agent advantage is the counterfactual:
    A_i = Q(s, a) − E_{a_i'~π_i}[Q(s, a_{−i}, a_i')]
Exact marginalization on the 25×4 radar task head is infeasible (4^25 evaluations), so we
sample K=8 counterfactual actions per agent (COMA-S). This is the standard engineering
practice for mixed discrete/continuous action spaces.

## 3. Key results (from `experiments/phase1.5_coma_seed42/`)
| Metric | Value | Health band |
|---|---|---|
| kr (final train) | TBD | curriculum floor 0.5 m |
| eval_kill_rate | TBD | > 0.5 |
| cum_red / blue / draw | TBD / TBD / TBD | — |
| aim_residual | TBD | < 0.1 m target |
| adv_std (last) | TBD | 1e-3 < x < 50 |
| cmd_policy_loss (last) | TBD | |x| > 1e-4 |

**Gate vs MAPPO**: TBD (PASS = cum_red ≥ 0.6, eval_kill_rate ≥ 0.5).

## 4. Repo structure (APP layout)
- `code/config.yaml` — frozen config (absolute `checkpoint_dir`)
- `code/.git_commit` — pinned SHA (WARN-only tripwire)
- `data/checkpoints/` — algorithm outputs (iter_*.pt, populated at runtime)
- `data/logs/` — training log (populated at runtime)
- `environment/README.md` — shared conda env spec
- `LICENSE` — symlink to repo root LICENSE
- `APP_PUBLICATION.json` — manifest

Entry point: repo-root `main.py` (pure-Python launcher; activates `fluxphased`
conda env, enables faulthandler, writes PID, then forwards to
`algo._shared.train_laser:main`).

## 5. Computational requirements
- GPU: 1x RTX 4090 (24 GB) — verified for sanity 5 iter; full 20 iter targeted < 18 GB peak
- Wall-clock: ~3.5–4 h for 20 PSRO iterations (target; COMA-S adds ~10–15% overhead vs MAPPO)
- Conda env: `fluxphased` (see `environment/README.md`)
- Disk: ~290 MB checkpoints + ~13 MB log per run

## 6. Reproduce
```bash
cd /home/ubuntu/CODE/FluxPhased-
python main.py --config algo/coma/code/config.yaml
# Outputs land in algo/coma/data/{checkpoints,logs}/
```

For the archived result (frozen commit + experiment artifacts):
see `experiments/phase1.5_coma_seed42/` (post full-run).

## 7. Citation
Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., & Whiteson, S. (2018).
*Counterfactual Multi-Agent Policy Gradients.* AAAI 2018.

Citation for this repo's exact variant: (to be filled when paper is public.)
