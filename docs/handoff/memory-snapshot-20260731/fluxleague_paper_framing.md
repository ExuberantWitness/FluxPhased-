---
name: fluxleague-paper-framing
description: "EAAI Q1 paper framing — FluxLeague as SOTA (AlphaStar-derived) + IPPO/MAPPO baselines + significant performance improvement. NOT a \"we fixed bug X\" paper."
metadata: 
  node_type: memory
  type: project
  originSessionId: f140dbce-c539-4be4-9cf7-6a1b64fdc269
---

**Paper story for EAAI Q1 top journal** (user-corrected framing, 2026-06-21):

> "世界一个IQ级别的多智能体对抗任务 + 我们提出一个新型的基于联赛的多智能体训练测试，性能相对现有算法有显著提升"

**Translation:** IQ-level adversarial multi-agent task + novel league-based multi-agent training framework + significant performance improvement over existing algorithms.

**Why:** User rejected initial framing of "we fixed fire head bug" as the contribution — that's a tactical fix, not a paper contribution. The PAPER contribution is the FluxLeague framework itself, benchmarked against AlphaStar-derived SOTA and IPPO/MAPPO baselines.

**How to apply:**
- FluxLeague = main contribution (AlphaStar-style 3-role league: MAIN + MAIN_EXPLOITER + LEAGUE_EXPLOITER, PSRO+Nash meta-solver)
- IPPO, MAPPO = BASELINES for comparison (not co-equal alternatives)
- Reference repos: github.com/google-deepmind/alphastar, github.com/liuruoze/mini-AlphaStar, github.com/kimbing2/AlphaStar_Implementation
- Kill-learning fixes (Tier 1.1/1.2/1.3) are engineering hygiene — ship as substrate, ablate kill-rate delta, NOT as standalone contribution
- Task domain: multi-static phased-array radar league (4 radars × 2 teams, 25km map, laser dwell-to-kill)

Related: [[fluxleague-kill-fix-tier1]]
