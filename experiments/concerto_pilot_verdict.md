# Concerto-RRM Pilot Verdict

**Result**: 0/4 criteria PASS

**Decision**: **0/4 PASS → retreat to Path A** (C1+C0 → IEEE TAES) per EAAI_RESEARCH_PLAN.md §9.

## Per-cell aggregate (mean across seeds)

| Method | Difficulty | n_seeds | QoS agg | detect | track | comm | jam | min dwell | wallclock(s) |
|---|---|---|---|---|---|---|---|---|---|
| classical | L0 | 5 | 0.890±0.016 | 1.000 | 1.000 | 0.646 | 0.912 | 0.120 | 13.4 |
| classical | L1 | 5 | 0.617±0.016 | 0.999 | 0.274 | 0.646 | 0.547 | 0.120 | 13.3 |
| classical | L3 | 5 | 0.554±0.028 | 0.999 | 0.069 | 0.646 | 0.502 | 0.120 | 13.4 |
| concerto_v1 | L0 | 5 | 0.859±0.013 | 1.000 | 1.000 | 0.525 | 0.911 | 0.120 | 111.4 |
| concerto_v1 | L1 | 5 | 0.509±0.013 | 0.998 | 0.042 | 0.524 | 0.473 | 0.120 | 108.8 |
| concerto_v1 | L3 | 5 | 0.518±0.024 | 0.999 | 0.055 | 0.524 | 0.495 | 0.120 | 110.4 |
| concerto_v2 | L0 | 5 | 0.890±0.016 | 1.000 | 1.000 | 0.646 | 0.912 | 0.120 | 12.7 |
| concerto_v2 | L1 | 5 | 0.448±0.004 | 0.998 | 0.042 | 0.279 | 0.473 | 0.120 | 111.2 |
| concerto_v2 | L3 | 5 | 0.453±0.021 | 0.999 | 0.055 | 0.265 | 0.495 | 0.120 | 111.4 |
| mappo | L0 | 5 | 0.000±0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 115.8 |
| mappo | L1 | 5 | 0.000±0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 115.9 |
| mappo | L3 | 5 | 0.000±0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 114.9 |

## Criterion checks

### C1 (classical sanity floor): **FAIL**
L0 classical QoS=0.890 (> 0.9 required: FAIL) | L3 classical QoS=0.554 (< 0.6 required: PASS)

### C2 (Concerto v2 beats classical + MAPPO at L3): **FAIL**
v2-classical gap=-0.101 (>0.1 required: FAIL, p=0.000: PASS) | v2-mappo gap=+0.453 (>0.05 required: PASS, p=0.000: PASS)

### C3 (no function collapse, dwell ≥ 0.05): **FAIL**
60 cells below floor 0.05: e.g. ['mappo/L0/seed42/detect=0.000', 'mappo/L0/seed42/track=0.000', 'mappo/L0/seed42/comm=0.000']

### C4 (Concerto faster than MAPPO): **FAIL**
v2 wallclock=111.4s / mappo wallclock=114.9s = 0.97 (< 0.7 required: FAIL)

## Decision tree

**0/4 PASS → retreat to Path A** (C1+C0 → IEEE TAES) per EAAI_RESEARCH_PLAN.md §9.

## Per-seed values (QoS aggregate)

| Method | Difficulty | Seed | QoS agg | n_rl | n_classical |
|---|---|---|---|---|---|
| classical | L0 | 42 | 0.898 | 0 | 60 |
| classical | L0 | 43 | 0.890 | 0 | 60 |
| classical | L0 | 44 | 0.911 | 0 | 60 |
| classical | L0 | 45 | 0.882 | 0 | 60 |
| classical | L0 | 46 | 0.868 | 0 | 60 |
| classical | L1 | 42 | 0.625 | 0 | 60 |
| classical | L1 | 43 | 0.617 | 0 | 60 |
| classical | L1 | 44 | 0.637 | 0 | 60 |
| classical | L1 | 45 | 0.609 | 0 | 60 |
| classical | L1 | 46 | 0.594 | 0 | 60 |
| classical | L3 | 42 | 0.544 | 0 | 60 |
| classical | L3 | 43 | 0.547 | 0 | 60 |
| classical | L3 | 44 | 0.601 | 0 | 60 |
| classical | L3 | 45 | 0.551 | 0 | 60 |
| classical | L3 | 46 | 0.527 | 0 | 60 |
| concerto_v1 | L0 | 42 | 0.872 | 120 | 380 |
| concerto_v1 | L0 | 43 | 0.856 | 120 | 380 |
| concerto_v1 | L0 | 44 | 0.873 | 120 | 380 |
| concerto_v1 | L0 | 45 | 0.850 | 120 | 380 |
| concerto_v1 | L0 | 46 | 0.843 | 120 | 380 |
| concerto_v1 | L1 | 42 | 0.523 | 120 | 380 |
| concerto_v1 | L1 | 43 | 0.506 | 120 | 380 |
| concerto_v1 | L1 | 44 | 0.524 | 120 | 380 |
| concerto_v1 | L1 | 45 | 0.500 | 120 | 380 |
| concerto_v1 | L1 | 46 | 0.494 | 120 | 380 |
| concerto_v1 | L3 | 42 | 0.512 | 120 | 380 |
| concerto_v1 | L3 | 43 | 0.508 | 120 | 380 |
| concerto_v1 | L3 | 44 | 0.559 | 120 | 380 |
| concerto_v1 | L3 | 45 | 0.514 | 120 | 380 |
| concerto_v1 | L3 | 46 | 0.498 | 120 | 380 |
| concerto_v2 | L0 | 42 | 0.898 | 0 | 60 |
| concerto_v2 | L0 | 43 | 0.890 | 0 | 60 |
| concerto_v2 | L0 | 44 | 0.911 | 0 | 60 |
| concerto_v2 | L0 | 45 | 0.882 | 0 | 60 |
| concerto_v2 | L0 | 46 | 0.868 | 0 | 60 |
| concerto_v2 | L1 | 42 | 0.452 | 489 | 11 |
| concerto_v2 | L1 | 43 | 0.451 | 489 | 11 |
| concerto_v2 | L1 | 44 | 0.448 | 489 | 11 |
| concerto_v2 | L1 | 45 | 0.444 | 489 | 11 |
| concerto_v2 | L1 | 46 | 0.445 | 489 | 11 |
| concerto_v2 | L3 | 42 | 0.431 | 499 | 1 |
| concerto_v2 | L3 | 43 | 0.442 | 499 | 1 |
| concerto_v2 | L3 | 44 | 0.485 | 499 | 1 |
| concerto_v2 | L3 | 45 | 0.461 | 499 | 1 |
| concerto_v2 | L3 | 46 | 0.448 | 499 | 1 |
| mappo | L0 | 42 | 0.000 | 500 | 0 |
| mappo | L0 | 43 | 0.000 | 500 | 0 |
| mappo | L0 | 44 | 0.000 | 500 | 0 |
| mappo | L0 | 45 | 0.000 | 500 | 0 |
| mappo | L0 | 46 | 0.000 | 500 | 0 |
| mappo | L1 | 42 | 0.000 | 500 | 0 |
| mappo | L1 | 43 | 0.000 | 500 | 0 |
| mappo | L1 | 44 | 0.000 | 500 | 0 |
| mappo | L1 | 45 | 0.000 | 500 | 0 |
| mappo | L1 | 46 | 0.000 | 500 | 0 |
| mappo | L3 | 42 | 0.000 | 500 | 0 |
| mappo | L3 | 43 | 0.000 | 500 | 0 |
| mappo | L3 | 44 | 0.000 | 500 | 0 |
| mappo | L3 | 45 | 0.000 | 500 | 0 |
| mappo | L3 | 46 | 0.000 | 500 | 0 |