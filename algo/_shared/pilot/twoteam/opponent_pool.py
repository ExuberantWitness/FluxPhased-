"""Two-team opponent pool with PFSP sampling for league self-play.

Design rationale (vs reusing `algo/_shared/self_play/opponent_pool.py`):
  - Existing pool is bound to FluxLeague's team/role/generation schema and
    requires `current_policy_id` to be a registered record. For twoteam league,
    the current training AC is in-memory (not a pool record), and twoteam is
    symmetric (no team filter), so the API mismatch is large.
  - We keep the AlphaStar f_hard(x) = (1-x)^p PFSP weighting + EMA update
    (validated in the existing pool) and drop the team/role coupling.

Pool contents can be heterogeneous ("kind" field):
  - "rule"      : StrongRule (factory returns a TwoTeamStrongRuleCommander)
  - "extreme"   : ExtremeCommander (factory returns the ExtremeCommander)
  - "script"    : candidate-exploit script (factory returns the script commander)
  - "checkpoint": a torch checkpoint path (load via TwoTeamCommanderActorCritic)

Win-rate semantics:
  Each record tracks `win_rate_vs_current` = EMA of how often the *current
  training agent* beats this opponent. Unknown (None) treated as "hardest"
  per AlphaStar PFSP — so newly-added self-snapshots get evaluated first.
"""

from __future__ import annotations
import os
import json
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Any


KINDS = ("rule", "extreme", "script", "checkpoint")


@dataclass
class PolicyRecord:
    """One entry in the twoteam opponent pool."""
    name: str                                              # unique id, e.g. "strong_rule", "self/iter050"
    kind: str                                              # one of KINDS
    checkpoint_path: Optional[str] = None                  # required for kind="checkpoint"
    factory: Optional[Callable[[], Any]] = None            # required for kind in {rule, extreme, script}
    win_rate_vs_current: Optional[float] = None            # EMA; None = unknown → treated as hardest
    games_played_vs_current: int = 0
    is_self_snapshot: bool = False
    created_at_iter: int = 0                               # for self-snapshots, the league iter

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"PolicyRecord.kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind == "checkpoint":
            if self.checkpoint_path is None:
                raise ValueError(f"checkpoint record {self.name!r} needs checkpoint_path")
        else:
            if self.factory is None:
                raise ValueError(f"{self.kind} record {self.name!r} needs factory")


class TwoTeamOpponentPool:
    """PFSP pool for twoteam league. Current training AC is NOT in the pool."""

    def __init__(
        self,
        population_cap: int = 30,
        pfsp_hardness_p: float = 1.0,
        ema_alpha: float = 0.1,
        rng_seed: Optional[int] = None,
        pfsp_var_mix: float = 0.0,
        ema_var_uniform_floor: float = 0.0,
    ):
        self.population_cap = int(population_cap)
        self.pfsp_hardness_p = float(pfsp_hardness_p)
        self.ema_alpha = float(ema_alpha)
        self.pfsp_var_mix = float(pfsp_var_mix)
        self.ema_var_uniform_floor = float(ema_var_uniform_floor)
        self.records: Dict[str, PolicyRecord] = {}
        self._self_snapshot_order: List[str] = []   # names of self-snapshots, in creation order
        if rng_seed is not None:
            self._rng = np.random.default_rng(rng_seed)
        else:
            self._rng = np.random.default_rng()

    def add(self, record: PolicyRecord) -> None:
        """Insert a record. Overwrites if name exists."""
        if record.name in self.records:
            existing = self.records[record.name]
            # Preserve win-rate EMA if overwriting with same name
            record.win_rate_vs_current = existing.win_rate_vs_current
            record.games_played_vs_current = existing.games_played_vs_current
            if existing.is_self_snapshot and existing.name in self._self_snapshot_order:
                self._self_snapshot_order.remove(existing.name)
        self.records[record.name] = record
        if record.is_self_snapshot:
            self._self_snapshot_order.append(record.name)
        # Evict if over cap (drop oldest self-snapshot, never evict rule/extreme/script)
        self._maybe_evict()

    def _maybe_evict(self) -> None:
        """Evict oldest self-snapshot if over cap. Never evict script/rule/extreme seeds."""
        while len(self.records) > self.population_cap and len(self._self_snapshot_order) > 1:
            # Keep at least 1 self-snapshot; evict the oldest
            oldest = self._self_snapshot_order[0]
            if len(self._self_snapshot_order) <= 1:
                break
            self._self_snapshot_order.pop(0)
            if oldest in self.records:
                del self.records[oldest]

    def sample_pfsp(self, exclude: Optional[str] = None) -> Optional[PolicyRecord]:
        """Sample one opponent, weighted by f_hard ⊕ f_var. None wr → hardest.

        PFSP weighting (AlphaStar Nature 2019, §4.5):
          f_hard(known_wr) = (1 - known_wr)^p                    # 夯难对手
          f_var(known_wr)  = known_wr * (1 - known_wr)           # ~50% wr 对手最高学习价值
          weights = (1 - var_mix) · f_hard + var_mix · f_var
          var_mix=0 → 旧行为 (纯 f_hard);var_mix=0.5 → 推荐 default.

        Health gate: 若池已知 wr 的方差 < ema_var_uniform_floor(默认 0.0=关),
        强制均匀采样一轮,防 PFSP 塌到单一对手。

        Args:
            exclude: optional name to exclude.

        Returns:
            One PolicyRecord, or None if pool is empty.
        """
        candidates = [r for r in self.records.values() if r.name != exclude]
        if not candidates:
            return None
        wr = np.array([
            (np.nan if r.win_rate_vs_current is None else r.win_rate_vs_current)
            for r in candidates
        ])
        unknown = np.isnan(wr)
        known_wr = np.where(unknown, 0.0, wr)

        # Health gate: ema_var < floor → 强制均匀
        if self.ema_var_uniform_floor > 0.0 and known_wr.size >= 2:
            known_only = wr[~unknown]
            if known_only.size >= 2 and float(np.var(known_only)) < self.ema_var_uniform_floor:
                probs = np.ones(len(candidates)) / len(candidates)
                idx = int(self._rng.choice(len(candidates), p=probs))
                return candidates[idx]

        f_hard = (1.0 - known_wr) ** self.pfsp_hardness_p
        f_var = known_wr * (1.0 - known_wr)
        weights = (1.0 - self.pfsp_var_mix) * f_hard + self.pfsp_var_mix * f_var
        if unknown.any():
            weights[unknown] = weights.max() if weights.max() > 0 else 1.0
        total = weights.sum()
        if total < 1e-12:
            probs = np.ones(len(candidates)) / len(candidates)
        else:
            probs = weights / total
        idx = int(self._rng.choice(len(candidates), p=probs))
        return candidates[idx]

    def update_win_rate(self, name: str, win: bool) -> None:
        """EMA update: first observation replaces (not blended toward 0.5).

        This mirrors the validated fix in `self_play/opponent_pool.py:update_win_rate`
        — the original "init=0.5 + EMA" collapsed PFSP priorities to uniform.
        """
        if name not in self.records:
            return
        rec = self.records[name]
        if rec.win_rate_vs_current is None:
            rec.win_rate_vs_current = float(win)
        else:
            a = self.ema_alpha
            rec.win_rate_vs_current = (1.0 - a) * rec.win_rate_vs_current + a * float(win)
        rec.games_played_vs_current += 1

    def get(self, name: str) -> Optional[PolicyRecord]:
        return self.records.get(name)

    def all_records(self) -> List[PolicyRecord]:
        return list(self.records.values())

    def num_records(self) -> int:
        return len(self.records)

    def ema_variance(self) -> float:
        """Variance of known win rates. Low → PFSP may be stuck uniform (R3 risk)."""
        known = [r.win_rate_vs_current for r in self.records.values()
                 if r.win_rate_vs_current is not None]
        if len(known) < 2:
            return 0.0
        return float(np.var(known))

    def summary(self) -> Dict[str, Dict]:
        """Return a JSON-friendly snapshot of pool state (for logging)."""
        out = {}
        for name, r in self.records.items():
            out[name] = {
                "kind": r.kind,
                "is_self_snapshot": r.is_self_snapshot,
                "win_rate_vs_current": r.win_rate_vs_current,
                "games_played": r.games_played_vs_current,
            }
        return out

    def save_metadata(self, path: str) -> None:
        """Save pool metadata (factories not serialized — those must be re-registered on load)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = {
            "pfsp_hardness_p": self.pfsp_hardness_p,
            "ema_alpha": self.ema_alpha,
            "population_cap": self.population_cap,
            "records": {
                name: {
                    "kind": r.kind,
                    "checkpoint_path": r.checkpoint_path,
                    "win_rate_vs_current": r.win_rate_vs_current,
                    "games_played_vs_current": r.games_played_vs_current,
                    "is_self_snapshot": r.is_self_snapshot,
                    "created_at_iter": r.created_at_iter,
                }
                for name, r in self.records.items()
            },
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)


def build_opponent_action_fn(record: PolicyRecord, device: str = "cuda"):
    """Construct a per-team `action_fn(env, team) -> action_dict` for a pool record.

    For checkpoint records: load AC, wrap with deterministic forward.
    For rule/extreme/script records: call `factory()`, use its `.get_action(env, team)`.

    Returns:
        action_fn callable + a `cleanup()` callable (to free GPU mem between snapshots).
    """
    if record.kind == "checkpoint":
        from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
        ac = TwoTeamCommanderActorCritic().to(device)
        ckpt = torch.load(record.checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("ac_state", ckpt)
        ac.load_state_dict(state)
        ac.eval()

        @torch.no_grad()
        def action_fn(env, team: int):
            obs = env.get_obs()
            o = obs["obs"][:, team]
            detect_t = env.get_detect_list()[:, team]   # WP-3 M0/M1
            priv = obs["privileged"][:, team]
            action, _ = ac.get_action_for_env(o, detect_t, priv, deterministic=True)
            return action

        def cleanup():
            nonlocal ac
            del ac
            torch.cuda.empty_cache()

        return action_fn, cleanup

    else:
        commander = record.factory()

        def action_fn(env, team: int):
            return commander.get_action(env, team)

        def cleanup():
            pass

        return action_fn, cleanup
