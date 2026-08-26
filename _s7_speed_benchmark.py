"""Isolated before/after S7 rollout speed benchmark.

The before implementation is loaded from a temporary package generated from
commit bf13807^; the after implementation is the current working tree.
Only collect_rollout is timed (no validation, checkpoint, disk logging, or
PPO update). CUDA is deliberately not used while the queued ablation owns it.
"""
from __future__ import annotations
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "_bench_s7_before"
SRC = ROOT / "env/gpu/array_face_s7"
if BENCH.exists():
    shutil.rmtree(BENCH)
shutil.copytree(SRC, BENCH)
for p in BENCH.glob("*.py"):
    s = p.read_text(encoding="utf-8")
    s = s.replace("env.gpu.array_face_s7", "_bench_s7_before")
    p.write_text(s, encoding="utf-8")
# Baseline env/trainer from the parent of the optimization commit.
import subprocess
subprocess.run(["git", "show", "bf13807^:env/gpu/array_face_s7/env.py"],
               check=True, stdout=(BENCH / "env.py").open("wb"))
subprocess.run(["git", "show", "bf13807^:experiments/array_face_s7/learning_repair/trainer_s7.py"],
               check=True, stdout=(BENCH / "trainer_s7.py").open("wb"))
for p in (BENCH / "env.py", BENCH / "trainer_s7.py"):
    s = p.read_text(encoding="utf-8").replace("env.gpu.array_face_s7", "_bench_s7_before")
    p.write_text(s, encoding="utf-8")

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s7 import EnvConfig as AfterEnvConfig, UPAConfig, N_CELLS_S7, N_BEAM_DIRS_S7
from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2
from experiments.array_face_s7.learning_repair.trainer_s7 import S7SelfPlayTrainer as AfterTrainer

specj = [HeadSpec("cell", "bernoulli", N_CELLS_S7, bernoulli_logit_bias=-3.0),
         HeadSpec("beam", "categorical", N_BEAM_DIRS_S7)]
specr = [HeadSpec("beam", "categorical", N_BEAM_DIRS_S7),
         HeadSpec("svc", "categorical", 2)]
manifest = ROOT / "experiments/array_face_s1/manifests/ppo_train.json"
train_seeds = [e["seed"] for e in json.loads((manifest).read_text())["entries"]]
physics = default_debug_physics_config(P_jam_W=0.1)

# Baseline package imports its own constants/env/trainer.
spec = importlib.util.spec_from_file_location("_bench_s7_before.trainer_s7", BENCH / "trainer_s7.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
BeforeTrainer = mod.S7SelfPlayTrainer
BeforeEnvConfig = importlib.import_module("_bench_s7_before").EnvConfig


def make_cfg(seed, E, H):
    return S2PPOConfigV2(
        profile="array_face_s7_v1", iterations=1, n_envs=E, horizon=H,
        actor_lr=3e-5, critic_lr=1e-3, target_kl=0.02, per_head_entropy=True,
        entropy_coef_per_head={"cell": 2e-2, "beam": 5e-3, "svc": 1e-2},
        entropy_anneal_frac_per_head={"cell": 0.7, "beam": 0.9, "svc": 0.5},
        use_privileged_critic=True, privileged_value_coef=0.5, distill_coef=0.1,
        seed=seed, train_seed=seed, device="cpu")


def make_env_cfg(cls, seed, E, H):
    return cls(n_envs=E, horizon=H, n_services=2, dt=1.0,
               active_budget_steps=63, duty_budget=1.0,
               arrival_rate_per_service=0.15, mission_tau_window=6,
               detects_required=1, potential_coef=0.05, gamma=0.99,
               device="cpu", seed=seed)


def run_case(Trainer, EnvCls, E, H, seed=20260801, reps=4):
    cfg = make_cfg(seed, E, H)
    ec = make_env_cfg(EnvCls, seed, E, H)
    kw = dict(cfg=cfg, env_cfg=ec, physics=physics, radar=UPAConfig(), jammer=UPAConfig(),
              train_seeds=train_seeds, manifest_path=manifest,
              out_dir=ROOT / "_bench_output", jammer_specs=specj, radar_specs=specr)
    # Use a throwaway trainer; weights are irrelevant for environment timing.
    tr = Trainer(**kw)
    tr._assign_scenarios_and_reset()
    # Warm up once outside measurements.
    tr.collect_rollout()
    times = []
    for _ in range(reps):
        tr._assign_scenarios_and_reset()
        t0 = time.perf_counter()
        tr.collect_rollout()
        if tr.cfg.device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def summarize(label, times, E, H):
    # discard the largest outlier only for the robust central estimate, but
    # report all samples so the result is auditable.
    return {"label": label, "E": E, "H": H, "samples_s": times,
            "median_s": sorted(times)[len(times)//2],
            "mean_s": sum(times)/len(times),
            "per_env_step_ms": sum(times)/len(times)/(E*H)*1000}

if __name__ == "__main__":
    torch.set_num_threads(1)
    all_out = []
    for E, H, reps in [(2, 64, 5), (16, 64, 3)]:
        before = summarize("before", run_case(BeforeTrainer, BeforeEnvConfig, E, H, reps=reps), E, H)
        after = summarize("after", run_case(AfterTrainer, AfterEnvConfig, E, H, reps=reps), E, H)
        speedup_mean = before["mean_s"] / after["mean_s"]
        speedup_median = before["median_s"] / after["median_s"]
        print(json.dumps({"before": before, "after": after,
                          "speedup_mean_x": speedup_mean,
                          "speedup_median_x": speedup_median}, indent=2), flush=True)
        all_out.append({"before": before, "after": after,
                        "speedup_mean_x": speedup_mean,
                        "speedup_median_x": speedup_median})
    (ROOT / "s7_speed_benchmark.json").write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print("wrote s7_speed_benchmark.json", flush=True)
    shutil.rmtree(BENCH, ignore_errors=True)
