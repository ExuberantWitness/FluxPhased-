"""Benchmark S7 fixed mission queue against list reference.

Only the mission backend changes; the detector loop, RNG order, physics and
trainer are identical. Uses CPU while the paper ablation owns the GPU.
"""
import json, time
from pathlib import Path
import torch
from env.gpu.array_face_s7 import EnvConfig, ArrayFaceS7VecEnv, UPAConfig
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config


def rollout(backend, E=16, H=64, reps=5):
    out=[]
    for rep in range(reps):
        cfg=EnvConfig(n_envs=E,horizon=H,active_budget_steps=63,
                      arrival_rate_per_service=.15,device='cpu',seed=20260801,
                      mission_backend=backend)
        env=ArrayFaceS7VecEnv(cfg,physics=default_debug_physics_config(P_jam_W=.1),radar=UPAConfig(),jammer=UPAConfig())
        env.reset(seed=20260801)
        g=torch.Generator().manual_seed(9000+rep)
        # One cell per jammer per step for the first 31/32 steps, then idle;
        # this is budget-valid for both replay passes and keeps inputs identical.
        actions=[]
        for t in range(H):
            cell=torch.zeros(E,2,25)
            if t < 32: cell[:,0, t % 25] = 1.0
            if t < 31: cell[:,1, (t+7) % 25] = 1.0
            jb=torch.randint(0,25,(E,2),generator=g)
            rb=torch.randint(0,25,(E,2),generator=g)
            rs=torch.randint(0,2,(E,2),generator=g)
            actions.append((cell,jb,rb,rs))
        # warmup separate by repeating deterministic actions once
        env.reset(seed=20260801)
        for cell,jb,rb,rs in actions: env.step(cell,jb,rb,rs)
        env.reset(seed=20260801)
        t0=time.perf_counter()
        for cell,jb,rb,rs in actions: env.step(cell,jb,rb,rs)
        out.append(time.perf_counter()-t0)
    return out

if __name__=='__main__':
    torch.set_num_threads(1)
    result={}
    for E in (2,16):
        b=rollout('list',E=E); f=rollout('fixed',E=E)
        result[E]={'list_s':b,'fixed_s':f,'list_mean_s':sum(b)/len(b),'fixed_mean_s':sum(f)/len(f),
                   'list_median_s':sorted(b)[len(b)//2],'fixed_median_s':sorted(f)[len(f)//2],
                   'speedup_mean_x':(sum(b)/len(b))/(sum(f)/len(f)),
                   'speedup_median_x':sorted(b)[len(b)//2]/sorted(f)[len(f)//2]}
        print(json.dumps({E:result[E]},indent=2),flush=True)
    Path('s7_speed_benchmark_fixed_queue.json').write_text(json.dumps(result,indent=2))
    print('wrote s7_speed_benchmark_fixed_queue.json')
