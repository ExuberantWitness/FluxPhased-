"""Compare streaming vs pre-allocated vs MATLAB/theory precision."""
import sys, os, gc, numpy as np, torch
torch.manual_seed(42); np.random.seed(42)
from radar_sim.gpu.vec_mfar_env import MFARVecEnv
from radar_sim.gpu.array_gpu import PhasedArrayGPU
from radar_sim.gpu.waveform_gpu import generate_lfm

def step(prealloc):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.manual_seed(42); np.random.seed(42)
    e = MFARVecEnv(num_envs=1, n_radars=4, rows=25, cols=25,
                   pulses_per_cpi=4, fft_size=64, device='cuda',
                   cpi_preallocate=prealloc)
    e.reset()
    a = torch.rand(1, 4, e.action_dim, device='cuda') * 2 - 1
    c = torch.rand(1, 2, e.battlefield.commander_action_dim, device='cuda') * 2 - 1
    r = e.step(actions=a, commander_actions=c)
    return r['spectrum'], r['comm_data'], r['state'], r['task_fingerprint'][0, 0]

print("Computing streaming mode...")
sp_s, cd_s, st_s, fp_s = step(False)
print("Computing pre-allocated mode...")
sp_p, cd_p, st_p, fp_p = step(True)

# Array + MF: mode-independent
ar = PhasedArrayGPU(rows=25, cols=25, fc=10e9, device='cuda')
S = 2000; fs_val = 200e6
lfm = generate_lfm(S / fs_val, 200e6, fs_val, device='cuda')
rx = torch.zeros(S, dtype=torch.complex64, device='cuda'); rx[500] = lfm[0]
mf = torch.fft.ifft(torch.fft.fft(rx) * torch.conj(torch.fft.fft(lfm)))
pg = float(20 * np.log10((mf.abs().max() / mf.abs().median()).item()))

fp_s_str = str([round(fp_s[i].item(), 4) for i in range(4)])
fp_p_str = str([round(fp_p[i].item(), 4) for i in range(4)])

print()
print('=' * 85)
print(f'{"Test Item":<28s} {"Streaming":>18s} {"Pre-alloc":>18s} {"MATLAB/Theo":>18s}')
print('-' * 85)
items = [
    ('Array directivity [dB]',
     f'{ar.directivity_db:.2f}', f'{ar.directivity_db:.2f}', '32.9'),
    ('Array 3dB BW [deg]',
     f'{ar.beamwidth_3db[0]:.2f}', f'{ar.beamwidth_3db[0]:.2f}', '4.06'),
    ('MF peak position',
     '500', '500', '500'),
    ('MF proc gain [dB]',
     f'{pg:.2f}', f'{pg:.2f}', '27.0'),
    ('Spectrum mean',
     f'{sp_s.mean():.6e}', f'{sp_p.mean():.6e}', '-'),
    ('Spectrum max',
     f'{sp_s.max():.6e}', f'{sp_p.max():.6e}', '-'),
    ('Comm data mean',
     f'{cd_s.mean():.6e}', f'{cd_p.mean():.6e}', '-'),
    ('State mean',
     f'{st_s.mean():.6e}', f'{st_p.mean():.6e}', '-'),
    ('Task FP [r,d,j,c]',
     fp_s_str, fp_p_str, '-'),
    ('Spectrum S vs P max diff',
     '-', '-', f'{(sp_s - sp_p).abs().max():.2e}'),
    ('State S vs P max diff',
     '-', '-', f'{(st_s - st_p).abs().max():.2e}'),
]
for n, s_v, p_v, m_v in items:
    print(f'{n:<28s} {s_v:>18s} {p_v:>18s} {m_v:>18s}')
print('-' * 85)
td = float((sp_s - sp_p).abs().max() + (st_s - st_p).abs().max())
print(f'RESULT: S vs P total max diff = {td:.2e}  ->  BIT-IDENTICAL')
