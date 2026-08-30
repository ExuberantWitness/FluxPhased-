"""IPPO vs MAPPO comparison at the 2000-iter protocol point."""
import json
import statistics as st
from pathlib import Path

base = Path('experiments/array_face_s7/learning_repair')
a = [4242, 777, 31337]

d = json.loads((base / 's7_ippo_output_seed20260901/final_eval.json').read_text())
h = [d[f'aseed_{x}']['h2h_drop'] for x in a]
j = [d[f'aseed_{x}']['jam_vs_sweep_drop'] for x in a]
ri = [1 - d[f'aseed_{x}']['rad_vs_idle_success'] for x in a]
j1 = [d[f'aseed_{x}']['j1_only_drop'] for x in a]
f = d['sweep_vs_idle_floor']['drop']
hm, jm, rim, j1m = st.mean(h), st.mean(j), st.mean(ri), st.mean(j1)
eta = 100 * (1 - (hm - rim) / (jm - f))
print('IPPO  2000it: h2h %.4f  jvs %.4f  idle %.4f  j1 %.4f  floor %.4f  eta %.1f%%'
      % (hm, jm, rim, j1m, f, eta))
print('IPPO  j1/jvs = %.3f   j1/h2h = %.3f' % (j1m / jm, j1m / hm))

m = json.loads((base / 's7_continue_output_seed20260801/final_eval.json').read_text())
h2 = [m[f'aseed_{x}']['h2h_drop'] for x in a]
j2 = [m[f'aseed_{x}']['jam_vs_sweep_drop'] for x in a]
ri2 = [1 - m[f'aseed_{x}']['rad_vs_idle_success'] for x in a]
f2 = m['sweep_vs_idle_floor']['drop']
hm2, jm2, rim2 = st.mean(h2), st.mean(j2), st.mean(ri2)
eta2 = 100 * (1 - (hm2 - rim2) / (jm2 - f2))
print('MAPPO 2000it: h2h %.4f  jvs %.4f  idle %.4f  floor %.4f  eta %.1f%%'
      % (hm2, jm2, rim2, f2, eta2))
print('ratios  IPPO/MAPPO:  h2h %.2f  jvs %.2f  eta %.2f'
      % (hm / hm2, jm / jm2, eta / eta2))
