import json
import statistics
from pathlib import Path

rows = [json.loads(l) for l in Path(
    'experiments/array_face_s7/learning_repair/s7_continue2_output_seed20260801/val_metrics.jsonl'
).read_text().splitlines() if l.strip()]
by_iter = {r['iter']: r for r in rows}
rows = [by_iter[i] for i in sorted(by_iter)]
w3 = [r for r in rows if 2400 < r['iter'] <= 2600]
print('window3 iters:', [r['iter'] for r in w3])
print('window3 jvs values:', [round(r['jam_vs_sweep_drop'], 6) for r in w3])
print('window3 jvs mean:', statistics.mean(r['jam_vs_sweep_drop'] for r in w3))
print('window3 h2h mean:', statistics.mean(r['h2h_drop'] for r in w3))
