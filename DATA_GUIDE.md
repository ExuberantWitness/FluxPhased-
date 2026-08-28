# FluxPhased 数据使用指南(供其他 Agent / 协作者)

> **目的**:一次性说清 S1→S7 全部已计算数据的位置、格式、用法、初步结论与已知坑。
> **日期**:2026-08-28 · **状态**:S6/S7 全部完成;R5-lite 训练中(预计 08-29 上午出数)。
> **分支**:`g3-bsta/array-face-s1`(主分支,一切论文数据在此)· 性能实验在 `s7/perf-aggressive`(勿用于论文数字)。
> 论文骨架与图表清单见 [paper/main.tex](paper/main.tex) 与 [paper/FIGURE_PLAN.md](paper/FIGURE_PLAN.md)。

---

## 1. 核心结论速查(全部为最终数字)

### 主结论:攻击方数量翻倍 → 防御遏制崩塌到约 1/3(且跨种子复现)

| 指标 | S6(1干扰方 vs 2雷达,3种子) | S7(2v2,3种子,各收敛至 3000 iter) |
|---|---:|---:|
| h2h(学习 vs 学习)drop | 0.0888 ± 0.0053 | **0.3366 ± 0.0143**(3.8×) |
| jam_vs_sweep(火力)drop | 0.2751 ± 0.0110 | **0.5294 ± 0.0215**(同预算 +92%) |
| rad_vs_idle drop | 0.011 | 0.0194 ± 0.0008(雷达能力无损) |
| 自然地板 sweep_vs_idle | 0.106 | 0.1174 ± 0.0011 |
| **floor 调整后中和率 η** | **63.7% ± 0.7%** | **23.0% ± 1.1%** |

η 公式:`η = 1 − (h2h − rad_idle_drop) / (jam_vs_sweep − floor)`。
η 的种子间极差仅 ±1.1pp —— 这是论文头号数字。

### 机制分解(共址消融,R3)

| 几何 | h2h | η | 效应归因 |
|---|---:|---:|---|
| S6 单干扰方 | 0.0888 | 63.7% | 基线 |
| 共址双干扰方(+60,+60) | 0.2927 ± 0.0119 | 28.4% | **数量主效应**(63.7→28.4) |
| 交叉火力(±60 vs ±20) | 0.3390 ± 0.0050 | 24.2% | **几何次级增益**(28.4→24.2) |

结论:**"单束压制即全部机制"的朴素读法被证伪**;双源持续压力+预算分割是主体,交叉火力是次级。

### 其他已确立结论

1. **收敛性**:均衡在 ~iter 1700 收敛;2000–3000 窗口四视图全部平坦(±0.005)。**1000-iter 预算会低估 h2h 约 25–30%**(0.26 vs 0.34)——引用任何数字必须注明预算。
2. **非循环**:三段续训控制(退火冻结)证明后期上升是真实收敛而非非传递循环 → 无需联赛机制解释 S7 动力学。
3. **适应权衡(R4)**:双打训练的雷达被单干扰方利用(j1_only 0.2086 ± 0.0745,三种子全部存在但幅度种子依赖 0.12–0.26)。种子 01 内部:j1_only 随专精化翻倍(0.12→0.24)。
4. **S6 跨机制不变式**:snr=22 与 snr=12 下 η 均约 60–64%(S6 的三种子复现)。
5. **行为指纹**:雷达 az±30° 扇区分工(贪心=随机一致);干扰对烧光全部预算(vs S6 节流);贪心模式坍缩 idle 为跨阶段不变式。

### 进行中(R5-lite,预计 08-29 上午)

单干扰方暴露比例 mix ∈ {25%, 50%, 75%} × 2000 iter,判读"对手类混合能否恢复单打鲁棒性"。参照点 = 0% mix @2000 iter(seed01:h2h 0.3282 / j1 0.2532 / η 20.2%)。

---

## 2. 数据目录地图(全部相对仓库根 `E:\DATA\vscode\FluxPhased`)

### S7(2v2,论文主数据)

| 目录(`experiments/array_face_s7/learning_repair/` 下) | 内容 | 用途 |
|---|---|---|
| `s7_selfplay_output_seed2026080{1,2,3}/` | 三种子 0–1000 iter(正常退火) | 早期轨迹、预算敏感性;**非收敛数字** |
| `s7_continue_output_seed20260801/` | 种子01 续至 2000 + `final_eval.json` | **0% mix 参照点**、收敛中段 |
| `s7_continue2_output_seed20260801/` | 种子01 续至 3000 + `final_eval.json` | **种子01 权威收敛数字** |
| `s7_seed02_cont_output_seed20260802/` | 种子02 → 3000 + `final_eval.json` | 三种子统计 |
| `s7_seed03_cont_output_seed20260803/` | 种子03 → 3000 + `final_eval.json` | 三种子统计 |
| `s7_ablation_output_seed20260811/` | 共址消融 2000 iter + 终评 | 机制分解 |
| `s7_r5_mix0p{25,5,75}_output_seed2026082{1,2,3}/` | R5 混合训练(生成中) | R5 章节 |

### S6(1v2 基线,`experiments/array_face_s6/learning_repair/` 下)

| 目录 | 内容 | 注意 |
|---|---|---|
| `s6_selfplay_output_seed20260729/` + final_eval.json | snr=**22**(stale 覆盖,诚实框) | 机制对照用,**勿混入 snr=12 统计** |
| `s6_selfplay_output_seed2026073{0,1}/` + final_eval.json | snr=12 复现对 | 与 S7 同机制对比的正确基线 |

### 每个输出目录的文件结构(通用)

```
<dir>/
  train_metrics.jsonl   # 每迭代一行:rollout_drop/success、双方熵、clip_frac、cumulative_transitions
  val_metrics.jsonl     # 每 val_every 轮:iter, h2h_drop, jam_vs_sweep_drop, rad_vs_idle_success, j1_only_drop(S7), elapsed_s
  run.log               # PowerShell 重定向,UTF-16LE!人类可读的验证行(iter X h2h=... j_ent=...)
  selfplay_latest.pt    # 最终 checkpoint(torch;未入 git,只在本地)
  final_eval.json       # 全协议终评(存在即权威)
  final_eval_run.log    # 终评日志(UTF-16)
```

### 关键脚本(仓库根)

| 脚本 | 用法 | 说明 |
|---|---|---|
| `_s7_final_eval.py` | `--seed N --out-dir DIR [--jammer-az '+60,+60'] [--device cpu]` | 全协议终评:64 验证种子 × 3 action seeds(4242/777/31337)× 4 视图 + 地板。**几何消融的目录必须传 `--jammer-az`**(见坑#4) |
| `_s7_analyze_traj.py` | 直接运行 | 解析所有 S7 run.log → 200-iter 季度统计表(内置去重与 UTF-16 解码) |
| `_s7_policy_extract.py` | `python _s7_policy_extract.py <seed> [cpu|cuda] [greedy|stochastic]` | 行为提取:波束直方图、svc 分工、交叉分配 JNR 矩阵 |
| `_s7_plot_curves_full.py` | 直接运行 | 图2:3000-iter 四视图三段曲线(输出 `experiments/array_face_s7/arms_race_curves_s7_full.png`) |
| `_s7_merge_stats.py` | 直接运行 | S7 1000-iter 协议三种子合并(注意:3000 收敛合并需用下文路径) |
| `_s7_speed_benchmark.py` | 直接运行 | 性能基准(论文不需要) |

---

## 3. 如何使用(加载模式示例)

### 3.1 读终评并计算 η

```python
import json, statistics
def load_eval(path):
    d = json.load(open(path)); a = (4242, 777, 31337)
    h  = statistics.mean([d[f"aseed_{x}"]["h2h_drop"] for x in a])
    j  = statistics.mean([d[f"aseed_{x}"]["jam_vs_sweep_drop"] for x in a])
    j1 = statistics.mean([d[f"aseed_{x}"]["j1_only_drop"] for x in a])
    ri = 1 - statistics.mean([d[f"aseed_{x}"]["rad_vs_idle_success"] for x in a])
    f  = d["sweep_vs_idle_floor"]["drop"]
    eta = 100 * (1 - (h - ri) / (j - f))
    return dict(h2h=h, jvs=j, j1=j1, rad_idle=ri, floor=f, eta=eta)

base = "experiments/array_face_s7/learning_repair/"
s7 = [load_eval(base + f"s7_continue2_output_seed20260801/final_eval.json"),
      load_eval(base + f"s7_seed02_cont_output_seed20260802/final_eval.json"),
      load_eval(base + f"s7_seed03_cont_output_seed20260803/final_eval.json")]
for k in ("h2h", "jvs", "eta"):
    v = [x[k] for x in s7]
    print(k, f"{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}")
```

### 3.2 三种子 3000-iter 合并(权威路径)

```python
paths = [
  "s7_continue2_output_seed20260801",      # seed 01 的 3000-iter 在 continue2
  "s7_seed02_cont_output_seed20260802",
  "s7_seed03_cont_output_seed20260803",
]
```

### 3.3 训练轨迹季度统计

直接运行 `_s7_analyze_traj.py`(已处理:UTF-16 解码、resume 重复段去重、续训目录)。
若自行解析 `run.log`:验证行格式
`iter <N> h2h_drop=<v> jam_vs_sweep=<v> rad_vs_idle_succ=<v> j1_only=<v> j_ent=<v> r_ent=<v>`。

---

## 4. 已知坑(必须读)

1. **run.log / final_eval_run.log 是 UTF-16LE**(PowerShell 重定向所致)。
   读取:`open(p,'rb').read().decode('utf-16', errors='replace')`,失败再退 utf-8。
2. **train_metrics.jsonl 含 resume 重复段**:机器休眠中断后 `--resume` 会重写最后检查点之后的迭代号。
   **任何统计必须按 `iteration` 去重(保留最后一次出现)**;行数 ≠ 迭代数。
3. **1000-iter 与 3000-iter 数字不可混用**:1000-iter 协议低估收敛 h2h 约 25–30%。
   引用规则:S7 主表只用 3000-iter 终评;1000-iter 仅作预算敏感性讨论。
4. **消融终评必须传几何参数**:共址消融目录若不传 `--jammer-az '+60,+60'` 会用默认交叉火力几何评估 → 数字作废。
   历史事故文件 `s7_ablation_output_seed20260811/final_eval_wrong_default_geometry.json` 保留作审计,**勿引用**;正确数字在 `final_eval.json`(2026-08-28 校正)。
5. **S6 种子 20260729 是 snr=22**(诚实框),snr=12 基线是 20260730/31。S6 vs S7 对比一律用 snr=12。
6. **j1_only 种子间方差大**(0.12–0.26):论文如实报告"全部存在、幅度种子依赖",不要只用均值。
7. **checkpoint(.pt)不在 git 里**(仅本地):复现训练用 `run_s7_selfplay.py --resume`;评估只需 final_eval.json。
8. **验证种子与训练种子分离**:64 train seeds(`ppo_train.json`)+ 64 validation seeds(`checkpoint_validation.json`),都在 `experiments/array_face_s1/manifests/`。终评用全部 64 验证种子;训练中验证只用前 16。
9. **性能分支数字勿入论文**:`s7/perf-aggressive` 的提速(1.4–1.5×)经过审计但改变 logits 微末(2.4e-7);论文数字全部出自主分支代码。
10. **机器休眠频发**:训练链有自愈(watchdog 每 20 分钟 + 原子检查点),墙钟时间含中断;判断完成以链日志 `ALL ... DONE` 标记为准。

---

## 5. 正在生成 / 待做

| 项 | 状态 | 说明 |
|---|---|---|
| R5-lite(3 比例 × 2000 iter + 终评) | 🔄 训练中,预计 08-29 上午 | 完成标记:`s7_r5.log` 出现 `ALL R5 DONE`;每个 mix 目录将有 final_eval.json;判读见 §1 末 |
| 论文图 1/3/4/5 | 待绘制 | 数据全齐,见 FIGURE_PLAN.md 的数据源列 |
| refs.bib | 待填充 | 12 篇文献清单在 main.tex 的 Related Work 注释中 |
| 摘要 / V-E 章节 | 等 R5 | 两种结尾的措辞已预写在 main.tex 注释里 |

## 6. 复现入口(速查)

```bash
# 环境
C:/Users/zhang/.conda/envs/fluxphased/python.exe   # torch 2.x, CUDA
export PYTHONPATH=E:/DATA/vscode/FluxPhased

# 回归门(改动任何 env/trainer 代码后必跑)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD pytest -q \
  tests/array_face/test_array_face_s7.py tests/array_face/test_array_factor_s7.py
# S6 门:tests/array_face/test_array_face_s6.py test_array_factor_s6.py
# 预期:S7 18/18,S6 17/17
```
