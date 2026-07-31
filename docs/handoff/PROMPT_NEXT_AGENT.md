# PROMPT — 给接手 agent 的复现操作手册

> **使用方法**:在新机器的 Claude Code 会话里,把本文件**整篇**作为第一条 user message 粘贴进去,然后让 agent 按 step 顺序执行。
>
> **本文件位置**:`docs/handoff/PROMPT_NEXT_AGENT.md`(在 `g3-bsta/array-face-s1` 分支)
> **配套文档**:[`HANDOFF_20260731.md`](./HANDOFF_20260731.md)(985 行,完整背景)

---

## 你的角色

你是 **Array-Face 项目**的接手 agent。前一个 agent 在另一台机器上做完了这些事:

1. ✅ S1(radar 5-cell 1D ULA + Rx AF)完成 + multi-seed PPO 跑完(3 broke out / 3 stuck)
2. ✅ S1 完整快照 + 实验 + 报告推送到 `g3-bsta/array-face-s1` 分支
3. ✅ S2(jammer 5-cell 1D ULA + Tx AF + MultiDiscrete)代码写完
4. ⚠️ **S2 有一个物理 bug 未修**(P_jam_W 语义冲突,见 handoff §10.1)
5. ✅ handoff 文档 + memory snapshot(43 个 .md)推送完毕

**你的任务**:按本 prompt 走完 6 个 step,完成 S2 修复 + 复现 + 验证 + 报告,**然后停下与用户讨论是否进 S3**(不要自动进)。

---

## 必读与必守(违反就回退重来)

### 必读(读不完不准动)

1. **本 prompt 全文**(你正在读)
2. **`docs/handoff/HANDOFF_20260731.md`**(985 行,完整背景)— 至少读完 §0 / §8 / §10 / §10.7 / §12
3. **`docs/handoff/memory-snapshot-20260731/MEMORY.md`**(43 行索引)— 至少扫一遍,知道有哪些历史 lesson

### 必守(硬约束,违反会激怒用户)

- **所有面向用户的文字必须中文**(commit message 可英文)— memory `chinese_only_responses.md`
- **不准调用 codex MCP**(在评审、头脑风暴、计划环节)— memory `feedback_no_codex.md`
- **不动 lite / fast-work / forensic / handoff / main 分支** — handoff §8.4
- **每次 push 必须先问用户授权**(单次授权 ≠ 永久)— handoff §8.3
- **不要 force-push、不要 rewrite history、不要 --no-verify** — handoff §8.3
- **每个 phase 必 multi-seed ≥ 3,跑完先 plot 与用户讨论** — handoff §8.5

---

## 输入条件(假设新机器已具备)

接手 agent 启动前,**确认**以下条件(任一缺失先停下问用户):

- [ ] 新机器装了 conda(或 miniconda),能 `conda create`
- [ ] NVIDIA GPU(最好 RTX PRO 6000 / Blackwell sm_120;否则 PyTorch 版本要调整)
- [ ] GitHub SSH key 已注册到 `ExuberantWitness` 账号(verify: `ssh -T git@github.com` 应返回 "Hi ExuberantWitness!")
- [ ] 网络通 GitHub(可能需要代理,见 handoff §5.3)

---

## 执行协议(6 个 step,严格顺序)

> **原则**:每个 step 都有验证条件。验证不过 → 不要进下一步 → 查 handoff §17 FAQ → 还不行就停下问用户。

### Step 0: 加载 memory + 通读 handoff(预计 15 分钟)

#### 0.1 仓库 + worktree

```bash
# 如果新机器还没 clone:
mkdir -p /home/ubuntu/CODE && cd /home/ubuntu/CODE
git clone git@github.com:ExuberantWitness/FluxPhased-.git
cd FluxPhased-
git worktree add /home/ubuntu/CODE/g3-bsta-fastwork g3-bsta/array-face-s1
cd /home/ubuntu/CODE/g3-bsta-fastwork

# 如果已 clone 但分支不对:
git fetch git@github.com:ExuberantWitness/FluxPhased-.git g3-bsta/array-face-s1
git checkout g3-bsta/array-face-s1
git pull  # 如果本地落后
```

**验证**:`git rev-parse HEAD` 应输出 `fcb2e10...` 或更新。

#### 0.2 装 memory

```bash
# 看本机项目路径(决定 memory 装哪)
pwd  # 期望 /home/ubuntu/CODE/g3-bsta-fastwork

# 把 memory 装到 Claude 的 project 目录
# 路径规则:把项目顶层目录(/home/ubuntu/CODE)的 / 换成 -
MEM_DIR="/home/ubuntu/.claude/projects/-home-ubuntu-CODE/memory"
mkdir -p "$MEM_DIR"
cp docs/handoff/memory-snapshot-20260731/*.md "$MEM_DIR/"

# 验证
ls "$MEM_DIR" | wc -l  # 期望: 43
cat "$MEM_DIR/MEMORY.md" | head -3  # 期望: 看到 "Wang 2025 全文参数" 等条目
```

#### 0.3 重启 Claude Code(让 memory 生效)

退出当前 Claude Code 会话,重新启动。新会话的 system context 里应包含:
```
Contents of /home/ubuntu/.claude/projects/-home-ubuntu-CODE/memory/MEMORY.md (user's auto-memory, persists across conversations):
- [Wang 2025 全文参数](wang2025_params.md) — ...
- ...(43 行)
```

**验证**:重启后问 Claude "列出当前 memory 里所有 user feedback 类的条目"。应能列出 3 个:
- `chinese_only_responses`
- `feedback_no_codex`
- `feedback_pool_randomization`

#### 0.4 通读 handoff

```bash
less docs/handoff/HANDOFF_20260731.md
```

**至少读完**:§0 / §8 / §10 / §10.7 / §12。其余可跳读。

**验证**:你能回答:
- S2 物理 bug 是什么?(答:`physics.P_jam_W` 在 physics.py 是 per-cell,在 env_config 是 total,7dB EIRP 偏差)
- 信号灯 GREEN 的 3 个硬门槛?(答:break-out rate ≥ 2/3, mean ∈ [0.17, 0.27], best ≥ 0.18)
- 5 个 protected branches?(答:main / fast-work / forensic / handoff / mfr-lite-fastwork)

→ **答不上来 → 重读 handoff 对应章节,不要进 Step 1**。

---

### Step 1: 跑测试 verify(预计 5 分钟)

```bash
cd /home/ubuntu/CODE/g3-bsta-fastwork
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fluxphased

# 1. S1 测试(已 commit,必 PASS)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD pytest -q \
  tests/array_face/test_array_factor_s1.py \
  tests/array_face/test_array_face_s1.py
# 期望: 18/18 PASS

# 2. S2 测试(修 bug 前应该全 PASS,因为测试用旧语义 P_jam_W=10.0)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD pytest -q \
  tests/array_face/test_array_factor_s2.py \
  tests/array_face/test_array_face_s2.py
# 期望: 23/23 PASS

# 3. lite regression(全 phase 必保持 75/75)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD pytest -q tests/g3_bsta_lite
# 期望: 75/75 PASS
```

**验证**:三个数字都对(18 / 23 / 75)。

**失败处理**:
- pytest 找不到 → FAQ Q1
- ImportError env → FAQ Q2
- S2 测试 FAIL 但你还没动代码 → FAQ Q3,需要先修 bug
- lite regression FAIL → FAQ Q13(紧急,你可能误改了 lite)

---

### Step 2: 修 S2 物理 bug(预计 30 分钟)

**目标**:把 `P_jam_W` 全部统一为**单 cell 功率(per-cell)**,默认 2.0 W(plan §2.2)。

#### 2.1 修改 5 处(按 handoff §10.2)

文件 + 行号(行号可能因前面编辑略有偏移,grep 定位):

```bash
# (1) env/gpu/array_face_s2/env.py L48
#     P_jam_W: float = 10.0 → P_jam_W: float = 2.0
grep -n "P_jam_W.*10.0\|P_jam_W: float" env/gpu/array_face_s2/env.py

# (2) env/gpu/array_face_s2/env.py L77, L101, L152, L284
#     能量预算 = E0_tokens × P_jam_W × N_cells × dt
#     需要在 EnvConfig 加 n_jammer_cells: int = 5 字段
grep -n "E0_tokens.*P_jam_W" env/gpu/array_face_s2/env.py

# (3) experiments/array_face_s2/learning_repair/run_s2_ppo.py L64, L72
#     P_jam_W=10.0 → P_jam_W=2.0
grep -n "P_jam_W" experiments/array_face_s2/learning_repair/run_s2_ppo.py

# (4) tests/array_face/test_array_factor_s2.py L71, L99, L122
#     P_jam_W=10.0 → P_jam_W=2.0
grep -n "P_jam_W" tests/array_face/test_array_factor_s2.py

# (5) tests/array_face/test_array_face_s2.py L18
#     _make_env 默认 P_jam_W=10.0 → P_jam_W=2.0
grep -n "P_jam_W" tests/array_face/test_array_face_s2.py
```

**Edit 顺序**:先改 `EnvConfig`(加 `n_jammer_cells`),再改能量预算公式(4 处),再改默认值,再改 runner 和 tests。

#### 2.2 跑 M0 micro-verify(必 PASS 才能进 2.3)

```python
# 存为 /tmp/m0_verify.py 然后 python /tmp/m0_verify.py
import torch
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s2 import (
    RadarULAConfig, JammerULAConfig, compute_jnr_db_s2,
)

physics = default_debug_physics_config(P_jam_W=2.0)
radar = RadarULAConfig()
jammer = JammerULAConfig()
E = 1
is_jam = torch.ones(E, dtype=torch.bool)

# 主瓣对准
jnr_main = compute_jnr_db_s2(
    physics, radar, jammer,
    jammer_active=is_jam,
    jammer_service_id=torch.zeros(E, dtype=torch.int64),
    victim_service_id=torch.zeros(E, dtype=torch.int64),
    radar_beam_az_idx=torch.tensor([2]),
    jammer_beam_az_idx=torch.tensor([2]),
)

# 旁瓣(jammer off-axis)
jnr_side = compute_jnr_db_s2(
    physics, radar, jammer,
    jammer_active=is_jam,
    jammer_service_id=torch.zeros(E, dtype=torch.int64),
    victim_service_id=torch.zeros(E, dtype=torch.int64),
    radar_beam_az_idx=torch.tensor([2]),
    jammer_beam_az_idx=torch.tensor([0]),
)

spread = jnr_main.item() - jnr_side.item()
print(f"JNR main: {jnr_main.item():.2f} dB")
print(f"JNR side: {jnr_side.item():.2f} dB")
print(f"Spread:   {spread:.2f} dB")

assert 10.0 < jnr_main.item() < 15.0, f"主瓣 JNR 超 [10,15] 范围"
assert spread >= 15.0, f"spread < 15 dB"
print("M0 PASS")
```

**期望输出**:
```
JNR main: 12.55 dB   (允许 ±0.5)
JNR side: -7.33 dB   (允许 ±1)
Spread:   19.88 dB   (允许 ±1)
M0 PASS
```

**失败**:
- JNR main 太高(> 20) → 你还是用了 P_jam_W=10.0,检查改动是否生效
- JNR main 太低(< 5) → N² 相干增益没加上,检查 physics.py L92-94
- Spread < 15 → AF 公式坏了,检查 array_factor.py

#### 2.3 重跑测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD pytest -q \
  tests/array_face/test_array_factor_s2.py \
  tests/array_face/test_array_face_s2.py
# 期望: 23/23 PASS

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD pytest -q tests/g3_bsta_lite
# 期望: 75/75 PASS

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD pytest -q \
  tests/array_face/test_array_factor_s1.py \
  tests/array_face/test_array_face_s1.py
# 期望: 18/18 PASS
```

**全 PASS 才能进 Step 3**。

---

### Step 3: 跑 S2 PPO(预计 90 分钟,3 seed × 25-30 min)

#### 3.1 准备

```bash
# 看清输出目录命名规则(S1 用过)
ls experiments/array_face_s1/learning_repair/s1_ppo_output_amend02_seed*/

# S2 用同命名
# run_s2_ppo.py --seed <N> 输出到 s2_ppo_output_amend02_seed<N>/
```

#### 3.2 顺序跑 3 seed

```bash
cd /home/ubuntu/CODE/g3-bsta-fastwork
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fluxphased

# 3 seed × 1000 iter × 16 envs × 64 steps = 3.07M transitions
for s in 20260729 20260730 20260801; do
  echo "===== seed $s starting at $(date) ====="
  python experiments/array_face_s2/learning_repair/run_s2_ppo.py --seed $s \
    2>&1 | tee experiments/array_face_s2/learning_repair/s2_run_seed${s}.log
done
```

**预算**:每 seed 约 25-30 min on RTX PRO 6000。总耗时 75-90 min。

#### 3.3 训练中监控(开新 terminal)

```bash
# 看最新一行
tail -1 experiments/array_face_s2/learning_repair/s2_ppo_output_amend02_seed20260729/val_metrics.jsonl | python3 -m json.tool

# 看 entropy 是否坍缩(应保持 > 0.5)
tail -50 experiments/array_face_s2/learning_repair/s2_ppo_output_amend02_seed20260729/train_metrics.jsonl | \
  python3 -c "import sys, json; [print(json.loads(l).get('entropy', 'N/A')) for l in sys.stdin]"

# 看 GPU 使用率(应 > 50%)
nvidia-smi dmon -s u -c 5
```

**异常处理**:
- loss NaN → FAQ Q4,kill 训练,降 actor_lr 重试
- entropy < 0.1 in iter < 100 → FAQ Q5
- GPU OOM → FAQ Q11
- 耗时 > 60 min/seed → FAQ Q15

#### 3.4 训练完成验证

```bash
# 每个 seed 应有:
for s in 20260729 20260730 20260801; do
  d=experiments/array_face_s2/learning_repair/s2_ppo_output_amend02_seed${s}
  echo "seed $s:"
  echo "  train rows: $(wc -l < $d/train_metrics.jsonl)"  # 期望: 1000
  echo "  val rows:   $(wc -l < $d/val_metrics.jsonl)"    # 期望: 100
  echo "  final val:  $(tail -1 $d/val_metrics.jsonl | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["val_macro_drop"])')"
done
```

**期望**(参考 S1):
- 1000 train rows / 100 val rows
- final val 在 0.09(stuck)到 0.24(broke out 最佳)之间

---

### Step 4: 应用信号灯标准(预计 15 分钟)

#### 4.1 计算 aggregate 指标

```python
# 存为 /tmp/evaluate_s2.py,python /tmp/evaluate_s2.py
import json
from pathlib import Path
from statistics import mean, stdev

SEEDS = [20260729, 20260730, 20260801]
BASE = Path("experiments/array_face_s2/learning_repair")

finals = []
peaks = []
break_iter = []
statuses = []

for s in SEEDS:
    val_path = BASE / f"s2_ppo_output_amend02_seed{s}" / "val_metrics.jsonl"
    if not val_path.exists():
        print(f"WARN: seed {s} val_metrics.jsonl missing")
        continue
    with open(val_path) as f:
        rows = [json.loads(line) for line in f]
    final_val = rows[-1]["val_macro_drop"]
    peak_val = max(r["val_macro_drop"] for r in rows)
    peak_iter = max(rows, key=lambda r: r["val_macro_drop"])["iter"]
    # broke out = 突破 0.12
    broke_at = next((r["iter"] for r in rows if r["val_macro_drop"] > 0.12), None)
    status = "broke" if broke_at is not None else "stuck"
    finals.append(final_val)
    peaks.append(peak_val)
    break_iter.append(broke_at)
    statuses.append(status)
    print(f"seed {s}: final={final_val:.4f} peak={peak_val:.4f}@{peak_iter} "
          f"broke_at={broke_at} status={status}")

print()
n_broke = statuses.count("broke")
n_total = len(statuses)
print(f"Break-out rate: {n_broke}/{n_total} ({100*n_broke/n_total:.0f}%)")

if n_broke > 0:
    broke_finals = [f for f, s in zip(finals, statuses) if s == "broke"]
    print(f"Broke-out mean final: {mean(broke_finals):.4f} ± {stdev(broke_finals):.4f}"
          if len(broke_finals) > 1
          else f"Broke-out mean final: {mean(broke_finals):.4f}")

print(f"Best seed final: {max(finals):.4f}")
print(f"Worst seed final: {min(finals):.4f}")
print(f"All-seed mean final: {mean(finals):.4f} ± {stdev(finals):.4f}"
      if len(finals) > 1 else f"All-seed mean: {mean(finals):.4f}")
```

#### 4.2 查决策表(handoff §10.7.7)

根据 4.1 输出,查信号灯:

| 信号灯 | break-out rate | mean | best | 含义 | 后续 |
|---|---|---|---|---|---|
| 🟢 GREEN | ≥ 2/3 | [0.17, 0.27] | ≥ 0.18 | S2 复现 S1 水平 | 进 S3 |
| 🟡 YELLOW | = 1/3 | [0.12, 0.17] | < 0.18 | S2 比 S1 难 | 与用户讨论 |
| 🔴 RED | = 0/3 | < 0.12 | — | S2 失败 | 根因分析 |

**记录信号灯结果**,Step 5 写报告用。

---

### Step 5: 写报告 + plot(预计 20 分钟)

#### 5.1 写 plot 脚本

参考 `experiments/array_face_s1/learning_repair/plot_amend02_multiseed.py` 写 S2 版:

```bash
cp experiments/array_face_s1/learning_repair/plot_amend02_multiseed.py \
   experiments/array_face_s2/learning_repair/plot_s2_multiseed.py
# 编辑:把路径里的 s1 → s2,标题 S1 → S2,基线参考线改 S1 mean 0.2205
```

跑:`python experiments/array_face_s2/learning_repair/plot_s2_multiseed.py`

**期望产出**:`s2_multiseed_performance.png`(3 seed mean ± std band + 与 S1 对比线)。

#### 5.2 写 REPORT.md

在 `experiments/array_face_s2/REPORT.md` 写完整报告。**必须包含**(handoff §10.6 + §10.7.4):

```markdown
# Array-Face S2 Report

> Phase: S2 — jammer 1D ULA + Tx AF + MultiDiscrete([3,5])
> Period: <开始-结束日期>
> Branch: g3-bsta/array-face-s1
> Plan ref: docs/array_face/INCREMENTAL_DESIGN.md §4

## 1. What S2 added (vs S1)
[物理增量 + 动作空间增量 + obs 增量]

## 2. M0/M1 verification (gate PASS)
- 10/10 physics tests PASS
- 13/13 env contract tests PASS
- lite regression: 75/75 PASS
- M0 micro-verify: JNR main = X.XX dB, spread = X.XX dB

## 3. PPO multi-seed results (1000 iter each)
[表格:seed / config / break_iter / final / peak / status]
[Aggregate: break-out rate, mean ± std, best]

## 4. Signal-light verdict
[GREEN / YELLOW / RED + 数据支撑]

## 5. Comparison vs S1
[curve overlay plot]
[S2 mean vs S1 mean (0.2205), delta in pp]

## 6. Lessons / surprises
[这次跑学到的东西]

## 7. Recommendation
[进 S3 / 改 Amend03 / 回退 / 等用户决策]
```

---

### Step 6: 停下,与用户讨论(强制)

**不要自动 commit / push S2 修复代码。不要自动进 S3。**

到这一步你应该:
1. ✅ S2 物理 bug 修复,所有测试 PASS
2. ✅ 3 seed × 1000 iter PPO 跑完
3. ✅ 信号灯判定有结果(GREEN/YELLOW/RED)
4. ✅ REPORT.md 写完,plot 生成

**给用户的报告模板**(中文,简洁):

```
S2 复现完成,信号灯判定:🟢/🟡/🔴 <颜色>

数据:
- Break-out rate: X/3
- Broke-out mean: 0.XXXX ± 0.XXXX(S1 ref: 0.2205 ± 0.0116,delta: ±X.X pp)
- Best seed: 0.XXXX(seed N,S1 best ref: 0.2372)

曲线:[贴 s2_multiseed_performance.png 路径]

建议:
- 🟢 → 进 S3(cell binding)
- 🟡 → 跑 Amend03 调整探索 / 接受 S2 比 S1 难 / 回退到 S1 重新审视
- 🔴 → 根因分析,不进 S3

下一步等你指示。
```

**等用户回复后再决定**:
- commit / push S2 修复代码(需要用户授权 push)
- 是否进 S3
- 是否需要调超参(Amend03)

---

## 何时停下问用户(覆盖所有 step)

立即停下问用户,不要自行决策的场景:

1. **Step 0 验证失败**:SSH 不通 / memory 装不上 / handoff 读不懂
2. **Step 1 lite regression FAIL**:你可能误改了 protected files
3. **Step 2 物理修复后 JNR 值仍不合理**:可能 plan §2.2 数字本身有问题
4. **Step 3 训练崩溃且重试无效**:GPU 问题 / 代码 bug / 物理没修干净
5. **Step 4 信号灯 RED**:必须根因分析,不进 S3
6. **Step 5 plot 发现曲线异常**:训练可能根本没学到东西
7. **想 push 任何 commit**:单次授权 ≠ 永久
8. **想动 protected branches**(main / fast-work / forensic / handoff / mfr-lite-fastwork)
9. **任何违反用户偏好的操作**(英文回复、调 codex 等)

---

## 报告 / 沟通风格(强制)

- **所有面向用户文字必须中文**(memory `chinese_only_responses.md`)
- 数字必带图(plot / screenshot)
- 选项给 A/B/C/D 加推荐 + 理由
- 不要"你看着办"式回答
- 不准调 codex MCP(memory `feedback_no_codex.md`)

---

## 常见错误(避免)

1. **跳过 M0 micro-verify 直接跑 PPO** → 浪费 90 min GPU
2. **跑 PPO 时改代码** → 训练结果不可复现
3. **3 seed 没跑完就报告** → 没法算 mean ± std
4. **忘记 lite regression** → 可能破坏 protected 文件未发现
5. **修完不写 REPORT** → 用户没法决策
6. **自动进 S3** → 违反 multi-seed gating 规则

---

## 附录:如果一切顺利的总耗时

| Step | 耗时 | 累计 |
|---|---|---|
| 0. 加载 memory + 读 handoff | 15 min | 15 min |
| 1. 跑测试 verify | 5 min | 20 min |
| 2. 修 S2 物理 bug | 30 min | 50 min |
| 3. 跑 S2 PPO(3 seed) | 90 min | 140 min |
| 4. 应用信号灯标准 | 15 min | 155 min |
| 5. 写报告 + plot | 20 min | 175 min |
| 6. 等用户回复 | 不定 | — |

**总计**:约 3 小时(假设 PPO 不出问题)。

---

## 附录:完整命令清单(可一键复制)

```bash
# === Phase A: setup ===
cd /home/ubuntu/CODE/g3-bsta-fastwork
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate fluxphased
export PYTHONPATH=$PWD

# === Phase B: tests ===
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/array_face/test_array_factor_s1.py tests/array_face/test_array_face_s1.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/array_face/test_array_factor_s2.py tests/array_face/test_array_face_s2.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/g3_bsta_lite

# === Phase C: M0 verify ===
python /tmp/m0_verify.py

# === Phase D: PPO (3 seeds) ===
for s in 20260729 20260730 20260801; do
  python experiments/array_face_s2/learning_repair/run_s2_ppo.py --seed $s
done

# === Phase E: evaluate ===
python /tmp/evaluate_s2.py

# === Phase F: plot ===
python experiments/array_face_s2/learning_repair/plot_s2_multiseed.py

# === Phase G: report + 等用户 ===
# 编辑 experiments/array_face_s2/REPORT.md
# 然后停下,等用户讨论
```

---

**本 prompt 版本**: v1(2026-07-31)
**配套 handoff**: [`HANDOFF_20260731.md`](./HANDOFF_20260731.md) v2(985 行)
**前置 commit**: `fcb2e10` 在分支 `g3-bsta/array-face-s1`
**预期产出**: 信号灯判定 + REPORT.md + s2_multiseed_performance.png(不自动 push)
