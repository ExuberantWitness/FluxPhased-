# PROMPT — 给接手 agent 的从零复现手册

> **使用方法**:在新机器的 Claude Code 会话第一条 user message,**整篇粘贴**这个文件。
> Agent 应该按 Phase A → Phase K 顺序执行,不跳步。
>
> **设计原则**:agent 在新机器上**什么都没有**的状态下也能跑通。Handoff 文档通过 raw URL 下载(无需 auth),memory snapshot 在 repo 里(公开 clone 即可)。

---

## 0. 你的角色

你是 **Array-Face 项目**的接手 agent。前一个 agent 在另一台机器上做完了:

1. ✅ S1(radar 5-cell 1D ULA + Rx AF)完成 + multi-seed PPO 跑完(3 broke out / 3 stuck,best 0.2372)
2. ✅ S1 完整快照 + 实验 + 报告推送到 `g3-bsta/array-face-s1` 分支
3. ✅ S2(jammer 5-cell 1D ULA + Tx AF + MultiDiscrete)代码写完
4. ⚠️ **S2 有一个物理 bug 未修**(P_jam_W 语义冲突)
5. ✅ handoff 文档 + memory snapshot(43 个 .md)推送完毕

**你的任务**:走完 Phase A → Phase K,完成 S2 修复 + 复现 + 验证 + 报告 + 核查点,**然后停下与用户讨论**。

---

## 1. 关键事实(开干前先记牢)

| 项 | 值 |
|---|---|
| GitHub repo | `ExuberantWitness/FluxPhased-`(public,可匿名 clone) |
| 目标分支 | `g3-bsta/array-face-s1` |
| 最新 commit | `2bb51ea`(截至 2026-07-31) |
| Handoff 路径 | `docs/handoff/HANDOFF_20260731.md`(1264 行) |
| Memory snapshot 路径 | `docs/handoff/memory-snapshot-20260731/`(43 个 .md) |
| 仓库 raw URL base | `https://raw.githubusercontent.com/ExuberantWitness/FluxPhased-/g3-bsta/array-face-s1/` |
| 用户偏好 | 中文回复 / 不调 codex MCP / 不动 protected 分支 / push 前必问 |

---

## 2. 必守约束(违反会激怒用户)

- **所有面向用户的文字必须中文**(memory `chinese_only_responses.md`)
- **不准调用 codex MCP**(评审 / 头脑风暴 / 计划环节)— memory `feedback_no_codex.md`
- **不动 lite / fast-work / forensic / handoff / main 分支**
- **每次 push 必须先问用户授权**(单次 ≠ 永久)
- **不 force-push、不 rewrite history、不用 --no-verify**
- **每个 phase 必 multi-seed ≥ 3,跑完先 plot 与用户讨论**

---

## 3. 输入条件(假设新机器已具备,缺则停下问用户)

- [ ] Linux 机器(Ubuntu 22.04+ 或类似)
- [ ] NVIDIA GPU(最好 RTX PRO 6000 / Blackwell sm_120)
- [ ] 网络通 GitHub(可能需要代理 — Phase A 会测)
- [ ] **sudo 权限**(可能需要装系统包)— 没有的话停下问用户

---

# Phase A: 环境检查 + 下载 handoff(预计 15 分钟)

> **目标**:确认基础工具就绪 + **下载并通读 handoff 文档**(整个流程的导航灯塔)。
> Handoff 通过 raw URL 下载,**不需要 GitHub 认证**(repo 是 public)。

## A.1 工具检查

```bash
echo "=== 工具检查 ==="
for cmd in git curl python3 conda ssh nvidia-smi; do
  if command -v $cmd >/dev/null 2>&1; then
    echo "  ✓ $cmd: $(command -v $cmd)"
  else
    echo "  ✗ $cmd: MISSING"
  fi
done

echo ""
echo "=== Python 版本 ==="
python3 --version 2>&1

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>&1 || echo "GPU 不可见"
```

**期望**:
- ✓ `git` ≥ 2.30
- ✓ `curl` 任意版本
- ✓ `python3` ≥ 3.10
- ✓ `conda` (miniconda 或 anaconda 都行)
- ✓ `ssh` OpenSSH 任意版本
- ✓ `nvidia-smi` 看到 GPU(NVIDIA driver 装好)

**如果缺**:
- `git` 缺:`sudo apt update && sudo apt install -y git`
- `conda` 缺:从 https://docs.conda.io/en/latest/miniconda.html 下载 Linux x86_64 installer,bash 安装
- `nvidia-smi` 缺:NVIDIA driver 没装,停下问用户(这超出了 prompt 范围)

## A.2 测试 GitHub 连通性(可能需要代理)

```bash
echo "=== 测试 GitHub API(无代理)==="
curl -sS --max-time 10 -o /dev/null -w "HTTP %{http_code} time %{time_total}s\n" \
  https://api.github.com 2>&1
```

**期望输出**:`HTTP 200 time < 1s`

**如果超时或失败** → 需要代理。问用户:

> 我需要访问 GitHub 下载仓库和 handoff 文档,但直连失败。
> 请告诉我代理地址(或确认无需代理):
> - 如果你已有代理(如 `http://127.0.0.1:6789`),告诉我地址 + 端口
> - 如果不需要代理,回复"无需代理"
> - 如果你的网络需要其他方式(vpn 等),告诉我配置方法

收到代理地址后:
```bash
export https_proxy=http://<addr>:<port>
export http_proxy=http://<addr>:<port>
export all_proxy=socks5://<addr>:<port>
# 重测
curl -sS --max-time 10 -o /dev/null -w "HTTP %{http_code}\n" https://api.github.com
```

**如果还是不通** → 停下,把 `curl -v` 输出贴给用户,问下一步。

## A.3 下载 handoff 文档(raw URL,无需 auth)

```bash
mkdir -p /tmp/array-face-bootstrap && cd /tmp/array-face-bootstrap

# 下载 handoff(公开 raw URL,无需 token)
curl -sS --max-time 30 -o HANDOFF_20260731.md \
  https://raw.githubusercontent.com/ExuberantWitness/FluxPhased-/g3-bsta/array-face-s1/docs/handoff/HANDOFF_20260731.md

# 验证
ls -la HANDOFF_20260731.md  # 期望 ~50KB
wc -l HANDOFF_20260731.md   # 期望 ~1264 行
head -3 HANDOFF_20260731.md # 期望 "# 承接文档 — Array-Face 多阶段路线交接"
```

**如果失败**:
- HTTP 404 → 分支名或路径写错,核对 `ExuberantWitness/FluxPhased-/-/g3-bsta/array-face-s1` 拼写
- HTTP 403 → GitHub rate limit(罕见,public repo),等 1 分钟重试
- 超时 → 代理问题,回 A.2

## A.4 通读 handoff

**必须读完**(不能跳):
- §0 TL;DR
- §1 项目背景
- §8 用户偏好与硬约束
- §10 当前任务:修 S2 物理 bug(包括 §10.7 信号灯制)
- §12 已知陷阱
- §17 故障排查 FAQ
- §18 术语表

可跳读:§2-§7,§11,§13-§16,§19

**读完应能回答**(自测):
1. S2 物理 bug 是什么?→ `physics.P_jam_W` 在 physics.py 当 per-cell,在 EnvConfig 当 total,7dB EIRP 偏差
2. 信号灯 GREEN 的 3 个硬门槛?→ break-out ≥ 2/3 + mean ∈ [0.17, 0.27] + best ≥ 0.18
3. S1 broke-out mean 是多少?→ 0.2205 ± 0.0116
4. 5 个 protected branches?→ main / mfr-lite-fastwork / forensic-output / handoff/evidence / 当前 array-face-s* 之外的全部
5. 用户用什么语言?→ 中文

**答不上 → 重读对应章节**。

---

# Phase B: 安装 memory(预计 5 分钟,仅 Claude Code 适用)

> **目标**:把 memory snapshot 装到 Claude 的项目目录,让新会话能加载历史 lesson。
>
> **环境适配**:
> - **Claude Code**(官方 CLI):memory 自动加载机制生效,本 phase 必须执行。
> - **ZCode / 其他第三方 agent runtime**:memory 自动加载**不生效**,本 phase 可跳过;
>   改为在 Phase F 之后**直接从 clone 出的 `docs/handoff/memory-snapshot-*/` 按需 Read**。
>   2026-07-31 跨机器复现就走了这条路径(ZCode/GLM-5.2),证明可行。

## B.1 确定项目顶层目录

Claude Code 的 memory 路径规则:**项目顶层目录的 `/` 换成 `-`**。

例如:
- 项目在 `/home/ubuntu/CODE/g3-bsta-fastwork` → 顶层是 `/home/ubuntu/CODE` → memory 路径 `/home/ubuntu/.claude/projects/-home-ubuntu-CODE/memory/`
- 项目在 `/home/foo/bar` → memory 路径 `/home/foo/.claude/projects/-home-foo-bar/memory/`

**先决定你打算把项目 clone 到哪**。推荐 `/home/$USER/CODE/g3-bsta-fastwork`(沿用前一个 agent 的路径)。

```bash
# 决定项目根(顶层 worktree parent)
PROJECT_PARENT="/home/$USER/CODE"   # 你之后会 clone 到 $PROJECT_PARENT/g3-bsta-fastwork
MEM_HASH=$(echo "$PROJECT_PARENT" | sed 's|^/||; s|/|-|g')
MEM_DIR="/home/$USER/.claude/projects/-$MEM_HASH/memory"

echo "Memory 装到: $MEM_DIR"
mkdir -p "$MEM_DIR"
```

## B.2 下载 memory snapshot(public raw URL)

```bash
# 拿到 memory 文件列表(GitHub API,public)
curl -sS --max-time 30 \
  "https://api.github.com/repos/ExuberantWitness/FluxPhased-/contents/docs/handoff/memory-snapshot-20260731?ref=g3-bsta/array-face-s1" \
  | python3 -c "
import json, sys
files = json.load(sys.stdin)
for f in files:
    print(f['name'])
" > /tmp/memory_files.txt

cat /tmp/memory_files.txt | head -5  # 期望: 看到 MEMORY.md 等
wc -l /tmp/memory_files.txt           # 期望: 43
```

**如果 API 失败**(rate limit / 网络):直接 clone 整个 repo(Phase C),从 clone 出的目录里 cp。

## B.3 批量下载 + 安装

```bash
# 批量下载每个 memory 文件
RAW_BASE="https://raw.githubusercontent.com/ExuberantWitness/FluxPhased-/g3-bsta/array-face-s1/docs/handoff/memory-snapshot-20260731"

while read fname; do
  curl -sS --max-time 15 -o "$MEM_DIR/$fname" "$RAW_BASE/$fname"
done < /tmp/memory_files.txt

# 验证
ls "$MEM_DIR" | wc -l                              # 期望: 43
head -3 "$MEM_DIR/MEMORY.md"                       # 期望: 看到 "Wang 2025 全文参数"
grep "arrayface_mappo_unban" "$MEM_DIR/MEMORY.md"  # 期望: 找到一行
```

## B.4 验证 memory 完整性

```bash
# 检查 MEMORY.md 里链接的文件都存在
MISSING=0
while IFS= read -r line; do
  # 提取 [name](file.md) 里的 file.md
  fname=$(echo "$line" | grep -oE '\]\([^)]+\.md\)' | sed 's/^](//;s/)$//')
  if [ -n "$fname" ] && [ ! -f "$MEM_DIR/$fname" ]; then
    echo "缺失: $fname"
    MISSING=$((MISSING+1))
  fi
done < <(grep "^- \[" "$MEM_DIR/MEMORY.md")
echo "缺失文件数: $MISSING"  # 期望: 0
```

**如果 MISSING > 0** → 重新下载,或直接 clone 整个 repo(Phase C)再 cp。

---

# Phase C: Clone 仓库 + 设置 worktree(预计 10 分钟)

## C.1 Clone(HTTPS,public repo 无需 auth)

```bash
mkdir -p "$PROJECT_PARENT" && cd "$PROJECT_PARENT"

# 如果还没 clone
if [ ! -d "FluxPhased-" ]; then
  git clone --depth 50 https://github.com/ExuberantWitness/FluxPhased-.git
fi

cd FluxPhased-

# 拉取目标分支
git fetch origin g3-bsta/array-face-s1
git checkout g3-bsta/array-face-s1
git log --oneline -6
# 期望最新 commit: 2bb51ea docs(handoff): v2 expansion + agent reproduction prompt
```

## C.2 创建 worktree(分离主仓库和工作目录)

```bash
# 在主 clone 里加一个 worktree,g3-bsta-fastwork 单独一个目录
git worktree add "$PROJECT_PARENT/g3-bsta-fastwork" g3-bsta/array-face-s1
# 如果已有这个 worktree 路径,git 会报错,git worktree list 看
cd "$PROJECT_PARENT/g3-bsta-fastwork"
pwd  # 期望: /home/<user>/CODE/g3-bsta-fastwork
git rev-parse HEAD  # 期望: 2bb51ea...
```

## C.3 (可选)配置 SSH key(后续 push 需要)

> **如果只跑测试和 PPO,不 push,可跳过本节**。
> 如果之后要 push(等用户授权),需要 SSH key。

```bash
# 看是否已有 SSH key
ls ~/.ssh/id_ed25519* 2>&1

# 如果没有,生成
if [ ! -f ~/.ssh/id_ed25519 ]; then
  ssh-keygen -t ed25519 -C "fluxphased-push-$(hostname)" -N "" -f ~/.ssh/id_ed25519
fi

# 配置 ~/.ssh/config(走 443 端口,某些网络环境封 22)
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat > ~/.ssh/config.github <<EOF
Host github.com
  Hostname ssh.github.com
  Port 443
  User git
EOF
# 如果 ~/.ssh/config 不存在或不含 github.com,合并进去
if [ ! -f ~/.ssh/config ] || ! grep -q "Host github.com" ~/.ssh/config; then
  cat ~/.ssh/config.github >> ~/.ssh/config
fi
chmod 600 ~/.ssh/config
rm ~/.ssh/config.github

# 显示公钥(用户要去 GitHub 注册)
echo "==============================================="
echo "把这整行复制到 https://github.com/settings/ssh/new:"
echo "==============================================="
cat ~/.ssh/id_ed25519.pub
echo "==============================================="
echo "Title 随便填(如 fluxphased-push-$(hostname)),Key 粘贴以上整行,保存。"
echo "完成后告诉我 'SSH 加好了',我继续。"
```

**强制停下,等用户**:用户需要在浏览器打开 `https://github.com/settings/ssh/new`,把公钥粘进去。

用户确认后:
```bash
# 验证 SSH
ssh -T -o StrictHostKeyChecking=accept-new git@github.com
# 期望: "Hi ExuberantWitness! You've successfully authenticated..."
```

**如果失败**:
- "Permission denied (publickey)" → key 没注册成功,让用户检查 GitHub settings/keys
- "ssh: connect to host ssh.github.com port 443" → 网络 / 代理问题,需要 `ProxyCommand` 在 ~/.ssh/config

---

# Phase D: 配置 conda env + 装依赖(预计 15-30 分钟,取决于网速)

## D.1 创建 conda env

```bash
# 如果 env 已存在,跳过
conda env list | grep fluxphased && echo "env 已存在,跳过创建" || \
  conda create -n fluxphased python=3.10 -y

source /home/$USER/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null \
  || source $(conda info --base)/etc/profile.d/conda.sh
conda activate fluxphased

which python  # 期望: /home/$USER/miniconda3/envs/fluxphased/bin/python
python --version  # 期望: Python 3.10.x
```

## D.2 装依赖

**Blackwell GPU(RTX PRO 6000 / RTX 50xx,sm_120)**:
```bash
pip install --upgrade pip
pip install torch==2.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
```

**Standard GPU(RTX 40xx / A100 / RTX 30xx,CUDA 12.1)**:
```bash
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

**判断 GPU 类型**:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# sm_120 = Blackwell → 用 2.12.0
# sm_89 / sm_90 = Ada / Hopper → 用 2.4.1
```

**其他依赖**:
```bash
pip install ittapi
pip install warp-lang==1.10.1
pip install pettingzoo==1.24.3
pip install gym==0.26.2
pip install gymnasium==1.1.1
pip install numpy==1.24.4 scipy==1.10.1 matplotlib==3.7.5
pip install pyyaml==6.0.3
pip install tensorboard==2.14.0 tqdm==4.67.1 pandas==2.0.3
pip install pytest
```

(中国大陆可加 `-i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`)

## D.3 验证 torch + GPU

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute cap: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
    x = torch.randn(1024, 1024, device='cuda')
    y = x @ x
    print(f'Matmul OK: {y.shape}, device={y.device}')
"
```

**期望**:
```
PyTorch: 2.12.0
CUDA available: True
GPU: NVIDIA RTX PRO 6000
Compute cap: sm_120
Matmul OK: torch.Size([1024, 1024]), device=cuda:0
```

**失败**:
- CUDA available False → driver 版本不够 / 装错了 torch 版本
- Matmul 报 sm_120 unsupported → torch 版本太老,需要 ≥ 2.11

---

# Phase E: 重启 Claude Code 让 memory 生效(预计 5 分钟)

> Memory 文件装好后,**当前 session 看不到**,必须重启 Claude Code。

## E.1 退出当前 session

```
/exit   # 或 Ctrl+D
```

## E.2 重启 Claude Code,在项目目录里

```bash
cd "$PROJECT_PARENT/g3-bsta-fastwork"
claude  # 或你的启动命令
```

## E.3 验证 memory 加载

新 session 启动后,system context 应包含:
```
Contents of /home/<user>/.claude/projects/.../memory/MEMORY.md (user's auto-memory, persists across conversations):
- [Wang 2025 全文参数](wang2025_params.md) — ...
- ...(43 行)
```

**自测**:问自己(或让 Claude 回答):
> 列出当前 memory 里所有 user feedback 类的条目。

期望回答:
- `chinese_only_responses`(中文回复)
- `feedback_no_codex`(不调 codex)
- `feedback_pool_randomization`(池级随机化)

**如果 memory 没加载** → handoff §17 Q12,路径不对。修路径后再次重启。

---

# Phase F: 跑测试 verify(预计 5 分钟)

> 现在你已在新 session 里,memory 已加载。下面所有命令在 worktree 根目录跑。

## F.1 设置环境

```bash
cd "$PROJECT_PARENT/g3-bsta-fastwork"  # = /home/$USER/CODE/g3-bsta-fastwork
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fluxphased
export PYTHONPATH=$PWD
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

## F.2 跑 S1 测试

```bash
pytest -q tests/array_face/test_array_factor_s1.py tests/array_face/test_array_face_s1.py
# 期望: 18 passed
```

## F.3 跑 S2 测试(修复前,用旧 P_jam_W=10.0,应全 PASS)

```bash
pytest -q tests/array_face/test_array_factor_s2.py tests/array_face/test_array_face_s2.py
# 期望: 23 passed
```

## F.4 lite regression(必须零破坏)

```bash
pytest -q tests/g3_bsta_lite
# 期望: 75 passed
```

**任一失败**:
- pytest 找不到 → handoff §17 Q1
- ImportError env → handoff §17 Q2
- S2 测试 FAIL(且你没动代码)→ handoff §17 Q3,正常,bug 还没修
- lite regression FAIL → handoff §17 Q13(紧急,可能误改 protected 文件)

---

# Phase G: 修 S2 物理 bug(预计 30 分钟)

> **目标**:把 `P_jam_W` 全部统一为**单 cell 功率(per-cell)**,默认 2.0 W(plan §2.2)。

## G.1 阅读相关代码

```bash
# 1. physics.py 当前实现(注意 P_jam_W 的语义)
sed -n '85,115p' env/gpu/array_face_s2/physics.py

# 2. EnvConfig 当前默认值
grep -n "P_jam_W\|n_jammer_cells" env/gpu/array_face_s2/env.py | head -10

# 3. runner 当前传值
grep -n "P_jam_W" experiments/array_face_s2/learning_repair/run_s2_ppo.py
```

## G.2 按 handoff §10.2 修 5 处

**(1) `env/gpu/array_face_s2/env.py` 加字段 + 改默认**

```python
# EnvConfig 类内(L42 起):
# 加字段:
n_jammer_cells: int = 5    # 5-cell ULA,与 JammerULAConfig 一致

# 改默认值:
P_jam_W: float = 2.0       # per-cell (plan §2.2 P_cell_W)
```

**(2) 能量预算 4 处**(`env.py` L77 / L101 / L152 / L284)

```python
# 改前:
self.E0 = float(self.E0_tokens) * float(self.P_jam_W) * float(self.dt)
# 改后(乘以 N_cells):
self.E0 = float(self.E0_tokens) * float(self.P_jam_W) * float(self.n_jammer_cells) * float(self.dt)
```

同样改 L101 / L152 / L284 的同类表达式(grep `P_jam_W.*self.dt` 全找出来)。

**(3) runner**:`run_s2_ppo.py` L64 + L72

```python
# 改前:
dt=1.0, P_jam_W=10.0,            # S2: 5 cells × 2.0 W
physics = default_debug_physics_config(P_jam_W=10.0)
# 改后:
dt=1.0, P_jam_W=2.0,             # S2 per-cell (plan §2.2)
physics = default_debug_physics_config(P_jam_W=2.0)
```

**(4) physics 测试**:`tests/array_face/test_array_factor_s2.py` L71 / L99 / L122

```python
# 全部:
physics = default_debug_physics_config(P_jam_W=10.0)
# 改为:
physics = default_debug_physics_config(P_jam_W=2.0)
```

**(5) env 测试**:`tests/array_face/test_array_face_s2.py` L18

```python
def _make_env(n_envs=4, horizon=16, profile="mdp_sanity_v1", P_jam_W=10.0):
# 改为:
def _make_env(n_envs=4, horizon=16, profile="mdp_sanity_v1", P_jam_W=2.0):
```

## G.3 M0 micro-verify(PPO 前必 PASS)

```bash
cat > /tmp/m0_verify.py <<'PYEOF'
import torch
import sys
sys.path.insert(0, '.')  # 让 from env import 能工作
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s2 import (
    RadarULAConfig, JammerULAConfig, compute_jnr_db_s2,
)

physics = default_debug_physics_config(P_jam_W=2.0)
radar = RadarULAConfig()
jammer = JammerULAConfig()
E = 1
is_jam = torch.ones(E, dtype=torch.bool)

jnr_main = compute_jnr_db_s2(
    physics, radar, jammer,
    jammer_active=is_jam,
    jammer_service_id=torch.zeros(E, dtype=torch.int64),
    victim_service_id=torch.zeros(E, dtype=torch.int64),
    radar_beam_az_idx=torch.tensor([2]),
    jammer_beam_az_idx=torch.tensor([2]),
)

jnr_side = compute_jnr_db_s2(
    physics, radar, jammer,
    jammer_active=is_jam,
    jammer_service_id=torch.zeros(E, dtype=torch.int64),
    victim_service_id=torch.zeros(E, dtype=torch.int64),
    radar_beam_az_idx=torch.tensor([2]),
    jammer_beam_az_idx=torch.tensor([0]),
)

spread = jnr_main.item() - jnr_side.item()
print(f"JNR main (broadside-broadside): {jnr_main.item():.2f} dB")
print(f"JNR side (jammer at -60deg):    {jnr_side.item():.2f} dB")
print(f"Spread:                          {spread:.2f} dB")

assert 65.0 < jnr_main.item() < 70.0, \
    f"主瓣 JNR {jnr_main.item():.2f} 超预期 [65,70],物理 bug 未修干净"
assert spread >= 15.0, \
    f"spread {spread:.2f} < 15 dB,AF 公式问题"
print("M0 PASS")
PYEOF

python /tmp/m0_verify.py
```

**期望**:
```
JNR main (broadside-broadside): 67.48 dB
JNR side (jammer at -60deg):    47.60 dB
Spread:                          19.88 dB
M0 PASS
```

> **重要**:67.48 dB 必须与 S1 baseline(P_jam_W=50W 单天线)**等值**,这是 S1/S2 物理可比
> 的硬保证。若你的实测偏离 [65, 70],**先走 §10.7.8(三方交叉验证)**,不要硬改代码。
> 历史 lesson:本 handoff v1 曾把期望值写成 12.55 dB(算术错),2026-07-31 跨机器复现
> 触发该 bug,ZCode 通过三方验证判定 handoff 错、代码对。详见 handoff §10.7.8。

**失败诊断**:
- JNR main > 20 → 还是用了 P_jam_W=10.0,检查改动是否生效
- JNR main < 5 → N² 相干增益没加,检查 physics.py L92-94
- Spread < 15 → AF 公式坏了,检查 array_factor.py
- 完全跑不通(ImportError 等)→ handoff §17 Q2

## G.4 重跑所有测试(必须全 PASS)

```bash
pytest -q tests/array_face/test_array_factor_s2.py tests/array_face/test_array_face_s2.py
# 期望: 23 passed

pytest -q tests/array_face/test_array_factor_s1.py tests/array_face/test_array_face_s1.py
# 期望: 18 passed(S1 零破坏)

pytest -q tests/g3_bsta_lite
# 期望: 75 passed(lite 零破坏)
```

**总 116 个测试全 PASS 才能进 Phase H**。

---

# Phase H: 跑 S2 PPO(预计 90 分钟,3 seed × 25-30 min)

## H.1 启动训练

```bash
cd "$PROJECT_PARENT/g3-bsta-fastwork"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fluxphased
export PYTHONPATH=$PWD

# 3 seed 顺序跑
for s in 20260729 20260730 20260801; do
  echo "===== seed $s starting at $(date) ====="
  python experiments/array_face_s2/learning_repair/run_s2_ppo.py --seed $s \
    2>&1 | tee experiments/array_face_s2/learning_repair/s2_run_seed${s}.log
done
```

预算:**3 seed × 25-30 min on RTX PRO 6000 = 75-90 min 总**。

## H.2 训练中监控(开新 terminal)

```bash
# 最新 val
tail -1 experiments/array_face_s2/learning_repair/s2_ppo_output_amend02_seed20260729/val_metrics.jsonl \
  | python3 -m json.tool

# entropy 是否坍缩(应 > 0.5)
tail -50 experiments/array_face_s2/learning_repair/s2_ppo_output_amend02_seed20260729/train_metrics.jsonl \
  | python3 -c "import sys, json; [print(f'entropy={json.loads(l).get(\"entropy\", \"N/A\"):.4f}') for l in sys.stdin]"

# GPU 使用率(应 > 50%)
nvidia-smi dmon -s u -c 5
```

**异常处理**:
- loss NaN → handoff §17 Q4,kill 训练,降 actor_lr 重试
- entropy < 0.1 in iter < 100 → handoff §17 Q5
- GPU OOM → handoff §17 Q11

## H.3 训练完成验证

```bash
for s in 20260729 20260730 20260801; do
  d="experiments/array_face_s2/learning_repair/s2_ppo_output_amend02_seed${s}"
  echo "seed $s:"
  echo "  train rows: $(wc -l < $d/train_metrics.jsonl)"  # 期望: 1000
  echo "  val rows:   $(wc -l < $d/val_metrics.jsonl)"    # 期望: 100
  echo "  final val:  $(tail -1 $d/val_metrics.jsonl | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())[\"val_macro_drop\"])')"
done
```

---

# Phase I: 应用信号灯标准(预计 15 分钟)

## I.1 计算 aggregate

```bash
cat > /tmp/evaluate_s2.py <<'PYEOF'
import json
from pathlib import Path
from statistics import mean, stdev

SEEDS = [20260729, 20260730, 20260801]
BASE = Path("experiments/array_face_s2/learning_repair")

finals, peaks, statuses = [], [], []

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
    broke_at = next((r["iter"] for r in rows if r["val_macro_drop"] > 0.12), None)
    status = "broke" if broke_at is not None else "stuck"
    finals.append(final_val)
    peaks.append(peak_val)
    statuses.append(status)
    print(f"seed {s}: final={final_val:.4f} peak={peak_val:.4f}@{peak_iter} "
          f"broke_at={broke_at} status={status}")

print()
n_broke = statuses.count("broke")
n_total = len(statuses)
print(f"Break-out rate: {n_broke}/{n_total} ({100*n_broke/n_total:.0f}%)")

if n_broke > 0:
    broke_finals = [f for f, s in zip(finals, statuses) if s == "broke"]
    if len(broke_finals) > 1:
        print(f"Broke-out mean final: {mean(broke_finals):.4f} ± {stdev(broke_finals):.4f}")
    else:
        print(f"Broke-out mean final: {broke_finals[0]:.4f} (n=1)")

print(f"Best seed final: {max(finals):.4f}")
print(f"All-seed mean: {mean(finals):.4f} ± {stdev(finals):.4f}" if len(finals) > 1 else f"All-seed mean: {mean(finals):.4f}")

# 信号灯判定
print("\n=== Signal-light ===")
if n_broke >= 2 and mean([f for f, s in zip(finals, statuses) if s == "broke"]) >= 0.17 and max(finals) >= 0.18:
    print("🟢 GREEN — 复现成功,可进 S3")
elif n_broke >= 1 and mean(finals) >= 0.12:
    print("🟡 YELLOW — 部分成功,与用户讨论")
else:
    print("🔴 RED — 失败,根因分析")
PYEOF

python /tmp/evaluate_s2.py
```

## I.2 查决策表(handoff §10.7.7)

| 信号灯 | break-out rate | mean | best | 后续 |
|---|---|---|---|---|
| 🟢 GREEN | ≥ 2/3 | [0.17, 0.27] | ≥ 0.18 | 进 S3 |
| 🟡 YELLOW | = 1/3 | [0.12, 0.17] | < 0.18 | 与用户讨论 |
| 🔴 RED | = 0/3 | < 0.12 | — | 根因分析 |

---

# Phase J: 报告 + 停下等用户(预计 20 分钟)

## J.1 生成 plot

```bash
# 参考 S1 plot 写 S2 版
cp experiments/array_face_s1/learning_repair/plot_amend02_multiseed.py \
   experiments/array_face_s2/learning_repair/plot_s2_multiseed.py

# 编辑 plot_s2_multiseed.py:
# - s1 → s2 路径
# - 标题 S1 → S2
# - 基线参考线改 S1 mean 0.2205
# 用 sed 或编辑器改
sed -i 's|s1_ppo_output_amend02_seed|s2_ppo_output_amend02_seed|g' \
   experiments/array_face_s2/learning_repair/plot_s2_multiseed.py
sed -i 's|S1 |S2 |g' \
   experiments/array_face_s2/learning_repair/plot_s2_multiseed.py

python experiments/array_face_s2/learning_repair/plot_s2_multiseed.py
# 产出: s2_multiseed_performance.png
```

## J.2 写 REPORT.md

在 `experiments/array_face_s2/REPORT.md` 写完整报告。**必须包含**:

```markdown
# Array-Face S2 Report

> Phase: S2 — jammer 1D ULA + Tx AF + MultiDiscrete([3,5])
> Period: <开始-结束日期>
> Branch: g3-bsta/array-face-s1
> Plan ref: docs/array_face/INCREMENTAL_DESIGN.md §4

## 1. What S2 added (vs S1)
[物理 + 动作空间 + obs 增量]

## 2. M0/M1 verification (gate PASS)
- 10/10 physics tests PASS
- 13/13 env contract tests PASS
- lite regression: 75/75 PASS
- M0 micro-verify: JNR main = X.XX dB, spread = X.XX dB

## 3. PPO multi-seed results (1000 iter each)
[表:seed / break_iter / final / peak / status]
[Aggregate:break-out rate / mean ± std / best]

## 4. Signal-light verdict
[GREEN / YELLOW / RED + 数据支撑]

## 5. Comparison vs S1
[curve overlay plot]
[S2 mean vs S1 mean (0.2205),delta in pp]

## 6. Lessons / surprises
[这次跑学到的东西]

## 7. Recommendation
[进 S3 / Amend03 / 回退 / 等用户决策]
```

## J.3 给用户的最终报告(中文,简洁)

```
S2 复现完成,信号灯:🟢/🟡/🔴 <颜色>

数据:
- Break-out rate: X/3
- Broke-out mean: 0.XXXX ± 0.XXXXX(S1 ref: 0.2205 ± 0.0116,delta: ±X.X pp)
- Best seed: 0.XXXX(seed N,S1 best ref: 0.2372)

曲线:[s2_multiseed_performance.png 路径]

建议:
- 🟢 → 进 S3(cell binding)
- 🟡 → Amend03 / 接受 / 回退
- 🔴 → 根因分析,不进 S3

报告写在 experiments/array_face_s2/REPORT.md。
等你指示是否 commit / push / 进 S3。
```

## J.4 强制 STOP

**不要**:
- ❌ 自动 commit / push(等用户授权)
- ❌ 自动进 S3(违反 multi-seed gating)
- ❌ 自动修改 protected branches
- ❌ "你看着办" 式回答

**要**:
- ✅ 给具体选项(A/B/C/D + 推荐 + 理由)
- ✅ 等用户回复

---

# Phase K: 写 REPRO_CHECKPOINT 给下一个核查 agent(预计 15 分钟)

> **目标**:把本次复现的关键决策、独立验证、对 handoff 的偏离与判定依据,写成一个
> **自包含 checkpoint 文件**,供下一个核查 agent(可能是另一个工具/机器)在 3 分钟内
> 判断"这次复现是否可信"。
>
> **背景**:2026-07-31 的跨机器复现中,前一个 agent(ZCode)侦测到 handoff §10.4/§10.7.3/§10.7.6
> 写的 JNR 期望值 12.55 dB 与代码实测 67.48 dB 严重冲突。它没有盲改代码,而是独立手算 +
> 三方交叉验证(S1/S2-fixed/S2-buggy),最后判定"代码对、handoff 错",并写了
> `REPRO_CHECKPOINT_20260731.md` 供主机端 agent 核查。这一做法挽救了整个复现,值得固化。

## K.1 何时写

- ✅ **任何 M0/micro-verify 失败但你判定是 handoff 错而不是代码错时**
- ✅ **任何 handoff 数字与代码实测冲突时**(即便不阻塞)
- ✅ **训练曲线与预期显著偏离时**(mean 偏离 S1 baseline > 5pp)
- ✅ **3 seed break-out rate < 2/3**(信号灯非 GREEN)
- ❌ 一切顺利、信号灯 GREEN、无任何偏离 → 不必写(Phase J 报告足够)

## K.2 文件路径与命名

```
docs/handoff/REPRO_CHECKPOINT_<YYYYMMDD>.md
```

`YYYYMMDD` 用本次复现**完成日**(不是开始日)。

## K.3 必含章节(模板)

```markdown
# S2 复现核查点 — REPRO_CHECKPOINT_<YYYYMMDD>

> **用途**:供另一个 agent 核查本次 S2 复现工作是否正确。
> 本文自包含:含背景、已完成工作、核心发现、可复现的验证命令、关键数据、文件位置。
> **生成时间**:<YYYY-MM-DD>
> **复现根目录**:<绝对路径>
> **分支**:<branch> @ <短 commit hash>

---

## 0. 核查 TL;DR(给核查 agent 的 3 分钟结论)
[4-5 条要点:物理修复 / 测试 / 训练状态 / 信号灯初判 / 未完成项]

## 1. 背景(为什么做这件事)
[仓库 / 分支 / 任务 / 已知 bug / 前一 phase 的 baseline]

## 2. 本次复现采用的方式
[全新 clone / 复用 worktree / env 复用还是新建 / 用户授权了什么]

## 3. 已完成工作(逐 Phase)
[表格:Phase | 内容 | 结果]

## 4. 物理修复改动清单(共 X 处)
[文件 + 行号 + 改动;附 grep 核查命令证明无残留]

## 5. 核心发现(handoff 与代码的冲突,如有)
### 5.1 现象
[M0 实测什么 vs handoff 期望什么,差异多少]

### 5.2 根因
[独立手算表格:项 | 实际值 | handoff 写的 | 问题]

### 5.3 三方交叉验证(决定性证据)
[S1 baseline / S2-fixed / S2-buggy 三个版本的对比表]

### 5.4 复现验证命令(核查 agent 可直接跑)
[完整的 python - <<'PY' ... PY 块,不需要核查 agent 自己设计实验]

## 6. Phase 9 训练进展(本文件生成时的快照)
[每 seed:状态 / val 末 / peak / break-out iter]
[与 S1 baseline 的对比表]
[初步信号灯 + 待聚合说明]

## 7. 文件位置索引(核查用)
[复现根 / 本文件 / 原 handoff / 原 prompt / memory snapshot / 改过的代码 / 训练输出]

## 8. 给核查 agent 的核查清单
[6-8 条可独立验证的断言,每条附命令]

### 8.1 如果核查发现我判断错了
[给核查 agent 留的反向选项 + 验证路径,不要让下一个 agent 盲信]

## 9. 可复现命令汇总
[环境激活 / 测试 / M0 micro-verify / 单 seed PPO]

## 10. 读取训练结果
[解析 train_metrics.jsonl / val_metrics.jsonl 的命令]

## 11. 未完成项
[明确的 TODO,不要让下一个 agent 猜]
```

## K.4 关键原则

1. **自包含**:核查 agent 不应需要回读你的会话历史;所有命令、数字、判定依据都在文件里。
2. **可证伪**:每个断言都附一条核查 agent 能直接复制粘贴运行的命令。
3. **不偏袒**:既写"我做对了什么",也写"如果我认为对的实际是错的,核查 agent 该怎么发现"。
4. **绝对路径**:复现根、文件路径全用绝对路径(如 `/home/ubuntu/repro/array-face-s1/...`)。
5. **commit hash + 分支**:写明白前 HEAD,核查 agent 才能 checkout 同一状态。
6. **不写感受**:不写"我觉得复现很成功",写"seed 20260729 val=0.2101,S1 ref 0.2205,delta -1pp 落在 ±5pp 容差内 → 复现 S1 水平"。

## K.5 写完之后

```bash
# 不要 commit / push(REPRO_CHECKPOINT 是给主机端核查 agent 看的,留在复现机本地)
# 但要把它在 Phase J 的最终报告里引用:

# 在给用户的报告末尾加:
"本机核查文件:<绝对路径>(供下一个核查 agent 使用)"
```

---

# 紧急停下场景(任何时候遇到都停)

立即停下问用户:

1. **Phase A 工具缺失且 sudo 装不上**
2. **Phase A 网络/代理问题用户答不上**
3. **Phase C SSH key 用户拒绝注册**
4. **Phase D conda env 装不起来 / torch CUDA 不可用**
5. **Phase E memory 装上但 session 加载不到**
6. **Phase F lite regression FAIL**(你可能误改 protected 文件)
7. **Phase G M0 micro-verify 反复失败**(可能 plan §2.2 数字本身有问题)
8. **Phase H 训练崩溃且重试无效**
9. **Phase I 信号灯 RED**(必须根因分析,不进 S3)
10. **想 push 任何 commit**(单次授权 ≠ 永久)
11. **想动 protected branches**
12. **任何违反用户偏好**(英文回复、调 codex 等)

---

# 完整命令 cheat-sheet(已通过验证的命令汇总)

```bash
# === 一次性环境变量 ===
PROJECT_PARENT="/home/$USER/CODE"
MEM_HASH=$(echo "$PROJECT_PARENT" | sed 's|^/||; s|/|-|g')
MEM_DIR="/home/$USER/.claude/projects/-$MEM_HASH/memory"

# === Phase A: 下载 handoff ===
mkdir -p /tmp/array-face-bootstrap && cd /tmp/array-face-bootstrap
curl -sS -o HANDOFF_20260731.md \
  https://raw.githubusercontent.com/ExuberantWitness/FluxPhased-/g3-bsta/array-face-s1/docs/handoff/HANDOFF_20260731.md

# === Phase B: 装 memory ===
mkdir -p "$MEM_DIR"
RAW_BASE="https://raw.githubusercontent.com/ExuberantWitness/FluxPhased-/g3-bsta/array-face-s1/docs/handoff/memory-snapshot-20260731"
for f in MEMORY.md arrayface_mappo_unban.md chinese_only_responses.md feedback_no_codex.md feedback_pool_randomization.md; do
  curl -sS -o "$MEM_DIR/$f" "$RAW_BASE/$f"
done
# (用 Phase B 的批量脚本下全部 43 个)

# === Phase C: clone + worktree ===
mkdir -p "$PROJECT_PARENT" && cd "$PROJECT_PARENT"
[ ! -d FluxPhased- ] && git clone --depth 50 https://github.com/ExuberantWitness/FluxPhased-.git
cd FluxPhased- && git fetch origin g3-bsta/array-face-s1 && git checkout g3-bsta/array-face-s1
git worktree add "$PROJECT_PARENT/g3-bsta-fastwork" g3-bsta/array-face-s1 2>/dev/null
cd "$PROJECT_PARENT/g3-bsta-fastwork"

# === Phase D-G: 测试 + 修复 + 验证 ===
source $(conda info --base)/etc/profile.d/conda.sh && conda activate fluxphased
export PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# (按 Phase G 修 5 处代码)
# (跑 M0 micro-verify)

pytest -q tests/array_face/test_array_factor_s1.py tests/array_face/test_array_face_s1.py  # 18
pytest -q tests/array_face/test_array_factor_s2.py tests/array_face/test_array_face_s2.py  # 23
pytest -q tests/g3_bsta_lite                                                               # 75

# === Phase H: PPO ===
for s in 20260729 20260730 20260801; do
  python experiments/array_face_s2/learning_repair/run_s2_ppo.py --seed $s
done

# === Phase I: 评估 ===
python /tmp/evaluate_s2.py

# === Phase J: 报告 + STOP ===
python experiments/array_face_s2/learning_repair/plot_s2_multiseed.py
# 编辑 experiments/array_face_s2/REPORT.md
# 停下,等用户
```

---

# 总耗时估计

| Phase | 内容 | 耗时 | 累计 |
|---|---|---|---|
| A | 下载 handoff + 通读 | 15 min | 15 |
| B | 装 memory | 5 min | 20 |
| C | clone + worktree + (可选 SSH) | 10 min | 30 |
| D | conda env + 依赖 | 15-30 min | 45-60 |
| E | 重启 session 验证 memory | 5 min | 50-65 |
| F | 跑测试 verify | 5 min | 55-70 |
| G | 修 S2 物理 + 验证 | 30 min | 85-100 |
| H | S2 PPO 3 seed | 90 min | 175-190 |
| I | 评估信号灯 | 15 min | 190-205 |
| J | 报告 + plot | 20 min | 210-225 |

**总计**: 约 3.5-4 小时(假设无大障碍,含依赖安装时间)。

---

**本 prompt 版本**: v2(2026-07-31,从零 bootstrap 重写)
**配套 handoff**: [`HANDOFF_20260731.md`](./HANDOFF_20260731.md) v2(1264 行)
**前置 commit**: `2bb51ea` 在分支 `g3-bsta/array-face-s1`
**预期产出**: 信号灯判定 + REPORT.md + s2_multiseed_performance.png(不自动 push)
