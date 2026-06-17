# 双机部署方案：PRO 6000 训练 + 4090 控制

把全尺寸 LASER 多智能体训练搬到 **PRO 6000 Blackwell（98GB）**，4090 这台作为控制端 / 跳板。
两条路线（B = VSCode Remote-SSH 接管 / C = ssh + tmux 脱离式长跑）**共用同一套 SSH 地基**，先做 §1，再二选一。

## 0. 角色与前提

| | 机器 A（控制端） | 机器 B（训练端） |
|---|---|---|
| 硬件 | RTX 4090 D 24GB | **RTX PRO 6000 Blackwell 98GB（sm_120）** |
| 角色 | 跳板、监控、代码源 | 跑全尺寸训练 |
| 现状 | 已有仓库 + GitHub 访问（代理 127.0.0.1:6789 + PAT） | 待配置 |

**B 端硬性要求**：NVIDIA 驱动支持 sm_120（CUDA ≥13.0）；OpenSSH server 运行中；A 能从网络到达 B。

**先收集**（填进下面命令的占位符）：B 的 `IP/主机名`、登录 `用户名`、是否同一局域网。

---

## 1. SSH 地基（B 和 C 都要）

**B 端** —— 开 SSH server、查 IP：
```bash
sudo apt install -y openssh-server          # 若未装
sudo systemctl enable --now ssh
ip -4 addr | grep inet                       # 记下局域网 IP，如 192.168.x.x
nvidia-smi -L                                # 确认 PRO 6000 可见
```

**A 端** —— 生成密钥、免密登录、写 ssh 别名：
```bash
ssh-keygen -t ed25519 -C "a-4090-to-pro6000" -f ~/.ssh/pro6000   # 一路回车
ssh-copy-id -i ~/.ssh/pro6000.pub <B_USER>@<B_IP>                 # 输一次 B 的密码
cat >> ~/.ssh/config <<'EOF'

Host pro6000
    HostName <B_IP>
    User <B_USER>
    IdentityFile ~/.ssh/pro6000
    ServerAliveInterval 30
    ServerAliveCountInterval 3
EOF
ssh pro6000 nvidia-smi        # 成功 = 地基通了
```
`ServerAliveInterval` 让连接自动保活，断网后能续。

---

## 2. B 端环境（Blackwell 专用，一次性）

依赖极简（numpy / torch / yaml / warp），关键是 **torch 必须支持 sm_120**：
```bash
ssh pro6000        # 或在 B 的终端里
git clone https://github.com/ExuberantWitness/FluxPhased-.git    # 已有则 git pull
cd FluxPhased- && git checkout evo/laser-fix

conda create -n fluxphased python=3.10 -y && conda activate fluxphased
# Blackwell：PyTorch ≥2.11 + CUDA ≥13.0（与 A 端 torch 2.11+cu130 对齐即可）
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install warp-lang numpy pyyaml

# 验证 sm_120 在编译目标里：
python -c "import torch; print(torch.cuda.get_arch_list())"      # 应含 'sm_120'
python -c "import torch; print(torch.cuda.get_device_name(0))"   # 应是 PRO 6000
```
> 若 B 端连 GitHub/pip 需代理，先 `export HTTPS_PROXY=...`（B 自己的代理，不是 A 的 127.0.0.1:6789）。
> GitHub 推送凭据：在 B 上 `git config credential.helper store` 后首次 push 输入 PAT（**勿把 token 写进任何提交文件**）。

---

## 3A. 路线 C：ssh + tmux 脱离式长跑（稳定性优先，推荐）

训练挂在 tmux 里，**断网/关笔记本都不影响**，随时 re-attach 看实时输出。

**A 端一条命令起跑（落盘已是持久盘，无需改）：**
```bash
ssh pro6000 "cd FluxPhased- && conda run -n fluxphased tmux new-session -d -s laser \
  'python -m training.train_laser --config configs/laser_25x25_config.yaml 2>&1 | tee logs/pro6000_run.log'"
```

**监控（每次独立短连接，抖动无所谓）：**
```bash
ssh pro6000 "tail -20 FluxPhased-/logs/pro6000_run.log"
ssh pro6000 "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"
ssh pro6000 -t "tmux attach -t laser"     # 看实时；Ctrl-b d 退出不杀任务
```
把 B 的 IP/用户名告诉我，**这套我可以从 4090 这边直接帮你启动和盯**。

## 3B. 路线 B：VSCode Remote-SSH 接管

1. A 的 VSCode 装扩展 **Remote - SSH**（`ms-vscode-remote.remote-ssh`）；
2. F1 → `Remote-SSH: Connect to Host` → 选 `pro6000`（读 §1 的 ssh config）；
3. 新窗口里 `打开文件夹` → `~/FluxPhased-`；
4. 在远端窗口装 Claude Code 扩展，之后**我的工具就原生在 PRO 6000 上跑**，改配置/启动/监控跟本地一样；
5. 训练仍建议放 tmux（§3A），这样即使 Remote-SSH 重连，任务也不断。

---

## 4. 全尺寸训练配置（98GB 才跑得开）

`configs/laser_25x25_config.yaml` 已为 PRO 6000 调好，与 4090 上的小配置区别：

| 参数 | 4090（local/p14） | PRO 6000（config） | 含义 |
|---|---|---|---|
| `num_envs` | 2 | **4 → 可上调** | 并行环境数；98GB 可试 8–16，看显存余量 |
| `max_steps_per_episode` | 500 | **600000** | 完整 60s 交战（4090 跑不开） |
| `psro_iterations` | 15–20 | 30 | 更充分 |
| `population_cap` | 10 | 50 | 更大对手池 |
| `n_eval_games` | 默认 | 10 | 评估更稳 |

**调显存**：起跑后 `nvidia-smi` 看占用，有余量就 `num_envs` 往上加（4090 上 num_envs=2 峰值 ~15GB，98GB 理论可到 ~12 env，建议从 8 起步逐步加）。
**EW 递进**（与 4090 同序，落盘改持久盘）：`laser_25x25_league` → `ew_race`（干扰对抗）→ `ew_exposure`（辐射暴露前沿，崩过那次）。

---

## 5. 检查清单

- [ ] `ssh pro6000 nvidia-smi` 通，能看到 PRO 6000
- [ ] B 端 `torch.cuda.get_arch_list()` 含 `sm_120`
- [ ] 仓库在 B 上 checkout 到 `evo/laser-fix`
- [ ] 训练在 tmux 里跑，`logs/*.log` 有 `[PSRO`/`[Eval` 行
- [ ] checkpoint_dir 指向持久盘（**不是 /tmp**，避免重蹈写满磁盘崩溃）
- [ ] 显存有余量则上调 `num_envs`
