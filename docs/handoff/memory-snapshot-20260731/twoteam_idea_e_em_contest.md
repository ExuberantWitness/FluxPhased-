---
name: twoteam-idea-e-em-contest
description: "Idea E 纯电磁持续对抗(kill 禁用)全程:P0' 探针发现 bc:bc 升级陷阱(BC 非 EM-纳什)+ bc:js 互相降级均衡存在;P1' 零和 EM reward FAIL(升级陷阱是吸引子);P2' = general-sum v4 + 池子随机化,进行中"
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

# Idea E 纯电磁持续对抗(2026-07-24~25)

**Why**: kill 指标(激光 50m + 暴露即杀伤 = 终端稀疏二值)被判定为 5 条负结果线的共同根因;规则教师的近纳什是 kill 链特异的。用户决策:kill 完全禁用(env kill_enabled=False),固定 horizon 持续对抗,蓝=规则池+RL 快照随机上,红=联赛种群,目标是演化出侦察/探测/通信/干扰的丰富战术。

**How to apply**: EM 博弈的 reward 设计必须避开"零和差分 + 固定对手"组合(见下 P1' 教训);评价 EM 策略用 tracks_ok/jam_on_me/emit_duty/exposure/comm_ok 稠密指标,不用 kill。

## P0' 零训练探针(7 cells × 双向 × 32 eps × 300 步)
- EM 指标稠密有方差(tracks_ok 0.0~0.96,jam 0.005~1.0)→ 奖励基底合格,冷启动梯度墙消失
- kill 模式 RL 快照在 EM 维度极弱(tracks 0.14 vs BC 0.81)→ EM 是真正不同的游戏
- **bc:bc 镜像 = 互相致盲僵局(jam=1.0, tracks≈0)→ 升级陷阱/囚徒困境结构;BC 在 EM 博弈非纳什,头部空间确认**
- **bc:js 对局双方 tracks 0.95+/0.95+、jam 0.005 → BC 干扰是反应式的(jamreact_tau 门控),对面安静它就安静,互相降级均衡客观存在**

## P1' FAIL(2026-07-25,高信息量)
- V1(零和差分:tracks+0.3·comm−0.2·暴露−0.1·发射,red−blue)/ V2(+0.5·施加干扰,退化对照)各 100-iter 联赛
- 训练后策略 EM 维度比 BC init 退化 10 倍:tracks 0.23/0.18 vs BC 0.79;js 规则在 reward 上双杀 RL(0.69 vs 0.43)
- **v1:rl、v1:v2 镜像 = 互相致盲(tracks 0.05/0.04)→ 零和差分下"弄瞎对面"≡"自己看清",升级是对每个固定对手的优势策略,升级陷阱成为吸引子**
- 根因二元:①零和差分 reward;②池成员固定参数不条件回应 → 降级策略无学习信号

## P2'(FAIL,2026-07-25)
- general-sum reward v4 + RandomizedRuleWrapper 全池随机化,100-iter;Gate(6 cells×双向×32eps)判决:
- tracks 0.15 vs BC 0.78(5× 差,决定性 FAIL);vs AJR 0.09 vs 0.38;最好 cell vs js 0.52 略胜 js 0.48
- 降级未涌现:emit_duty 0.98;v4:v1 仍互相致盲(0.04/0.04);comm_ok=0.000 全局(自致残,非被干扰)
- **交叉负结果(最重要)**:v1(零和)0.23 / v2 0.18 / v4(general-sum+随机池)0.15 —— 三 reward 变体同样从 BC-init(0.79)退化 ⇒ **reward 设计不是瓶颈;PPO 从 BC-init 微调 100-iter 内必然打断检测→跟踪链,与 reward 无关**。Idea E reward 修复路线关闭。
- 预算 ~7/8 GPU-h 已用;下一步方向决策交用户(Idea D 诚实负结果论文素材已齐 vs 新方向)

## 深度归因(2026-07-25):为什么连提升趋势都没有 — 问题设置四层缺陷
1. **reward 四项死三项(代码级)**: exposure 纯 emission 累积(env:907)→ 全员 emit≈1 → 常数 0.024 零梯度;comm_link_ok = 两孔径 comm alloc 均≥0.10 硬阈值 all()(env:703)→ BC 住上方(0.81)克隆住下方(0.000)悬崖无坡度零梯度;−0.1·emit 被 tracks(1.0)量级压制。活跃 reward 只剩 tracks。
2. **tracks 梯度信噪比极低**: IMM-PDAF 协方差积分量,单步动作影响~0.001 ≪ episode 间方差;轨迹 0.002→0.19→0.15 只学到"别招 BC 反应式干扰";vs js 100-iter 不动(0.514→0.519)。
3. **联赛仪器挂 kill 上**: wr_vs_opp 恒 0.50 → PFSP ema_var=0 → 联赛退化为随机抽对手。
4. **起点错配**: kill 模式克隆 EM 起点 tracks=0.002 非 0.79;"退化 10×"叙事错误,实际是"从 0.002 爬 0.19 后停滞"。
**修复清单(ROI 序)**: comm 软阈值(1 行)/ 删 exposure+emit 死项 / tracks_ok→PBRS trace_P 势差(br_trainer 现成 shape_trace_P_reduction_bonus)/ EM 模式重做 BC 预训练 / PFSP+eval 换 em_score。
**方法论教训**: 搬文献 reward 成分前,必须先验证每个成分在自己 env 里是"活"的(有方差、有策略梯度路径);搬游戏框架时,联赛的配套仪器(PFSP/eval/快照)必须同步换指标。

## 开发教训
- **commander action dict 的 key 是可选的**:extreme/exploit 不发 channel_select/beam_target,任何 wrapper 后处理必须 `if key in act` 守卫(本次 smoke 'channel_select' KeyError;Phase B.1 D3 已踩过一次,这是第二次)
- 跨协议数字不可比:Tier 2.1 的 5σ  headline 在同协议对照下消失(见 [[twoteam-tier21-adaptive-outcome]])
- 相关:[[twoteam-tier223-pfspfix-outcome]]、[[feedback-pool-randomization]]
