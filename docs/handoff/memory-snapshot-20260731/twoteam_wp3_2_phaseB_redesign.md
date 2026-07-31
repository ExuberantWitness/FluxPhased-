---
name: twoteam-wp3-2-phaseb-redesign
description: "WP-3.2 Phase B.1 — 推翻\"PPO毁BC sweep\"前提;4层缺陷链全修复;bc5 G1 kill 0.48-0.61;G2 PASS(RL kill=BC 57-62%,训练不再摧毁BC);G3 ablation 启动"
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

# WP-3.2 Phase B/B.1 根因与修复(2026-07-21~22)

**Phase A/B 全部 11 次失败跑的前提被推翻**: "PPO 摧毁了 BC 的 sweep 行为"是错的——
神经 BC base 从来就没有学会扫(H3 判决: 纯 BC ckpt crossplay kill 0.016 vs teacher 0.906)。
所有失败跑都是从一个坏掉的固定波束策略出发的。

## 四层缺陷链(全部定量验证并修复)

1. **D1 BC beam loss**: 联合 Beta-NLL 把 beam 头逼成退化常数(α=71.7/β=0.5 → az 钉死
   3.098 rad,网络忽略全部 obs)。确定性扫掠教师需要回归而非似然。
   修复: (sin,cos) L2 回归(beam_sincos_head),val_beam 0.027 → 0.0007(20-40×)。
   Beta-mean 参数化会压缩极值,不能用。

2. **D3 信道链 3 bug**(连环,任一都致命):
   - BlindClassical ECCM 规则把**所有孔径跳到同一信道** → 队内共信道自干扰
     JNR 1e3-7e5 → 双雷达全盲。修复: 分集跳频(孔径 k 跳 +k+1),任意起点一步恢复分集。
   - combine_team_actions 只在**双方都**给 channel_select 时才堆叠 → vs extreme 对手时
     静默丢弃 → env 永不应用 → 全部钉在 ch0。修复: 单方给时另一方补"保持当前信道"。
   - BC 采集数据因此完全没有有效 ECCM 示例(bc4 chan_match 0.017)。
   - 修复后 teacher vs extreme kill 0.5(之前 0);bc5 G1 crossplay kill 0.484(低干扰)/
     0.609(高干扰),vs 修复后 teacher baseline 0.875/0.922。

3. **D4 forward/evaluate clamp 不一致毒化 PPO ratio**(最严重,存在于全部 11 次先前跑):
   BC 训练的 freq_hop 头极度集中于 u=0(教师 hop_low=1.0),α≈0.5 时 Beta 密度在
   u→0 发散;**49% 样本**落在 evaluate_actions 的 [1e-4,1-1e-4] clamp 区内,而 forward
   不 clamp → lp_old≠lp_new 系统性偏差 ~1.2/孔径 → iter-0 kl=1.26(应为 0),ratio 从
   第一步就是错的。修复: forward 在 logp 和 action 计算**前** clamp u(3 处:
   freq_hop/beam Beta/residual Beta)。修后 warmup kl=-0.0005。

4. **ResiP 结构**(arXiv:2407.16677,唯一主流 residual-RL 结构): 冻结**整个** BC actor
   (trunk+detect_mlp+6头),独立 residual_trunk(detach 输入),residual 头初始化
   Beta(1.2,1.2)(SLowRL 恒等精神)。Phase B 的 R3"共享 trunk+冻 head"架构层面就是错的
   (trunk 漂移 0.15 → 函数空间 beam 偏 0.618 rad = 8.2×HPBW)。

## 关键教训

- **评估基线会移动**: 修复 teacher ECCM 后,同一 crossplay 的 BC kill 从 0.172 涨到 0.922
  (高干扰)——teacher 之前只是靠对手更烂而显得强。
- **静态干扰测试失效**: pytest 干扰测试原断言 kill≤0.3,但分集跳频正当击败静态同信道
  干扰(真实 ECCM 行为)。按测试自身 docstring 的 spec 授权改为**跟随干扰**(复制教师
  确定性 ECCM 规则预测下一信道)+ 全 duty + jam 0.6 → collapse 性质保留(kill 0.963→PASS)。
- **教师自对弈是 0-0 对称 floor**: 镜像策略信道锁步,互相致盲,det=0。评估 RL 的
  真实目标是对教师取得非对称优势(打破锁步),不是复制教师。
- **kill 链机制**: beam→detection→tracker_init→trace_P<tau_track→激光门→radar_E→kill;
  HPBW=0.075 rad,e_kill=2.0。任一环节断则全链归零。

## 状态

bc5(val_beam 0.0007)冻结为 residual base;G0 microverify 6/6;pytest 109/109。

**G2 PASS(2026-07-22)**: 100-iter joint,bc5 初始化,Δ=0.0375,beam_only,warmup 5,
Beta-KL 0.3。训练 124 min 健康(kl 全程 0.002-0.01,clip≤0.07,warmup it<5 kl=0)。
Crossplay gate(RL kill ≥ BC×50%):低干扰 RL 0.531 vs BC 0.859(62%)✅;
高干扰 RL 0.500 vs BC 0.875(57%)✅。对比 Phase B 的 0.000-0.016——"训练摧毁 BC"
问题彻底解决。但 RL 未超越 BC(residual 几乎未动,kl 极小),未达 STRONG PASS(80%)。
G1 纯 bc5 为 0.484/0.609,G2 与之统计无差(±0.5 std, n=64)——ResiP 保住了 BC
行为但 100 iter 内 residual 没学出非对称优势。
另修: resfit_head_mode 入 ckpt(G3 "all" 配置 eval 保真必需,此前会静默回退 beam_only)。

G3 ablation(2026-07-22 完成): Δ∈{0.019,0.0375,0.087}×head-mode∈{beam_only,all}
6 配置×100-iter joint 全过 gate(RL kill 0.50-0.61 = BC 54-71%),但统计上互相无差
(SEM≈0.06,n=64)且与纯 bc5(G1 0.484/0.609)无差。**100 iter 内 Δ 与 head-mode
都不是有效杠杆**——residual 在 kl≈0.002-0.01 保守更新下基本没动。结构修复(不再
摧毁 BC)与"学出非对称优势"是两回事。G3b(4 PPO mode × 最优配置): hd/ippo/mappo
全 ≈0.5,PPO mode 同样无区分。消融(joint ±warmup ±beta-KL,n_envs=128):
nowarmup 0.493/nokl 0.571/plain 0.531 vs 控制 0.578,kl 轨迹全部 0.001-0.006 无爆
——**D4 clamp 修复才是真正的承重稳定件,warmup/beta-KL 是冗余保险**(便宜,保留)。
RL 能力曲线(iter 0/50/100 × 10 跑): 全部平坦(wp3_phaseB1_rl_curve.png)。
名义最优 d0.0375_all(均值 0.578)。Commits ccf18be, 3f66aa2(twoteam/bc-ppo)。
后续见 [[twoteam-d2-multiteacher-pilot]]。
相关: [[twoteam-wp3-2-phaseA-fail]] [[twoteam-wp3-1-beam-sweep-collapse]]
