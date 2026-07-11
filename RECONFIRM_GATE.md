# R2 前 Re-confirm 门(给 PRO6000)— ~1-2 GPU-h,放全网格前的最后一道便宜裁决

**为什么**:预飞行门暴露两件事——① **α_eff bug** 使之前 WP2 的 MAPPO 暗中是纯 IPPO → **prior headline(n4_L1 双赢)全部失效,必须用修好的代码重测**;② sanity 显示**经典在 n8 完胜**(8.0 vs MAPPO 5.58)、**MAPPO 在 n4_L3 输经典**(3.54 < 3.583)。**在烧 1120-ep 全网格前,先用修好的代码确认"学习赢经典(kill 或 survival-Pareto)"还成不成立。**

## Task A — 核心主张重confirm(命门,60 评估)
- 方法:{**MAPPO(α_eff 已修)**, **IPPO**, **强模块化经典**};
- 格子:{`n4_L0`, `n4_L1-τ1`, `n4_L3-trained`, `n8_L0`};
- 5 seed;**每格同时报 kill 和 survival**。

## Task B — league-PFSP L3(硬上限 ~1-2 GPU-h,可并行)
- 目标:input-adaptive 的 L3,drop vs L1-τ1 **≥0.05** 且策略输出随 red task histogram 变化(证真自适应);
- **硬上限**:预算内做不到 → 接受"常数最优 jammer",论文如实叫 `trained worst-case (near-constant)`,**不钻 league 兔子洞**。

## 门判据
| 结果 | 决定 |
|---|---|
| **PASS**:学习(MAPPO 或 IPPO)在 n4_L1 **kill 赢经典**;**或** 清晰 **survival-Pareto 占优**(≥同 kill 且 survival 显著更高)@ n4_L1/L3 | **APPROVE 全 R2(1120)** |
| **FAIL**:修好后学习在 n4 **kill 也只平/输经典 且 无 survival 优势** | **STOP,别烧 R2**,重构框架/退 |

## PASS 后的诚实框架(论文这么写,已锁定)
1. **headline = "学习式指挥官"报 MAPPO 和 IPPO 两者**(MAPPO 助 scale、IPPO 在重 EW 更稳),**CTDE-under-noise 脆弱是诚实发现,不是减分**;
2. **操作包线框架**:经典在 n8 规模占优、学习在 n4-中等 EW 带扩展包线——**如实说"学习在哪帮、经典在哪够用"**,不吹全面碾压;
3. **L3 = trained worst-case (near-constant)**,jammer-adaptivity 增益弱本身如实报。

## 回报
贴回:4 格 × {MAPPO, IPPO, 经典} 的 **kill + survival(per-seed + mean + CI)** + league-L3 结果(drop vs L1-τ1 + 是否 input-adaptive)。据此判 APPROVE 全 R2 / STOP 重构。
