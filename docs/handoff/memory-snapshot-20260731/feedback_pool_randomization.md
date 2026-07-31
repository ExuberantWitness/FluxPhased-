---
name: feedback-pool-randomization
description: "联赛/对手池的随机化必须池级推广,不能只做进单个成员 — 用户 2026-07-25 纠正:'不要搞固定的重干扰规则池,我之前让你引入随机性'"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

# 对手池随机化要池级,不是单成员

用户原话(2026-07-25):"不要搞'面对固定的重干扰规则池',我之前让你引入随机性,你为什么不?"

**规则**:当用户要求"对手随机性/随机化"时,默认含义是**整个对手池每个成员都有 episode 级参数/风格随机化**,而不是只做一个带随机化的对手(如 AdaptiveJamRuleCommander)放进池里、其余成员保持确定性固定参数。

**Why**: P1' 联赛中我只做了"每 episode 随机抽池成员"(PFSP 抽样),但每个规则成员本身参数固定。用户指出这不算引入随机性——固定成员让降级/条件策略没有学习信号,是 P1' 升级陷阱的两个根因之一。Tier 2.1 计划里的三路随机化(风格切换/探测/强度抖动)当时只做进了 AJR 一个成员,没有推广。

**How to apply**: 涉及对手池/联赛训练时:①新随机化机制默认做成可包任意成员的 wrapper(如 RandomizedRuleWrapper 模式,种子派生自 env._reset_count 保持可复现);②池初始化时对 rule/extreme/exploit 全家族生效;③向用户汇报时明确说明"哪些成员有随机化、哪些没有"。相关:[[twoteam-idea-e-em-contest]]
