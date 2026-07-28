# PRO6000 Agent Implementation Specification

当前固定版见 [PRO6000_AGENT_IMPLEMENTATION_SPEC_20260728_023103.md](PRO6000_AGENT_IMPLEMENTATION_SPEC_20260728_023103.md)。

当前状态：`BLOCK_PPO_PROVENANCE`。PRO6000 目前仅获授权执行 P0 source handoff 与 symbol resolution。

执行原则：先验证 `SOURCE_HANDOFF.json`；再确认 emitter/receiver/resource/selectivity 的物理绑定；之后才做 IQ calibration、平台约束的 budgeted target allocation、masked categorical、oracle/headroom。没有 untouched all-script robust headroom 不运行 PPO。

可直接复制给 executor 的指令见 [PRO6000_EXECUTION_PROMPT.md](PRO6000_EXECUTION_PROMPT.md)。
