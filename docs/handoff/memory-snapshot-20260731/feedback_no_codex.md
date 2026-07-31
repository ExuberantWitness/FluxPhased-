---
name: feedback-no-codex
description: 用户明确禁止使用 codex MCP(mcp__codex__codex / codex-reply)做评审或头脑风暴;idea-discovery 等 pipeline 中需要外部模型评审的环节由 Claude 自己完成
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

不要使用 codex MCP 工具(mcp__codex__codex / mcp__codex__codex-reply)。

**Why**: 2026-07-24 idea-discovery Phase 2 中我用 codex 做头脑风暴,用户立即中断并指示"不要使用codex"。

**How to apply**: research pipeline(idea-creator / research-review / novelty-check 等)中所有标注"GPT-5.5 via Codex MCP"的环节,改由我自己(Claude)直接完成分析、头脑风暴与评审,不再调用 codex 工具。
