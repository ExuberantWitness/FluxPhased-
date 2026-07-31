---
name: chinese-only-responses
description: "User reads Chinese, not English; respond in Chinese for all user-facing communication"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 902a7f7f-2d60-4a53-b927-2a75af5c8fc4
---

User reads Chinese, not English. All user-facing text (questions, summaries, status reports, AskUserQuestion content) must be in Chinese.

**Why:** User said verbatim "用中文说，我英文不好" on 2026-07-14 when shown an English AskUserQuestion. They can't parse long English option text.

**How to apply:**
- AskUserQuestion: questions, option labels, descriptions — all Chinese
- Status updates and end-of-turn summaries — Chinese
- Code, file paths, log output, technical identifiers — keep as-is (don't translate)
- Markdown reports destined for user reading — Chinese prose, English/code where natural
- Memory files — Chinese is fine (user-facing); keep code identifiers in English
