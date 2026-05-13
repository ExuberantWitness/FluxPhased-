"""Structured evaluation report generation.

Aggregates results from all evaluation modules into a unified report
that can be exported as dict, JSON, or Markdown.
"""

import json
import os
from datetime import datetime
from typing import Optional


class EvaluationReport:
    """Aggregated evaluation report from all metric modules."""

    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.perception: dict = {}
        self.combat: dict = {}
        self.game: dict = {}
        self.comm: dict = {}
        self.sensitivity: dict = {}
        self.cde: dict = {}
        self.timing: dict = {}
        self.metadata: dict = {}

    def to_dict(self) -> dict:
        """Flat dict representation of all metrics."""
        return {
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "perception": self.perception,
            "combat": self.combat,
            "game": self.game,
            "comm": self.comm,
            "sensitivity": self.sensitivity,
            "cde": self.cde,
            "timing": self.timing,
        }

    def to_json(self, path: str):
        """Serialize to JSON file."""
        d = self.to_dict()
        # Convert any non-serializable types
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(i) for i in obj]
            elif hasattr(obj, "item"):
                return obj.item()
            elif isinstance(obj, float) and (obj != obj):  # NaN check
                return None
            return obj

        d = make_serializable(d)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

    def to_markdown(self, path: str):
        """Generate human-readable Markdown report."""
        lines = [
            f"# FluxPhased 效能评估报告",
            f"",
            f"**时间**: {self.timestamp}",
            f"",
        ]

        if self.metadata:
            lines.append("## 环境配置")
            lines.append("")
            for k, v in self.metadata.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        sections = [
            ("感知效能", self.perception),
            ("作战决策质量", self.combat),
            ("对抗博弈", self.game),
            ("通信质量", self.comm),
            ("敏感性分析", self.sensitivity),
            ("CDE 综合指标", self.cde),
            ("处理延迟", self.timing),
        ]

        for title, data in sections:
            if not data:
                continue
            lines.append(f"## {title}")
            lines.append("")
            self._flatten_dict(data, lines, indent=0)
            lines.append("")

        content = "\n".join(lines)
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        return content

    def _flatten_dict(self, d, lines, indent=0):
        """Recursively flatten nested dicts into markdown lines."""
        prefix = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{prefix}- **{k}**:")
                self._flatten_dict(v, lines, indent + 1)
            elif isinstance(v, list):
                lines.append(f"{prefix}- **{k}**: [{len(v)} items]")
            elif isinstance(v, float):
                if v != v:  # NaN
                    lines.append(f"{prefix}- **{k}**: N/A")
                else:
                    lines.append(f"{prefix}- **{k}**: {v:.4f}")
            else:
                lines.append(f"{prefix}- **{k}**: {v}")

    def summary(self) -> str:
        """One-line summary string."""
        parts = []
        if self.perception:
            ra = self.perception.get("range_accuracy", "N/A")
            parts.append(f"range_acc={ra}")
        if self.combat:
            kr = self.combat.get("kill_rate", "N/A")
            parts.append(f"kill_rate={kr}")
        if self.game:
            wr = self.game.get("win_rate_red", "N/A")
            parts.append(f"win_rate_red={wr}")
        if self.cde:
            cde = self.cde.get("cde", "N/A")
            parts.append(f"CDE={cde}")
        return " | ".join(parts) if parts else "No metrics computed"
