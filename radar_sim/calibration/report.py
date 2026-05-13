"""Calibration report generation."""

import os
import numpy as np
from typing import Optional

from .estimator import EstimationResult


class CalibrationReport:
    """Generate human-readable calibration reports."""

    def generate_markdown(self, result: EstimationResult) -> str:
        """Generate markdown report from estimation result."""
        lines = [
            "# FluxPhased Sim2Real 参数标定报告",
            "",
            f"**优化方法**: {result.method}",
            f"**收敛状态**: {'成功' if result.success else '未收敛'} — {result.message}",
            f"**评估次数**: {result.n_evaluations}",
            "",
            "## 参数估计结果",
            "",
            "| 参数 | 真实值 | 估计值 | 误差 | 误差% |",
            "|------|--------|--------|------|-------|",
        ]

        for name in result.estimated_params:
            true_v = result.true_params.get(name, float('nan'))
            est_v = result.estimated_params[name]
            err = est_v - true_v
            err_pct = abs(err / abs(true_v)) * 100 if true_v != 0 else float('inf')
            lines.append(f"| {name} | {true_v:.4f} | {est_v:.4f} | {err:+.4f} | {err_pct:.1f}% |")

        if result.convergence_history:
            lines.extend([
                "",
                "## 收敛历史",
                "",
                f"- 初始残差: {result.convergence_history[0]:.6f}",
                f"- 最终残差: {result.convergence_history[-1]:.6f}",
                f"- 残差降低: {(1 - result.convergence_history[-1] / max(result.convergence_history[0], 1e-30)) * 100:.1f}%",
            ])

        if result.covariance is not None:
            lines.extend([
                "",
                "## 参数协方差矩阵",
                "",
                "```",
            ])
            for row in result.covariance:
                lines.append("  " + "  ".join(f"{v:.2e}" for v in row))
            lines.append("```")

        return "\n".join(lines)

    def save_report(self, result: EstimationResult, path: str):
        """Save report to markdown file."""
        md = self.generate_markdown(result)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)

    def plot_convergence(self, result: EstimationResult, path: Optional[str] = None):
        """Plot convergence history."""
        if not result.convergence_history:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(result.convergence_history, 'b-', linewidth=0.8)
        ax.set_xlabel('Evaluation')
        ax.set_ylabel('Residual')
        ax.set_title(f'Convergence ({result.method})')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
