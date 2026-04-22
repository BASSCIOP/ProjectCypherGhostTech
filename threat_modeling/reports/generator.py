"""Report generation for threat analysis results.

Supports:
- Detailed JSON reports with full technique breakdowns
- Markdown summaries with risk scores and recommendations
- Protocol comparison reports (both JSON and Markdown)
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from threat_modeling.data.adversaries import get_adversary, list_adversaries
from threat_modeling.data.protocols import get_protocol
from threat_modeling.data.techniques import list_techniques
from threat_modeling.engines.scoring import AnalysisResult, ScoringEngine, TechniqueResult


class ReportGenerator:
    """Generates JSON and Markdown reports from analysis results.

    Usage::

        engine = ScoringEngine()
        gen = ReportGenerator(engine)

        # Single analysis report
        result = engine.analyze("tor", "nation_state")
        json_str = gen.to_json(result)
        md_str = gen.to_markdown(result)

        # Comparison report
        comparison = engine.compare(["tor", "vpn", "proxy"])
        md_cmp = gen.comparison_markdown(comparison)
    """

    def __init__(self, engine: Optional[ScoringEngine] = None):
        self.engine = engine or ScoringEngine()

    # ------------------------------------------------------------------
    # JSON Reports
    # ------------------------------------------------------------------

    def to_json(self, result: AnalysisResult, indent: int = 2) -> str:
        """Generate a detailed JSON report from an AnalysisResult.

        Args:
            result: The analysis result to serialize.
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        data = self._result_to_dict(result)
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def comparison_json(
        self,
        comparison: Dict[str, List[AnalysisResult]],
        indent: int = 2,
    ) -> str:
        """Generate a JSON comparison report.

        Args:
            comparison: Output from ScoringEngine.compare().
            indent: JSON indentation level.

        Returns:
            JSON string containing the full comparison.
        """
        data = {
            "report_type": "protocol_comparison",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocols": {},
        }
        for pid, results in comparison.items():
            proto = get_protocol(pid)
            data["protocols"][pid] = {
                "name": proto.name,
                "adversary_analyses": [
                    self._result_to_dict(r) for r in results
                ],
            }
        return json.dumps(data, indent=indent, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Markdown Reports
    # ------------------------------------------------------------------

    def to_markdown(self, result: AnalysisResult) -> str:
        """Generate a Markdown summary report from an AnalysisResult.

        Args:
            result: The analysis result.

        Returns:
            Markdown formatted string.
        """
        lines: List[str] = []
        lines.append(f"# Threat Analysis: {result.protocol_name}")
        lines.append(f"**Adversary Tier:** {result.adversary_name}")
        lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")

        # Summary box
        lines.append("## Risk Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| **Overall Risk Score** | **{result.overall_risk_score}/100** |")
        lines.append(f"| **Risk Level** | **{result.risk_level}** |")
        lines.append(f"| **90% Confidence Interval** | {result.confidence_low} – {result.confidence_high} |")
        lines.append("")

        # Technique breakdown
        lines.append("## Detection Technique Breakdown")
        lines.append("")
        lines.append("| # | Technique | Base Effectiveness | Adversary Capability | Detection Probability | Difficulty |")
        lines.append("|---|-----------|-------------------|---------------------|----------------------|------------|")

        sorted_results = sorted(
            result.technique_results,
            key=lambda r: r.detection_probability,
            reverse=True,
        )
        for i, tr in enumerate(sorted_results, 1):
            lines.append(
                f"| {i} | {tr.technique_name} | "
                f"{tr.base_effectiveness:.0%} | "
                f"{tr.adversary_capability:.0%} | "
                f"**{tr.detection_probability:.0%}** | "
                f"{tr.difficulty} |"
            )
        lines.append("")

        # Top threats
        top_threats = [tr for tr in sorted_results if tr.detection_probability >= 0.10][:5]
        if top_threats:
            lines.append("## Top Threats")
            lines.append("")
            for i, tr in enumerate(top_threats, 1):
                lines.append(
                    f"{i}. **{tr.technique_name}** — "
                    f"{tr.detection_probability:.0%} detection probability "
                    f"({tr.difficulty} difficulty)"
                )
            lines.append("")

        # Mitigations
        lines.append("## Mitigation Recommendations")
        lines.append("")
        for i, mit in enumerate(result.mitigation_recommendations, 1):
            lines.append(f"{i}. {mit}")
        lines.append("")

        return "\n".join(lines)

    def comparison_markdown(
        self,
        comparison: Dict[str, List[AnalysisResult]],
    ) -> str:
        """Generate a Markdown comparison report across protocols.

        Args:
            comparison: Output from ScoringEngine.compare().

        Returns:
            Markdown formatted string with side-by-side comparison tables.
        """
        lines: List[str] = []
        lines.append("# Protocol Comparison Report")
        lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")

        protocol_ids = list(comparison.keys())
        if not protocol_ids:
            lines.append("_No protocols to compare._")
            return "\n".join(lines)

        # Determine adversary tiers from first protocol's results
        adversary_ids = [r.adversary_id for r in comparison[protocol_ids[0]]]

        # Overall risk score comparison table
        lines.append("## Overall Risk Scores (0-100)")
        lines.append("")
        header = "| Adversary Tier | " + " | ".join(
            get_protocol(pid).name for pid in protocol_ids
        ) + " |"
        separator = "|" + "|".join(["---"] * (len(protocol_ids) + 1)) + "|"
        lines.append(header)
        lines.append(separator)

        for adv_id in adversary_ids:
            adv = get_adversary(adv_id)
            row = f"| {adv.name} |"
            for pid in protocol_ids:
                result = next(
                    r for r in comparison[pid] if r.adversary_id == adv_id
                )
                score = result.overall_risk_score
                level = result.risk_level
                row += f" **{score}** ({level}) |"
            lines.append(row)
        lines.append("")

        # Risk level legend
        lines.append("> **Risk Levels:** Low (0-24) · Moderate (25-49) · High (50-74) · Critical (75-100)")
        lines.append("")

        # Per-adversary detailed comparison
        for adv_id in adversary_ids:
            adv = get_adversary(adv_id)
            lines.append(f"## Detailed: vs {adv.name}")
            lines.append("")

            # Table header
            tech_header = "| Technique |"
            tech_sep = "|---|"
            for pid in protocol_ids:
                proto = get_protocol(pid)
                tech_header += f" {proto.name} |"
                tech_sep += "---|"
            lines.append(tech_header)
            lines.append(tech_sep)

            # Get technique names from first result
            first_result = next(
                r for r in comparison[protocol_ids[0]] if r.adversary_id == adv_id
            )
            technique_ids = [tr.technique_id for tr in first_result.technique_results]

            for tech_id in technique_ids:
                row = ""
                tech_name = ""
                for pid in protocol_ids:
                    result = next(
                        r for r in comparison[pid] if r.adversary_id == adv_id
                    )
                    tr = next(
                        t for t in result.technique_results if t.technique_id == tech_id
                    )
                    tech_name = tr.technique_name
                    row += f" {tr.detection_probability:.0%} |"
                lines.append(f"| {tech_name} |{row}")
            lines.append("")

        # Summary recommendations
        lines.append("## Summary")
        lines.append("")

        # Find safest and riskiest protocol per adversary
        for adv_id in adversary_ids:
            adv = get_adversary(adv_id)
            results_for_adv = []
            for pid in protocol_ids:
                result = next(
                    r for r in comparison[pid] if r.adversary_id == adv_id
                )
                results_for_adv.append(result)

            safest = min(results_for_adv, key=lambda r: r.overall_risk_score)
            riskiest = max(results_for_adv, key=lambda r: r.overall_risk_score)
            lines.append(
                f"- **vs {adv.name}:** Lowest risk = {safest.protocol_name} "
                f"({safest.overall_risk_score}), Highest risk = {riskiest.protocol_name} "
                f"({riskiest.overall_risk_score})"
            )
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # File output helpers
    # ------------------------------------------------------------------

    def save_report(
        self,
        result: AnalysisResult,
        path: str,
        fmt: str = "json",
    ) -> str:
        """Save a single analysis report to a file.

        Args:
            result: The analysis result.
            path: Output file path.
            fmt: 'json' or 'markdown'.

        Returns:
            The output file path.
        """
        content = self.to_json(result) if fmt == "json" else self.to_markdown(result)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def save_comparison(
        self,
        comparison: Dict[str, List[AnalysisResult]],
        path: str,
        fmt: str = "json",
    ) -> str:
        """Save a comparison report to a file.

        Args:
            comparison: Output from ScoringEngine.compare().
            path: Output file path.
            fmt: 'json' or 'markdown'.

        Returns:
            The output file path.
        """
        if fmt == "json":
            content = self.comparison_json(comparison)
        else:
            content = self.comparison_markdown(comparison)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_dict(result: AnalysisResult) -> dict:
        """Convert an AnalysisResult to a serializable dictionary."""
        return {
            "protocol": {
                "id": result.protocol_id,
                "name": result.protocol_name,
            },
            "adversary": {
                "id": result.adversary_id,
                "name": result.adversary_name,
            },
            "overall_risk_score": result.overall_risk_score,
            "risk_level": result.risk_level,
            "confidence_interval": {
                "low": result.confidence_low,
                "high": result.confidence_high,
                "confidence_level": "90%",
            },
            "technique_results": [
                {
                    "technique_id": tr.technique_id,
                    "technique_name": tr.technique_name,
                    "base_effectiveness": tr.base_effectiveness,
                    "adversary_capability": tr.adversary_capability,
                    "detection_probability": tr.detection_probability,
                    "difficulty": tr.difficulty,
                }
                for tr in result.technique_results
            ],
            "mitigation_recommendations": result.mitigation_recommendations,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
