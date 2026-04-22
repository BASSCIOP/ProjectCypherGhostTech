"""Rule-based scoring engine for detection probability analysis.

Calculates detection probabilities for each anonymity protocol against each
adversary tier by combining:
1. Base technique effectiveness against the protocol
2. Adversary capability to employ each technique
3. Aggregation into overall risk scores with confidence intervals

Scoring formula per technique:
    detection_prob = technique_effectiveness(protocol) * adversary_capability(technique)

Overall risk score (0-100):
    Combines individual technique probabilities using a union-of-independent-events
    model, then scales to 0-100.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from threat_modeling.data.adversaries import AdversaryTier, get_adversary, list_adversaries
from threat_modeling.data.techniques import DetectionTechnique, list_techniques
from threat_modeling.data.protocols import AnonymityProtocol, get_protocol, list_protocols


@dataclass
class TechniqueResult:
    """Result of evaluating one detection technique.

    Attributes:
        technique_id: ID of the detection technique.
        technique_name: Human-readable technique name.
        base_effectiveness: Raw technique effectiveness against the protocol (0-1).
        adversary_capability: How well the adversary can employ this technique (0-1).
        detection_probability: Combined probability (base * capability).
        difficulty: Technique difficulty level.
    """
    technique_id: str
    technique_name: str
    base_effectiveness: float
    adversary_capability: float
    detection_probability: float
    difficulty: str


@dataclass
class AnalysisResult:
    """Complete analysis result for one protocol vs one adversary tier.

    Attributes:
        protocol_id: The analysed protocol.
        protocol_name: Human-readable protocol name.
        adversary_id: The adversary tier.
        adversary_name: Human-readable adversary name.
        technique_results: Per-technique detection probabilities.
        overall_risk_score: Aggregated risk score (0-100).
        confidence_low: Lower bound of 90% confidence interval.
        confidence_high: Upper bound of 90% confidence interval.
        risk_level: Qualitative label ('Low', 'Moderate', 'High', 'Critical').
        mitigation_recommendations: Suggested mitigations.
    """
    protocol_id: str
    protocol_name: str
    adversary_id: str
    adversary_name: str
    technique_results: List[TechniqueResult]
    overall_risk_score: float
    confidence_low: float
    confidence_high: float
    risk_level: str
    mitigation_recommendations: List[str]


class ScoringEngine:
    """Rule-based scoring engine for anonymity threat analysis.

    Usage::

        engine = ScoringEngine()
        result = engine.analyze("tor", "nation_state")
        print(result.overall_risk_score)  # 0-100
    """

    # Confidence interval half-width as fraction of score, by adversary tier.
    # Nation-states have narrower intervals (more certain assessment) while
    # lower tiers have wider intervals (more uncertainty).
    _CONFIDENCE_HALFWIDTH = {
        "isp":          0.15,
        "corporate":    0.12,
        "apt":          0.10,
        "nation_state": 0.08,
    }

    def analyze(
        self,
        protocol_id: str,
        adversary_id: str,
    ) -> AnalysisResult:
        """Analyze a single protocol against a single adversary tier.

        Args:
            protocol_id: Protocol identifier (e.g., 'tor', 'vpn').
            adversary_id: Adversary tier identifier (e.g., 'isp', 'nation_state').

        Returns:
            AnalysisResult with per-technique breakdowns and overall risk score.

        Raises:
            KeyError: If protocol or adversary ID is not recognized.
        """
        protocol = get_protocol(protocol_id)
        adversary = get_adversary(adversary_id)
        techniques = list_techniques()

        technique_results: List[TechniqueResult] = []

        for tech in techniques:
            base_eff = tech.effectiveness_against(protocol_id)
            adv_cap = adversary.effectiveness_for(tech.id)
            det_prob = base_eff * adv_cap

            technique_results.append(TechniqueResult(
                technique_id=tech.id,
                technique_name=tech.name,
                base_effectiveness=round(base_eff, 4),
                adversary_capability=round(adv_cap, 4),
                detection_probability=round(det_prob, 4),
                difficulty=tech.difficulty,
            ))

        # Aggregate: probability of at least one technique succeeding
        # P(detect) = 1 - product(1 - p_i)
        probs = [tr.detection_probability for tr in technique_results]
        p_none = 1.0
        for p in probs:
            p_none *= (1.0 - p)
        overall_prob = 1.0 - p_none

        risk_score = round(overall_prob * 100, 1)

        # Confidence interval
        hw_frac = self._CONFIDENCE_HALFWIDTH.get(adversary_id, 0.12)
        hw = risk_score * hw_frac
        conf_low = round(max(0.0, risk_score - hw), 1)
        conf_high = round(min(100.0, risk_score + hw), 1)

        risk_level = self._risk_label(risk_score)
        mitigations = self._generate_mitigations(protocol, adversary, technique_results)

        return AnalysisResult(
            protocol_id=protocol.id,
            protocol_name=protocol.name,
            adversary_id=adversary.id,
            adversary_name=adversary.name,
            technique_results=technique_results,
            overall_risk_score=risk_score,
            confidence_low=conf_low,
            confidence_high=conf_high,
            risk_level=risk_level,
            mitigation_recommendations=mitigations,
        )

    def compare(
        self,
        protocol_ids: List[str],
        adversary_id: Optional[str] = None,
    ) -> Dict[str, List[AnalysisResult]]:
        """Compare multiple protocols against one or all adversary tiers.

        Args:
            protocol_ids: List of protocol identifiers to compare.
            adversary_id: If provided, compare against a single adversary tier.
                          If None, compare against all adversary tiers.

        Returns:
            Dict mapping protocol_id -> list of AnalysisResult (one per adversary).
        """
        adversaries = (
            [get_adversary(adversary_id)]
            if adversary_id
            else list_adversaries()
        )
        results: Dict[str, List[AnalysisResult]] = {}
        for pid in protocol_ids:
            results[pid] = [
                self.analyze(pid, adv.id) for adv in adversaries
            ]
        return results

    def analyze_all(self) -> Dict[str, List[AnalysisResult]]:
        """Analyze all protocols against all adversary tiers.

        Returns:
            Dict mapping protocol_id -> list of AnalysisResult.
        """
        return self.compare([p.id for p in list_protocols()])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_label(score: float) -> str:
        """Convert a 0-100 risk score to a qualitative label."""
        if score < 25:
            return "Low"
        elif score < 50:
            return "Moderate"
        elif score < 75:
            return "High"
        else:
            return "Critical"

    @staticmethod
    def _generate_mitigations(
        protocol: AnonymityProtocol,
        adversary: AdversaryTier,
        technique_results: List[TechniqueResult],
    ) -> List[str]:
        """Generate mitigation recommendations based on top threats.

        Recommendations target the highest-probability detection techniques
        and are tailored to the protocol and adversary tier.
        """
        # Sort by detection probability descending
        sorted_results = sorted(
            technique_results, key=lambda r: r.detection_probability, reverse=True
        )

        mitigations: List[str] = []

        # Mitigation lookup by technique ID
        _MITIGATION_DB = {
            "traffic_volume": (
                "Use traffic padding or constant-rate transmission to obscure "
                "volume-based correlation. Consider cover traffic generators."
            ),
            "timing_correlation": (
                "Add random delays and jitter to traffic flows. Use protocols "
                "with built-in traffic shaping. Avoid predictable communication "
                "patterns."
            ),
            "protocol_fingerprint": (
                "Use pluggable transports or obfuscation layers (e.g., obfs4 for "
                "Tor) to disguise protocol fingerprints. Tunnel anonymity traffic "
                "inside common protocols (HTTPS, WebSocket)."
            ),
            "entry_exit_corr": (
                "Use multi-hop routing with diverse relay selection. Avoid using "
                "anonymity networks where the adversary controls both the entry "
                "and exit network segments. Prefer networks with high traffic volume."
            ),
            "dns_leakage": (
                "Ensure all DNS queries route through the anonymity tunnel. Use "
                "DNS-over-HTTPS (DoH) or DNS-over-TLS (DoT). Verify configuration "
                "with leak testing tools. Disable fallback resolvers."
            ),
            "webrtc_leakage": (
                "Disable WebRTC in the browser or use a browser that blocks WebRTC "
                "leaks by default (e.g., Tor Browser). Use browser extensions to "
                "prevent STUN requests."
            ),
            "ip_reputation": (
                "Use bridge relays or unlisted entry points not present in public "
                "blocklists. Rotate VPN servers frequently. Use residential IP "
                "addresses when available."
            ),
            "behavioral_pattern": (
                "Vary browsing habits and session patterns. Use separate anonymity "
                "sessions for different activities. Avoid logging into personally "
                "identifiable accounts while using anonymity tools."
            ),
            "circuit_analysis": (
                "Use protocols that frequently rotate circuits/tunnels. Prefer "
                "networks with large user bases. Avoid long-lived circuits."
            ),
            "metadata_analysis": (
                "Minimize session duration and frequency patterns. Use different "
                "anonymity tools for different activities. Avoid establishing "
                "regular schedules for anonymous communication."
            ),
            "side_channel_timing": (
                "Keep anonymity software updated to patch known side-channel "
                "vulnerabilities. Use hardened operating systems with process "
                "isolation (e.g., Whonix, Tails)."
            ),
            "infra_mapping": (
                "Use bridge nodes or private entry points not listed in public "
                "directories. Prefer decentralized networks where infrastructure "
                "is harder to enumerate comprehensively."
            ),
        }

        # Add mitigations for top-5 threats with probability > 0.05
        seen = set()
        for result in sorted_results:
            if result.detection_probability < 0.05:
                continue
            if result.technique_id in seen:
                continue
            seen.add(result.technique_id)
            mitigation = _MITIGATION_DB.get(result.technique_id)
            if mitigation:
                mitigations.append(
                    f"[{result.technique_name}] (detection prob: "
                    f"{result.detection_probability:.0%}): {mitigation}"
                )
            if len(mitigations) >= 5:
                break

        # Add general advice for high-capability adversaries
        if adversary.id in ("apt", "nation_state"):
            mitigations.append(
                "[General] Against advanced adversaries: use defense-in-depth "
                "by combining multiple anonymity tools, practicing strict "
                "operational security (OPSEC), and compartmentalizing activities "
                "across separate anonymity sessions and identities."
            )

        if not mitigations:
            mitigations.append(
                "Risk is low for this protocol/adversary combination. "
                "Maintain standard anonymity hygiene and keep software updated."
            )

        return mitigations
