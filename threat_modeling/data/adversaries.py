"""Adversary tier definitions and capability matrices.

Models four tiers of adversaries with increasing sophistication:
- ISP-Level: Passive network observers with access to traffic metadata
- Corporate Security: Enterprise security teams with DPI and endpoint visibility
- Advanced Persistent Threat (APT): Well-funded threat groups with advanced tooling
- Nation-State: State-sponsored actors with global surveillance capabilities

References:
- MITRE ATT&CK: T1040 (Network Sniffing), T1071 (Application Layer Protocol)
- MITRE ATT&CK: T1573 (Encrypted Channel), T1090 (Proxy)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AdversaryTier:
    """Represents an adversary tier with capabilities and resources.

    Attributes:
        id: Unique identifier (e.g., 'isp', 'corporate', 'apt', 'nation_state').
        name: Human-readable name.
        description: Detailed description of the adversary.
        resources: Description of resources available.
        capabilities: List of high-level capability descriptions.
        typical_use_cases: Common scenarios for this adversary tier.
        technique_capabilities: Mapping of technique_id -> effectiveness (0.0-1.0)
            indicating how well this adversary can employ each detection technique.
        mitre_references: Relevant MITRE ATT&CK technique IDs.
    """
    id: str
    name: str
    description: str
    resources: str
    capabilities: tuple  # frozen requires immutable
    typical_use_cases: tuple
    technique_capabilities: Dict[str, float] = field(default_factory=dict)
    mitre_references: tuple = ()

    def can_employ(self, technique_id: str) -> bool:
        """Check if this adversary can employ a given detection technique."""
        return self.technique_capabilities.get(technique_id, 0.0) > 0.0

    def effectiveness_for(self, technique_id: str) -> float:
        """Return how effectively this adversary can use a technique (0.0-1.0)."""
        return self.technique_capabilities.get(technique_id, 0.0)


# ---------------------------------------------------------------------------
# Adversary Tier Definitions
# ---------------------------------------------------------------------------

_ADVERSARY_TIERS: Dict[str, AdversaryTier] = {}


def _register(tier: AdversaryTier) -> AdversaryTier:
    _ADVERSARY_TIERS[tier.id] = tier
    return tier


ISP_LEVEL = _register(AdversaryTier(
    id="isp",
    name="ISP-Level",
    description=(
        "Internet Service Providers and similar network operators that have "
        "direct access to subscriber traffic at the network edge. They can "
        "observe connection metadata (source/destination IPs, ports, timing, "
        "volume) but typically lack deep packet inspection on encrypted flows. "
        "Their primary advantage is persistent, passive visibility over all "
        "subscriber traffic."
    ),
    resources=(
        "Network taps and mirroring infrastructure; NetFlow/IPFIX collectors; "
        "DNS resolver logs; subscriber identity databases; limited DPI appliances "
        "for traffic classification. Budget: moderate. Staff: network engineers "
        "with some security analysts."
    ),
    capabilities=(
        "Passive traffic monitoring and metadata collection",
        "DNS query logging and analysis",
        "Basic traffic classification (protocol identification)",
        "Volume and timing analysis on subscriber connections",
        "IP reputation lookups against known anonymity infrastructure",
        "Compliance-driven data retention and lawful intercept",
    ),
    typical_use_cases=(
        "Detecting use of known VPN/Tor endpoints",
        "Bandwidth throttling of anonymity traffic",
        "Lawful intercept compliance",
        "Identifying subscribers connecting to flagged IP ranges",
    ),
    technique_capabilities={
        "traffic_volume":       0.70,
        "timing_correlation":   0.30,
        "protocol_fingerprint": 0.40,
        "entry_exit_corr":      0.15,
        "dns_leakage":          0.85,
        "webrtc_leakage":       0.10,
        "ip_reputation":        0.75,
        "behavioral_pattern":   0.25,
        "circuit_analysis":     0.10,
        "metadata_analysis":    0.55,
        "side_channel_timing":  0.05,
        "infra_mapping":        0.30,
    },
    mitre_references=(
        "T1040",   # Network Sniffing
        "T1071.1", # Web Protocols
        "T1590",   # Gather Victim Network Information
    ),
))

CORPORATE_SECURITY = _register(AdversaryTier(
    id="corporate",
    name="Corporate Security",
    description=(
        "Enterprise security operations centers (SOCs) and dedicated security "
        "teams within organizations. They have visibility into internal network "
        "traffic via firewalls, proxies, and endpoint agents. They can perform "
        "deep packet inspection on managed networks and correlate with endpoint "
        "telemetry."
    ),
    resources=(
        "Next-generation firewalls with DPI; SIEM platforms; endpoint detection "
        "and response (EDR) agents; web proxy logs; SSL/TLS interception (for "
        "managed devices); threat intelligence feeds. Budget: moderate to high. "
        "Staff: SOC analysts, threat hunters, incident responders."
    ),
    capabilities=(
        "Deep packet inspection on managed network segments",
        "SSL/TLS interception for managed endpoints",
        "Endpoint telemetry (process, DNS, connection logs)",
        "Correlation across network and host-based indicators",
        "WebRTC and DNS leak detection via controlled resolvers",
        "Behavioral analytics and anomaly detection",
    ),
    typical_use_cases=(
        "Detecting policy-violating anonymity tool usage",
        "Identifying data exfiltration over encrypted tunnels",
        "Insider threat detection",
        "Monitoring for unauthorized VPN or proxy usage",
    ),
    technique_capabilities={
        "traffic_volume":       0.65,
        "timing_correlation":   0.40,
        "protocol_fingerprint": 0.75,
        "entry_exit_corr":      0.20,
        "dns_leakage":          0.90,
        "webrtc_leakage":       0.80,
        "ip_reputation":        0.80,
        "behavioral_pattern":   0.60,
        "circuit_analysis":     0.15,
        "metadata_analysis":    0.65,
        "side_channel_timing":  0.10,
        "infra_mapping":        0.40,
    },
    mitre_references=(
        "T1040",   # Network Sniffing
        "T1071",   # Application Layer Protocol
        "T1573",   # Encrypted Channel
        "T1090",   # Proxy
        "T1557",   # Adversary-in-the-Middle
    ),
))

APT = _register(AdversaryTier(
    id="apt",
    name="Advanced Persistent Threat (APT)",
    description=(
        "Well-funded and technically sophisticated threat groups, often with "
        "access to zero-day exploits, custom tooling, and long-term operational "
        "patience. APTs can compromise infrastructure at multiple network points "
        "and perform advanced correlation attacks. They may have partial access "
        "to ISP-level data through compromised infrastructure."
    ),
    resources=(
        "Custom exploit frameworks; compromised network infrastructure (routers, "
        "switches); implants on target and intermediary systems; access to some "
        "ISP-level data via compromises; advanced traffic analysis tools; "
        "machine learning pipelines for behavioral analysis. Budget: high. "
        "Staff: elite offensive security researchers and developers."
    ),
    capabilities=(
        "Advanced traffic analysis with ML-based correlation",
        "Compromise of anonymity relay infrastructure",
        "Entry/exit traffic correlation across multiple vantage points",
        "Side-channel attacks against protocol implementations",
        "Long-term behavioral profiling",
        "Infrastructure enumeration and mapping of anonymity networks",
        "Custom protocol fingerprinting against obfuscated traffic",
    ),
    typical_use_cases=(
        "Targeted deanonymization of specific individuals",
        "Long-term surveillance of anonymity network users",
        "Compromise of anonymity relay nodes",
        "Attribution of anonymous communications",
    ),
    technique_capabilities={
        "traffic_volume":       0.80,
        "timing_correlation":   0.75,
        "protocol_fingerprint": 0.85,
        "entry_exit_corr":      0.65,
        "dns_leakage":          0.90,
        "webrtc_leakage":       0.85,
        "ip_reputation":        0.85,
        "behavioral_pattern":   0.80,
        "circuit_analysis":     0.60,
        "metadata_analysis":    0.80,
        "side_channel_timing":  0.55,
        "infra_mapping":        0.70,
    },
    mitre_references=(
        "T1040",   # Network Sniffing
        "T1557",   # Adversary-in-the-Middle
        "T1599",   # Network Boundary Bridging
        "T1090.3", # Multi-hop Proxy
        "T1583",   # Acquire Infrastructure
        "T1592",   # Gather Victim Host Information
    ),
))

NATION_STATE = _register(AdversaryTier(
    id="nation_state",
    name="Nation-State",
    description=(
        "State-sponsored intelligence agencies with global surveillance "
        "capabilities, legal authority to compel ISP cooperation, and access "
        "to undersea cable taps and internet exchange points. They can perform "
        "large-scale traffic analysis, compel infrastructure operators, and "
        "deploy sophisticated attacks against anonymity systems."
    ),
    resources=(
        "Global SIGINT infrastructure (undersea cable taps, IXP monitoring); "
        "legal authority for lawful intercept and data retention mandates; "
        "supercomputing resources for cryptanalysis; teams of world-class "
        "researchers; ability to operate or compromise anonymity relays at "
        "scale; collaboration with allied intelligence agencies. Budget: "
        "virtually unlimited. Staff: thousands of analysts and researchers."
    ),
    capabilities=(
        "Global passive traffic collection and correlation",
        "Large-scale entry/exit correlation across national boundaries",
        "Operation of malicious anonymity relays (Sybil attacks)",
        "Cryptanalytic capabilities against weakened implementations",
        "Legal compulsion of service providers for data access",
        "Advanced side-channel and timing attacks at scale",
        "Complete infrastructure mapping of anonymity networks",
        "Cross-protocol correlation and long-term pattern analysis",
    ),
    typical_use_cases=(
        "Mass surveillance programs",
        "Targeted deanonymization for national security",
        "Mapping entire anonymity network infrastructures",
        "Signals intelligence collection",
        "Counter-intelligence operations",
    ),
    technique_capabilities={
        "traffic_volume":       0.90,
        "timing_correlation":   0.90,
        "protocol_fingerprint": 0.95,
        "entry_exit_corr":      0.85,
        "dns_leakage":          0.95,
        "webrtc_leakage":       0.90,
        "ip_reputation":        0.95,
        "behavioral_pattern":   0.90,
        "circuit_analysis":     0.80,
        "metadata_analysis":    0.90,
        "side_channel_timing":  0.75,
        "infra_mapping":        0.90,
    },
    mitre_references=(
        "T1040",   # Network Sniffing
        "T1557",   # Adversary-in-the-Middle
        "T1599",   # Network Boundary Bridging
        "T1090",   # Proxy
        "T1583",   # Acquire Infrastructure
        "T1584",   # Compromise Infrastructure
        "T1590",   # Gather Victim Network Information
        "T1600",   # Weaken Encryption
    ),
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_adversary(adversary_id: str) -> AdversaryTier:
    """Retrieve an adversary tier by ID.

    Args:
        adversary_id: One of 'isp', 'corporate', 'apt', 'nation_state'.

    Returns:
        The corresponding AdversaryTier object.

    Raises:
        KeyError: If the adversary ID is not found.
    """
    if adversary_id not in _ADVERSARY_TIERS:
        available = ", ".join(sorted(_ADVERSARY_TIERS.keys()))
        raise KeyError(
            f"Unknown adversary tier '{adversary_id}'. Available: {available}"
        )
    return _ADVERSARY_TIERS[adversary_id]


def list_adversaries() -> List[AdversaryTier]:
    """Return all adversary tiers ordered by increasing capability."""
    order = ["isp", "corporate", "apt", "nation_state"]
    return [_ADVERSARY_TIERS[k] for k in order if k in _ADVERSARY_TIERS]
