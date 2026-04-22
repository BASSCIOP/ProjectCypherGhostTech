"""Detection technique definitions and effectiveness data.

Covers 12 core network traffic analysis and deanonymization techniques
used by various adversary tiers. Each technique includes effectiveness
ratings against each anonymity protocol.

References:
- MITRE ATT&CK: T1040 (Network Sniffing), T1071 (Application Layer Protocol)
- Academic: "Users Get Routed: Traffic Correlation on Tor" (Johnson et al.)
- Academic: "Website Fingerprinting at Internet Scale" (Panchenko et al.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DetectionTechnique:
    """A detection or deanonymization technique.

    Attributes:
        id: Unique technique identifier.
        name: Human-readable name.
        description: Detailed explanation of the technique.
        difficulty: Difficulty level to execute ('low', 'medium', 'high', 'very_high').
        resources_required: Description of resources needed.
        protocol_effectiveness: Mapping of protocol_id -> base effectiveness (0.0-1.0)
            before considering adversary capability multipliers.
        mitre_references: Relevant MITRE ATT&CK technique IDs.
    """
    id: str
    name: str
    description: str
    difficulty: str
    resources_required: str
    protocol_effectiveness: Dict[str, float] = field(default_factory=dict)
    mitre_references: tuple = ()

    def effectiveness_against(self, protocol_id: str) -> float:
        """Return base effectiveness of this technique against a protocol."""
        return self.protocol_effectiveness.get(protocol_id, 0.0)


# ---------------------------------------------------------------------------
# Technique Definitions
# ---------------------------------------------------------------------------

_TECHNIQUES: Dict[str, DetectionTechnique] = {}


def _reg(t: DetectionTechnique) -> DetectionTechnique:
    _TECHNIQUES[t.id] = t
    return t


TRAFFIC_VOLUME = _reg(DetectionTechnique(
    id="traffic_volume",
    name="Traffic Volume Analysis",
    description=(
        "Analyzes the volume and patterns of network traffic to correlate "
        "anonymous sessions with real identities. Large or distinctive data "
        "transfers can be matched across entry and exit points by comparing "
        "byte counts and flow sizes over time windows."
    ),
    difficulty="low",
    resources_required=(
        "Network taps or flow collectors (NetFlow/IPFIX); storage for traffic "
        "metadata; basic statistical analysis tools."
    ),
    protocol_effectiveness={
        "tor":       0.35,
        "vpn":       0.55,
        "vpn_chain": 0.40,
        "proxy":     0.70,
        "i2p":       0.30,
    },
    mitre_references=("T1040", "T1071"),
))

TIMING_CORRELATION = _reg(DetectionTechnique(
    id="timing_correlation",
    name="Timing Correlation Analysis",
    description=(
        "Correlates the timing of packets entering and exiting an anonymity "
        "network to link sender and receiver. By injecting timing perturbations "
        "or observing natural timing patterns, an adversary at both ends can "
        "match flows with high confidence."
    ),
    difficulty="high",
    resources_required=(
        "Observation points at both entry and exit of the anonymity network; "
        "high-precision timestamps (microsecond or better); statistical "
        "correlation tools; significant compute for large-scale analysis."
    ),
    protocol_effectiveness={
        "tor":       0.45,
        "vpn":       0.65,
        "vpn_chain": 0.35,
        "proxy":     0.75,
        "i2p":       0.30,
    },
    mitre_references=("T1040", "T1557"),
))

PROTOCOL_FINGERPRINT = _reg(DetectionTechnique(
    id="protocol_fingerprint",
    name="Protocol Fingerprinting / Deep Packet Inspection",
    description=(
        "Uses deep packet inspection (DPI) to identify the specific anonymity "
        "protocol in use based on packet structure, handshake patterns, TLS "
        "characteristics, and traffic signatures. Even encrypted protocols "
        "leak identifiable patterns in packet sizes, timing, and handshake "
        "sequences."
    ),
    difficulty="medium",
    resources_required=(
        "DPI appliances or software (e.g., nDPI, commercial NGFW); signature "
        "databases for anonymity protocols; moderate compute for real-time "
        "classification."
    ),
    protocol_effectiveness={
        "tor":       0.50,
        "vpn":       0.60,
        "vpn_chain": 0.45,
        "proxy":     0.80,
        "i2p":       0.40,
    },
    mitre_references=("T1071", "T1573"),
))

ENTRY_EXIT_CORRELATION = _reg(DetectionTechnique(
    id="entry_exit_corr",
    name="Entry/Exit Correlation",
    description=(
        "Monitors both the entry point (where traffic enters the anonymity "
        "network) and exit point (where it leaves) to correlate flows. This "
        "is particularly effective against circuit-based anonymity systems "
        "when the adversary controls or observes both ends of the circuit."
    ),
    difficulty="very_high",
    resources_required=(
        "Observation capability at multiple network vantage points; control "
        "or monitoring of entry and exit relays; global traffic visibility "
        "or cooperation from multiple network operators."
    ),
    protocol_effectiveness={
        "tor":       0.40,
        "vpn":       0.70,
        "vpn_chain": 0.30,
        "proxy":     0.80,
        "i2p":       0.25,
    },
    mitre_references=("T1040", "T1557", "T1599"),
))

DNS_LEAKAGE = _reg(DetectionTechnique(
    id="dns_leakage",
    name="DNS Leakage Detection",
    description=(
        "Detects when DNS queries bypass the anonymity tunnel and are sent "
        "directly to the user's configured DNS resolver, revealing the "
        "actual destinations being accessed. Misconfigured clients or "
        "applications frequently leak DNS queries outside the tunnel."
    ),
    difficulty="low",
    resources_required=(
        "DNS resolver logs; network monitoring at the DNS level; basic "
        "log analysis tools."
    ),
    protocol_effectiveness={
        "tor":       0.15,
        "vpn":       0.45,
        "vpn_chain": 0.30,
        "proxy":     0.75,
        "i2p":       0.10,
    },
    mitre_references=("T1071.004",),  # DNS
))

WEBRTC_LEAKAGE = _reg(DetectionTechnique(
    id="webrtc_leakage",
    name="WebRTC Leakage Detection",
    description=(
        "Exploits WebRTC's STUN/TURN requests that can bypass proxy and VPN "
        "configurations, revealing the user's real IP address. WebRTC in "
        "browsers may enumerate local network interfaces and send requests "
        "outside the anonymity tunnel."
    ),
    difficulty="low",
    resources_required=(
        "Web application with WebRTC capabilities; STUN server to receive "
        "leaked requests; basic web development knowledge."
    ),
    protocol_effectiveness={
        "tor":       0.10,
        "vpn":       0.50,
        "vpn_chain": 0.35,
        "proxy":     0.80,
        "i2p":       0.10,
    },
    mitre_references=("T1071.001",),  # Web Protocols
))

IP_REPUTATION = _reg(DetectionTechnique(
    id="ip_reputation",
    name="IP/ASN Reputation Analysis",
    description=(
        "Identifies anonymity tool usage by checking whether the source or "
        "destination IP addresses belong to known anonymity infrastructure "
        "(Tor exit nodes, VPN provider ranges, known proxy servers). Uses "
        "IP reputation databases and ASN classification."
    ),
    difficulty="low",
    resources_required=(
        "IP reputation databases (commercial or open-source Tor exit lists, "
        "VPN IP databases); ASN lookup services; automated checking tools."
    ),
    protocol_effectiveness={
        "tor":       0.70,
        "vpn":       0.55,
        "vpn_chain": 0.45,
        "proxy":     0.60,
        "i2p":       0.30,
    },
    mitre_references=("T1590.005",),  # IP Addresses
))

BEHAVIORAL_PATTERN = _reg(DetectionTechnique(
    id="behavioral_pattern",
    name="Behavioral Pattern Recognition",
    description=(
        "Analyzes user behavior patterns (browsing habits, session timing, "
        "typing cadence, mouse movements) to fingerprint and track users "
        "across anonymity sessions. Even with IP anonymity, unique behavioral "
        "patterns can deanonymize users over time."
    ),
    difficulty="high",
    resources_required=(
        "Long-term traffic or session data; machine learning models for "
        "pattern recognition; web application instrumentation for behavioral "
        "data collection; significant training data."
    ),
    protocol_effectiveness={
        "tor":       0.50,
        "vpn":       0.55,
        "vpn_chain": 0.50,
        "proxy":     0.60,
        "i2p":       0.45,
    },
    mitre_references=("T1592",),  # Gather Victim Host Information
))

CIRCUIT_ANALYSIS = _reg(DetectionTechnique(
    id="circuit_analysis",
    name="Circuit / Connection Analysis",
    description=(
        "Maps and analyzes the circuits or connection paths through anonymity "
        "networks. By observing relay selection patterns, circuit build timing, "
        "and connection establishment sequences, adversaries can narrow down "
        "or identify specific users."
    ),
    difficulty="very_high",
    resources_required=(
        "Control or observation of multiple relay/exit nodes; deep understanding "
        "of the anonymity protocol internals; long-term data collection; "
        "significant analytical capability."
    ),
    protocol_effectiveness={
        "tor":       0.35,
        "vpn":       0.20,
        "vpn_chain": 0.25,
        "proxy":     0.15,
        "i2p":       0.30,
    },
    mitre_references=("T1090.3", "T1599"),  # Multi-hop Proxy, Network Boundary Bridging
))

METADATA_ANALYSIS = _reg(DetectionTechnique(
    id="metadata_analysis",
    name="Metadata Analysis",
    description=(
        "Collects and analyzes connection metadata (timestamps, session "
        "durations, data volumes, connection frequencies) without inspecting "
        "payload content. Metadata alone can reveal communication patterns, "
        "associate sessions, and support deanonymization when combined with "
        "other techniques."
    ),
    difficulty="low",
    resources_required=(
        "Flow data collectors; session logging infrastructure; basic data "
        "analytics tools; storage for historical metadata."
    ),
    protocol_effectiveness={
        "tor":       0.40,
        "vpn":       0.50,
        "vpn_chain": 0.35,
        "proxy":     0.65,
        "i2p":       0.35,
    },
    mitre_references=("T1040", "T1590"),
))

SIDE_CHANNEL_TIMING = _reg(DetectionTechnique(
    id="side_channel_timing",
    name="Side-Channel Timing Attacks",
    description=(
        "Exploits timing side-channels in protocol implementations to extract "
        "information or correlate traffic. This includes measuring processing "
        "delays, observing scheduling patterns, and leveraging CPU cache "
        "timing to infer activities within anonymity software."
    ),
    difficulty="very_high",
    resources_required=(
        "Deep knowledge of target protocol implementation; ability to measure "
        "precise timing (nanosecond-level); controlled experimental setup; "
        "potentially co-located infrastructure for cache-based attacks."
    ),
    protocol_effectiveness={
        "tor":       0.25,
        "vpn":       0.20,
        "vpn_chain": 0.15,
        "proxy":     0.20,
        "i2p":       0.20,
    },
    mitre_references=("T1499.004",),  # Application or System Exploitation
))

INFRA_MAPPING = _reg(DetectionTechnique(
    id="infra_mapping",
    name="Infrastructure Mapping",
    description=(
        "Enumerates and maps the infrastructure of anonymity networks, "
        "including relay nodes, exit points, directory servers, and hidden "
        "service directories. Knowledge of the full network topology enables "
        "targeted attacks on specific relays or statistical deanonymization."
    ),
    difficulty="high",
    resources_required=(
        "Active and passive network scanning tools; participation in the "
        "anonymity network (running relays); analysis of public directory "
        "information; long-term monitoring for infrastructure changes."
    ),
    protocol_effectiveness={
        "tor":       0.55,
        "vpn":       0.40,
        "vpn_chain": 0.30,
        "proxy":     0.50,
        "i2p":       0.35,
    },
    mitre_references=("T1590", "T1583", "T1584"),
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_technique(technique_id: str) -> DetectionTechnique:
    """Retrieve a detection technique by ID.

    Raises:
        KeyError: If the technique ID is not found.
    """
    if technique_id not in _TECHNIQUES:
        available = ", ".join(sorted(_TECHNIQUES.keys()))
        raise KeyError(
            f"Unknown technique '{technique_id}'. Available: {available}"
        )
    return _TECHNIQUES[technique_id]


def list_techniques() -> List[DetectionTechnique]:
    """Return all detection techniques in definition order."""
    return list(_TECHNIQUES.values())
