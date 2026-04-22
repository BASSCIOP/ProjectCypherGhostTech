"""Anonymity protocol definitions.

Models five common network anonymity protocols with their characteristics,
strengths, weaknesses, and resistance properties.

References:
- MITRE ATT&CK: T1090 (Proxy), T1090.3 (Multi-hop Proxy)
- MITRE ATT&CK: T1573 (Encrypted Channel)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AnonymityProtocol:
    """An anonymity / privacy protocol.

    Attributes:
        id: Unique identifier (e.g., 'tor', 'vpn').
        name: Human-readable protocol name.
        description: How the protocol works.
        strengths: Tuple of protocol strengths.
        weaknesses: Tuple of protocol weaknesses.
        characteristics: Key technical characteristics.
        typical_use_cases: Common uses.
        anonymity_layers: Number of encryption / routing layers.
        decentralized: Whether the protocol is decentralized.
        mitre_references: Relevant MITRE ATT&CK technique IDs.
    """
    id: str
    name: str
    description: str
    strengths: tuple
    weaknesses: tuple
    characteristics: Dict[str, str] = field(default_factory=dict)
    typical_use_cases: tuple = ()
    anonymity_layers: int = 1
    decentralized: bool = False
    mitre_references: tuple = ()


# ---------------------------------------------------------------------------
# Protocol Definitions
# ---------------------------------------------------------------------------

_PROTOCOLS: Dict[str, AnonymityProtocol] = {}


def _reg(p: AnonymityProtocol) -> AnonymityProtocol:
    _PROTOCOLS[p.id] = p
    return p


TOR = _reg(AnonymityProtocol(
    id="tor",
    name="Tor (The Onion Router)",
    description=(
        "Tor routes traffic through a series of three volunteer-operated relays "
        "(guard, middle, exit), applying layered encryption at each hop. The "
        "client negotiates a circuit through the Tor network, and each relay "
        "only knows the immediately preceding and following hop. Exit relays "
        "connect to the final destination on the open internet. Tor also "
        "supports onion services (.onion) for end-to-end encrypted, server-"
        "anonymous communication without exit relays."
    ),
    strengths=(
        "Strong anonymity through multi-hop onion routing",
        "Large and diverse volunteer relay network (~6,000+ relays)",
        "Well-studied and peer-reviewed protocol design",
        "Supports onion services for mutual anonymity",
        "Resistant to single-point-of-failure compromise",
        "Free and open-source with active development community",
        "Built-in pluggable transports for censorship circumvention",
    ),
    weaknesses=(
        "Vulnerable to traffic correlation by global adversaries",
        "Exit relay can observe unencrypted traffic to destination",
        "Performance overhead (latency) due to multi-hop routing",
        "Tor exit nodes are publicly listed (easy IP reputation blocking)",
        "Guard node selection creates a persistent entry point",
        "Sybil attacks possible if adversary operates many relays",
        "Browser-based attacks (JavaScript, WebRTC) can bypass Tor",
    ),
    characteristics={
        "routing": "Onion routing with 3 hops (guard, middle, exit)",
        "encryption": "Layered AES-128-CTR with per-hop TLS",
        "network_type": "Overlay network on TCP",
        "directory": "Centralized directory authorities (9 trusted)",
        "latency": "High (300-1000ms typical added latency)",
        "throughput": "Low to moderate (limited by relay bandwidth)",
        "ip_hiding": "Source IP hidden from destination; destination hidden from source via onion services",
    },
    typical_use_cases=(
        "Anonymous web browsing",
        "Whistleblowing and journalism",
        "Circumventing censorship",
        "Accessing onion services",
        "Privacy-preserving research",
    ),
    anonymity_layers=3,
    decentralized=True,
    mitre_references=("T1090.3", "T1573"),
))

VPN = _reg(AnonymityProtocol(
    id="vpn",
    name="VPN (Single Provider)",
    description=(
        "A Virtual Private Network creates an encrypted tunnel between the "
        "user's device and a VPN server operated by the provider. All traffic "
        "is routed through this single server, which assigns a new IP address "
        "to the user. The VPN provider can see the user's real IP and all "
        "traffic metadata (and content if not end-to-end encrypted). Common "
        "protocols include WireGuard, OpenVPN, and IKEv2/IPsec."
    ),
    strengths=(
        "Simple to set up and use",
        "Encrypts all traffic between client and VPN server",
        "Hides real IP from destination servers",
        "Good performance with modern protocols (WireGuard)",
        "Wide availability and provider choice",
        "Effective against local network observers",
    ),
    weaknesses=(
        "Single point of trust: VPN provider sees all traffic",
        "Provider may log user activity (trust-based model)",
        "VPN server IP ranges are often known and blocked",
        "DNS and WebRTC leaks are common with poor implementations",
        "No protection if the VPN provider is compromised or compelled",
        "Traffic patterns are preserved through the tunnel",
        "Single hop provides limited anonymity against network-level adversaries",
    ),
    characteristics={
        "routing": "Single-hop tunnel to VPN server",
        "encryption": "ChaCha20/AES-256 (WireGuard) or AES-256-GCM (OpenVPN)",
        "network_type": "Point-to-point tunnel (UDP or TCP)",
        "directory": "N/A (centralized provider)",
        "latency": "Low (10-50ms typical added latency)",
        "throughput": "High (near line-speed with WireGuard)",
        "ip_hiding": "Source IP hidden from destination; visible to VPN provider",
    },
    typical_use_cases=(
        "General privacy from local network/ISP",
        "Accessing geo-restricted content",
        "Securing public Wi-Fi connections",
        "Remote work and corporate access",
    ),
    anonymity_layers=1,
    decentralized=False,
    mitre_references=("T1573", "T1090"),
))

VPN_CHAIN = _reg(AnonymityProtocol(
    id="vpn_chain",
    name="VPN Chain (Multi-hop VPN)",
    description=(
        "Routes traffic through two or more VPN servers operated by different "
        "providers in sequence, creating a multi-hop encrypted tunnel. Each "
        "VPN server only knows the previous and next hop, similar in concept "
        "to onion routing but with trusted providers at each hop. This "
        "reduces the risk of a single compromised provider deanonymizing "
        "the user."
    ),
    strengths=(
        "No single provider sees both the user's real IP and traffic destination",
        "Higher anonymity than single VPN due to trust distribution",
        "More resistant to provider compromise or legal compulsion",
        "Each hop adds a layer of IP obfuscation",
        "Can mix providers across jurisdictions for legal diversity",
    ),
    weaknesses=(
        "Complex setup and configuration",
        "Significant performance degradation (cumulative latency)",
        "All providers in the chain must be operational",
        "Still vulnerable if all providers collude or are compromised",
        "Traffic volume and timing patterns may persist through the chain",
        "Limited provider support for native multi-hop",
        "Troubleshooting connectivity issues is difficult",
    ),
    characteristics={
        "routing": "Multi-hop through 2+ VPN servers (different providers)",
        "encryption": "Nested encryption layers (one per hop)",
        "network_type": "Cascaded point-to-point tunnels",
        "directory": "N/A (manual configuration of chain)",
        "latency": "Moderate to high (cumulative per-hop latency)",
        "throughput": "Moderate (limited by slowest hop)",
        "ip_hiding": "Source IP hidden from all but first hop; destination hidden from all but last hop",
    },
    typical_use_cases=(
        "High-anonymity requirements beyond single VPN",
        "Distributing trust across multiple providers and jurisdictions",
        "Layered defense for sensitive communications",
    ),
    anonymity_layers=2,
    decentralized=False,
    mitre_references=("T1090.3", "T1573"),
))

PROXY = _reg(AnonymityProtocol(
    id="proxy",
    name="Proxy (SOCKS/HTTP)",
    description=(
        "A proxy server acts as an intermediary for client requests. SOCKS "
        "proxies operate at the transport layer (supporting any TCP/UDP "
        "traffic), while HTTP proxies operate at the application layer. The "
        "proxy forwards requests on behalf of the client, masking the client's "
        "IP from the destination. However, the proxy operator has full "
        "visibility into the connection, and most proxies provide minimal "
        "encryption."
    ),
    strengths=(
        "Simple to configure and widely supported",
        "Low latency and overhead",
        "Application-specific routing (per-app proxy support)",
        "SOCKS5 supports both TCP and UDP",
        "Useful for simple IP masking requirements",
    ),
    weaknesses=(
        "Minimal or no encryption (especially HTTP proxies)",
        "Single point of trust and failure",
        "DNS leaks are very common",
        "WebRTC leaks bypass proxy settings easily",
        "Only proxies configured applications (not system-wide by default)",
        "Proxy server sees all traffic in cleartext (HTTP proxy)",
        "No traffic padding or timing obfuscation",
        "IP addresses of proxy servers are easily enumerated",
    ),
    characteristics={
        "routing": "Single-hop through proxy server",
        "encryption": "None (HTTP proxy) or optional TLS wrapping (SOCKS5+TLS)",
        "network_type": "Application-layer relay",
        "directory": "N/A (single server)",
        "latency": "Very low (5-20ms typical added latency)",
        "throughput": "High (limited only by proxy server capacity)",
        "ip_hiding": "Source IP hidden from destination; fully visible to proxy operator",
    },
    typical_use_cases=(
        "Simple IP masking",
        "Bypassing basic geo-restrictions",
        "Application-specific traffic routing",
        "Web scraping and automation",
    ),
    anonymity_layers=1,
    decentralized=False,
    mitre_references=("T1090",),
))

I2P = _reg(AnonymityProtocol(
    id="i2p",
    name="I2P (Invisible Internet Project)",
    description=(
        "I2P is a fully decentralized, packet-based anonymity network that "
        "uses garlic routing (bundling multiple encrypted messages together). "
        "Unlike Tor, I2P is designed primarily for internal services "
        "(eepsites) rather than outproxying to the regular internet. Traffic "
        "uses unidirectional tunnels: separate inbound and outbound tunnel "
        "chains, each typically 3 hops. The distributed network database "
        "(netDB) enables peer discovery without centralized directory servers."
    ),
    strengths=(
        "Fully decentralized with distributed peer discovery (netDB)",
        "Garlic routing bundles messages for traffic analysis resistance",
        "Unidirectional tunnels increase correlation attack difficulty",
        "Designed for internal network services (strong for in-network privacy)",
        "Packet-based (supports UDP natively, unlike Tor)",
        "All participating nodes route traffic (no dedicated exit/relay split)",
        "Short-lived tunnels (10 minutes default) limit observation windows",
    ),
    weaknesses=(
        "Smaller network than Tor (fewer peers, less traffic to blend with)",
        "Limited outproxy support (not designed for regular internet access)",
        "Higher latency than VPN due to multi-hop garlic routing",
        "Less studied than Tor in academic literature",
        "Resource-intensive for participants (all nodes route traffic)",
        "Bootstrap/startup time is significant (tunnel building)",
        "Ecosystem of services is smaller than Tor's .onion ecosystem",
    ),
    characteristics={
        "routing": "Garlic routing with unidirectional tunnels (3+ hops each)",
        "encryption": "Layered ElGamal/AES+SessionTag encryption per hop",
        "network_type": "Packet-based overlay network (UDP and TCP)",
        "directory": "Distributed network database (netDB) via Kademlia DHT",
        "latency": "High (variable, typically 500ms-2s)",
        "throughput": "Low to moderate",
        "ip_hiding": "Source IP hidden through unidirectional tunnel chains; strong for in-network services",
    },
    typical_use_cases=(
        "Accessing I2P internal services (eepsites)",
        "Anonymous peer-to-peer communication",
        "Private messaging and forums",
        "Decentralized application hosting",
    ),
    anonymity_layers=3,
    decentralized=True,
    mitre_references=("T1090.3", "T1573"),
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_protocol(protocol_id: str) -> AnonymityProtocol:
    """Retrieve a protocol by ID.

    Args:
        protocol_id: One of 'tor', 'vpn', 'vpn_chain', 'proxy', 'i2p'.

    Raises:
        KeyError: If the protocol ID is not found.
    """
    if protocol_id not in _PROTOCOLS:
        available = ", ".join(sorted(_PROTOCOLS.keys()))
        raise KeyError(
            f"Unknown protocol '{protocol_id}'. Available: {available}"
        )
    return _PROTOCOLS[protocol_id]


def list_protocols() -> List[AnonymityProtocol]:
    """Return all anonymity protocols in definition order."""
    order = ["tor", "vpn", "vpn_chain", "proxy", "i2p"]
    return [_PROTOCOLS[k] for k in order if k in _PROTOCOLS]
