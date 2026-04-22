"""Network Anonymity Threat Modeling SDK.

A security research tool for modeling adversary capabilities against
network anonymity protocols. Evaluates detection probabilities across
multiple adversary tiers and detection techniques.

This is an educational and defensive security research tool.
"""

__version__ = "1.0.0"
__author__ = "Threat Modeling SDK"

from threat_modeling.data.adversaries import AdversaryTier, get_adversary, list_adversaries
from threat_modeling.data.techniques import DetectionTechnique, get_technique, list_techniques
from threat_modeling.data.protocols import AnonymityProtocol, get_protocol, list_protocols
from threat_modeling.engines.scoring import ScoringEngine
from threat_modeling.reports.generator import ReportGenerator

__all__ = [
    "AdversaryTier",
    "get_adversary",
    "list_adversaries",
    "DetectionTechnique",
    "get_technique",
    "list_techniques",
    "AnonymityProtocol",
    "get_protocol",
    "list_protocols",
    "ScoringEngine",
    "ReportGenerator",
]
