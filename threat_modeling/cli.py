"""Command-line interface for the Network Anonymity Threat Modeling SDK.

Commands:
    analyze <protocol> --adversary <tier>   Analyze a single protocol
    compare <protocol1> <protocol2> ...     Compare multiple protocols
    list-adversaries                        Show all adversary tiers
    list-protocols                          Show all supported protocols
    list-techniques                         Show all detection techniques
    report <protocol> --format --output     Generate reports to file
"""

import argparse
import sys
from typing import List, Optional

from threat_modeling import __version__
from threat_modeling.data.adversaries import get_adversary, list_adversaries
from threat_modeling.data.protocols import get_protocol, list_protocols
from threat_modeling.data.techniques import list_techniques
from threat_modeling.engines.scoring import ScoringEngine
from threat_modeling.reports.generator import ReportGenerator


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="threat-model",
        description=(
            "Network Anonymity Threat Modeling SDK — Analyze anonymity protocols "
            "against adversary capabilities."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  threat-model analyze tor --adversary nation_state\n"
            "  threat-model compare tor vpn proxy\n"
            "  threat-model compare tor vpn --adversary apt\n"
            "  threat-model report tor --format markdown --output report.md\n"
            "  threat-model list-adversaries\n"
            "  threat-model list-protocols\n"
            "  threat-model list-techniques\n"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- analyze --
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze a single protocol against an adversary tier",
    )
    p_analyze.add_argument(
        "protocol",
        help="Protocol to analyze (tor, vpn, vpn_chain, proxy, i2p)",
    )
    p_analyze.add_argument(
        "--adversary", "-a",
        default="nation_state",
        help="Adversary tier (isp, corporate, apt, nation_state). Default: nation_state",
    )

    # -- compare --
    p_compare = subparsers.add_parser(
        "compare",
        help="Compare multiple protocols",
    )
    p_compare.add_argument(
        "protocols",
        nargs="+",
        help="Protocols to compare (e.g., tor vpn proxy)",
    )
    p_compare.add_argument(
        "--adversary", "-a",
        default=None,
        help="Specific adversary tier (default: all tiers)",
    )

    # -- report --
    p_report = subparsers.add_parser(
        "report",
        help="Generate a report file",
    )
    p_report.add_argument(
        "protocol",
        help="Protocol to analyze",
    )
    p_report.add_argument(
        "--adversary", "-a",
        default="nation_state",
        help="Adversary tier. Default: nation_state",
    )
    p_report.add_argument(
        "--format", "-f",
        choices=["json", "markdown"],
        default="json",
        help="Output format (json or markdown). Default: json",
    )
    p_report.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path. Default: <protocol>_<adversary>_report.<ext>",
    )

    # -- list-adversaries --
    subparsers.add_parser(
        "list-adversaries",
        help="List all adversary tiers",
    )

    # -- list-protocols --
    subparsers.add_parser(
        "list-protocols",
        help="List all supported anonymity protocols",
    )

    # -- list-techniques --
    subparsers.add_parser(
        "list-techniques",
        help="List all detection techniques",
    )

    return parser


def _risk_bar(score: float, width: int = 20) -> str:
    """Create a simple text-based risk bar."""
    filled = int(score / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Handle the 'analyze' command."""
    engine = ScoringEngine()
    try:
        result = engine.analyze(args.protocol, args.adversary)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  THREAT ANALYSIS: {result.protocol_name}")
    print(f"  Adversary: {result.adversary_name}")
    print(f"{'='*60}\n")

    print(f"  Overall Risk Score: {result.overall_risk_score}/100 "
          f"{_risk_bar(result.overall_risk_score)}")
    print(f"  Risk Level:         {result.risk_level}")
    print(f"  90% Confidence:     {result.confidence_low} – {result.confidence_high}")
    print()

    print("  Detection Techniques (sorted by probability):")
    print(f"  {'─'*56}")
    sorted_results = sorted(
        result.technique_results,
        key=lambda r: r.detection_probability,
        reverse=True,
    )
    for tr in sorted_results:
        prob_pct = f"{tr.detection_probability:.0%}"
        print(f"    {tr.technique_name:<40} {prob_pct:>5}  ({tr.difficulty})")
    print()

    print("  Mitigation Recommendations:")
    print(f"  {'─'*56}")
    for i, mit in enumerate(result.mitigation_recommendations, 1):
        # Wrap long lines
        lines = _wrap_text(mit, 56)
        print(f"    {i}. {lines[0]}")
        for line in lines[1:]:
            print(f"       {line}")
    print()


def _cmd_compare(args: argparse.Namespace) -> None:
    """Handle the 'compare' command."""
    engine = ScoringEngine()
    gen = ReportGenerator(engine)

    try:
        comparison = engine.compare(args.protocols, args.adversary)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print the markdown comparison to stdout
    md = gen.comparison_markdown(comparison)
    print(md)


def _cmd_report(args: argparse.Namespace) -> None:
    """Handle the 'report' command."""
    engine = ScoringEngine()
    gen = ReportGenerator(engine)

    try:
        result = engine.analyze(args.protocol, args.adversary)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    ext = "json" if args.format == "json" else "md"
    output = args.output or f"{args.protocol}_{args.adversary}_report.{ext}"

    gen.save_report(result, output, fmt=args.format)
    print(f"Report saved to: {output}")


def _cmd_list_adversaries(_args: argparse.Namespace) -> None:
    """Handle the 'list-adversaries' command."""
    print(f"\n{'='*60}")
    print("  ADVERSARY TIERS")
    print(f"{'='*60}\n")

    for adv in list_adversaries():
        print(f"  [{adv.id}] {adv.name}")
        # Print first 80 chars of description
        desc = adv.description[:120] + ("..." if len(adv.description) > 120 else "")
        print(f"    {desc}")
        print(f"    MITRE ATT&CK: {', '.join(adv.mitre_references)}")
        print()


def _cmd_list_protocols(_args: argparse.Namespace) -> None:
    """Handle the 'list-protocols' command."""
    print(f"\n{'='*60}")
    print("  ANONYMITY PROTOCOLS")
    print(f"{'='*60}\n")

    for proto in list_protocols():
        layers = f"{proto.anonymity_layers} layer(s)"
        decentralized = "decentralized" if proto.decentralized else "centralized"
        print(f"  [{proto.id}] {proto.name}")
        print(f"    {layers}, {decentralized}")
        desc = proto.description[:120] + ("..." if len(proto.description) > 120 else "")
        print(f"    {desc}")
        print()


def _cmd_list_techniques(_args: argparse.Namespace) -> None:
    """Handle the 'list-techniques' command."""
    print(f"\n{'='*60}")
    print("  DETECTION TECHNIQUES")
    print(f"{'='*60}\n")

    for tech in list_techniques():
        print(f"  [{tech.id}] {tech.name}")
        print(f"    Difficulty: {tech.difficulty}")
        print(f"    MITRE ATT&CK: {', '.join(tech.mitre_references)}")
        desc = tech.description[:120] + ("..." if len(tech.description) > 120 else "")
        print(f"    {desc}")
        print()


def _wrap_text(text: str, width: int) -> List[str]:
    """Simple word-wrap for terminal output."""
    words = text.split()
    lines: List[str] = []
    current_line = ""
    for word in words:
        if current_line and len(current_line) + 1 + len(word) > width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = f"{current_line} {word}" if current_line else word
    if current_line:
        lines.append(current_line)
    return lines or [""]


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "analyze":          _cmd_analyze,
        "compare":          _cmd_compare,
        "report":           _cmd_report,
        "list-adversaries": _cmd_list_adversaries,
        "list-protocols":   _cmd_list_protocols,
        "list-techniques":  _cmd_list_techniques,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
