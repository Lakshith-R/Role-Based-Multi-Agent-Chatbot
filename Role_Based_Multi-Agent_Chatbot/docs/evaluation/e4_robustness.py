"""
E4: Fallback Rate & Robustness — Zero API Calls Version
=========================================================
Reads from the existing local log file (logs/workflow_logs.txt) and computes
robustness metrics from REAL user interactions. NO extra API calls needed.

Metrics:
 - Overall fallback rate
 - Fallback rate per agent
 - Average confidence score per agent
 - Low-confidence routing rate (confidence < 0.6)

Outputs:
 - docs/evaluation/results/robustness_report.json
"""

import sys
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "docs" / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT.parent / "Agentic_Student_Assistant-main" / "logs" / "workflow_logs.txt"
LOW_CONFIDENCE_THRESHOLD = 0.6


def parse_log_file(log_path: Path) -> list:
    """Parse workflow_logs.txt into a list of interaction dicts."""
    if not log_path.exists():
        return []

    entries = []
    current = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "Query:" in line:
                current["query"] = line.split("Query:", 1)[-1].strip()
            elif "Routed Agent:" in line:
                current["agent"] = line.split("Routed Agent:", 1)[-1].strip()
            elif "Confidence:" in line:
                try:
                    current["confidence"] = float(re.search(r"[\d.]+", line).group())
                except Exception:
                    pass
            elif "Fallback Used:" in line:
                current["is_fallback"] = "yes" in line.lower()
            elif "=" * 20 in line and current.get("query"):
                entries.append(current)
                current = {}
    if current.get("query"):
        entries.append(current)
    return entries


def run_robustness():
    """Compute robustness and fallback metrics from log file."""
    print("🛡️  E4: Robustness & Fallback Evaluation (from log file — zero API calls)")
    print("=" * 60)

    entries = parse_log_file(LOG_FILE)

    if not entries:
        print("⚠️  No log data found. Run the Streamlit app first, then re-run.")
        return {}

    total = len(entries)
    fallbacks = [e for e in entries if e.get("is_fallback")]
    fallback_rate = len(fallbacks) / total if total else 0

    # Per agent stats
    by_agent = defaultdict(list)
    for e in entries:
        agent = e.get("agent", "unknown")
        by_agent[agent].append(e)

    # Confidence stats
    conf_entries = [e for e in entries if "confidence" in e]
    avg_confidence = (sum(e["confidence"] for e in conf_entries) / len(conf_entries)) if conf_entries else 0
    low_conf = [e for e in conf_entries if e["confidence"] < LOW_CONFIDENCE_THRESHOLD]
    low_conf_rate = len(low_conf) / len(conf_entries) if conf_entries else 0

    # Per agent breakdown
    per_agent = {}
    print(f"\n{'Agent':<15} {'Queries':>8} {'Fallbacks':>10} {'Fallback%':>10} {'Avg Conf':>10}")
    print("-" * 60)
    for agent, interactions in sorted(by_agent.items()):
        agent_fallbacks = sum(1 for e in interactions if e.get("is_fallback"))
        agent_confs = [e["confidence"] for e in interactions if "confidence" in e]
        agent_avg_conf = round(sum(agent_confs) / len(agent_confs), 3) if agent_confs else 0
        fb_rate = round(agent_fallbacks / len(interactions), 3) if interactions else 0
        per_agent[agent] = {
            "queries": len(interactions),
            "fallbacks": agent_fallbacks,
            "fallback_rate": fb_rate,
            "avg_confidence": agent_avg_conf
        }
        print(f"{agent:<15} {len(interactions):>8} {agent_fallbacks:>10} {fb_rate:>9.1%} {agent_avg_conf:>10.3f}")

    # Heuristic: Count how many edge cases (very short, very long, or special characters)
    # were correctly routed to fallback or handled gracefully with high confidence
    edge_cases = [e for e in entries if len(e.get("query", "")) < 5 or len(e.get("query", "")) > 100 or "?" not in e.get("query", "")]
    if edge_cases:
        edge_handled = sum(1 for e in edge_cases if e.get("is_fallback") or e.get("confidence", 0) > 0.8)
        edge_case_accuracy = edge_handled / len(edge_cases)
    else:
        edge_case_accuracy = 1.0

    print(f"\n📊 Overall Fallback Rate:          {fallback_rate:.2%}  ({len(fallbacks)}/{total})")
    print(f"📊 Average Router Confidence:      {avg_confidence:.3f}")
    print(f"📊 Low-Confidence Routing Rate:    {low_conf_rate:.2%}  (conf < {LOW_CONFIDENCE_THRESHOLD})")
    print(f"📊 Edge Case Handling Accuracy:    {edge_case_accuracy:.2%}")

    result = {
        "total_interactions": total,
        "overall_fallback_rate": round(fallback_rate, 4),
        "average_confidence": round(avg_confidence, 4),
        "low_confidence_rate": round(low_conf_rate, 4),
        "edge_case_accuracy": round(edge_case_accuracy, 4),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "per_agent": per_agent,
        "source": "log_file"
    }

    out_path = RESULTS_DIR / "robustness_report.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")
    return result


if __name__ == "__main__":
    run_robustness()
