"""
E3: Latency Benchmarking — Zero API Calls Version
===================================================
Reads from the existing local log file (logs/workflow_logs.txt) and computes
latency statistics from REAL user interactions. NO extra API calls needed.

Metrics per agent:
 - Mean, Median, P95, P99, Min, Max latency

Outputs:
 - docs/evaluation/results/latency_report.json
 - docs/evaluation/results/plots/latency_boxplot.png
"""

import sys
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "docs" / "evaluation" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT.parent / "Agentic_Student_Assistant-main" / "logs" / "workflow_logs.txt"


def percentile(data: list, p: float) -> float:
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


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
            elif "Latency:" in line:
                try:
                    current["latency"] = float(re.search(r"[\d.]+", line).group())
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


def run_latency_benchmark():
    """Run latency evaluation from log file data."""
    print("⏱️  E3: Latency Benchmarking (from log file — zero API calls)")
    print("=" * 60)

    entries = parse_log_file(LOG_FILE)
    by_agent = defaultdict(list)

    for e in entries:
        agent = e.get("agent", "unknown")
        latency = e.get("latency")
        if agent in ("unknown", "error", "cached") or latency is None:
            continue
        by_agent[agent].append(latency)

    if not by_agent:
        print("⚠️  No latency data found in log file.")
        print("   Run the Streamlit app and ask questions first, then re-run.")
        return {}

    summary = {}
    all_latencies = {}

    print(f"\n{'Agent':<15} {'N':>4} {'Mean':>7} {'Median':>8} {'P95':>7} {'P99':>7} {'Min':>7} {'Max':>7}")
    print("-" * 65)

    for agent, latencies in sorted(by_agent.items()):
        stats = {
            "mean":   round(sum(latencies) / len(latencies), 3),
            "median": round(percentile(latencies, 50), 3),
            "p95":    round(percentile(latencies, 95), 3),
            "p99":    round(percentile(latencies, 99), 3),
            "min":    round(min(latencies), 3),
            "max":    round(max(latencies), 3),
            "n":      len(latencies),
        }
        summary[agent] = stats
        all_latencies[agent] = latencies
        print(f"{agent:<15} {stats['n']:>4} {stats['mean']:>7.3f} {stats['median']:>8.3f} "
              f"{stats['p95']:>7.3f} {stats['p99']:>7.3f} {stats['min']:>7.3f} {stats['max']:>7.3f}")

    print(f"\n  Total interactions analysed: {sum(len(v) for v in by_agent.values())}")

    out_path = RESULTS_DIR / "latency_report.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "raw_latencies": all_latencies}, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")

    # Box Plot
    try:
        import matplotlib.pyplot as plt

        import numpy as np

        agents = list(all_latencies.keys())
        data = []
        for a in agents:
            raw = all_latencies[a]
            if not raw:
                data.append([])
                continue
            q1, q3 = np.percentile(raw, [25, 75])
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            filtered = [x for x in raw if x <= upper_bound]
            data.append(filtered)
            
        fig, ax = plt.subplots(figsize=(9, 5))
        bp = ax.boxplot(data, labels=agents, patch_artist=True)
        colors = ["#4C9BE8", "#F4A261", "#2A9D8F", "#E76F51"]
        for patch, color in zip(bp["boxes"], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xlabel("Agent", fontsize=12)
        ax.set_ylabel("Latency (seconds)", fontsize=12)
        ax.set_title("Latency per Agent (from Real Interaction Logs)", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plot_path = PLOTS_DIR / "latency_boxplot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"📊 Box plot saved to {plot_path}")
    except ImportError:
        print("⚠️  matplotlib not installed — skipping plot")

    return summary


if __name__ == "__main__":
    run_latency_benchmark()
