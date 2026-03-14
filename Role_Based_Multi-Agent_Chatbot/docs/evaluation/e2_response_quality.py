"""
E2: Response Quality — Zero API Calls Version
==============================================
Reads from the existing local log file (logs/workflow_logs.txt) and scores
response quality using keyword-based heuristics. NO extra LLM/API calls.

Metrics per agent:
 - Avg Response Length (completeness proxy)
 - Query-Response Keyword Overlap (relevance proxy)
 - Response Quality Score (0-10 composite)

Outputs:
 - docs/evaluation/results/response_quality.json
 - docs/evaluation/results/plots/response_quality_bar.png
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

STOPWORDS = {"a", "an", "the", "is", "it", "in", "on", "for", "to", "of", "and", "or", "my", "me", "i"}


def parse_log_file(log_path: Path) -> list:
    """Parse workflow_logs.txt into a list of interaction dicts."""
    if not log_path.exists():
        print(f"⚠️  Log file not found at {log_path}")
        print("   Run the Streamlit app and ask some questions first to generate logs.")
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
            elif "Confidence:" in line:
                try:
                    current["confidence"] = float(re.search(r"[\d.]+", line).group())
                except Exception:
                    pass
            elif "Fallback Used:" in line:
                current["is_fallback"] = "yes" in line.lower()
            elif "Final Answer:" in line:
                # Next lines are the answer — grab next line
                current["answer_start"] = True
            elif current.get("answer_start") and line and not line.startswith("="):
                current["answer"] = current.get("answer", "") + " " + line
            elif "=" * 20 in line and current.get("query"):
                entries.append(current)
                current = {}

    if current.get("query"):
        entries.append(current)

    return entries


def keyword_overlap(query: str, answer: str) -> float:
    """Compute keyword overlap between query and answer (relevance score 0-1)."""
    q_words = {w.lower() for w in re.findall(r"\b\w{4,}\b", query) if w.lower() not in STOPWORDS}
    a_words = {w.lower() for w in re.findall(r"\b\w{4,}\b", answer)}
    if not q_words:
        return 0.5
    overlap = q_words & a_words
    return min(len(overlap) / len(q_words), 1.0)


def length_score(answer: str) -> float:
    """Score response completeness based on length (0-1, max at 500 chars)."""
    return min(len(answer.strip()) / 500, 1.0)


def run_response_quality():
    """Run keyword-based response quality evaluation from log file."""
    print("🧑‍⚖️  E2: Response Quality Evaluation (from log file — zero API calls)")
    print("=" * 60)

    entries = parse_log_file(LOG_FILE)

    if not entries:
        # No log data — use synthetic fallback values for demo purposes
        print("⚠️  No log data found. Using placeholder values.")
        print("   Tip: Chat with the app first, then re-run this evaluation.")
        synth = {
            "papers":     {"relevance": 7.2, "completeness": 6.8, "overall": 7.0},
            "books":      {"relevance": 7.8, "completeness": 7.5, "overall": 7.65},
            "job_market": {"relevance": 8.1, "completeness": 7.0, "overall": 7.55},
            "documents":  {"relevance": 6.5, "completeness": 6.0, "overall": 6.25},
        }
        result = {"agent_averages": synth, "source": "placeholder", "details": {}}
        out_path = RESULTS_DIR / "response_quality.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Placeholder results saved to {out_path}")
        _plot_bar(synth)
        return result

    # Group by agent
    by_agent = defaultdict(list)
    for e in entries:
        agent = e.get("agent", "unknown")
        if agent in ("unknown", "error", "cached"):
            continue
        by_agent[agent].append(e)

    agent_averages = {}
    details = {}

    for agent, interactions in by_agent.items():
        scores = []
        for e in interactions:
            query = e.get("query", "")
            answer = e.get("answer", "")
            rel = keyword_overlap(query, answer) * 10   # 0-10
            comp = length_score(answer) * 10             # 0-10
            overall = round((rel + comp) / 2, 2)
            scores.append({"relevance": round(rel, 2), "completeness": round(comp, 2), "overall": overall})

        avg = {
            "relevance":    round(sum(s["relevance"] for s in scores) / len(scores), 2),
            "completeness": round(sum(s["completeness"] for s in scores) / len(scores), 2),
            "overall":      round(sum(s["overall"] for s in scores) / len(scores), 2),
            "n":            len(scores)
        }
        agent_averages[agent] = avg
        details[agent] = scores
        print(f"  📌 {agent:<15} Relevance: {avg['relevance']:>5}/10 | Completeness: {avg['completeness']:>5}/10 | Overall: {avg['overall']:>5}/10  (n={avg['n']})")

    print(f"\n  Total interactions analysed: {len(entries)}")

    result = {"agent_averages": agent_averages, "source": "log_file", "details": details}
    out_path = RESULTS_DIR / "response_quality.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")
    _plot_bar(agent_averages)
    return result


def _plot_bar(agent_averages: dict):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        agents = list(agent_averages.keys())
        x = np.arange(len(agents))
        width = 0.28
        metrics = ["relevance", "completeness", "overall"]
        colors = ["#4C9BE8", "#F4A261", "#2A9D8F"]

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = [agent_averages[a].get(metric, 0) for a in agents]
            bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=color, alpha=0.85)
            ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=2)

        ax.set_xlabel("Agent", fontsize=12)
        ax.set_ylabel("Score (0-10)", fontsize=12)
        ax.set_title("Per-Agent Response Quality (Keyword-Based Scoring)", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(agents, fontsize=11)
        ax.set_ylim(0, 12)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plot_path = PLOTS_DIR / "response_quality_bar.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"📊 Bar chart saved to {plot_path}")
    except ImportError:
        print("⚠️  matplotlib not installed — skipping plot")


if __name__ == "__main__":
    run_response_quality()
