"""
E1: Router Accuracy Evaluation
================================
Runs 30 labelled test queries through the RouterAgent and measures:
 - Per-class Precision, Recall, F1
 - Overall accuracy
 - Confidence score distribution
 - Confusion matrix chart

Outputs:
 - docs/evaluation/results/router_accuracy.json
 - docs/evaluation/results/plots/confusion_matrix.png
"""

import sys
import json
import os
from pathlib import Path
from collections import defaultdict

# ─── Path Setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from docs.evaluation.evaluation_datasets import ROUTER_TEST_SET

RESULTS_DIR = ROOT / "docs" / "evaluation" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_router_accuracy():
    """Run the routing accuracy evaluation."""
    from agentic_student_assistant.core.orchestration.router_agent import RouterAgent

    print("🔍 E1: Router Accuracy Evaluation")
    print("=" * 60)

    router = RouterAgent()

    y_true = []
    y_pred = []
    confidences = []
    details = []

    for i, item in enumerate(ROUTER_TEST_SET):
        query = item["query"]
        expected = item["expected"]

        try:
            decision = router.route(query)
            predicted = decision.agent
            confidence = decision.confidence
        except Exception as e:
            print(f"  ⚠️  Error on query '{query[:40]}': {e}")
            predicted = "error"
            confidence = 0.0

        y_true.append(expected)
        y_pred.append(predicted)
        confidences.append(confidence)

        correct = predicted == expected
        status = "✅" if correct else "❌"
        print(f"  {status} [{i+1:02d}] {query[:45]:<45} | Expected: {expected:<12} | Got: {predicted:<12} | Conf: {confidence:.2f}")

        details.append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
            "correct": correct
        })

    # ── Metrics ───────────────────────────────────────────────────────────────
    labels = sorted(set(y_true))
    overall_accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "support": y_true.count(label)}

    avg_confidence = sum(confidences) / len(confidences)

    # ── Print Summary ─────────────────────────────────────────────────────────
    print(f"\n📊 Overall Accuracy: {overall_accuracy:.2%}")
    print(f"📊 Average Confidence: {avg_confidence:.3f}")
    print(f"\n{'Agent':<15} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 55)
    for label, m in per_class.items():
        print(f"{label:<15} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {m['support']:>8}")

    # ── Save Results ──────────────────────────────────────────────────────────
    result = {
        "overall_accuracy": round(overall_accuracy, 4),
        "average_confidence": round(avg_confidence, 4),
        "per_class_metrics": per_class,
        "details": details
    }
    out_path = RESULTS_DIR / "router_accuracy.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")

    # ── Confusion Matrix Plot ─────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        n = len(labels)
        matrix = np.zeros((n, n), dtype=int)
        label_idx = {l: i for i, l in enumerate(labels)}
        for t, p in zip(y_true, y_pred):
            if p in label_idx:
                matrix[label_idx[t]][label_idx[p]] += 1

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted Agent", fontsize=12)
        ax.set_ylabel("True Agent", fontsize=12)
        ax.set_title("Router Agent — Confusion Matrix", fontsize=14, fontweight="bold")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() * 0.5 else "black", fontsize=12)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plot_path = PLOTS_DIR / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"📊 Confusion matrix saved to {plot_path}")
    except ImportError:
        print("⚠️  matplotlib not installed — skipping plot")

    return result


if __name__ == "__main__":
    run_router_accuracy()
