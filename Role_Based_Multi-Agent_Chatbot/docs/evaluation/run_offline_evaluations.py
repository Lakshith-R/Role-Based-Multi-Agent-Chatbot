"""
run_offline_evaluations.py — Offline Evaluation Runner
==================================================
Runs only the evaluation scripts that DO NOT consume API credits (E2, E3, E4)
and generates the full thesis evaluation report (E6).

Usage:
    cd Agentic_Student_Assistant-main
    .\.venv\Scripts\python.exe docs/evaluation/run_offline_evaluations.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EVALUATIONS = [
    ("E2 — Response Quality",      "docs.evaluation.e2_response_quality",  "run_response_quality"),
    ("E3 — Latency Benchmark",     "docs.evaluation.e3_latency_benchmark", "run_latency_benchmark"),
    ("E4 — Robustness Testing",    "docs.evaluation.e4_robustness",        "run_robustness"),
]

HEADER = """
╔══════════════════════════════════════════════════════════════╗
║        Agentic Student Assistant — Offline Evaluation        ║
║                  Running E2, E3, E4, E6                      ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(HEADER)
    total_start = time.time()
    passed = []
    failed = []

    for label, module_path, fn_name in EVALUATIONS:
        print(f"\n{'─' * 65}")
        print(f"▶  Running: {label}")
        print(f"{'─' * 65}")
        t0 = time.time()
        try:
            import importlib
            mod = importlib.import_module(module_path)
            fn = getattr(mod, fn_name)
            fn()
            elapsed = time.time() - t0
            print(f"\n✅  {label} completed in {elapsed:.1f}s")
            passed.append(label)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n❌  {label} failed after {elapsed:.1f}s: {e}")
            failed.append((label, str(e)))

    # ── E6: Visualiser (runs after all results are saved) ────────────────────
    print(f"\n{'─' * 65}")
    print(f"▶  Running: E6 — Results Visualiser & Report Generator")
    print(f"{'─' * 65}")
    try:
        from docs.evaluation.e6_visualiser import run_visualiser
        run_visualiser()
        passed.append("E6 — Visualiser")
    except Exception as e:
        print(f"❌  E6 Visualiser failed: {e}")
        failed.append(("E6 — Visualiser", str(e)))

    # ── Final Report ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'═' * 65}")
    print(f"  OFFLINE EVALUATION COMPLETE — Total time: {total_elapsed:.1f}s")
    print(f"  ✅ Passed: {len(passed)}/{len(passed) + len(failed)}")
    if failed:
        print(f"  ❌ Failed: {len(failed)}")
        for label, err in failed:
            print(f"     - {label}: {err[:80]}")
    print(f"\n  📁 Results saved to: docs/evaluation/results/")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()
