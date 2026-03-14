"""
E5: Multi-Turn Context Retention
==================================
Tests whether the assistant correctly uses prior conversation context
in follow-up questions. Runs 3 scripted multi-turn dialogues and scores
each final response for keyword presence + contextual relevance.

Outputs:
 - docs/evaluation/results/context_retention.json
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from docs.evaluation.evaluation_datasets import MULTITURN_CONVERSATIONS

RESULTS_DIR = ROOT / "docs" / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_context_retention():
    """Run the multi-turn context retention evaluation."""
    from agentic_student_assistant.core.orchestration.main_graph import app as agent_graph

    print("🔄  E5: Multi-Turn Context Retention")
    print("=" * 60)

    results = []
    total_score = 0

    for conv in MULTITURN_CONVERSATIONS:
        print(f"\n  📝 Scenario: {conv['name']}")
        print(f"     {conv['description']}")
        chat_history = []

        # Execute each turn
        for i, turn in enumerate(conv["turns"]):
            if turn["role"] == "user":
                query = turn["content"]
                print(f"  👤 Turn {i+1}: {query}")

                try:
                    result = agent_graph.invoke({
                        "query": query,
                        "chat_history": chat_history
                    })
                    response = result.get("result", "")
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    response = ""

                if "__AGENT_RESPONSE__" in [t["content"] for t in conv["turns"]]:
                    # Replace placeholder in next assistant turn
                    for t in conv["turns"]:
                        if t["role"] == "assistant" and t["content"] == "__AGENT_RESPONSE__":
                            t["content"] = response
                            break

                chat_history.append(("user", query))

            elif turn["role"] == "assistant":
                response = turn["content"]
                chat_history.append(("assistant", response))
                if response != "__AGENT_RESPONSE__":
                    print(f"  🤖 Response: {response[:120]}...")

        # Score the final response
        final_response = chat_history[-1][1] if chat_history and chat_history[-1][0] == "assistant" else ""
        keywords = conv.get("eval_keywords", [])
        hits = [kw for kw in keywords if kw.lower() in final_response.lower()]
        keyword_score = len(hits) / len(keywords) if keywords else 0

        # Heuristic: a contextually aware response should be ≥ 50 chars
        length_ok = len(final_response) >= 50
        context_score = round((keyword_score + (1 if length_ok else 0)) / 2, 3)
        total_score += context_score

        print(f"\n  📊 Keyword Score: {keyword_score:.2%} ({len(hits)}/{len(keywords)} keywords matched: {hits})")
        print(f"  📊 Context Score: {context_score:.2%}")

        results.append({
            "scenario": conv["name"],
            "description": conv["description"],
            "eval_keywords": keywords,
            "matched_keywords": hits,
            "keyword_score": round(keyword_score, 3),
            "context_score": context_score,
            "final_response_length": len(final_response),
            "final_response_preview": final_response[:200]
        })

    overall = round(total_score / len(MULTITURN_CONVERSATIONS), 3)
    print(f"\n📊 Overall Context Retention Score: {overall:.2%}")

    out = {"overall_score": overall, "scenarios": results}
    out_path = RESULTS_DIR / "context_retention.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")
    return out


if __name__ == "__main__":
    run_context_retention()
