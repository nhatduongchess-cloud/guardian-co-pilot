"""
run_eval.py
===========

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: EVAL  ·  spec §1 ("thêm Eval harness") / judging §"giá trị thực tế"

A tiny, rubric-based evaluation of the reasoning engine over a golden set of
explanation contexts (golden_set.jsonl). It exists to make one point to the
reviewers: we MEASURE explanation quality, we do not demo-and-pray.

THE RUBRIC (5 checks per case, each worth 1 point)
--------------------------------------------------
  1. non_empty      : an answer was produced at all.
  2. le_2_sentences : at most two sentences (schema §3.4).
  3. grounded       : every NUMBER in the answer appears in the facts — the
                      anti-hallucination guarantee (Constraint #3).
  4. vietnamese     : the answer contains Vietnamese diacritics.
  5. on_topic       : every expected keyword is present.

It runs with the DETERMINISTIC fallback by default (so results are stable and
reproducible), and automatically with the real Phi-3 model if one is loaded —
letting you compare template vs LLM quality with the same yardstick.

RUN (from personalization_engine/):
    python -m eval.run_eval
    python -m eval.run_eval --json     # machine-readable summary
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import argparse
import json
import re
import sys
from pathlib import Path

from models.explanation import ExplanationContext, ExplanationResponse
from reasoning.reasoning_engine import ReasoningEngine

_GOLDEN = Path(__file__).with_name("golden_set.jsonl")

# Vietnamese-specific letters (a quick "is this actually Vietnamese?" signal).
_VIET_CHARS = "ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ"


# ===========================================================================
# SECTION 1 — RUBRIC CHECKS
# ===========================================================================
def _numbers(text: str) -> set[str]:
    return {t.replace(",", ".") for t in re.findall(r"\d+(?:[.,]\d+)?", text)}


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?=\s|$)", text))


def score_case(
    context: ExplanationContext,
    response: ExplanationResponse,
    expect_keywords: list[str],
) -> dict:
    """Return a dict of {check_name: bool} plus the produced text."""
    text = response.text_vi
    facts = " ".join([
        context.situation_summary,
        context.decision_taken,
        context.driver_memory_snippet,
    ])

    checks = {
        "non_empty": bool(text.strip()),
        "le_2_sentences": _sentence_count(text) <= 2,
        "grounded": _numbers(text).issubset(_numbers(facts)),
        "vietnamese": any(ch in _VIET_CHARS for ch in text.lower()),
        "on_topic": all(kw.lower() in text.lower() for kw in expect_keywords),
    }
    return {"text": text, "checks": checks, "score": sum(checks.values())}


# ===========================================================================
# SECTION 2 — RUNNER
# ===========================================================================
def run(as_json: bool = False) -> int:
    """Evaluate every golden case. Returns a process exit code (0 = all pass)."""
    if not _GOLDEN.exists():
        print(f"Golden set not found: {_GOLDEN}", file=sys.stderr)
        return 2

    engine = ReasoningEngine()
    mode = "phi3-onnx" if engine.backend_available() else "deterministic-fallback"

    cases = [json.loads(line) for line in _GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]

    results = []
    total_points = 0
    max_points = 0
    for case in cases:
        context = ExplanationContext.model_validate(case["context"])
        response = engine.explain(context)
        result = score_case(context, response, case.get("expect_keywords", []))
        result["id"] = case["id"]
        results.append(result)
        total_points += result["score"]
        max_points += len(result["checks"])

    perfect = sum(1 for r in results if r["score"] == len(r["checks"]))

    if as_json:
        print(json.dumps({
            "mode": mode,
            "cases": len(results),
            "perfect_cases": perfect,
            "points": total_points,
            "max_points": max_points,
            "results": results,
        }, ensure_ascii=False, indent=2))
    else:
        _print_report(mode, results, perfect, total_points, max_points)

    # Exit non-zero if any case is not perfect — makes this usable in CI.
    return 0 if perfect == len(results) else 1


def _print_report(mode, results, perfect, total_points, max_points) -> None:
    print("=" * 72)
    print(f"Reasoning Engine evaluation  ·  backend = {mode}")
    print("=" * 72)
    for r in results:
        flags = "".join("✓" if v else "✗" for v in r["checks"].values())
        status = "PASS" if r["score"] == len(r["checks"]) else "FAIL"
        print(f"[{status}] {r['id']:<28} {flags}  ({r['score']}/{len(r['checks'])})")
        print(f"        → {r['text']}")
        if r["score"] != len(r["checks"]):
            failed = [k for k, v in r["checks"].items() if not v]
            print(f"        ! failed: {', '.join(failed)}")
    print("-" * 72)
    print(f"Perfect cases: {perfect}/{len(results)}   "
          f"Total: {total_points}/{max_points} points "
          f"({100 * total_points / max_points:.0f}%)")
    print("Legend: non_empty · le_2_sentences · grounded · vietnamese · on_topic")


# ===========================================================================
# SECTION 3 — CLI
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the reasoning engine.")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    raise SystemExit(run(as_json=args.json))
