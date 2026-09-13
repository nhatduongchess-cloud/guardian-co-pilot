"""
reasoning package — the Reasoning Engine (§2.3).

    fallback_templates.py -> deterministic Vietnamese sentences (never fail)
    prompt_templates.py   -> few-shot prompt assembly for Phi-3
    phi3_runtime.py       -> ONNX Runtime GenAI wrapper (lazy, degrades safely)
    context_builder.py    -> assemble a grounded ExplanationContext
    reasoning_engine.py   -> orchestrate: try Phi-3, gate on confidence,
                             fall back to a template, guarantee an answer
"""

from reasoning.reasoning_engine import ReasoningEngine

__all__ = ["ReasoningEngine"]
