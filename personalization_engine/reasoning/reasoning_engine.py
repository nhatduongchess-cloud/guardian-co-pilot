"""
reasoning_engine.py
===================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: REASONING  ·  §2.3 / Constraint #3

The orchestrator. It turns a grounded `ExplanationContext` into a final
`ExplanationResponse`, deciding — safely — whether to trust the LLM or fall
back to a deterministic template.

THE PIPELINE (top to bottom = most-preferred to safest)
-------------------------------------------------------
    1. If a language model is available, build the prompt and generate.
    2. VALIDATE the output with a deterministic gate:
         - non-empty, not too long, trimmed to <=2 sentences,
         - contains no NUMBER that is absent from the grounded facts
           (the anti-hallucination guard for safety values).
    3. If validation passes -> use it (confidence 0.85).
    4. If the LLM is unavailable, errors, or fails validation ->
       render the fallback template (confidence 1.0, it cannot be wrong).

The result is that `GetExplanation` ALWAYS returns a grounded sentence, and a
misbehaving LLM can only ever downgrade us to the template — never emit a
wrong safety claim (Constraint #1, #3).
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import re
from typing import Optional

from models.explanation import ExplanationContext, ExplanationResponse
from reasoning import fallback_templates, prompt_templates
from reasoning.phi3_runtime import LLMBackend, Phi3Error, Phi3Runtime


# Confidence we assign to an LLM answer that passed every deterministic check.
_LLM_CONFIDENCE = 0.85
# Longest acceptable explanation (defensive cap; ~2 short Vietnamese sentences).
_MAX_CHARS = 320


class ReasoningEngine:
    """
    Owns the LLM backend (or lack of one) and the fallback logic. Inject a
    backend for tests; leave it None to auto-build the lazy Phi-3 runtime.
    """

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        max_new_tokens: int = 96,
        temperature: float = 0.3,
    ) -> None:
        # Default backend is the lazy, safe Phi-3 runtime — cheap to construct,
        # loads the model only on first real use.
        self._backend: LLMBackend = backend if backend is not None else Phi3Runtime()
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    # -- introspection (for /health) ---------------------------------------
    def backend_available(self) -> bool:
        try:
            return self._backend.is_available()
        except Exception:
            return False

    # -- the one public method ---------------------------------------------
    def explain(self, context: ExplanationContext) -> ExplanationResponse:
        """Produce a grounded ExplanationResponse for a context. Never raises."""
        # Path A: try the LLM if it is genuinely available.
        if self.backend_available():
            candidate = self._try_llm(context)
            if candidate is not None:
                return ExplanationResponse(
                    text_vi=candidate,
                    confidence=_LLM_CONFIDENCE,
                    used_fallback_template=False,
                    audio_ready=True,
                )

        # Path B: deterministic fallback (always correct, always available).
        return ExplanationResponse(
            text_vi=fallback_templates.render_fallback(context),
            confidence=1.0,
            used_fallback_template=True,
            audio_ready=True,
        )

    # =======================================================================
    # PRIVATE
    # =======================================================================
    def _try_llm(self, context: ExplanationContext) -> Optional[str]:
        """Generate + validate. Return clean text, or None to trigger fallback."""
        prompt = prompt_templates.build_phi3_prompt(context)
        try:
            raw = self._backend.generate(
                prompt,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
            )
        except Phi3Error:
            return None
        except Exception:
            # Any unexpected backend error must never escape — fall back.
            return None

        cleaned = self._sanitize(raw)
        if not self._passes_gate(cleaned, context):
            return None
        return cleaned

    @staticmethod
    def _sanitize(text: str) -> str:
        """Trim, collapse whitespace, and keep at most the first 2 sentences."""
        text = re.sub(r"\s+", " ", text).strip()
        # Split into sentences, but ONLY treat '.'/'!'/'?' as a boundary when it
        # is followed by whitespace or end-of-string. This keeps decimals like
        # "0.42" intact instead of splitting them into "0." and "42".
        sentences = re.findall(r".+?[.!?](?=\s|$)", text)
        if sentences:
            text = " ".join(s.strip() for s in sentences[:2]).strip()
        return text

    @classmethod
    def _passes_gate(cls, text: str, context: ExplanationContext) -> bool:
        """
        The deterministic confidence gate. Reject the LLM output unless it is:
          * non-empty and not absurdly long,
          * free of any NUMBER not present in the grounded facts (this is the
            hard anti-hallucination check for safety-relevant values).
        """
        if not text or len(text) > _MAX_CHARS:
            return False

        allowed = cls._numbers_in(cls._facts_text(context))
        produced = cls._numbers_in(text)
        # Every number the model produced must be justified by the facts.
        for n in produced:
            if n not in allowed:
                return False
        return True

    @staticmethod
    def _facts_text(context: ExplanationContext) -> str:
        return " ".join([
            context.situation_summary,
            context.decision_taken,
            context.driver_memory_snippet,
        ])

    @staticmethod
    def _numbers_in(text: str) -> set[str]:
        """Extract numeric tokens (e.g. '0.42', '1.1', '3'), normalising commas."""
        raw = re.findall(r"\d+(?:[.,]\d+)?", text)
        return {tok.replace(",", ".") for tok in raw}


# ===========================================================================
# SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    from reasoning.phi3_runtime import DisabledLLMBackend

    # Force the fallback path so this runs anywhere (no model needed).
    engine = ReasoningEngine(backend=DisabledLLMBackend())
    ctx = ExplanationContext(
        event_id="evt_1",
        event_type="FATIGUE_ALERT",
        situation_summary="PERCLOS 0.42, cao hơn mức bình thường của bạn.",
        decision_taken="Guardian đã phát cảnh báo để nhắc bạn.",
        driver_memory_snippet="Chuyến trước bạn cũng có dấu hiệu mệt vào buổi tối.",
        personalization_state="PERSONALIZED",
    )
    resp = engine.explain(ctx)
    print("backend available:", engine.backend_available())
    print("used_fallback    :", resp.used_fallback_template)
    print("text_vi          :", resp.text_vi)
