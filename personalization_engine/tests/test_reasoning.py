"""
test_reasoning.py
=================

Unit tests for VERTICAL 1 -> REASONING.

The safety-critical properties we pin here (Constraint #3):
  * the fallback template is always grounded, non-empty, and <=2 sentences;
  * the engine NEVER returns empty;
  * the confidence gate REJECTS any LLM output that invents a number absent
    from the grounded facts, falling back to the template instead.

We drive the LLM path with a FAKE backend so these tests run anywhere, with no
model downloaded (the real Phi-3 runtime degrades to the same fallback path).
"""

import re

import pytest

from models.explanation import ExplanationContext
from models.driver_memory import PersonalizationState
from reasoning import fallback_templates
from reasoning.phi3_runtime import DisabledLLMBackend, LLMBackend, Phi3Error
from reasoning.reasoning_engine import ReasoningEngine


# ---------------------------------------------------------------------------
# a fake backend we can script to return anything
# ---------------------------------------------------------------------------
class FakeBackend(LLMBackend):
    def __init__(self, reply: str, available: bool = True, raise_error: bool = False):
        self._reply = reply
        self._available = available
        self._raise = raise_error

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt, max_new_tokens=96, temperature=0.3):
        if self._raise:
            raise Phi3Error("boom")
        return self._reply


def _ctx(state=PersonalizationState.PERSONALIZED, snippet="Chuyến trước bạn cũng mệt"):
    return ExplanationContext(
        event_id="evt_1",
        event_type="FATIGUE_ALERT",
        situation_summary="PERCLOS 0.42, cao hơn mức bình thường của bạn.",
        decision_taken="Guardian đã phát cảnh báo để nhắc bạn.",
        driver_memory_snippet=snippet,
        personalization_state=state,
    )


def _sentence_count(text: str) -> int:
    # Count sentence-final punctuation only when followed by whitespace or the
    # end of the string, so a decimal like "0.42" is not miscounted as a break.
    return len(re.findall(r"[.!?](?=\s|$)", text))


# ===========================================================================
# GROUP 1 — FALLBACK TEMPLATES
# ===========================================================================


def test_fallback_is_nonempty_and_grounded():
    text = fallback_templates.render_fallback(_ctx())
    assert text.strip()
    # Grounded: the measured value from the situation appears verbatim.
    assert "0.42" in text
    # Reflects the decision that was taken.
    assert "cảnh báo" in text


def test_fallback_at_most_two_sentences():
    text = fallback_templates.render_fallback(_ctx())
    assert _sentence_count(text) <= 2


@pytest.mark.parametrize("state,marker", [
    (PersonalizationState.COLD_START, "mặc định"),
    (PersonalizationState.WARMING, "đang học"),
    (PersonalizationState.PERSONALIZED, "quen thuộc"),
])
def test_fallback_tone_changes_with_state(state, marker):
    text = fallback_templates.render_fallback(_ctx(state=state))
    assert marker in text


def test_fallback_without_history_still_valid():
    text = fallback_templates.render_fallback(_ctx(snippet=""))
    assert text.strip()
    assert _sentence_count(text) <= 2


# ===========================================================================
# GROUP 2 — THE ENGINE: FALLBACK PATH
# ===========================================================================


def test_engine_uses_fallback_when_backend_unavailable():
    engine = ReasoningEngine(backend=DisabledLLMBackend())
    resp = engine.explain(_ctx())
    assert resp.used_fallback_template is True
    assert resp.text_vi.strip()
    assert resp.confidence == 1.0


def test_engine_falls_back_when_backend_errors():
    engine = ReasoningEngine(backend=FakeBackend("x", raise_error=True))
    resp = engine.explain(_ctx())
    assert resp.used_fallback_template is True


def test_engine_never_returns_empty():
    for backend in [DisabledLLMBackend(), FakeBackend(""), FakeBackend("   ")]:
        resp = ReasoningEngine(backend=backend).explain(_ctx())
        assert resp.text_vi.strip()


# ===========================================================================
# GROUP 3 — THE ENGINE: LLM PATH + CONFIDENCE GATE
# ===========================================================================


def test_engine_accepts_grounded_llm_output():
    good = "Mắt bạn nhắm nhiều hơn bình thường nên Guardian nhắc bạn nghỉ ngơi."
    engine = ReasoningEngine(backend=FakeBackend(good))
    resp = engine.explain(_ctx())
    assert resp.used_fallback_template is False
    assert resp.text_vi == good
    assert resp.confidence < 1.0  # LLM answers carry <1.0 confidence


def test_engine_rejects_hallucinated_number():
    # 9.9 does not appear in the grounded facts -> must be rejected -> fallback.
    bad = "Guardian phát hiện PERCLOS 9.9 nên đã dừng xe khẩn cấp."
    engine = ReasoningEngine(backend=FakeBackend(bad))
    resp = engine.explain(_ctx())
    assert resp.used_fallback_template is True
    assert "9.9" not in resp.text_vi


def test_engine_accepts_number_present_in_facts():
    ok = "Guardian thấy PERCLOS 0.42 nên nhắc bạn nghỉ."
    engine = ReasoningEngine(backend=FakeBackend(ok))
    resp = engine.explain(_ctx())
    assert resp.used_fallback_template is False


def test_engine_trims_to_two_sentences():
    verbose = "Câu một. Câu hai. Câu ba thừa thãi. Câu bốn."
    engine = ReasoningEngine(backend=FakeBackend(verbose))
    resp = engine.explain(_ctx())
    # The engine keeps at most the first two sentences of an accepted answer.
    assert _sentence_count(resp.text_vi) <= 2


def test_engine_rejects_absurdly_long_output():
    huge = "an toàn " * 200  # far over the char cap, no sentence punctuation
    engine = ReasoningEngine(backend=FakeBackend(huge))
    resp = engine.explain(_ctx())
    assert resp.used_fallback_template is True
