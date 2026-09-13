"""
prompt_templates.py
===================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: REASONING  ·  §2.3

This file builds the PROMPT for Phi-3-mini. It is deliberately separated from
the runtime (phi3_runtime.py) and from the decision of WHAT to say
(context_builder.py). The division of labour is the whole safety story:

    context_builder  decides the FACTS  (deterministic, grounded)
    prompt_templates decides the FRAMING (fixed instructions + few-shot)
    phi3_runtime     decides the WORDS   (LLM, low temperature)

The model is never asked to reason about safety. It is asked to do one narrow
thing: rephrase already-decided Vietnamese facts into one natural, warm
sentence for a driver. The system prompt and few-shot examples pin it there.

FORMAT
------
We emit Phi-3's official chat template string:
    <|system|>...<|end|><|user|>...<|end|><|assistant|>
so ONNX Runtime GenAI can consume it directly.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from models.explanation import ExplanationContext


# ===========================================================================
# SECTION 1 — THE SYSTEM INSTRUCTION (the guardrails, in Vietnamese)
# ===========================================================================
SYSTEM_PROMPT = (
    "Bạn là trợ lý an toàn Guardian trên xe hơi. Nhiệm vụ DUY NHẤT của bạn là "
    "diễn đạt lại các dữ kiện đã cho thành MỘT lời nhắc ngắn gọn, thân thiện "
    "bằng tiếng Việt cho tài xế.\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. CHỈ dùng dữ kiện được cung cấp. TUYỆT ĐỐI không thêm số liệu, nguyên "
    "nhân hay hành động không có trong dữ kiện.\n"
    "2. Tối đa 2 câu. Ngắn gọn, bình tĩnh, không gây hoảng.\n"
    "3. Không đưa ra mệnh lệnh điều khiển xe; chỉ giải thích và khuyến nghị nhẹ nhàng.\n"
    "4. Viết hoàn toàn bằng tiếng Việt có dấu."
)


# ===========================================================================
# SECTION 2 — FEW-SHOT EXAMPLES (4 grounded pairs)
# ===========================================================================
# Each example shows the model exactly how to turn a facts-block into a good
# sentence. They cover different event types and personalisation states so the
# model generalises the STYLE, not the content.
# ---------------------------------------------------------------------------
_FEWSHOT: list[tuple[str, str]] = [
    (
        "Sự kiện: cảnh báo mệt mỏi.\n"
        "Tình huống: PERCLOS 0.42, cao hơn mức bình thường của bạn.\n"
        "Quyết định: Guardian đã phát cảnh báo nhắc nghỉ ngơi.\n"
        "Lịch sử: Chuyến trước bạn cũng mệt vào buổi tối.\n"
        "Trạng thái cá nhân hoá: PERSONALIZED.",
        "Mắt bạn nhắm nhiều hơn bình thường nên Guardian nhắc bạn nghỉ ngơi một chút. "
        "Giống chuyến tối hôm trước, buổi tối bạn thường dễ mệt hơn.",
    ),
    (
        "Sự kiện: nguy cơ va chạm vật cản.\n"
        "Tình huống: Thời gian tới va chạm chỉ còn 1.1 giây với một xe phía trước.\n"
        "Quyết định: Guardian đã hỗ trợ phanh.\n"
        "Lịch sử: (không có).\n"
        "Trạng thái cá nhân hoá: COLD_START.",
        "Xe phía trước quá gần nên Guardian đã hỗ trợ phanh để giữ khoảng cách an toàn. "
        "Vì mới quen bạn, Guardian đang dùng ngưỡng an toàn mặc định.",
    ),
    (
        "Sự kiện: mặt đường trơn.\n"
        "Tình huống: Ước lượng độ bám đường thấp (0.28).\n"
        "Quyết định: Guardian đã nhắc giảm tốc độ.\n"
        "Lịch sử: (không có).\n"
        "Trạng thái cá nhân hoá: WARMING.",
        "Mặt đường đang khá trơn nên Guardian nhắc bạn giảm tốc để chắc tay lái hơn. "
        "Guardian vẫn đang học thói quen lái của bạn để điều chỉnh phù hợp.",
    ),
    (
        "Sự kiện: lệch làn.\n"
        "Tình huống: Xe drift nhẹ sang làn bên phải.\n"
        "Quyết định: Guardian đã phát cảnh báo lệch làn.\n"
        "Lịch sử: Bạn thường giữ làn rất tốt.\n"
        "Trạng thái cá nhân hoá: PERSONALIZED.",
        "Xe hơi lệch sang phải nên Guardian nhắc bạn chỉnh lại làn. "
        "Bình thường bạn giữ làn rất tốt nên đây chỉ là nhắc nhở nhẹ.",
    ),
]


# ===========================================================================
# SECTION 3 — CONTEXT -> FACTS BLOCK
# ===========================================================================
def facts_block(context: ExplanationContext) -> str:
    """Render an ExplanationContext into the same facts layout as the few-shots."""
    history = context.driver_memory_snippet.strip() or "(không có)"
    return (
        f"Sự kiện: {context.event_type.value}.\n"
        f"Tình huống: {context.situation_summary}\n"
        f"Quyết định: {context.decision_taken}\n"
        f"Lịch sử: {history}\n"
        f"Trạng thái cá nhân hoá: {context.personalization_state.value}."
    )


# ===========================================================================
# SECTION 4 — BUILD THE PHI-3 CHAT PROMPT
# ===========================================================================
def build_phi3_prompt(context: ExplanationContext) -> str:
    """
    Assemble the full Phi-3 chat-template string: system rules, the few-shot
    demonstrations, then the real facts block, ending on an open assistant turn.
    """
    parts: list[str] = [f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"]
    for user_facts, assistant_reply in _FEWSHOT:
        parts.append(f"<|user|>\n{user_facts}<|end|>\n")
        parts.append(f"<|assistant|>\n{assistant_reply}<|end|>\n")
    parts.append(f"<|user|>\n{facts_block(context)}<|end|>\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


# ===========================================================================
# SECTION 5 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    ctx = ExplanationContext(
        event_id="evt_1",
        event_type="FATIGUE_ALERT",
        situation_summary="PERCLOS 0.42, cao hơn mức bình thường của bạn.",
        decision_taken="Guardian đã phát cảnh báo nhắc bạn nghỉ ngơi.",
        driver_memory_snippet="Chuyến trước bạn cũng có dấu hiệu mệt vào buổi tối.",
        personalization_state="PERSONALIZED",
    )
    print(build_phi3_prompt(ctx))
