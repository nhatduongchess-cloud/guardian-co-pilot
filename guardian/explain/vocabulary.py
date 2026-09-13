"""
vocabulary.py
=============

THE WORDS THE CAR IS ALLOWED TO USE.

Kept apart from the explanation logic on purpose: adding a language should be a
translation job, not a code change. Every user-visible string in the Slow Path
lives in this file.

Vietnamese is the primary language - the whole point of the feature is that the
driver hears the reason in the language they trust - and English is here because
the project is read by people who do not speak Vietnamese.
"""

from __future__ import annotations

VI = "vi"
EN = "en"
LANGUAGES = (VI, EN)

#: Decimal separator. Vietnamese writes 1,4 where English writes 1.4, and a
#: safety message that looks foreign is a safety message people distrust.
DECIMAL_SEP = {VI: ",", EN: "."}

#: What the car just did. Keyed by BehaviorMode value.
ACTION = {
    VI: {
        "emergency_brake": "Phanh khẩn cấp",
        "assist_brake": "Phanh hỗ trợ",
        "warn": "Cảnh báo",
        "inform": "Nhắc nhở",
        "cruise": "Chạy bình thường",
    },
    EN: {
        "emergency_brake": "Emergency braking",
        "assist_brake": "Assisted braking",
        "warn": "Warning",
        "inform": "Notice",
        "cruise": "Normal driving",
    },
}

#: Detected object classes. The detector speaks COCO; the driver does not.
OBJECT = {
    VI: {
        "car": "xe ô tô", "vehicle": "xe phía trước", "truck": "xe tải",
        "bus": "xe buýt", "bike": "xe máy", "motorcycle": "xe máy",
        "bicycle": "xe đạp", "walker": "người đi bộ", "person": "người đi bộ",
        "obstacle": "vật cản",
    },
    EN: {
        "car": "a car", "vehicle": "the vehicle ahead", "truck": "a lorry",
        "bus": "a bus", "bike": "a motorbike", "motorcycle": "a motorbike",
        "bicycle": "a bicycle", "walker": "a pedestrian", "person": "a pedestrian",
        "obstacle": "an obstacle",
    },
}

#: Driver state, as the driver would describe it about themselves.
DRIVER_STATE = {
    VI: {
        "alert": "tỉnh táo",
        "drowsy": "buồn ngủ",
        "microsleep": "ngủ gật",
        "yawning": "đang ngáp",
        "distracted": "mất tập trung",
    },
    EN: {
        "alert": "alert",
        "drowsy": "drowsy",
        "microsleep": "micro-sleeping",
        "yawning": "yawning",
        "distracted": "distracted",
    },
}

#: Safety Kernel check names, as a driver would hear them. The kernel's own
#: check names are English engineering labels; these are what gets spoken.
CHECK = {
    VI: {
        "system health": "kiểm tra tình trạng hệ thống",
        "rule-based safety": "kiểm tra quy tắc an toàn",
        "collision check": "kiểm tra nguy cơ va chạm",
        "comfort & drivability": "kiểm tra độ êm khi lái",
    },
    EN: {
        "system health": "the system-health check",
        "rule-based safety": "the safety-rule check",
        "collision check": "the collision check",
        "comfort & drivability": "the drivability check",
    },
}

#: Phrases assembled by the explainer. `{}` placeholders are filled positionally
#: in explainer.py; keep the order when translating.
PHRASE = {
    VI: {
        "unknown_object": "vật cản",
        "scene_factor": "{0} phía trước, cách {1} m, TTC {2} giây.",
        "scene_factor_no_ttc": "{0} phía trước, cách {1} m.",
        "driver_factor": "Tài xế {0}.",
        "driver_factor_reason": "Tài xế {0} — {1}.",
        "driver_evidence_eyes": "mắt nhắm trung bình {0}, PERCLOS {1}%",
        "driver_evidence_closure": "nhắm mắt liên tục {0} giây",
        "driver_evidence_phone": "tay cầm điện thoại {0}% thời gian",
        "driver_reaction": "Cần khoảng {0} giây để tài xế phản ứng.",
        "context_wet": "Mặt đường ướt nên quãng đường phanh dài hơn.",
        "context_night": "Trời tối, tầm nhìn giảm.",
        "context_wet_night": "Mặt đường ướt và trời tối — quãng đường phanh dài hơn.",
        "speed_factor": "Tốc độ {0} km/h.",
        "kernel_vetoed": "Safety Kernel chặn lệnh ở {0}.",
        "kernel_escalated": "Safety Kernel tăng mức can thiệp ở {0}.",
        "kernel_dampened": "Safety Kernel giảm mức can thiệp ở {0}.",
        "kernel_vetoed_plain": "Safety Kernel chặn lệnh.",
        "kernel_escalated_plain": "Safety Kernel tăng mức can thiệp.",
        "kernel_dampened_plain": "Safety Kernel giảm mức can thiệp.",
        "failsafe": "Hệ thống chuyển sang chế độ an toàn.",
        "earlier_because_driver": (
            "Guardian can thiệp sớm hơn bình thường vì tài xế {0}."
        ),
        "headline_intervention": "{0} {1}%: {2}",
        "headline_warning": "{0}: {1}",
        "headline_normal": "Không có nguy hiểm. Tài xế {0}.",
        "no_reason": "không có yếu tố nào đáng lưu ý.",
    },
    EN: {
        "unknown_object": "an obstacle",
        "scene_factor": "{0} ahead at {1} m, TTC {2} s.",
        "scene_factor_no_ttc": "{0} ahead at {1} m.",
        "driver_factor": "The driver is {0}.",
        "driver_factor_reason": "The driver is {0} — {1}.",
        "driver_evidence_eyes": "average eye closure {0}, PERCLOS {1}%",
        "driver_evidence_closure": "eyes closed {0} s in a row",
        "driver_evidence_phone": "phone in hand {0}% of the time",
        "driver_reaction": "About {0} s needed for the driver to react.",
        "context_wet": "The road is wet, so braking distance is longer.",
        "context_night": "It is dark and visibility is reduced.",
        "context_wet_night": "Wet road and darkness — braking distance is longer.",
        "speed_factor": "Speed {0} km/h.",
        "kernel_vetoed": "Safety kernel vetoed the command at {0}.",
        "kernel_escalated": "Safety kernel escalated the command at {0}.",
        "kernel_dampened": "Safety kernel dampened the command at {0}.",
        "kernel_vetoed_plain": "Safety kernel vetoed the command.",
        "kernel_escalated_plain": "Safety kernel escalated the command.",
        "kernel_dampened_plain": "Safety kernel dampened the command.",
        "failsafe": "System fell back to a safe mode.",
        "earlier_because_driver": (
            "Guardian acted earlier than usual because the driver is {0}."
        ),
        "headline_intervention": "{0} {1}%: {2}",
        "headline_warning": "{0}: {1}",
        "headline_normal": "No hazard. The driver is {0}.",
        "no_reason": "nothing of note.",
    },
}


def number(value: float, decimals: int, lang: str) -> str:
    """Format a number in the reading conventions of `lang`."""
    text = f"{value:.{decimals}f}"
    return text.replace(".", DECIMAL_SEP[lang]) if DECIMAL_SEP[lang] != "." else text


def lookup(table: dict, lang: str, key: str, default: str) -> str:
    """Translate `key`, falling back rather than raising: a missing word must
    never be the reason a safety message fails to appear."""
    return table.get(lang, table[EN]).get(key, default)
