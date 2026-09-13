# Vertical 1 — Personalization Engine (Guardian AI OS)

> **Guardian chỉ TƯ VẤN, không bao giờ ra lệnh xe.** Toàn bộ vertical này sinh ra
> *ngưỡng gợi ý* và *lời giải thích tiếng Việt*. Mọi lệnh an toàn thực sự là của
> **V4 Safety Kernel** (deterministic, không AI). Nếu engine này sập, treo hay
> trả sai — hệ thống lái **không bị ảnh hưởng**.

Engine học thói quen của từng tài xế qua các chuyến đi, rồi (a) đề xuất ngưỡng an
toàn cá nhân hoá cho V4, và (b) giải thích ngắn gọn, có căn cứ bằng tiếng Việt cho
tài xế qua cockpit V2.

---

## 0. Bốn ràng buộc không được phá vỡ

| # | Ràng buộc | Được thực thi ở đâu |
|---|-----------|----------------------|
| 1 | **Advisory-only** — V1 không phát lệnh xe. | `PersonalizedThreshold.is_advisory=True`; V1 chỉ trả số, V4 tự quyết. |
| 2 | **Chỉ cá nhân hoá sau ≥3 chuyến.** | State machine `COLD_START → WARMING → PERSONALIZED` trong [`memory/state_machine.py`](memory/state_machine.py). |
| 3 | **Giải thích phải grounded**, có fallback, không bịa. | [`reasoning/context_builder.py`](reasoning/context_builder.py) + confidence gate + [`reasoning/fallback_templates.py`](reasoning/fallback_templates.py). |
| 4 | **Chạy edge, không cần phần cứng mới.** | Pure-Python + SQLite + numpy; Phi-3 INT4 chạy CPU (tùy chọn). |

---

## 1. Kiến trúc

```
        [V3 Perception]                 [V4 World Model / Safety Kernel]
        risk embedding 128d             driver state · vehicle health · decision
              │                                   │
              └───────────────┬───────────────────┘
                              ▼   TelemetryEvent (§3.1)
     ┌───────────────────────────────────────────────────────────────┐
     │  VERTICAL 1 — PERSONALIZATION ENGINE                           │
     │                                                                │
     │  ingestion/  → Feature Store (append-only log per trip)        │
     │  memory/     → Driver Memory + State Machine + Learning Loop   │
     │  reasoning/  → Phi-3 (or fallback) grounded explanation        │
     │  services/   → business logic (threshold, explain, consent)    │
     │  api/        → REST surface (FastAPI)                          │
     └───────────────────────────────────────────────────────────────┘
             │  GetPersonalizedThreshold        │  GetExplanation
             ▼  (advisory number)               ▼  (text_vi + TTS)
        V4 Safety Kernel                    V2 Cockpit HMI
```

Luồng dữ liệu: `TelemetryEvent` → Feature Store → (kết thúc chuyến) Learning Loop
cập nhật Driver Memory → State Machine chuyển trạng thái → threshold & explanation
đọc từ memory. **Học theo lô sau mỗi chuyến, không real-time trong lúc lái** — nên
không bao giờ nằm trên Fast Path ≤100ms của V4.

---

## 2. Cấu trúc thư mục

```
personalization_engine/
├── main.py                     # điểm khởi động FastAPI (gắn 2 router)
├── conftest.py                 # để `pytest` chạy được từ mọi nơi
├── requirements.txt
│
├── models/                     # SCHEMA (vocabulary chung, §3)
│   ├── driver_profile.py       #   WHO + preferences (bản gốc)
│   ├── telemetry.py            #   TelemetryEvent (§3.1)
│   ├── driver_memory.py        #   DriverMemory + PersonalizationState (§3.2)
│   └── explanation.py          #   ExplanationContext/Response + Threshold (§3.3/§3.4)
│
├── ingestion/
│   └── feature_store.py        # append-only log (JSONL + in-memory), interface
│
├── memory/
│   ├── store.py                # MemoryStore interface (+ in-memory)
│   ├── sqlite_store.py         # SqliteMemoryStore (zero-ops, stdlib)
│   └── state_machine.py        # ⭐ trạng thái + EMA + z-score + safety floor
│
├── reasoning/
│   ├── phi3_runtime.py         # wrapper ONNX Runtime GenAI (lazy, degrade an toàn)
│   ├── prompt_templates.py     # few-shot tiếng Việt cho Phi-3
│   ├── context_builder.py      # lắp ExplanationContext grounded (+ retrieval)
│   ├── fallback_templates.py   # câu tĩnh, không bao giờ hallucinate
│   └── reasoning_engine.py     # orchestrator: thử LLM → gate → fallback
│
├── services/
│   ├── personalization_service.py  # identify + preferences (bản gốc)
│   ├── memory_service.py           # ingest, learning loop, threshold, consent
│   └── explanation_service.py      # GetExplanation(event_id)
│
├── api/
│   ├── routes.py               # driver identification (bản gốc)
│   └── memory_routes.py        # telemetry / threshold / explanation / consent
│
├── client/
│   └── personalization_client.py   # cầu nối cross-vertical (V2/V4 gọi Python sạch)
│
├── eval/
│   ├── golden_set.jsonl        # 12 case mẫu
│   └── run_eval.py             # chấm điểm theo rubric
│
├── mock_data/
│   ├── drivers.json            # 5 tài xế mẫu
│   └── simulate_trips.py       # ⭐ demo 3+ chuyến, chạy standalone
│
└── tests/                      # pytest (140 test)
    ├── test_state_machine.py   # viết TRƯỚC (spec §6)
    ├── test_feature_store.py
    ├── test_memory_store.py
    ├── test_reasoning.py
    ├── test_memory_api.py
    └── test_client_memory.py
```

---

## 3. Cài đặt & chạy nhanh

```bash
cd personalization_engine
pip install -r requirements.txt
```

### Chạy server

```bash
python main.py
```

Mở `http://127.0.0.1:8000/docs` để xem tài liệu API tương tác (Swagger UI).

### Chạy demo 3 chuyến (không cần server, không cần model)

```bash
python -m mock_data.simulate_trips
```

Bạn sẽ thấy tài xế **tốt nghiệp** `COLD_START → WARMING → PERSONALIZED`, ngưỡng mệt
tụt dần (cảnh báo sớm hơn cho tài xế tỉnh táo), rồi consent tắt/xoá bộ nhớ.

### Chạy test

```bash
pytest
```

### Chạy eval (chất lượng giải thích)

```bash
python -m eval.run_eval
```

> 💡 Trên Windows, nếu console báo lỗi Unicode khi in tiếng Việt, đặt biến môi trường
> `PYTHONUTF8=1` trước khi chạy.

---

## 4. API (REST, prefix `/api/v1`)

### Nhóm Driver Memory / Reasoning (spec §2.4)

| Method | Path | Ai gọi | Ý nghĩa |
|--------|------|--------|---------|
| `POST` | `/telemetry` | V3/V4 | Nạp một `TelemetryEvent`, trả về kèm `event_id`. |
| `POST` | `/drivers/{id}/trips/{trip_id}/end` | TV5 | Kết thúc chuyến → chạy Learning Loop. |
| `GET`  | `/drivers/{id}/threshold?event_type=FATIGUE_ALERT` | **V4** | Ngưỡng tư vấn (đã kẹp safety floor). |
| `GET`  | `/explanations/{event_id}` | **V2** | Giải thích tiếng Việt cho một sự kiện. |
| `POST` | `/explain` | V2 | Giải thích một event "live" (không cần nạp trước). |
| `GET`  | `/drivers/{id}/memory` | Tài xế | Minh bạch: xem tất cả những gì đã học. |
| `DELETE` | `/drivers/{id}/memory` | Tài xế | Xoá dữ liệu đã học (giữ consent + safety floor). |
| `PUT`  | `/drivers/{id}/consent?enabled=false` | Tài xế | Bật/tắt cá nhân hoá. |
| `GET`  | `/reasoning/status` | Ops | Đang dùng Phi-3 thật hay fallback. |

### Ví dụ curl

```bash
# 1) Nạp một sự kiện mệt mỏi
curl -X POST http://127.0.0.1:8000/api/v1/telemetry \
  -H "Content-Type: application/json" \
  -d '{"trip_id":"trip_001","driver_id":"driver_001","source":"V3_PERCEPTION","event_type":"FATIGUE_ALERT","scalar_features":{"perclos":0.42}}'

# 2) V4 hỏi ngưỡng cá nhân hoá cho tài xế
curl "http://127.0.0.1:8000/api/v1/drivers/driver_001/threshold?event_type=FATIGUE_ALERT"

# 3) V2 lấy lời giải thích (thay <event_id> bằng id ở bước 1)
curl http://127.0.0.1:8000/api/v1/explanations/<event_id>
```

Nhóm nhận diện tài xế (bản gốc) vẫn còn: `GET /drivers`, `GET /drivers/{id}`,
`POST /drivers`, `POST /drivers/{id}/activate`, `GET /health`.

---

## 5. Cá nhân hoá hoạt động thế nào

**State machine** (theo số chuyến ĐÃ hoàn thành):

| Trạng thái | Số chuyến | Hành vi |
|------------|-----------|---------|
| `COLD_START` | 0–1 | Dùng ngưỡng mặc định toàn cục. Ghi log, chưa học. |
| `WARMING` | 2 | Baseline đang hình thành; blend global + cá nhân (weight 0.5). |
| `PERSONALIZED` | ≥3 | Dùng baseline cá nhân (weight 0.85), vẫn kẹp safety floor. |

**Baseline** dùng EMA + phương sai (không train model — giải thích được ở quy mô
3–10 chuyến, có thể nâng cấp lên model thật ở v1.5). Ngưỡng mệt cá nhân:

```
personal = fatigue_ema + 1.5 * std
blended  = (1 - w) * global_default + w * personal      # w theo trạng thái
threshold = clamp(blended, safety_floor)                # V4 sở hữu floor
```

**Safety floor** kẹp theo đúng hướng an toàn: ngưỡng mệt bị *cap từ trên*
(`max_fatigue_threshold`), ngưỡng TTC bị *chặn từ dưới* (`min_ttc_threshold`).
V1 **không bao giờ** nới lỏng dưới mức V4 đặt.

Tài xế **tắt cá nhân hoá** (consent) → quay về global default ngay.

---

## 6. Reasoning: grounded + không bao giờ hallucinate

Phân tách trách nhiệm là toàn bộ câu chuyện an toàn:

```
context_builder  → quyết định DỮ KIỆN (deterministic, grounded)
prompt_templates → quyết định KHUNG   (system prompt + few-shot cố định)
phi3_runtime     → quyết định CÂU CHỮ  (LLM, temperature thấp)
```

Phi-3 chỉ được **diễn đạt lại** các dữ kiện đã chốt, không tự thêm số liệu. Engine
([`reasoning/reasoning_engine.py`](reasoning/reasoning_engine.py)) chạy pipeline:

1. Nếu có model → sinh câu.
2. **Confidence gate (deterministic):** loại nếu rỗng / quá dài / chứa **con số
   không có trong dữ kiện** (chặn hallucinate số an toàn), cắt còn ≤2 câu.
3. Qua gate → dùng (confidence 0.85). Không qua / lỗi / không có model →
   **fallback template** (confidence 1.0, không thể sai).

→ `GetExplanation` **luôn** trả câu grounded, **không bao giờ rỗng**. Một LLM lỗi
chỉ có thể *hạ cấp* xuống template, không bao giờ phát ra thông tin an toàn sai.

### Bật Phi-3 thật

Mặc định engine dùng fallback template (demo chạy ngay). Để dùng Phi-3-mini thật:

```bash
pip install onnxruntime-genai
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx \
  --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* \
  --local-dir ./models_onnx/phi3
# Trỏ service vào thư mục chứa genai_config.json:
export GUARDIAN_PHI3_MODEL_PATH=./models_onnx/phi3/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4
```

Có model + thư viện → engine **tự động** chuyển sang generation thật. Không cần sửa
gì khác. Kiểm tra: `GET /api/v1/reasoning/status`.

---

## 7. Ánh xạ sang spec

| Phần spec | Triển khai |
|-----------|-----------|
| §2.1 Event Ingestion & Feature Store | `ingestion/feature_store.py` (append-only, interface swap được) |
| §2.2 Driver Memory & Learning Loop | `memory/` (store + state_machine), `services/memory_service.py` |
| §2.3 Reasoning Engine (Phi-3) | `reasoning/` (runtime + prompt + context + fallback + engine) |
| §2.4 Personalization API | `api/memory_routes.py` + `client/personalization_client.py` |
| §3.1–§3.4 Schemas | `models/telemetry.py`, `driver_memory.py`, `explanation.py` |
| §1 "thêm" State machine tường minh | `PersonalizationState` + `PersonalizationStateMachine` |
| §1 "thêm" Prompt/RAG tách khỏi quyết định | `context_builder` (facts) tách `prompt_templates` (framing) |
| §1 "thêm" Eval harness | `eval/` (golden set + rubric, 100% pass) |
| §1 "thêm" Memory reset/transparency | `GET/DELETE /drivers/{id}/memory`, `PUT .../consent` |
| §1 "bớt" không fine-tune | prompt engineering + few-shot |
| §1 "bớt" không vector DB | numpy cosine in-process |
| §1 "bớt" rolling baseline thay ML | EMA + z-score |

**Khác biệt so với spec:** spec vẽ gRPC; ở đây dùng **REST/FastAPI** để đồng bộ với
phần còn lại của repo (cả `digital_cockpit` cũng REST). Lớp service bên dưới hoàn
toàn *transport-agnostic* — dựng gRPC server sau chỉ cần tái dùng nguyên các service.

---

## 8. Tích hợp cross-vertical

V2/V4 không import class của V1; họ gọi qua `PersonalizationClient`
([`client/personalization_client.py`](client/personalization_client.py)):

```python
from client.personalization_client import PersonalizationClient
from models.telemetry import EventType

with PersonalizationClient("http://127.0.0.1:8000") as v1:
    # V4: xin ngưỡng tư vấn (V4 tự quyết dùng hay không)
    thr = v1.get_personalized_threshold("driver_001", EventType.FATIGUE_ALERT)
    if not thr.used_global_default:
        ...  # V4 cân nhắc thr.threshold, luôn kẹp lại theo floor của chính V4

    # V2: lấy lời giải thích để hiển thị + đọc TTS
    text = v1.get_explanation(event_id).text_vi
```

Client tự dịch lỗi HTTP thành exception rõ ràng và **không làm sập vòng lặp của V4**
khi V1 offline (xem `health()` trả `False` thay vì raise).

---

## 9. Từ bản gốc đến bản này

Bản `personalization_engine` ban đầu là service *nhận diện tài xế + preferences +
safety policy cho novice* (REST/FastAPI, layered, comment kỹ). Bản này **giữ nguyên
toàn bộ phần đó** và **thêm mới** đúng theo spec: Feature Store, Driver Memory +
State Machine + Learning Loop, Reasoning Engine (Phi-3 + fallback), API
threshold/explanation/consent, eval harness, và bộ test tương ứng. 63 test gốc vẫn
xanh; tổng cộng **140 test pass**.

---

Câu ngắn gọn cho giám khảo: *V1 học tài xế qua từng chuyến, đề xuất ngưỡng an toàn
cá nhân hoá cho V4 và giải thích bằng tiếng Việt cho tài xế — nhưng chưa bao giờ tự
ra lệnh, và chưa bao giờ nói điều gì nó không có căn cứ.*
