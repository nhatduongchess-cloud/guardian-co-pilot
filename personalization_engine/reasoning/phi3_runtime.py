"""
phi3_runtime.py
===============

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: REASONING  ·  §2.3 / §4

A thin wrapper around Microsoft Phi-3-mini-4k-instruct (INT4) served through
ONNX Runtime GenAI. It is the ONLY file that knows onnxruntime_genai exists;
everything above it talks to the abstract `LLMBackend` interface, so the LLM
can be swapped or removed without touching the engine.

DESIGN RULES (why this file looks the way it does)
--------------------------------------------------
1. LAZY + SAFE. The heavy model is loaded on first use, inside a try/except.
   If onnxruntime_genai is not installed, or the model files are missing, the
   backend simply reports `is_available() == False` and the reasoning engine
   uses the deterministic fallback templates. The service must run on a laptop
   with NO model downloaded — that is exactly the reality on a laptop (spec §8,
   "latency / model risk" mitigations).

2. NEVER CRASH THE VERTICAL. A failure to load or generate raises a narrow
   `Phi3Error`, which the engine catches and turns into a fallback. Constraint
   #1: the LLM going wrong must never affect anything.

3. SHARED RUNTIME. We use ONNX Runtime GenAI on purpose — V3's perception
   pipeline already ships ONNX, so the team maintains ONE runtime (spec §4).

ENABLING THE REAL MODEL
-----------------------
    pip install onnxruntime-genai        # or onnxruntime-genai-directml on GPU
    # download the INT4 CPU build (~2.3 GB) into ./models_onnx/phi3 :
    huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx \
        --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* \
        --local-dir ./models_onnx/phi3
    # point the service at it (folder that contains genai_config.json):
    set GUARDIAN_PHI3_MODEL_PATH=./models_onnx/phi3/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4

With the env var set and the library installed, the engine automatically
switches from templates to real generation. Nothing else changes.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Optional


# ===========================================================================
# SECTION 1 — ERRORS + THE BACKEND INTERFACE
# ===========================================================================


class Phi3Error(Exception):
    """Raised for any Phi-3 load/generate failure. Always caught upstream."""


class LLMBackend(ABC):
    """
    The seam between the reasoning engine and any language model. Implement
    this to plug in a different model (a bigger LLM, a remote API, a mock)
    without changing a line of engine code.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """True only if this backend can actually generate right now."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 96,
                 temperature: float = 0.3) -> str:
        """Generate a completion for a prompt. Raise Phi3Error on failure."""
        raise NotImplementedError


# ===========================================================================
# SECTION 2 — DEFAULT (DISABLED) BACKEND — used when no LLM is configured
# ===========================================================================


class DisabledLLMBackend(LLMBackend):
    """A no-op backend that is never available. The engine then uses templates."""

    def is_available(self) -> bool:
        return False

    def generate(self, prompt: str, max_new_tokens: int = 96,
                 temperature: float = 0.3) -> str:
        raise Phi3Error("No LLM backend configured; using fallback templates.")


# ===========================================================================
# SECTION 3 — THE REAL PHI-3 BACKEND (ONNX Runtime GenAI)
# ===========================================================================


class Phi3Runtime(LLMBackend):
    """
    Loads Phi-3-mini INT4 via onnxruntime_genai and generates text.

    The constructor does NOT touch the model — construction must be cheap and
    infallible so the service boots instantly. The model is loaded on the
    first `is_available()`/`generate()` call and the result is cached.
    """

    # Sensible default location; overridable by env var (see module docstring).
    _DEFAULT_PATH = os.getenv(
        "GUARDIAN_PHI3_MODEL_PATH",
        str(Path(__file__).resolve().parents[1] / "models_onnx" / "phi3"),
    )

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path or self._DEFAULT_PATH
        self._og = None          # the onnxruntime_genai module (lazy)
        self._model = None       # the loaded model (lazy)
        self._tokenizer = None   # the tokenizer (lazy)
        self._load_attempted = False
        self._load_ok = False
        self._last_error: Optional[str] = None

    # -- lazy loading -------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        """Try once to import the lib and load the model. Cache the outcome."""
        if self._load_attempted:
            return self._load_ok
        self._load_attempted = True
        try:
            import onnxruntime_genai as og  # heavy, optional dependency

            if not Path(self._model_path).exists():
                raise Phi3Error(
                    f"Phi-3 model folder not found: {self._model_path}. "
                    "See phi3_runtime.py docstring to download it."
                )
            self._og = og
            self._model = og.Model(self._model_path)
            self._tokenizer = og.Tokenizer(self._model)
            self._load_ok = True
        except Exception as exc:  # ImportError, model errors, anything
            # Record the reason but DO NOT raise — the engine will use fallbacks.
            self._last_error = str(exc)
            self._load_ok = False
        return self._load_ok

    # -- interface ----------------------------------------------------------
    def is_available(self) -> bool:
        return self._ensure_loaded()

    @property
    def last_error(self) -> Optional[str]:
        """Why the model is unavailable (for /health and logs). None if OK."""
        return self._last_error

    def generate(self, prompt: str, max_new_tokens: int = 96,
                 temperature: float = 0.3) -> str:
        """
        Run Phi-3 on `prompt` and return only the newly generated text.

        Low temperature (0.2-0.3) keeps the wording stable across runs, which
        matters for a safety product — we do not want the same event explained
        wildly differently each time (spec §2.3).
        """
        if not self._ensure_loaded():
            raise Phi3Error(self._last_error or "Phi-3 not available.")

        og = self._og
        try:
            input_tokens = self._tokenizer.encode(prompt)

            params = og.GeneratorParams(self._model)
            # `set_search_options` signature is stable across ORT-GenAI versions.
            params.set_search_options(
                max_length=len(input_tokens) + max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0.0,
            )

            generator = og.Generator(self._model, params)
            # Newer API (>=0.4) feeds tokens explicitly; guard for older builds.
            if hasattr(generator, "append_tokens"):
                generator.append_tokens(input_tokens)
            elif hasattr(params, "input_ids"):
                params.input_ids = input_tokens  # legacy path

            new_token_ids: list[int] = []
            while not generator.is_done():
                generator.generate_next_token()
                seq = generator.get_sequence(0)
                if len(seq) > len(input_tokens):
                    new_token_ids = list(seq[len(input_tokens):])
                # Stop early once we clearly have >=2 sentences worth of tokens.
                if len(new_token_ids) >= max_new_tokens:
                    break

            text = self._tokenizer.decode(new_token_ids) if new_token_ids else ""
            return _clean_completion(text)
        except Phi3Error:
            raise
        except Exception as exc:
            raise Phi3Error(f"Phi-3 generation failed: {exc}") from exc


# ===========================================================================
# SECTION 4 — HELPERS
# ===========================================================================
def _clean_completion(text: str) -> str:
    """Trim the Phi-3 turn markers and stray whitespace from a completion."""
    for marker in ("<|end|>", "<|user|>", "<|assistant|>", "<|system|>"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


# ===========================================================================
# SECTION 5 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    rt = Phi3Runtime()
    print("model path :", rt._model_path)
    print("available  :", rt.is_available())
    if not rt.is_available():
        print("reason     :", rt.last_error)
        print("-> reasoning engine will use deterministic fallback templates.")
    else:
        print(rt.generate("<|user|>\nChào bạn<|end|>\n<|assistant|>\n"))
