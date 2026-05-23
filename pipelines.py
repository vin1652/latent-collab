"""
latent-collab / pipelines.py
-----------------------------
Orchestrates the two two-agent pipelines and holds the result dataclass.

  run_latent_mas  →  LatentThinker → LatentVerifier
  run_text_mas    →  TextThinker   → TextVerifier
"""

import re
import time
from typing import Optional
from dataclasses import dataclass

from agents import (
    AgentOutput,
    LatentThinker, LatentVerifier,
    TextThinker,   TextVerifier,
)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    question        : str
    expected        : float
    predicted       : str          # raw text from Verifier
    is_correct      : bool
    thinker_tokens  : int          # 0 for LatentMAS (nothing written to context)
    verifier_tokens : int
    total_tokens    : int
    total_time      : float
    thinker_time    : float
    verifier_time   : float
    mode            : str          # "latent" | "text"


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_number(text: str) -> Optional[float]:
    """
    Pull the final numeric answer out of an answer string.

    Strategy (in order):
      1. LaTeX \\boxed{N} — Qwen Math models emit this format.
      2. Keyword-anchored: number right after "Final Answer:", "Answer:", "result:".
      3. First number in the text — the verifier prompt ends with "Final Answer:"
         so the model outputs the answer before any step-by-step elaboration.
    """
    def to_float(s: str) -> Optional[float]:
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return None

    # 1. \boxed{720} or \boxed{28.00}
    boxed = re.search(r"\\boxed\{(-?\d[\d,]*\.?\d*)\}", text)
    if boxed:
        v = to_float(boxed.group(1))
        if v is not None:
            return v

    # 2. Keyword-anchored
    anchored = re.search(
        r"(?:final\s+answer|answer|result)\s*[:\$]?\s*\$?\s*(-?\d[\d,]*\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if anchored:
        v = to_float(anchored.group(1))
        if v is not None:
            return v

    # 3. First number — comes immediately after "Final Answer:" in the prompt
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    for candidate in nums:
        v = to_float(candidate)
        if v is not None:
            return v

    return None


def is_correct(predicted_text: str, expected: float, tolerance: float = 0.05) -> bool:
    """
    Check whether the extracted number from `predicted_text` matches `expected`.
    Uses a relative tolerance of 5 % (or absolute 0.01 for near-zero values).
    """
    num = extract_number(predicted_text)
    if num is None:
        return False
    if abs(expected) < 1e-6:
        return abs(num - expected) < 0.01
    return abs(num - expected) / abs(expected) < tolerance


# ── LatentMAS pipeline ────────────────────────────────────────────────────────

def run_latent_mas(
    question : str,
    expected : float,
    thinker  : LatentThinker,
    verifier : LatentVerifier,
    save_states: bool = False,
) -> PipelineResult:
    t0 = time.perf_counter()

    thinker_out  = thinker.think(question, save_states=save_states)
    verifier_out = verifier.verify(question, thinker_out)

    total_time = time.perf_counter() - t0

    return PipelineResult(
        question       = question,
        expected       = expected,
        predicted      = verifier_out.answer,
        is_correct     = is_correct(verifier_out.answer, expected),
        thinker_tokens = 0,                           # latent — no text output
        verifier_tokens= verifier_out.tokens_generated,
        total_tokens   = verifier_out.tokens_generated,
        total_time     = total_time,
        thinker_time   = thinker_out.time_elapsed,
        verifier_time  = verifier_out.time_elapsed,
        mode           = "latent",
    )


# ── TextMAS pipeline (baseline) ───────────────────────────────────────────────

def run_text_mas(
    question : str,
    expected : float,
    thinker  : TextThinker,
    verifier : TextVerifier,
) -> PipelineResult:
    t0 = time.perf_counter()

    thinker_out  = thinker.think(question)
    verifier_out = verifier.verify(question, thinker_out)

    total_time = time.perf_counter() - t0

    return PipelineResult(
        question       = question,
        expected       = expected,
        predicted      = verifier_out.answer,
        is_correct     = is_correct(verifier_out.answer, expected),
        thinker_tokens = thinker_out.tokens_generated,
        verifier_tokens= verifier_out.tokens_generated,
        total_tokens   = thinker_out.tokens_generated + verifier_out.tokens_generated,
        total_time     = total_time,
        thinker_time   = thinker_out.time_elapsed,
        verifier_time  = verifier_out.time_elapsed,
        mode           = "text",
    )
