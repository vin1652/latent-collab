"""
latent-collab / config.py
-------------------------
Central configuration for the Thinker-Verifier experiment.

Recommended model progression:
  1. "gpt2"                          -- tiny, no login, CPU-friendly, weak reasoning
  2. "Qwen/Qwen2.5-0.5B-Instruct"   -- much better, still small, needs HF login or public download
  3. "Qwen/Qwen2.5-1.5B-Instruct"   -- even better, ~3 GB, GPU recommended
"""

import torch

# ── Model ──────────────────────────────────────────────────────────────────────
# MODEL_NAME = "gpt2"                              # tiny, weak reasoning — smoke-tests only
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"        # general instruction-tuned, weak at math
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"    # math-specialized fine-tune, strong arithmetic

# ── Latent generation ──────────────────────────────────────────────────────────
N_LATENT_STEPS   = 30    # how many hidden-state feedback steps the Thinker takes
N_TEXT_TOKENS    = 80    # max chain-of-thought tokens for the TextMAS Thinker
MAX_ANSWER_TOKENS = 20   # max tokens the Verifier generates — short so model outputs answer first
LAMBDA_REG       = 0.01  # ridge-regression regularisation for Wₐ alignment matrix

# ── Evaluation ─────────────────────────────────────────────────────────────────
N_PROBLEMS = 10          # how many problems to run (max 20, increase as desired)

# ── Hardware ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
