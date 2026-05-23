"""
latent-collab / evaluate.py
----------------------------
Main evaluation script.  Run with:

    python evaluate.py

Loads the model once, runs both pipelines on N problems side-by-side, and
prints a detailed comparison table plus a summary.

Metrics reported:
  • Accuracy      — fraction of problems with the correct numeric answer
  • Output tokens — tokens actually written to the context (0 for LatentThinker)
  • Wall-clock time
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from alignment   import compute_alignment_matrix, alignment_quality
from agents      import LatentThinker, LatentVerifier, TextThinker, TextVerifier
from pipelines   import run_latent_mas, run_text_mas, PipelineResult
from problems    import PROBLEMS


# ── Helpers ───────────────────────────────────────────────────────────────────

def truncate(text: str, width: int) -> str:
    return text[:width - 1] + "…" if len(text) > width else text

def pct(a: float, b: float) -> str:
    if b == 0:
        return "  n/a"
    return f"{(a - b) / b * 100:+.1f}%"

def tick(ok: bool) -> str:
    return "✓" if ok else "✗"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  latent-collab  |  model: {config.MODEL_NAME}")
    print(f"  device: {config.DEVICE}  |  latent steps: {config.N_LATENT_STEPS}")
    print(f"{'─'*60}\n")

    print("Loading tokenizer and model…")
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float32,
    ).to(config.DEVICE)
    model.eval()

    # ── Alignment matrix ────────────────────────────────────────────────────
    print("Computing alignment matrix Wₐ…")
    Wa    = compute_alignment_matrix(model, lambda_reg=config.LAMBDA_REG)
    qual  = alignment_quality(model, Wa)
    print(f"  Wₐ shape : {tuple(Wa.shape)}")
    print(f"  Alignment relative error: {qual:.4f}  (lower = better)\n")

    # ── Build agents ────────────────────────────────────────────────────────
    l_thinker  = LatentThinker (model, tok, Wa, n_latent_steps=config.N_LATENT_STEPS)
    l_verifier = LatentVerifier(model, tok,     max_new_tokens =config.MAX_ANSWER_TOKENS)
    t_thinker  = TextThinker   (model, tok,     max_new_tokens =config.N_TEXT_TOKENS)
    t_verifier = TextVerifier  (model, tok,     max_new_tokens =config.MAX_ANSWER_TOKENS)

    problems = PROBLEMS[:config.N_PROBLEMS]
    print(f"Running {len(problems)} problems…\n")

    # ── Per-problem table ────────────────────────────────────────────────────
    W = 110
    COL = {"#": 4, "Q": 42, "Exp": 8, "L-ans": 14, "T-ans": 14,
           "L✓": 4, "T✓": 4, "L-tok": 6, "T-tok": 6, "L-t": 7, "T-t": 7}

    header = (
        f"{'#':<{COL['#']}} {'Question':<{COL['Q']}} {'Expected':>{COL['Exp']}}"
        f"  {'Latent answer':<{COL['L-ans']}} {'Text answer':<{COL['T-ans']}}"
        f"  {'L✓':<{COL['L✓']}} {'T✓':<{COL['T✓']}}"
        f"  {'L-tok':>{COL['L-tok']}} {'T-tok':>{COL['T-tok']}}"
        f"  {'L-s':>{COL['L-t']}} {'T-s':>{COL['T-t']}}"
    )
    print("=" * W)
    print(header)
    print("=" * W)

    latent_results = []
    text_results   = []

    for i, prob in enumerate(problems):
        q = prob["question"]
        a = prob["answer"]

        lr = run_latent_mas(q, a, l_thinker, l_verifier)
        tr = run_text_mas  (q, a, t_thinker, t_verifier)

        latent_results.append(lr)
        text_results.append(tr)

        row = (
            f"{i+1:<{COL['#']}} {truncate(q, COL['Q']):<{COL['Q']}} {a:>{COL['Exp']}.2f}"
            f"  {truncate(lr.predicted, COL['L-ans']):<{COL['L-ans']}}"
            f" {truncate(tr.predicted, COL['T-ans']):<{COL['T-ans']}}"
            f"  {tick(lr.is_correct):<{COL['L✓']}} {tick(tr.is_correct):<{COL['T✓']}}"
            f"  {lr.total_tokens:>{COL['L-tok']}} {tr.total_tokens:>{COL['T-tok']}}"
            f"  {lr.total_time:>{COL['L-t']}.2f} {tr.total_time:>{COL['T-t']}.2f}"
        )
        print(row)
        sys.stdout.flush()

    print("=" * W)

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(problems)
    l_acc   = sum(r.is_correct       for r in latent_results) / n * 100
    t_acc   = sum(r.is_correct       for r in text_results  ) / n * 100
    l_toks  = sum(r.total_tokens     for r in latent_results)
    t_toks  = sum(r.total_tokens     for r in text_results  )
    l_time  = sum(r.total_time       for r in latent_results)
    t_time  = sum(r.total_time       for r in text_results  )

    print(f"\n{'SUMMARY':=^60}")
    print(f"{'Metric':<28} {'LatentMAS':>10} {'TextMAS':>10} {'Δ':>10}")
    print("-" * 60)
    print(f"{'Accuracy':<28} {l_acc:>9.1f}% {t_acc:>9.1f}% {pct(l_acc, t_acc):>10}")
    print(f"{'Output tokens (total)':<28} {l_toks:>10} {t_toks:>10} {pct(l_toks, t_toks):>10}")
    print(f"{'Wall-clock time (s)':<28} {l_time:>10.1f} {t_time:>10.1f} {pct(l_time, t_time):>10}")
    print("=" * 60)
    print(
        f"\nNote: LatentMAS Thinker writes 0 output tokens — all reasoning is "
        f"in the {config.N_LATENT_STEPS} latent steps (hidden-state loop)."
    )
    print(
        f"TextMAS Thinker writes up to {config.N_TEXT_TOKENS} tokens of "
        f"chain-of-thought before the Verifier reads them.\n"
    )

    print("Tip: run  python peek.py  to visualise the Thinker's latent thoughts.\n")


if __name__ == "__main__":
    main()
