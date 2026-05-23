"""
latent-collab / peek.py
------------------------
"What is the model thinking?"

Runs the LatentThinker on one problem and, for each latent step, decodes
the hidden state into its nearest-neighbour vocabulary tokens.  This gives
you a window into the model's internal reasoning without it ever producing
a real word.

Run with:
    python peek.py
    python peek.py --problem 3   (use the 4th problem, 0-indexed)
    python peek.py --steps 50    (use 50 latent steps instead of default)

How the decoding works
-----------------------
At each latent step the model produces a hidden state h ∈ ℝ^{d_h}.
We project h into vocabulary space:  logits = h @ W_out.T
The top-k tokens by logit score are the model's "nearest neighbours"
— the concepts closest to what it's thinking at that step.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from alignment import compute_alignment_matrix
from agents    import LatentThinker
from problems  import PROBLEMS


# ── Nearest-neighbour decoding ────────────────────────────────────────────────

def decode_hidden_state(h: torch.Tensor, model, tokenizer, top_k: int = 5):
    """
    h     : 1-D tensor of shape (d_h,)
    Returns a list of (token_string, logit_score) for the top-k nearest tokens.
    """
    W_out = model.lm_head.weight.detach().float()   # (V, d_h)
    h_f   = h.float()                               # (d_h,)

    logits   = h_f @ W_out.T                        # (V,)
    top      = logits.topk(top_k)
    results  = []
    for idx, score in zip(top.indices.tolist(), top.values.tolist()):
        token = tokenizer.decode([idx])
        results.append((token, score))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Peek at latent thoughts.")
    parser.add_argument("--problem", type=int, default=0,
                        help="Problem index (0-based, default 0)")
    parser.add_argument("--steps", type=int, default=config.N_LATENT_STEPS,
                        help=f"Latent steps (default {config.N_LATENT_STEPS})")
    parser.add_argument("--topk", type=int, default=5,
                        help="Nearest neighbours to show per step (default 5)")
    parser.add_argument("--every", type=int, default=5,
                        help="Print every Nth step (default 5)")
    args = parser.parse_args()

    problem = PROBLEMS[args.problem]
    print(f"\n{'─'*70}")
    print(f"  peek.py  |  model: {config.MODEL_NAME}  |  {args.steps} latent steps")
    print(f"{'─'*70}")
    print(f"\nProblem: {problem['question']}")
    print(f"Expected answer: {problem['answer']}\n")

    # ── Load model ──────────────────────────────────────────────────────────
    print("Loading model…")
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, torch_dtype=torch.float32
    ).to(config.DEVICE)
    model.eval()

    Wa = compute_alignment_matrix(model, lambda_reg=config.LAMBDA_REG)

    # ── Run LatentThinker, saving all hidden states ─────────────────────────
    thinker = LatentThinker(model, tok, Wa, n_latent_steps=args.steps)
    output  = thinker.think(problem["question"], save_states=True)

    print(f"\nLatent states captured: {len(output.latent_states)}")
    print(f"KV-cache sequence length: {output.kv_length} tokens\n")

    # ── Decode each saved hidden state ───────────────────────────────────────
    print(f"{'Step':<6}  {'Top-{} nearest tokens'.format(args.topk)}")
    print("─" * 70)

    for i, h in enumerate(output.latent_states):
        if i % args.every != 0 and i != len(output.latent_states) - 1:
            continue

        label    = "init" if i == 0 else f"  {i:3d}"
        nn_pairs = decode_hidden_state(h, model, tok, top_k=args.topk)
        tokens   = "  |  ".join(
            f"'{tok_str.strip()}'({score:.1f})"
            for tok_str, score in nn_pairs
        )
        print(f"{label:<6}  {tokens}")

    print("─" * 70)
    print(
        "\nInterpretation guide:\n"
        "  • Each row shows what vocabulary concepts are geometrically\n"
        "    closest to the model's internal state at that latent step.\n"
        "  • These are NOT tokens the model outputted — they are never\n"
        "    decoded.  The model is reasoning in a richer continuous space.\n"
        "  • Watch for numeric tokens, operator tokens, or domain words\n"
        "    appearing as the model processes the problem.\n"
    )

    # ── Now let the Verifier read the latent cache and answer ────────────────
    from agents import LatentVerifier
    verifier = LatentVerifier(model, tok, max_new_tokens=config.MAX_ANSWER_TOKENS)
    v_out    = verifier.verify(problem["question"], output)

    print(f"Verifier answer : {v_out.answer}")
    print(f"Expected        : {problem['answer']}")
    print()


if __name__ == "__main__":
    main()
