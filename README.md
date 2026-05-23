# latent-collab

**What if AI agents skipped words entirely and passed their raw thoughts to each other?**

This is a hands-on toy implementation of the core idea behind
[**"Latent Collaboration in Multi-Agent Systems" (LatentMAS)**](https://arxiv.org/abs/2511.20639)
(Zou et al., 2024, Princeton / UIUC / Stanford).

The paper shows that LLM agents can collaborate entirely in **continuous latent space** —
passing KV-cache hidden states instead of text — and get:

- ↑ up to 14.6% higher accuracy
- ↓ 70–83% fewer output tokens
- ↑ 4× faster end-to-end inference

This repo lets you reproduce the key mechanism on any HuggingFace causal LM.

---

## How it works

```
Standard MAS (TextMAS):
  Thinker ──[generates 80 tokens of CoT text]──▶  Verifier ──▶ answer

Latent MAS (LatentMAS):
  Thinker ──[30 latent steps, hidden-state loop]──▶ KV-cache ──▶ Verifier ──▶ answer
              ↑ zero tokens written                    ↑ lossless
```

### The 3 key mechanisms

**1. Latent thought generation** (inside each agent)

Instead of `hidden_state → decode token → embed token → next step`,
LatentMAS does:

```
hidden_state → align → feed directly as next input → next step
```

The alignment matrix **Wₐ** maps the output hidden space back to input embedding space,
derived from the model's own weights in closed form — no training needed:

```
Wₐ = (W_out ᵀ W_out + λI)⁻¹  W_out ᵀ  W_in
```

**2. Lossless KV-cache transfer** (across agents)

After the Thinker finishes its latent loop, its entire KV-cache is handed to the Verifier.
The Verifier's attention layers can now "see" every internal representation the Thinker
computed — not just what it said in words.  Mathematically proven to be information-lossless.

**3. Zero intermediate tokens**

The Thinker never calls the language model head during its reasoning.
Output token count for the Thinker = **0**.

---

## Quickstart

```bash
git clone https://github.com/your-username/latent-collab
cd latent-collab
pip install -r requirements.txt

# Run the full evaluation (TextMAS vs LatentMAS side-by-side)
python evaluate.py

# Peek at what the model is "thinking" at each latent step
python peek.py
python peek.py --problem 2 --steps 50 --topk 5 --every 3
```

---

## Project structure

```
latent-collab/
├── config.py       # model name, latent steps, device — start here
├── alignment.py    # compute Wₐ from model weights (the math heart of the paper)
├── agents.py       # LatentThinker, LatentVerifier, TextThinker, TextVerifier
├── pipelines.py    # orchestrate both MAS modes, collect metrics
├── problems.py     # 20 arithmetic word problems with known answers
├── evaluate.py     # side-by-side comparison table + summary
└── peek.py         # decode hidden states → nearest vocabulary tokens
```

---

## Recommended model progression

| Model | Size | Notes |
|---|---|---|
| `gpt2` | 117 M | Instant, no login, weak reasoner — good for mechanics |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5 B | Much better, still CPU-friendly |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 B | Noticeably stronger, GPU recommended |

Change `MODEL_NAME` in `config.py` to switch.

---

## What you'll observe

Running `python peek.py` shows output like:

```
Step    Top-5 nearest tokens
──────────────────────────────────────────────────────────────────
init    'calculate'(42.1)  | 'total'(38.9) | 'sum'(37.4) | ...
   5    '×'(51.2)          | 'multiply'(49.7) | '3'(44.1) | ...
  10    '720'(61.3)        | 'result'(58.8) | 'answer'(55.2) | ...
  25    'therefore'(63.1)  | 'so'(61.8)  | 'hence'(59.4) | ...
  30    '720'(72.4)        | 'cookies'(68.2) | 'total'(65.7) | ...
```

The model is moving through concept-space — from "this is a multiplication problem"
toward the answer — without ever committing to a word.

---

## The theory (simplified)

A single latent hidden state h ∈ ℝ^{d_h} encodes up to d_h continuous dimensions
of information.  A single discrete token encodes log₂|V| bits.

For Qwen3-8B (d_h = 4096, |V| ≈ 150k):
one latent step ≈ **4096 / log₂(150000) ≈ 240×** more information than one token.

This is Theorem 3.1 from the paper, proven under the Linear Representation Hypothesis.

---

## What this repo does NOT cover

- Multi-GPU / distributed setups
- The hierarchical MAS setting (multiple domain-specialist agents + summariser)
- Training-based improvements on top of the latent loop
- Cross-model KV-cache transfer (same model is used for Thinker and Verifier here)

For the full system, see the authors' code: https://github.com/Gen-Verse/LatentMAS

---

## Reference

```bibtex
@article{zou2024latent,
  title   = {Latent Collaboration in Multi-Agent Systems},
  author  = {Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and ...},
  journal = {arXiv:2511.20639},
  year    = {2024}
}
```
