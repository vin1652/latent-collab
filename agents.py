"""
latent-collab / agents.py
--------------------------
Four agent classes that implement the two sides of the experiment:

    LatentThinker   — reasons entirely in hidden-state space, no text output
    LatentVerifier  — reads Thinker's KV-cache, generates a text answer
    TextThinker     — standard chain-of-thought generation (baseline)
    TextVerifier    — reads Thinker's text, generates a text answer (baseline)

The LatentThinker + LatentVerifier pair is the LatentMAS pipeline.
The TextThinker   + TextVerifier   pair is the TextMAS   baseline.
"""

import time
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ── Shared output dataclass ───────────────────────────────────────────────────

@dataclass
class AgentOutput:
    answer          : str                     # decoded text (empty for LatentThinker)
    tokens_generated: int                     # output tokens written to the context
    time_elapsed    : float                   # wall-clock seconds
    kv_cache        : Optional[object] = None # past_key_values handed off to Verifier
    kv_length       : int = 0                 # number of tokens represented in kv_cache
    latent_states   : List[torch.Tensor] = field(default_factory=list)  # for peek.py


# ── LatentThinker ─────────────────────────────────────────────────────────────

class LatentThinker:
    """
    Performs N latent thought steps:
      1. Encodes the role-prompt + question in the usual way.
      2. Takes the last-layer hidden state h_t.
      3. Aligns it back to input space:  e = h_t @ Wₐ
      4. Feeds e as the next input embedding (skipping decode + re-embed).
      5. Repeats N times, accumulating the KV-cache.

    Returns AgentOutput with kv_cache and kv_length set.
    No tokens are generated — zero output tokens.
    """

    ROLE = (
        "You are a precise mathematical analyst. "
        "Carefully think through the following problem:\n\n"
        "Problem: {question}\n\nAnalysis:"
    )

    def __init__(self, model, tokenizer, alignment_matrix: torch.Tensor,
                 n_latent_steps: int = 30):
        self.model    = model
        self.tok      = tokenizer
        self.Wa       = alignment_matrix          # (d_h, d_h)
        self.n_steps  = n_latent_steps
        self.device   = next(model.parameters()).device

    def think(self, question: str, save_states: bool = False) -> AgentOutput:
        prompt     = self.ROLE.format(question=question)
        input_ids  = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

        latent_states: List[torch.Tensor] = []
        t0 = time.perf_counter()

        with torch.no_grad():
            # ── Initial forward pass on the text prompt ──────────────────────
            out = self.model(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True,
            )
            # h : (1, 1, d_h)  — last-layer hidden state at final token position
            h        = out.hidden_states[-1][:, -1:, :]
            kv_cache = out.past_key_values

            if save_states:
                latent_states.append(h.squeeze().cpu())

            # ── Latent thought loop ───────────────────────────────────────────
            for _ in range(self.n_steps):
                # Project h from output space → input embedding space
                e = h @ self.Wa                  # (1, 1, d_h)

                out = self.model(
                    inputs_embeds=e,
                    past_key_values=kv_cache,
                    use_cache=True,
                    output_hidden_states=True,
                )
                h        = out.hidden_states[-1][:, -1:, :]
                kv_cache = out.past_key_values

                if save_states:
                    latent_states.append(h.squeeze().cpu())

        elapsed   = time.perf_counter() - t0
        kv_length = kv_cache[0][0].shape[-2]    # sequence length stored in cache

        return AgentOutput(
            answer="[latent — no text]",
            tokens_generated=0,
            time_elapsed=elapsed,
            kv_cache=kv_cache,
            kv_length=kv_length,
            latent_states=latent_states,
        )


# ── LatentVerifier ────────────────────────────────────────────────────────────

class LatentVerifier:
    """
    Receives the Thinker's KV-cache and generates a final text answer.

    Mechanism:
      - Prepend Thinker's KV-cache as context (layer-wise K/V injection).
      - Encode Verifier's own role prompt + question at positions AFTER the cache.
      - Greedy-decode the answer token-by-token (manual loop for control).

    The KV-cache encapsulates both the original question context AND the
    latent thoughts, giving the Verifier lossless access to Thinker's reasoning.
    """

    ROLE = (
        "Based on the detailed analysis above, provide only the final "
        "numerical answer to:\n\nProblem: {question}\n\nFinal Answer:"
    )

    def __init__(self, model, tokenizer, max_new_tokens: int = 50):
        self.model          = model
        self.tok            = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device         = next(model.parameters()).device

    def verify(self, question: str, thinker_output: AgentOutput) -> AgentOutput:
        prompt    = self.ROLE.format(question=question)
        input_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

        t0 = time.perf_counter()
        generated: List[int] = []

        with torch.no_grad():
            # ── Feed verifier prompt conditioned on thinker's KV-cache ───────
            # HuggingFace automatically offsets position_ids by len(past_kv)
            out      = self.model(
                input_ids=input_ids,
                past_key_values=thinker_output.kv_cache,
                use_cache=True,
            )
            kv_cache = out.past_key_values
            logits   = out.logits[:, -1, :]        # (1, vocab_size)

            # ── Greedy decode ─────────────────────────────────────────────────
            for _ in range(self.max_new_tokens):
                next_id = logits.argmax(dim=-1, keepdim=True)   # (1, 1)
                token   = next_id.item()
                generated.append(token)

                if token == self.tok.eos_token_id:
                    break

                out      = self.model(
                    input_ids=next_id,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
                kv_cache = out.past_key_values
                logits   = out.logits[:, -1, :]

        elapsed = time.perf_counter() - t0
        text    = self.tok.decode(generated, skip_special_tokens=True).strip()

        return AgentOutput(
            answer=text,
            tokens_generated=len(generated),
            time_elapsed=elapsed,
        )


# ── TextThinker ───────────────────────────────────────────────────────────────

class TextThinker:
    """
    Baseline: generates an explicit chain-of-thought in text.
    The CoT text is what the TextVerifier will read.
    """

    ROLE = (
        "You are a precise mathematical analyst. "
        "Carefully think through the following problem step by step:\n\n"
        "Problem: {question}\n\nStep-by-step reasoning:"
    )

    def __init__(self, model, tokenizer, max_new_tokens: int = 80):
        self.model          = model
        self.tok            = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device         = next(model.parameters()).device

    def think(self, question: str, **_) -> AgentOutput:
        prompt    = self.ROLE.format(question=question)
        input_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        elapsed   = time.perf_counter() - t0

        new_ids   = output_ids[0][input_ids.shape[1]:]
        cot_text  = self.tok.decode(new_ids, skip_special_tokens=True).strip()

        return AgentOutput(
            answer=cot_text,
            tokens_generated=len(new_ids),
            time_elapsed=elapsed,
        )


# ── TextVerifier ──────────────────────────────────────────────────────────────

class TextVerifier:
    """
    Baseline: reads the Thinker's chain-of-thought text and generates an answer.
    Token count = CoT tokens + answer tokens (the full TextMAS cost).
    """

    ROLE = (
        "You are a precise mathematical analyst. "
        "Carefully think through the following problem step by step:\n\n"
        "Problem: {question}\n\n"
        "Step-by-step reasoning: {cot}\n\n"
        "Based on the reasoning above, provide only the final numerical answer:\n\n"
        "Final Answer:"
    )

    def __init__(self, model, tokenizer, max_new_tokens: int = 50):
        self.model          = model
        self.tok            = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device         = next(model.parameters()).device

    def verify(self, question: str, thinker_output: AgentOutput) -> AgentOutput:
        prompt    = self.ROLE.format(question=question, cot=thinker_output.answer)
        input_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        new_ids     = output_ids[0][input_ids.shape[1]:]
        answer_text = self.tok.decode(new_ids, skip_special_tokens=True).strip()

        return AgentOutput(
            answer=answer_text,
            tokens_generated=len(new_ids),
            time_elapsed=elapsed,
        )
