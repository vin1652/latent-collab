"""
latent-collab / alignment.py
-----------------------------
Computes the alignment matrix Wₐ that maps a model's last-layer hidden states
back into its input embedding space — training-free, derived from the model's
own weight matrices.

Background (from the paper Section 3.1)
----------------------------------------
A transformer encodes input tokens via W_in (vocab → d_h) and projects its
final hidden state to vocabulary logits via W_out (d_h → vocab).

When we feed hidden states h directly as next-step inputs, the distribution
can be out of range for the shallow embedding layers.  The fix is a linear
projection Wₐ that maps h → e (a valid input embedding):

    e = h @ Wₐ,   where  Wₐ ≈ W_out⁻¹ W_in

Because W_out is non-square (vocab_size × d_h) its true inverse doesn't exist.
We instead solve a ridge regression:

    min_{Wₐ}  ‖W_out Wₐ − W_in‖²_F  +  λ ‖Wₐ‖²_F

Closed-form solution:

    Wₐ = (W_out ᵀ W_out + λI)⁻¹  W_out ᵀ  W_in

Shape: (d_h × d_h)  — computed once and reused for every latent step.
"""

import torch


def compute_alignment_matrix(model, lambda_reg: float = 0.01) -> torch.Tensor:
    """
    Derive Wₐ from the model's embedding weights (no training required).

    Args:
        model      : a HuggingFace CausalLM (GPT-2, Qwen, Llama, …)
        lambda_reg : ridge-regression regularisation strength (default 0.01)

    Returns:
        Wₐ  : FloatTensor of shape (d_h, d_h) on the same device as the model
    """
    with torch.no_grad():
        # W_in  : (vocab_size, d_h)  — token → embedding
        W_in  = model.get_input_embeddings().weight.detach().float()
        # W_out : (vocab_size, d_h)  — hidden state → logits
        W_out = model.lm_head.weight.detach().float()

        d_h = W_out.shape[1]
        device = W_out.device

        # A = W_out ᵀ @ W_out + λI   shape: (d_h, d_h)
        A = W_out.T @ W_out + lambda_reg * torch.eye(d_h, device=device, dtype=torch.float32)

        # B = W_out ᵀ @ W_in          shape: (d_h, d_h)
        B = W_out.T @ W_in

        # Solve A @ Wₐ = B  (more numerically stable than explicit inverse)
        Wa = torch.linalg.solve(A, B)

    return Wa.to(model.dtype)


def alignment_quality(model, Wa: torch.Tensor) -> float:
    """
    Sanity-check: measure how well Wₐ reconstructs W_in from W_out.
    Returns the relative Frobenius-norm error ‖W_out Wₐ − W_in‖_F / ‖W_in‖_F.
    A value close to 0 means excellent alignment.
    """
    with torch.no_grad():
        W_in  = model.get_input_embeddings().weight.detach().float()
        W_out = model.lm_head.weight.detach().float()
        Wa_f  = Wa.float()
        error = torch.norm(W_out @ Wa_f - W_in, p="fro")
        base  = torch.norm(W_in, p="fro")
        return (error / base).item()
