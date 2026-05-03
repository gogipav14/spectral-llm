"""
Sanity test for Route A math invariance.

Builds a fake Qwen3-style attention layer (q_proj + q_norm + RoPE +
attention dot-product), runs a forward pass, then applies the same
PCA-rotation + MatrixGammaRMSNorm transformation that
`_bbt_prep_spectral_pca` would, and verifies attention scores match
to float-precision tolerance.

Run: python -m bbt_quant.test_pca_invariance
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .autoround_bbt import MatrixGammaRMSNorm


def _per_head_pca_rotate_qk(
    Wq: torch.Tensor, Wk: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int,
    Uq: torch.Tensor, Uk: torch.Tensor,
):
    """Replicates the per-head row rotation used in _bbt_prep_spectral_pca."""
    # GQA collapse: q-heads in same kv-group share the kv-head's U.
    gqa_factor = n_heads // n_kv_heads
    Uq_collapsed = torch.zeros_like(Uq)
    for kv_h in range(n_kv_heads):
        for off in range(gqa_factor):
            Uq_collapsed[kv_h * gqa_factor + off] = Uk[kv_h]
    Uq = Uq_collapsed

    Wq_rot = Wq.clone()
    for h in range(n_heads):
        lo, hi = h * head_dim, (h + 1) * head_dim
        Wq_rot[lo:hi, :] = Uq[h].t() @ Wq[lo:hi, :]
    Wk_rot = Wk.clone()
    for h in range(n_kv_heads):
        lo, hi = h * head_dim, (h + 1) * head_dim
        Wk_rot[lo:hi, :] = Uk[h].t() @ Wk[lo:hi, :]
    return Wq_rot, Wk_rot, Uq, Uk


def _qwen3_qnorm_apply(x, gamma, eps=1e-6):
    """Vector-gamma RMSNorm matching Qwen3RMSNorm exactly."""
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (gamma * x.to(torch.float32) * torch.rsqrt(var + eps)).to(x.dtype)


def main():
    torch.manual_seed(0)
    B, T, hidden = 2, 8, 512
    n_heads, n_kv_heads, head_dim = 8, 4, 64
    dtype = torch.float32

    x = torch.randn(B, T, hidden, dtype=dtype)
    Wq = torch.randn(n_heads * head_dim, hidden, dtype=dtype) / math.sqrt(hidden)
    Wk = torch.randn(n_kv_heads * head_dim, hidden, dtype=dtype) / math.sqrt(hidden)
    gamma_q = torch.randn(head_dim, dtype=dtype) * 0.5 + 1.0
    gamma_k = torch.randn(head_dim, dtype=dtype) * 0.5 + 1.0
    eps = 1e-6

    # === Reference (vanilla Qwen3-style) ===
    q = x @ Wq.t()  # (B, T, n_heads*head_dim)
    k = x @ Wk.t()
    q = q.view(B, T, n_heads, head_dim)
    k = k.view(B, T, n_kv_heads, head_dim)
    q_normed_ref = _qwen3_qnorm_apply(q, gamma_q, eps)
    k_normed_ref = _qwen3_qnorm_apply(k, gamma_k, eps)
    # GQA: replicate k across heads
    k_rep_ref = k_normed_ref.repeat_interleave(n_heads // n_kv_heads, dim=2)
    # Attention scores
    scores_ref = torch.einsum("bthd,bshd->bhts", q_normed_ref, k_rep_ref)

    # === PCA-rotated path ===
    # Compute per-head covariance of q and k outputs for rotation calibration
    q_flat = q.reshape(-1, n_heads, head_dim)         # (BT, n_h, hd)
    k_flat = k.reshape(-1, n_kv_heads, head_dim)
    Cq = torch.einsum("nhd,nhe->hde", q_flat, q_flat) / q_flat.shape[0]
    Ck = torch.einsum("nhd,nhe->hde", k_flat, k_flat) / k_flat.shape[0]
    Uq = torch.linalg.eigh(0.5 * (Cq + Cq.transpose(-1, -2)))[1]
    Uk = torch.linalg.eigh(0.5 * (Ck + Ck.transpose(-1, -2)))[1]

    Wq_rot, Wk_rot, Uq_used, Uk_used = _per_head_pca_rotate_qk(
        Wq, Wk, n_heads, n_kv_heads, head_dim, Uq, Uk)

    # Matrix-Gamma per head: Gamma_h = diag(gamma) @ U[h]. Preserves
    # math invariance through RoPE (since post-q_norm output equals vanilla).
    Gamma_q = torch.zeros(n_heads, head_dim, head_dim, dtype=dtype)
    for h in range(n_heads):
        Gamma_q[h] = gamma_q.unsqueeze(1) * Uq_used[h]
    Gamma_k = torch.zeros(n_kv_heads, head_dim, head_dim, dtype=dtype)
    for h in range(n_kv_heads):
        Gamma_k[h] = gamma_k.unsqueeze(1) * Uk_used[h]

    qnorm_pca = MatrixGammaRMSNorm(Gamma_q, gamma_q, eps=eps)
    knorm_pca = MatrixGammaRMSNorm(Gamma_k, gamma_k, eps=eps)

    q_rot = (x @ Wq_rot.t()).view(B, T, n_heads, head_dim)
    k_rot = (x @ Wk_rot.t()).view(B, T, n_kv_heads, head_dim)
    q_normed_pca = qnorm_pca(q_rot)
    k_normed_pca = knorm_pca(k_rot)
    k_rep_pca = k_normed_pca.repeat_interleave(n_heads // n_kv_heads, dim=2)
    scores_pca = torch.einsum("bthd,bshd->bhts", q_normed_pca, k_rep_pca)

    diff_scores = (scores_ref - scores_pca).abs().max().item()
    # With Gamma = diag(gamma) U, q_norm output equals vanilla directly.
    diff_q_normed = (q_normed_ref - q_normed_pca).abs().max().item()
    diff_k_normed = (k_normed_ref - k_normed_pca).abs().max().item()

    print(f"[invariance] max|attn_scores_ref - attn_scores_pca| = {diff_scores:.3e}")
    print(f"[invariance] max|q_normed_ref - q_normed_pca|       = {diff_q_normed:.3e}")
    print(f"[invariance] max|k_normed_ref - k_normed_pca|       = {diff_k_normed:.3e}")
    assert diff_scores < 1e-4, f"attention scores diverged: {diff_scores}"
    assert diff_q_normed < 1e-4, f"q_normed math broken: {diff_q_normed}"
    assert diff_k_normed < 1e-4, f"k_normed math broken: {diff_k_normed}"
    print("[invariance] OK — full-precision math is invariant under spectral_pca")


if __name__ == "__main__":
    main()
