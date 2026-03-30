"""Tests for the DenoiserBlock."""

import torch

from src.models.diffusion.denoiser_block import DenoiserBlock


B = 4
SEQ_LEN = 10
DENOISER_DIM = 32
NUM_HEADS = 2
FF_DIM = 128
COND_LEN = 8


def _make_block():
    return DenoiserBlock(DENOISER_DIM, NUM_HEADS, FF_DIM, dropout=0.0)


def test_output_shape_matches_input():
    block = _make_block()
    x = torch.randn(B, SEQ_LEN, DENOISER_DIM)
    t_emb = torch.randn(B, DENOISER_DIM)
    cond = torch.randn(B, COND_LEN, DENOISER_DIM)
    out = block(x, t_emb, cond)
    assert out.shape == (B, SEQ_LEN, DENOISER_DIM)


def test_at_init_output_approx_input():
    """With alpha gates at zero, self-attn and cross-attn residuals vanish.
    Output should be approximately x + FFN(LN(x))."""
    block = _make_block()
    block.eval()
    x = torch.randn(B, SEQ_LEN, DENOISER_DIM)
    t_emb = torch.randn(B, DENOISER_DIM)
    cond = torch.randn(B, COND_LEN, DENOISER_DIM)

    with torch.no_grad():
        out = block(x, t_emb, cond)
        # The attention residuals should be gated to zero, so output = x + FFN(LN(x))
        expected = x + block.ffn(block.ln_ffn(x))
    assert torch.allclose(out, expected, atol=1e-5)


def test_conditioning_mask_respected():
    """Perturbing masked conditioning positions should not change output."""
    block = _make_block()
    block.eval()

    x = torch.randn(B, SEQ_LEN, DENOISER_DIM)
    t_emb = torch.randn(B, DENOISER_DIM)
    cond = torch.randn(B, COND_LEN, DENOISER_DIM)

    # Mask out the last 4 positions
    mask = torch.zeros(B, COND_LEN, dtype=torch.bool)
    mask[:, COND_LEN // 2:] = True

    with torch.no_grad():
        out1 = block(x, t_emb, cond, conditioning_mask=mask)

        # Perturb the masked positions
        cond2 = cond.clone()
        cond2[:, COND_LEN // 2:] = torch.randn(B, COND_LEN // 2, DENOISER_DIM) * 100.0
        out2 = block(x, t_emb, cond2, conditioning_mask=mask)

    assert torch.allclose(out1, out2, atol=1e-5)
