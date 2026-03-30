"""Tests for FrozenEncoder."""

import pytest
import torch

from src.models.encoder.frozen_encoder import FrozenEncoder

MODEL_NAME = "bert-base-uncased"
B, SEQ_LEN = 2, 8


@pytest.fixture(scope="module")
def encoder():
    """Module-scoped frozen encoder to avoid repeated downloads."""
    return FrozenEncoder(model_name=MODEL_NAME, unfreeze_top_n=0)


@pytest.fixture(scope="module")
def sample_inputs(encoder):
    """Random token ids within the BERT vocab range."""
    vocab_size = encoder.get_embedding_table().shape[0]
    input_ids = torch.randint(0, vocab_size, (B, SEQ_LEN))
    attention_mask = torch.ones(B, SEQ_LEN, dtype=torch.long)
    return input_ids, attention_mask


# --------------------------------------------------------------------- #
# Tests                                                                  #
# --------------------------------------------------------------------- #


def test_all_frozen(encoder):
    """When unfreeze_top_n=0 every encoder parameter should be frozen."""
    for name, param in encoder.named_parameters():
        assert not param.requires_grad, f"{name} should be frozen"


def test_output_shape(encoder, sample_inputs):
    """Encoded output must have shape (B, seq_len, hidden_dim)."""
    input_ids, attention_mask = sample_inputs
    hidden_dim = encoder.get_embedding_table().shape[1]

    output = encoder.encode(input_ids, attention_mask)

    assert output.shape == (B, SEQ_LEN, hidden_dim)


def test_determinism(encoder, sample_inputs):
    """Same input must produce identical output across two calls."""
    input_ids, attention_mask = sample_inputs

    out1 = encoder.encode(input_ids, attention_mask)
    out2 = encoder.encode(input_ids, attention_mask)

    assert torch.equal(out1, out2)


def test_no_gradient_flow(encoder, sample_inputs):
    """Backward from the encoder output should not produce gradients."""
    input_ids, attention_mask = sample_inputs

    output = encoder.encode(input_ids, attention_mask)
    # output is produced under torch.no_grad(), so grad_fn is None.
    assert output.grad_fn is None, "Output should have no grad_fn"

    for name, param in encoder.named_parameters():
        assert param.grad is None, f"{name} should have no gradient"


def test_embedding_table_shape(encoder):
    """Embedding table shape should be (vocab_size, hidden_dim)."""
    table = encoder.get_embedding_table()
    assert table.ndim == 2
    # BERT-base-uncased vocab is ~30522; just check it's reasonable.
    assert table.shape[0] > 10000
    assert table.shape[1] > 0


def test_stays_eval(encoder):
    """Calling .train() should keep the encoder in eval mode."""
    encoder.train()
    assert not encoder.training, "Encoder should remain in eval mode"
    assert not encoder.bert.training, "Inner BERT should remain in eval mode"
