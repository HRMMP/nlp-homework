import unittest

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest("torch is not installed") from exc

from torch import nn

from machine_translation.model import TransformerSeq2Seq, causal_mask, greedy_decode


def make_model():
    return TransformerSeq2Seq(
        src_vocab_size=17,
        tgt_vocab_size=19,
        pad_id=0,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_length=12,
    )


def test_forward_shape_and_target_vocabulary_dimension():
    model = make_model()
    src_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    decoder_input_ids = torch.tensor([[1, 6, 0, 0], [1, 7, 8, 9]])

    logits = model(src_ids, decoder_input_ids)

    assert logits.shape == (2, 4, 19)


def test_forward_builds_boolean_padding_masks():
    class CaptureTransformer(nn.Module):
        def forward(self, src, tgt, **kwargs):
            self.kwargs = kwargs
            return tgt

    model = make_model()
    capture = CaptureTransformer()
    model.transformer = capture
    src_ids = torch.tensor([[1, 0, 0]])
    decoder_input_ids = torch.tensor([[1, 5, 0]])

    model(src_ids, decoder_input_ids)

    assert capture.kwargs["src_key_padding_mask"].dtype == torch.bool
    assert capture.kwargs["src_key_padding_mask"].tolist() == [[False, True, True]]
    assert capture.kwargs["tgt_key_padding_mask"].tolist() == [[False, False, True]]
    assert capture.kwargs["memory_key_padding_mask"].tolist() == [[False, True, True]]
    assert capture.kwargs["tgt_mask"].dtype == torch.bool


def test_causal_mask_forbids_future_positions_only():
    mask = causal_mask(4, torch.device("cpu"))

    assert mask.dtype == torch.bool
    assert torch.equal(
        mask,
        torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [False, False, False, False],
            ]
        ),
    )


def test_future_decoder_tokens_do_not_change_earlier_logits():
    torch.manual_seed(3)
    model = make_model().eval()
    src_ids = torch.tensor([[1, 2, 0]])
    prefix = torch.tensor([[1, 6, 7, 8]])
    changed_future = torch.tensor([[1, 6, 12, 13]])

    with torch.no_grad():
        original_logits = model(src_ids, prefix)
        changed_logits = model(src_ids, changed_future)

    torch.testing.assert_close(original_logits[:, :2], changed_logits[:, :2])


class StepwiseLogitModel(nn.Module):
    def __init__(self, vocab_size=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.grad_enabled_during_forward = []

    def forward(self, src_ids, decoder_input_ids):
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        step = decoder_input_ids.size(1)
        logits = torch.full(
            (src_ids.size(0), step, self.vocab_size), -100.0, device=src_ids.device
        )
        if step == 1:
            logits[0, -1, 2] = 10.0  # First row ends immediately.
            logits[1, -1, 4] = 10.0
        elif step == 2:
            logits[:, -1, 5] = 10.0
        else:
            logits[:, -1, 2] = 10.0
        return logits


def test_greedy_decode_includes_eos_pads_finished_rows_and_restores_mode():
    model = StepwiseLogitModel().train()

    result = greedy_decode(
        model,
        torch.tensor([[1], [3]]),
        bos_id=1,
        eos_id=2,
        pad_id=0,
        max_new_tokens=5,
    )

    assert result.tolist() == [[2, 0, 0], [4, 5, 2]]
    assert model.training
    assert model.grad_enabled_during_forward == [False, False, False]


def test_greedy_decode_stops_at_maximum_length_without_eos():
    class NeverEndingModel(nn.Module):
        def forward(self, src_ids, decoder_input_ids):
            logits = torch.zeros(src_ids.size(0), decoder_input_ids.size(1), 8)
            logits[..., 3] = 1.0
            return logits

    result = greedy_decode(
        NeverEndingModel(),
        torch.tensor([[1], [2]]),
        bos_id=1,
        eos_id=2,
        pad_id=0,
        max_new_tokens=2,
    )

    assert result.tolist() == [[3, 3], [3, 3]]


def test_greedy_decode_zero_length_returns_empty_batch_tensor():
    result = greedy_decode(
        StepwiseLogitModel(),
        torch.tensor([[1], [2]]),
        bos_id=1,
        eos_id=2,
        pad_id=0,
        max_new_tokens=0,
    )

    assert result.shape == (2, 0)
