import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal position features to a batch of embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_length: int = 5000):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        angles = positions * frequencies.unsqueeze(0)
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        length = embeddings.size(1)
        if length > self.encoding.size(1):
            raise ValueError(
                f"sequence length {length} exceeds max_length {self.encoding.size(1)}"
            )
        return self.dropout(embeddings + self.encoding[:, :length].to(embeddings.dtype))


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Return a bool attention mask where True entries forbid attention."""
    return torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1
    )


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        pad_id,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        max_length,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.src_position = SinusoidalPositionalEncoding(d_model, dropout, max_length)
        self.tgt_position = SinusoidalPositionalEncoding(d_model, dropout, max_length)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.embedding_scale = math.sqrt(d_model)

    def forward(
        self, src_ids: torch.Tensor, decoder_input_ids: torch.Tensor
    ) -> torch.Tensor:
        src_padding_mask = src_ids.eq(self.pad_id)
        tgt_padding_mask = decoder_input_ids.eq(self.pad_id)
        tgt_mask = causal_mask(decoder_input_ids.size(1), decoder_input_ids.device)

        src = self.src_position(self.src_embedding(src_ids) * self.embedding_scale)
        tgt = self.tgt_position(
            self.tgt_embedding(decoder_input_ids) * self.embedding_scale
        )
        hidden = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.output_projection(hidden)


def greedy_decode(
    model,
    src_ids: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
) -> torch.Tensor:
    """Greedily decode, returning generated tokens without the BOS token."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    batch_size = src_ids.size(0)
    generated = torch.empty((batch_size, 0), dtype=torch.long, device=src_ids.device)
    if max_new_tokens == 0:
        return generated

    decoder_input_ids = torch.full(
        (batch_size, 1), bos_id, dtype=torch.long, device=src_ids.device
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=src_ids.device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(src_ids, decoder_input_ids)
                next_ids = logits[:, -1, :].argmax(dim=-1)
                next_ids = next_ids.masked_fill(finished, pad_id)
                generated = torch.cat((generated, next_ids.unsqueeze(1)), dim=1)
                finished |= next_ids.eq(eos_id)
                if bool(finished.all()):
                    break
                decoder_input_ids = torch.cat(
                    (decoder_input_ids, next_ids.unsqueeze(1)), dim=1
                )
    finally:
        model.train(was_training)
    return generated
