"""XLM-R cross-encoder with two heads: classification and projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .labels import NUM_LABELS


class DarcNliModel(nn.Module):
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        *,
        projection_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)

        # The tokenizer is saved with the checkpoint so a run can be scored
        # without re-downloading anything.
        self.tokenizer_name = model_name
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, NUM_LABELS)
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, projection_dim),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        try:
            outputs = self.encoder(**kwargs)
        except TypeError:
            # RoBERTa-family encoders (XLM-R) do not accept token_type_ids.
            kwargs.pop("token_type_ids", None)
            outputs = self.encoder(**kwargs)
        # <s> is the first token for RoBERTa-family models (XLM-R).
        cls = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(cls)
        projections = F.normalize(self.projection(cls), dim=-1)
        return logits, projections

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, directory: Path, *, tokenizer: Any, metadata: dict) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "model.pt")
        self.encoder.config.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        import json

        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load_checkpoint(cls, directory: Path, *, tokenizer: Any | None = None) -> "DarcNliModel":
        directory = Path(directory)
        state = torch.load(directory / "model.pt", map_location="cpu")
        model = cls(str(directory), projection_dim=int(state["projection.2.weight"].shape[0]))
        model.load_state_dict(state)
        return model


def build_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    return tokenizer
