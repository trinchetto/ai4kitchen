"""CLIP model utilities for ai4kitchen."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from torch import nn
from transformers import CLIPConfig, CLIPModel, CLIPProcessor


class ClipRecipeModule(nn.Module):
    """Thin wrapper around Hugging Face's CLIP model for recipe generation tasks."""

    def __init__(
        self,
        model_name_or_path: str | None = "openai/clip-vit-base-patch32",
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model, self.processor = self._initialize_clip(model_name_or_path)

    def _initialize_clip(
        self, model_name_or_path: str | None
    ) -> Tuple[CLIPModel, Optional[CLIPProcessor]]:
        processor: CLIPProcessor | None = None

        if model_name_or_path is None:
            return CLIPModel(CLIPConfig()), processor

        try:
            model = CLIPModel.from_pretrained(model_name_or_path)
        except Exception:  # pragma: no cover - offline fallback
            model = CLIPModel(CLIPConfig())

        try:
            processor = CLIPProcessor.from_pretrained(model_name_or_path)
        except Exception:  # pragma: no cover - offline fallback
            processor = None

        return model, processor

    def forward(self, batch: Dict[str, Any]) -> Any:
        """Run a forward pass through CLIP using pre-tokenized inputs."""

        if not isinstance(batch, dict):
            raise TypeError("forward expects a mapping with CLIP inputs.")

        clip_inputs = {
            key: value
            for key, value in batch.items()
            if key in {"pixel_values", "input_ids", "attention_mask", "position_ids"}
        }
        if not clip_inputs:
            raise ValueError("batch did not contain any CLIP-compatible keys.")

        return self.model(**clip_inputs)

    def encode_image(self, pixel_values: Any) -> Any:
        """Encode image features using the CLIP vision tower."""

        if hasattr(self.model, "get_image_features"):
            return self.model.get_image_features(pixel_values=pixel_values)
        raise AttributeError(
            "Underlying CLIP model does not provide image feature extraction."
        )

    def encode_text(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
        """Encode text features using the CLIP text tower."""

        if hasattr(self.model, "get_text_features"):
            return self.model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
        raise AttributeError(
            "Underlying CLIP model does not provide text feature extraction."
        )

    @property
    def projection_dim(self) -> int:
        """Return the shared projection dimension used by the CLIP text/image towers."""

        if hasattr(self.model.config, "projection_dim"):
            return int(self.model.config.projection_dim)

        if hasattr(self.model, "text_projection"):
            return int(self.model.text_projection.shape[-1])

        raise AttributeError(
            "Unable to determine CLIP projection dimension from the loaded model."
        )

    def freeze(self) -> None:
        """Freeze all CLIP parameters (both towers) to disable gradient updates."""

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()
