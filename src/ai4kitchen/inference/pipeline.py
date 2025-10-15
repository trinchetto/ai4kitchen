"""Inference pipeline for generating recipes from images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class RecipeGenerationPipeline:
    """High-level wrapper combining preprocessing, model inference, and decoding."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        """Generate a recipe for the provided image."""

        _ = image_path  # Placeholder until preprocessing is implemented
        return {"title": "", "ingredients": [], "instructions": []}
