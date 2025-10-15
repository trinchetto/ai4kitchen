"""Dataset definitions for ai4kitchen."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class IngredientExample:
    """Container for a single ingredient-to-recipe training example."""

    image_path: str
    ingredients: List[str]
    recipe_text: str


class IngredientRecipeDataset:
    """Placeholder dataset for loading ingredient and recipe pairs."""

    def __init__(self, manifest_path: str) -> None:
        self.manifest_path = manifest_path

    def __len__(self) -> int:  # pragma: no cover - placeholder until dataset is implemented
        return 0

    def __getitem__(self, index: int) -> IngredientExample:  # pragma: no cover - placeholder
        raise NotImplementedError
