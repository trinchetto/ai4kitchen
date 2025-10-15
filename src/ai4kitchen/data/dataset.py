"""Dataset definitions and helpers for ai4kitchen."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

from datasets import load_dataset

# Default Hugging Face dataset used for fine-tuning.
_DEFAULT_DATASET_NAME = "kaggle_food_recipes"


@dataclass
class IngredientExample:
    """Container for a single ingredient-to-recipe training example."""

    image_path: str
    ingredients: List[str]
    recipe_text: str


class IngredientRecipeDataset:
    """Dataset wrapper around Hugging Face's ``kaggle_food_recipes`` corpus."""

    def __init__(
        self,
        split: str,
        dataset_name: str = _DEFAULT_DATASET_NAME,
        cache_dir: str | None = None,
    ) -> None:
        """Load the requested split from the Hugging Face dataset.

        Parameters
        ----------
        split:
            Hugging Face split selector. Supports slice syntax such as
            ``train[:90%]`` or ``train[-10%:]``.
        dataset_name:
            Name of the dataset on the Hugging Face Hub. Defaults to
            ``kaggle_food_recipes``.
        cache_dir:
            Optional location for the Hugging Face dataset cache.
        """

        self.split = split
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self._dataset = self._load_dataset()

    def _load_dataset(self) -> Any:
        """Load the configured split from Hugging Face datasets."""

        return load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> IngredientExample:
        raw_example = self._dataset[index]
        image_path = self._extract_image_path(raw_example)
        ingredients = self._extract_ingredients(raw_example)
        recipe_text = self._extract_recipe_text(raw_example)
        return IngredientExample(image_path=image_path, ingredients=ingredients, recipe_text=recipe_text)

    @staticmethod
    def _extract_image_path(example: Any) -> str:
        """Return the dataset-provided image name if available."""

        value = example.get("Image_Name")
        return str(value) if value else ""

    @staticmethod
    def _extract_ingredients(example: Any) -> List[str]:
        """Gather a normalized list of ingredient strings."""

        raw_ingredients = example.get("Cleaned_Ingredients")
        if not raw_ingredients:
            return []
        if isinstance(raw_ingredients, str):
            stripped = raw_ingredients.strip()
            return [stripped] if stripped else []
        if isinstance(raw_ingredients, Iterable) and not isinstance(raw_ingredients, (str, bytes)):
            return [str(item).strip() for item in raw_ingredients if str(item).strip()]
        return [str(raw_ingredients)]

    @staticmethod
    def _extract_recipe_text(example: Any) -> str:
        """Normalize instruction fields into a single text blob."""

        raw_instructions = example.get("Instructions")
        if not raw_instructions:
            return ""
        if isinstance(raw_instructions, str):
            return raw_instructions.strip()
        if isinstance(raw_instructions, Iterable) and not isinstance(raw_instructions, (str, bytes)):
            parts = [str(item).strip() for item in raw_instructions if str(item).strip()]
            if parts:
                return "\n".join(parts)
        return str(raw_instructions)

    @staticmethod
    def _extract_title(example: Any) -> str:
        """Return the recipe title if present."""

        value = example.get("Title")
        return str(value).strip() if value else ""
