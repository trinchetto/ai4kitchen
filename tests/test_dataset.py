"""Tests for :mod:`ai4kitchen.data.dataset`."""

from __future__ import annotations

from typing import Any

import pytest

from ai4kitchen.data.dataset import IngredientExample, IngredientRecipeDataset


@pytest.fixture
def fake_dataset() -> list[dict[str, Any]]:
    return [
        {
            "Image_Name": "first.jpg",
            "Cleaned_Ingredients": ["flour", " sugar "],
            "Instructions": ["Mix ingredients.", "Bake for 20 minutes."],
        },
        {
            "Image_Name": "second.jpg",
            "Cleaned_Ingredients": "",
            "Instructions": "Stir well.",
        },
    ]


def test_dataset_len(monkeypatch: pytest.MonkeyPatch, fake_dataset: list[dict[str, Any]]) -> None:
    """Verify that ``__len__`` proxies the size of the underlying Hugging Face dataset."""

    captured_kwargs: dict[str, Any] = {}

    def fake_load_dataset(name: str, split: str, cache_dir: str | None) -> list[dict[str, Any]]:
        captured_kwargs["name"] = name
        captured_kwargs["split"] = split
        captured_kwargs["cache_dir"] = cache_dir
        return fake_dataset

    monkeypatch.setattr("ai4kitchen.data.dataset.load_dataset", fake_load_dataset)

    dataset = IngredientRecipeDataset(split="train[:2]")

    assert len(dataset) == len(fake_dataset)
    assert captured_kwargs == {"name": "kaggle_food_recipes", "split": "train[:2]", "cache_dir": None}


def test_dataset_getitem(monkeypatch: pytest.MonkeyPatch, fake_dataset: list[dict[str, Any]]) -> None:
    """Ensure ``__getitem__`` returns an ``IngredientExample`` with normalized fields."""

    monkeypatch.setattr("ai4kitchen.data.dataset.load_dataset", lambda *args, **kwargs: fake_dataset)

    dataset = IngredientRecipeDataset(split="train[:2]")
    example = dataset[0]

    assert isinstance(example, IngredientExample)
    assert example.image_path == "first.jpg"
    assert example.ingredients == ["flour", "sugar"]
    assert example.recipe_text == "Mix ingredients.\nBake for 20 minutes."


def test_extract_image_path_direct_call() -> None:
    """Ensure `_extract_image_path` returns the exact image name string from the row."""

    result = IngredientRecipeDataset._extract_image_path({"Image_Name": "some-image.png"})
    assert result == "some-image.png"


def test_extract_ingredients_normalizes_iterables() -> None:
    """Ensure `_extract_ingredients` trims strings and discards empties from iterable input."""

    result = IngredientRecipeDataset._extract_ingredients({"Cleaned_Ingredients": [" flour ", ""]})
    assert result == ["flour"]


def test_extract_recipe_text_joins_steps() -> None:
    """Ensure `_extract_recipe_text` joins multiple instructions with newlines."""

    result = IngredientRecipeDataset._extract_recipe_text({"Instructions": ["Step 1 ", " Step 2"]})
    assert result == "Step 1\nStep 2"


def test_extract_title_strips_whitespace() -> None:
    """Ensure `_extract_title` returns the recipe title with surrounding whitespace removed."""

    result = IngredientRecipeDataset._extract_title({"Title": "  Fancy Dish  "})
    assert result == "Fancy Dish"
