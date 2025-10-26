"""Tests for :mod:`ai4kitchen.data.ingredients_normalization` static helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Dynamically import the IngredientNormalizer class from the ingredients_normalization module
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai4kitchen"
    / "data"
    / "ingredients_normalization.py"
)
spec = importlib.util.spec_from_file_location("ingredients_normalization", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec for {MODULE_PATH}")
ingredients_normalization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingredients_normalization)
IngredientNormalizer = ingredients_normalization.IngredientNormalizer


def test_strip_accents_removes_diacritics() -> None:
    result = IngredientNormalizer._strip_accents("crème brûlée jalapeño")
    assert result == "creme brulee jalapeno"


def test_join_multiwords_replaces_known_phrases() -> None:
    text = "drizzle olive oil with fish sauce"
    result = IngredientNormalizer._join_multiwords(text, {"olive oil", "fish sauce"})
    assert result == "drizzle olive_oil with fish_sauce"


def test_remove_quantities_strips_numbers() -> None:
    text = "add 1-1/2 cups tomatoes and 3.5 oz chicken"
    result = IngredientNormalizer._remove_quantities(text)
    assert "1" not in result and "3" not in result


def test_remove_parentheticals_and_punctuation_replaces_with_spaces() -> None:
    text = "chopped (fresh) basil - room-temperature"
    result = IngredientNormalizer._remove_parentheticals_and_punctuation(text)
    assert result == "chopped  fresh  basil   room temperature"


def test_apply_synonyms_maps_known_terms() -> None:
    text = "powdered sugar with cilantro"
    result = IngredientNormalizer._apply_synonyms(
        text, {"powdered sugar": "icing sugar", "cilantro": "coriander"}
    )
    assert result == "icing sugar with coriander"


def test_tokenize_extracts_letter_sequences() -> None:
    text = "olive oil and chef's salt"
    result = IngredientNormalizer._tokenize(text)
    assert result == ["olive", "oil", "and", "chef's", "salt"]


def test_drop_clutter_removes_units_and_prep_words() -> None:
    text = "olive oil cups chopped parsley"
    result = IngredientNormalizer._drop_clutter(text)
    assert result == ["olive", "oil", "parsley"]


def test_normalize_ingredient_produces_core_tokens() -> None:
    result = IngredientNormalizer.normalize_ingredient("1 cup chopped cilantro (fresh)")
    assert result == ["coriander"]


def test_normalize_ingredient_list_deduplicates_in_order() -> None:
    ingredients = [
        "1 cup Olive Oil",
        "2 tablespoons olive oil",
        "Chopped fresh cilantro",
    ]
    result = IngredientNormalizer.normalize_ingredient_list(ingredients)
    assert result == ["olive", "oil", "coriander"]
