"""Smoke tests for the ai4kitchen project structure."""
from __future__ import annotations

from ai4kitchen.models.recipe_module import ClipRecipeModule
from ai4kitchen.inference.pipeline import RecipeGenerationPipeline


def test_clip_recipe_module_instantiation() -> None:
    module = ClipRecipeModule()
    assert module is not None


def test_pipeline_predict_output_shape() -> None:
    pipeline = RecipeGenerationPipeline(model=ClipRecipeModule())
    prediction = pipeline.predict("sample.jpg")
    assert {"title", "ingredients", "instructions"} <= set(prediction)
