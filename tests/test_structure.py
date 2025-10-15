"""Smoke tests for the ai4kitchen project structure."""

from __future__ import annotations

from ai4kitchen.models.recipe_module import ClipRecipeModule
from ai4kitchen.training import ClipRecipeFineTuner
from ai4kitchen.inference.pipeline import RecipeGenerationPipeline


def test_clip_recipe_module_instantiation() -> None:
    module = ClipRecipeModule()
    assert isinstance(module, ClipRecipeModule)


def test_clip_recipe_module_projection_dim_property() -> None:
    module = ClipRecipeModule()
    projection_dim = module.projection_dim
    assert isinstance(projection_dim, int)
    assert projection_dim > 0


def test_clip_recipe_finetuner_instantiation_and_freeze() -> None:
    finetuner = ClipRecipeFineTuner()

    clip_params = list(finetuner.clip.model.parameters())
    assert clip_params and all(not param.requires_grad for param in clip_params)

    fusion_params = list(finetuner.fusion_head.parameters())
    assert fusion_params and all(param.requires_grad for param in fusion_params)


def test_pipeline_predict_output_shape() -> None:
    pipeline = RecipeGenerationPipeline(model=ClipRecipeModule())
    prediction = pipeline.predict("sample.jpg")
    assert {"title", "ingredients", "instructions"} <= set(prediction)
