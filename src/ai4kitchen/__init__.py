"""Top-level package for ai4kitchen."""
from .models.recipe_module import ClipRecipeModule
from .inference.pipeline import RecipeGenerationPipeline

__all__ = ["ClipRecipeModule", "RecipeGenerationPipeline"]
