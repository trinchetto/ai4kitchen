"""Entry point for launching ai4kitchen training."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl

from ai4kitchen.models.recipe_module import ClipRecipeModule


def main(config_path: str = "config/training.yaml") -> None:
    """Launch a Lightning training run using the provided config file."""

    _ = Path(config_path)
    model = ClipRecipeModule()
    trainer = pl.Trainer()
    trainer.fit(model)


if __name__ == "__main__":
    main()
