#!/usr/bin/env python
"""Run a short ClipRecipeFineTuner training loop and verify loss decreases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ai4kitchen.models.recipe_module import ClipRecipeModule
from ai4kitchen.training.clip_finetuner import ClipRecipeFineTuner


@dataclass
class TrainingSummary:
    """Container for tracking per-epoch losses."""

    epoch_losses: List[float]

    @property
    def initial_loss(self) -> float:
        return self.epoch_losses[0]

    @property
    def final_loss(self) -> float:
        return self.epoch_losses[-1]


class SyntheticClipBatchDataset(Dataset[Dict[str, torch.Tensor]]):
    """Pre-generated synthetic CLIP inputs for deterministic training."""

    def __init__(
        self,
        size: int,
        image_size: int = 224,
        sequence_length: int = 77,
        vocab_size: int = 1000,
        seed: int = 0,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.pixel_values = torch.randn(
            size,
            3,
            image_size,
            image_size,
            generator=generator,
        ).float()
        self.input_ids = torch.randint(
            0,
            vocab_size,
            (size, sequence_length),
            generator=generator,
            dtype=torch.long,
        )
        self.attention_mask = torch.ones(size, sequence_length, dtype=torch.long)

    def __len__(self) -> int:
        return self.pixel_values.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "pixel_values": self.pixel_values[index],
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
        }


class TrainLossTracker(pl.Callback):
    """Lightning callback that records training loss after each epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_losses: List[float] = []

    def on_train_epoch_end(  # type: ignore[override]
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        for key in ("train_loss_epoch", "train_loss", "loss"):
            loss = trainer.callback_metrics.get(key)
            if loss is not None:
                self.epoch_losses.append(float(loss.detach().cpu().item()))
                break


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight ClipRecipeFineTuner training loop on synthetic data "
            "and verify the loss decreases."
        )
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to run."
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=16,
        help="Number of synthetic examples to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the training loop.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="Optimizer learning rate for the fusion head.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed applied before data/model initialization.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="PyTorch Lightning accelerator setting (e.g., cpu, gpu, auto).",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Optional number of devices to use (defaults to Lightning's auto selection).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point that orchestrates the training smoke test."""

    args = parse_args(argv)
    pl.seed_everything(args.seed, workers=True)

    dataset = SyntheticClipBatchDataset(size=args.dataset_size, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    clip_model = ClipRecipeModule(model_name_or_path=None)
    model = ClipRecipeFineTuner(clip_model=clip_model, learning_rate=args.learning_rate)

    loss_tracker = TrainLossTracker()

    trainer_kwargs: Dict[str, Any] = {
        "max_epochs": args.epochs,
        "accelerator": args.accelerator,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "deterministic": True,
    }
    if args.devices is not None:
        trainer_kwargs["devices"] = args.devices

    trainer = pl.Trainer(callbacks=[loss_tracker], **trainer_kwargs)
    trainer.fit(model, train_dataloaders=dataloader)

    if not loss_tracker.epoch_losses:
        raise SystemExit("Training did not produce any loss values.")

    summary = TrainingSummary(
        epoch_losses=loss_tracker.epoch_losses,
    )

    print(
        "Loss progression:",
        ", ".join(f"{loss:.6f}" for loss in summary.epoch_losses),
    )

    if summary.final_loss >= summary.initial_loss:
        raise SystemExit(
            f"Training loss did not decrease: {summary.initial_loss:.6f} -> {summary.final_loss:.6f}"
        )


if __name__ == "__main__":
    main()
