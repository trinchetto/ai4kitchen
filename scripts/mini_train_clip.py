#!/usr/bin/env python
"""Run a short ClipRecipeFineTuner training loop and verify loss decreases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Sequence

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


def move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Move all tensor values in the batch onto the requested device."""

    return {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def run_training_loop(
    epochs: int,
    dataloader: DataLoader[Dict[str, torch.Tensor]],
    model: ClipRecipeFineTuner,
    learning_rate: float,
    device: torch.device,
) -> TrainingSummary:
    """Execute a simple optimization loop and collect per-epoch losses."""

    optimizer = torch.optim.Adam(model.fusion_head.parameters(), lr=learning_rate)
    epoch_losses: List[float] = []

    model.train()
    for epoch in range(epochs):
        losses: List[float] = []
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            batch_on_device = move_batch_to_device(batch, device)
            loss = model.training_step(batch_on_device, 0)
            if not isinstance(loss, torch.Tensor):
                raise TypeError("training_step did not return a torch.Tensor loss.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        mean_loss = sum(losses) / max(len(losses), 1)
        epoch_losses.append(mean_loss)
        print(f"Epoch {epoch + 1}/{epochs}: mean loss = {mean_loss:.6f}")

    return TrainingSummary(epoch_losses=epoch_losses)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight ClipRecipeFineTuner training loop on synthetic data "
            "and verify the loss decreases."
        )
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to run.")
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
        "--device",
        type=str,
        default="cpu",
        help="Torch device identifier (default: cpu).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point that orchestrates the training smoke test."""

    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dataset = SyntheticClipBatchDataset(size=args.dataset_size, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    clip_model = ClipRecipeModule(model_name_or_path=None)
    model = ClipRecipeFineTuner(clip_model=clip_model, learning_rate=args.learning_rate)
    model.to(device)

    summary = run_training_loop(
        epochs=args.epochs,
        dataloader=dataloader,
        model=model,
        learning_rate=args.learning_rate,
        device=device,
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
