"""LightningModule for fine-tuning CLIP on recipe generation tasks."""

from __future__ import annotations

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ai4kitchen.models.recipe_module import ClipRecipeModule


class ClipRecipeFineTuner(pl.LightningModule):
    """LightningModule that keeps CLIP frozen and trains a lightweight fusion head."""

    def __init__(
        self,
        clip_model: ClipRecipeModule | None = None,
        learning_rate: float = 1e-4,
        fusion_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.clip = clip_model or ClipRecipeModule()
        self.clip.freeze()

        projection_dim = self.clip.projection_dim
        hidden_dim = fusion_hidden_dim or projection_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, batch: Dict[str, Any]) -> Any:
        """Forward pass delegated to the underlying CLIP model."""

        return self.clip(batch)

    def encode_image(self, pixel_values: Any) -> Any:
        """Proxy to the CLIP image encoder."""

        return self.clip.encode_image(pixel_values)

    def encode_text(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
        """Proxy to the CLIP text encoder."""

        return self.clip.encode_text(input_ids, attention_mask=attention_mask)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Compute contrastive loss between fused embeddings and CLIP text embeddings."""

        with torch.no_grad():
            outputs = self.forward(batch)

        if not hasattr(outputs, "image_embeds") or not hasattr(outputs, "text_embeds"):
            raise AttributeError("CLIP forward output missing image or text embeddings.")

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        fusion_input = torch.cat([image_embeds, text_embeds], dim=-1)
        fusion_embeds = self.fusion_head(fusion_input)

        fusion_embeds = F.normalize(fusion_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        device = fusion_embeds.device
        dtype = fusion_embeds.dtype

        logit_scale_param = getattr(self.clip.model, "logit_scale", None)
        if logit_scale_param is not None and hasattr(logit_scale_param, "detach"):
            logit_scale = logit_scale_param.detach().exp()
            logit_scale = logit_scale.to(device=device, dtype=dtype)
        else:
            logit_scale = torch.ones(1, device=device, dtype=dtype)

        logits_per_fusion = logit_scale * fusion_embeds @ text_embeds.t()
        logits_per_text = logits_per_fusion.t()

        batch_size = fusion_embeds.size(0)
        targets = torch.arange(batch_size, device=device)

        loss_img = F.cross_entropy(logits_per_fusion, targets)
        loss_txt = F.cross_entropy(logits_per_text, targets)
        loss = 0.5 * (loss_img + loss_txt)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer to update only the fusion head."""

        optimizer = torch.optim.Adam(
            self.fusion_head.parameters(), lr=self.learning_rate
        )
        return {"optimizer": optimizer}
