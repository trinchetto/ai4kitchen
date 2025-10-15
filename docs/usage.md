# Using ai4kitchen

1. Prepare manifests describing training and validation samples under the
   `data/` directory.
2. Update `config/training.yaml` with dataset paths and hyperparameters.
3. Launch training with `python scripts/train.py`.
4. Export and run inference with the utilities inside `ai4kitchen/inference`.
