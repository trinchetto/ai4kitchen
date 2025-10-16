# ai4kitchen

ai4kitchen converts images of dishes or pantry ingredients into fully fledged
recipes by combining vision and language models.

## Documentation

- [Overview](docs/overview.md)
- [Usage](docs/usage.md)
- [Code Structure](docs/structure.md)

## Getting Started

Install dependencies and run the lightweight test suite:

```bash
source scripts/setup.sh 
```
to keep the environment active in your shell when it finishes.

## Project Goals

- Fine-tune a CLIP backbone with a recipe generation head using PyTorch
  Lightning.
- Provide clear documentation for usage, design decisions, and repository
  layout.
- Maintain lightweight continuous integration to keep the project healthy.
