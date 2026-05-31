# FLUX.2 Klein Pipeline

Generate and edit images with the Black Forest Labs [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model on Apple Silicon with MLX.

## Features

- Unified generation and editing — text-to-image by default; uploading one or more images switches to editing automatically
- Two speed/quality modes: Distilled (4 steps) and Base (50 steps)
- Native Apple Silicon performance via MLX — no PyTorch required
- Two-column studio layout: controls on the left, generated image on the right
- Multi-image upload for editing and compositing workflows
- Auto-dimension: width/height sliders adjust to match the first input image's aspect ratio
- Optional vision-aware prompt upsampling via SmolVLM-500M-Instruct — a toggle in Advanced Settings; the VLM can see uploaded images when enhancing edit prompts (loaded on first use)
- Clickable examples — text-to-image prompts plus an editing example with bundled input images
- Per-step progress bar during inference
- Configurable seed, dimensions, guidance scale, and inference steps in Advanced Settings

## Requirements

- Apple Silicon Mac (M1+)
- Python 3.12+

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install dependencies: `uv sync`
3. Run the application: `uv run streamlit run streamlit_app.py`

Models are downloaded automatically on first use (~8GB per FLUX.2 Klein variant, ~1GB for SmolVLM).

## Testing

Run the unit tests (no GPU or model download required):

```bash
uv run pytest
```
