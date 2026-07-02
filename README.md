# FLUX.2 Klein Studio

Streamlit application for generating and editing images using Black Forest Labs [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) on Apple Silicon with MLX.

## Features

- Unified generation and editing — text-to-image by default; uploading one or more images switches to editing automatically
- Two speed/quality modes: Distilled (4 steps) and Base (50 steps)
- Native Apple Silicon performance via MLX — inference runs on MLX, not PyTorch
- Two-column studio layout: controls on the left, generated image on the right
- Enter to run — the prompt row is a borderless form, so pressing Enter submits the run
- Multi-image upload for editing and compositing workflows
- Auto-dimension: width/height sliders adjust to match the first input image's aspect ratio
- Optional vision-aware prompt upsampling via SmolVLM-500M-Instruct — a toggle in Advanced settings; the VLM can see uploaded images when enhancing edit prompts (loaded on first use)
- Clickable examples — text-to-image prompts plus an editing example with bundled input images (loading one replaces any manual upload)
- Per-step progress bar shown inside the output frame during inference, with labeled spinners for first-time model loads and prompt enhancement
- Configurable seed, dimensions, guidance scale, and inference steps in Advanced settings
- Native light/dark theme via `.streamlit/config.toml` (no custom CSS), with WCAG AA-compliant link contrast in both modes
- Graceful failure handling — empty runs are blocked; unreadable uploads and generation errors surface inline instead of crashing the app

## Requirements

- Apple Silicon Mac (M1+)
- Python 3.12+

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install dependencies: `uv sync`
3. Run the application: `uv run streamlit run streamlit_app.py`

Models are downloaded automatically on first use (~8GB per FLUX.2 Klein variant, ~1GB for SmolVLM).

## Development

Lint, format, type-check, and run the unit tests (no GPU or model download required):

```bash
uv run ruff check .   # Lint
uv run ruff format .  # Format
uv run ty check .     # Type check
uv run pytest         # Unit tests
```
