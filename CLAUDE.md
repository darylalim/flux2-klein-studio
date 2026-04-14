# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX.2 Klein Pipeline is a single-file Streamlit web application that generates and edits images using [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (4B parameters) from Black Forest Labs. Runs natively on Apple Silicon via [mflux](https://github.com/filipstrand/mflux) (diffusion) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (VLM). Two task modes: Generate (text-to-image) and Edit (multi-image editing). Two speed/quality options: Fast (distilled, 4 steps) and Quality (base, 50 steps). Optional vision-aware prompt upsampling via [SmolVLM-500M-Instruct](https://huggingface.co/mlx-community/SmolVLM-500M-Instruct-bf16) — the VLM can see uploaded images when enhancing editing prompts.

## Setup

```bash
uv sync
```

Requires Apple Silicon (M1+) and Python 3.12+.

## Running

```bash
uv run streamlit run streamlit_app.py
```

## Architecture

Everything lives in `streamlit_app.py`, structured in four sections:

1. **Model initialization** — Two model classes from mflux: `Flux2Klein` for text-to-image and `Flux2KleinEdit` for multi-image editing. MLX manages Apple Silicon unified memory automatically. Four `@st.cache_resource`-cached getters: `_get_model_distilled()`, `_get_model_base()`, `_get_edit_model_distilled()`, `_get_edit_model_base()`. `MODELS` maps mode names to txt2img getters; `EDIT_MODELS` maps mode names to edit getters. `MODE_DEFAULTS` holds per-mode defaults (`"Fast"`: 4 steps, CFG 1.0; `"Quality"`: 50 steps, CFG 4.0).
2. **Prompt upsampling** — `_get_vlm()` loads SmolVLM-500M-Instruct-bf16 via `mlx_vlm.load()`, cached with `@st.cache_resource`, returns a `(model, processor, config)` triple. Two system prompts: `UPSAMPLE_PROMPT_TEXT_ONLY` (text-to-image, capped at 120 words) and `UPSAMPLE_PROMPT_WITH_IMAGES` (image editing, concrete language, preserve unchanged elements). `upsample_prompt(prompt, image_list=None)` selects the system prompt based on whether images are provided, formats messages via `mlx_vlm.prompt_utils.apply_chat_template`, generates via `mlx_vlm.generate()`, and strips `<end_of_utterance>` tokens from the output. Loaded lazily on first use. `_resolve_prompt(prompt, image_list, auto_enhance)` wraps the auto-enhance decision: enhances only when `auto_enhance` is true, returns `(prompt, was_enhanced)` tuple.
3. **Inference** — `infer()` takes prompt, seed, dimensions (512–1440px), mode, optional `image_list`, and optional `progress_callback`. Defaults resolve from `MODE_DEFAULTS[mode]`. Selects `Flux2KleinEdit` from `EDIT_MODELS` when `image_list` is provided, otherwise `Flux2Klein` from `MODELS`. Calls `model.generate_image()` which returns a `GeneratedImage`; the PIL Image is extracted via `.image`. When `progress_callback` is provided, registers an `InLoopCallback` via `model.callbacks.register()` for per-step progress reporting, then removes it from `model.callbacks.in_loop` in a `finally` block so reporters do not accumulate on cached models across Runs. `_dimensions_from_images()` calculates output dimensions from the first uploaded image's aspect ratio (larger side 1024, rounds to 32, clamps 512–1440).
4. **UI** — Behind `if __name__ == "__main__"`: `st.title` header (`FLUX.2 Klein Pipeline`, no caption), a two-column pill row with a Task pill (`Generate` / `Edit`, default `Generate`, key `task_pills`, `label_visibility="collapsed"`) and a Quality pill (`Fast` / `Quality`, default `Fast`, key `mode_pills`, `label_visibility="collapsed"`). UI branches on `task_mode`: Generate renders a full-width prompt `st.text_area` (placeholder `Describe the image…`); Edit renders prompt (left) + file uploader (right) in `st.columns(2)` (prompt placeholder `Describe the edit…`, uploader label `Input images` with collapsed visibility). An `Enhance prompt` checkbox (`key="auto_enhance_checkbox"`, unchecked by default, no help tooltip) sits below the prompt area. A `Settings` expander contains seed, dimensions, guidance scale, and inference steps. The Run button is disabled in Edit mode when no images are uploaded (`run_disabled = task_mode == "Edit" and not image_list`). A `st.progress` bar shows per-step inference progress. Switching task mode causes the file uploader to leave/re-enter the widget tree, naturally clearing uploaded state. Width/height sliders auto-update to match uploaded image aspect ratio; guidance scale and steps update when mode changes.

## Commands

```bash
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format
uv run ruff format --check .     # Check formatting only
uv run ty check .                # Type check
uv run pytest                    # Run all tests
uv run pytest tests/test_streamlit_app.py  # Run a single test file
```

## Gotchas

### mflux / FLUX.2 Klein

- **Two model classes: `Flux2Klein` (text-to-image) and `Flux2KleinEdit` (multi-image editing).** Both created via `model_config=ModelConfig.flux2_klein_4b()`. No repo ID strings or token parameters.
- **`Flux2Klein.generate_image()` takes `image_path` (singular, `Path | str | None`) for img2img.** `Flux2KleinEdit.generate_image()` takes `image_paths` (plural, `list[Path | str] | None`) for multi-image editing.
- **`generate_image()` returns a `GeneratedImage` wrapper, not a PIL Image.** Access the PIL Image via `.image`.
- **MLX manages device placement automatically.** Apple Silicon unified memory is used directly.
- **Progress callbacks use `model.callbacks.register()`.** Register an object with a `call_in_loop(self, t, seed, prompt, latents, config, time_steps)` method. `CallbackRegistry` has no `unregister()` — callers must pop the instance from `model.callbacks.in_loop` themselves, or reporters accumulate on `@st.cache_resource`-cached models and fire on every subsequent Run.
- **The guidance parameter is named `guidance`**, not `guidance_scale`.
- **FLUX.2 Klein does not support negative prompts.**
- **Quality mode uses different defaults than Fast mode.** Quality (base model): 50 steps, CFG 4.0. Fast (distilled model): 4 steps, CFG 1.0. The underlying model variants are still named distilled and base internally; `"Fast"` and `"Quality"` are the user-facing `MODE_DEFAULTS` keys.

### mlx-vlm / SmolVLM

- **Use `mlx_vlm.load()` to get `(model, processor)` and `mlx_vlm.utils.load_config()` for config.** Config is required by `apply_chat_template`.
- **`mlx_vlm.generate()` handles tokenization and decoding internally.** Access result via `result.text`. SmolVLM appends `<end_of_utterance>` tokens that must be stripped from output.
- **`apply_chat_template` takes `num_images` instead of embedding image tokens in messages.** Pass images as a flat list to the `image` parameter of `generate()`.
- **`temperature > 0` implies sampling.** Use `temperature=0.0` for greedy decoding.
- **Use `max_tokens`** to control output length.

### General

- **Apple Silicon is the primary target.** The app uses MLX (mflux + mlx-vlm) which requires Apple Silicon or Linux CUDA. CPU-only and Windows are not supported.
- **All models share memory via MLX unified memory.** FLUX.2 Klein Distilled + Base (txt2img and edit variants) + SmolVLM in bfloat16. All loaded lazily via `@st.cache_resource`.
- **Streamlit widget state cannot be modified after instantiation.** Use `on_click` callbacks to set `st.session_state` keys for widgets, not direct assignment after the widget is created.
