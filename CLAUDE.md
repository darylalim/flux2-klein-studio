# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX.2 Klein Studio is a single-file Streamlit web app that generates and edits images with the Black Forest Labs [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model on Apple Silicon with MLX. Built on [mflux](https://github.com/filipstrand/mflux) (diffusion) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (VLM). Unified generation and editing: text-to-image by default, switching to multi-image editing automatically when input images are supplied. Two speed/quality modes, shown in the UI as Distilled (4 steps) and Base (50 steps) (internally `Fast`/`Quality`). Optional vision-aware prompt upsampling via [SmolVLM-500M-Instruct](https://huggingface.co/mlx-community/SmolVLM-500M-Instruct-bf16) — the VLM can see uploaded images when enhancing editing prompts.

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

Application code lives in `streamlit_app.py`, structured in four sections (with the test suite in `tests/` and bundled example images in `examples/`):

1. **Model initialization** — Two model classes from mflux: `Flux2Klein` for text-to-image and `Flux2KleinEdit` for multi-image editing. MLX manages Apple Silicon unified memory automatically. Four `@st.cache_resource`-cached getters: `_get_model_distilled()`, `_get_model_base()`, `_get_edit_model_distilled()`, `_get_edit_model_base()`. `MODELS` maps mode names to txt2img getters; `EDIT_MODELS` maps mode names to edit getters. `MODE_DEFAULTS` holds per-mode defaults (`"Fast"`: 4 steps, CFG 1.0; `"Quality"`: 50 steps, CFG 4.0).
2. **Prompt upsampling** — `_get_vlm()` loads SmolVLM-500M-Instruct-bf16 via `mlx_vlm.load()`, cached with `@st.cache_resource`, returns a `(model, processor, config)` triple. Two system prompts: `UPSAMPLE_PROMPT_TEXT_ONLY` (text-to-image, capped at 120 words) and `UPSAMPLE_PROMPT_WITH_IMAGES` (image editing, concrete language, preserve unchanged elements). `upsample_prompt(prompt, image_list: list | None = None)` selects the system prompt based on whether images are provided, formats messages via `mlx_vlm.prompt_utils.apply_chat_template`, generates via `mlx_vlm.generate()`, and strips `<end_of_utterance>` tokens from the output. Loaded lazily on first use. `_resolve_prompt(prompt, image_list, auto_enhance)` wraps the auto-enhance decision: enhances only when `auto_enhance` is true, returns `(prompt, was_enhanced)` tuple.
3. **Inference** — `infer()` takes prompt, seed, dimensions (256–1024px), mode, optional `image_list`, and optional `progress_callback`. Defaults resolve from `MODE_DEFAULTS[mode]`. Selects `Flux2KleinEdit` from `EDIT_MODELS` when `image_list` is provided, otherwise `Flux2Klein` from `MODELS`. Calls `model.generate_image()` which returns a `GeneratedImage`; the PIL Image is extracted via `.image`. When `progress_callback` is provided, registers an `InLoopCallback` via `model.callbacks.register()` for per-step progress reporting, then removes it from `model.callbacks.in_loop` in a `finally` block so reporters do not accumulate on cached models across Runs. `_dimensions_from_images()` calculates output dimensions from the first uploaded image's aspect ratio (larger side 1024, rounds to 32, clamps 256–1024).
4. **UI** — Behind `if __name__ == "__main__"` (`layout="wide"`, modeled on the FLUX.2 [klein] Gradio space): `st.title` header (`FLUX.2 Klein Studio`) plus a one-line `st.markdown` description with `[model]`/`[blog]` links. A two-column studio layout via `st.columns([2, 3])` — **left** holds all controls, **right** holds the output (one reserved `st.empty()` slot, `result_slot`; during a run it renders a per-step `st.progress` bar inside an `st.status` within a bordered `st.container`, and otherwise shows the result image or a native bordered placeholder container with theme-aware `:gray[...]` text — no raw HTML). Left column, top to bottom: a prompt `st.text_input` (placeholder `Enter your prompt`, key `prompt_input`) + a `Run` button inline via `st.container(horizontal=True)` (the input stretches, the button hugs its icon+label content so it never wraps); an `Input image(s) (optional)` expander (`icon=":material/image:"`) wrapping the file uploader (`accept_multiple_files=True`, label collapsed) — there is **no Task toggle**, uploading any image routes `infer()` to `Flux2KleinEdit` automatically; a `Mode` `st.segmented_control` (key `mode_radio`, `required=True` so the active segment can't be deselected — otherwise a deselect returns `None`, blanks the control, and silently resets the steps/guidance sliders; options `Distilled (4 steps)`/`Base (50 steps)` from `MODE_LABEL_LIST`, mapped back to internal `Fast`/`Quality` via `LABEL_TO_MODE`); an `Advanced settings` expander (collapsed by default, `icon=":material/tune:"`) holding a `Prompt upsampling` toggle (key `auto_enhance_toggle`) + `st.caption`, a `Randomize seed` toggle + a `Seed` `st.number_input` (disabled while Randomize is on), Width/Height sliders, and Number of inference steps/Guidance scale sliders; an `Examples` section of clickable `EXAMPLE_PROMPTS` buttons (truncated labels via `_truncate`, full prompt on hover) that fill the prompt box via `_set_example_prompt` (sets `st.session_state.prompt_input`, clears any example images); and an `Editing examples` section from `EDIT_EXAMPLES` (`(prompt, [image paths])` tuples, verbatim from the Gradio space) — each row is a prompt button + thumbnail `st.image`, and clicking calls `_load_edit_example` to set the prompt and store the bundled image paths in `st.session_state.example_images`. Because `st.file_uploader` cannot be set programmatically, `image_list` falls back to `example_images` (opened with `Image.open`) when no manual upload is present, so editing examples route through `Flux2KleinEdit` exactly like uploads; a manual upload clears `example_images`, and a `Clear example images` button (in the uploader expander, which auto-expands once when an example is loaded via a one-shot `expand_input_once` flag) removes them. Example images are bundled under `examples/` (`woman1.webp`, `cat_window.webp`, `bird.webp`), resolved via `_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"`. The Run button is always enabled (editing is implicit), but an empty request (no prompt and no image) is guarded with a warning instead of running inference; corrupt uploads (`OSError`) raise an `st.warning`, and an `infer()` failure sets the `st.status` to `state="error"` and shows an `st.error` in the controls column (so it survives the output slot being re-rendered) rather than crashing the page. Width/height sliders auto-update to match the first input image's aspect ratio (uploaded or example); guidance scale and steps update when the mode changes. Theming is native via `.streamlit/config.toml` (matched `[theme.light]`/`[theme.dark]` palettes, Inter font, `baseRadius`; no custom CSS — the in-app appearance switcher stays available); `st.set_page_config` sets `page_icon=":material/auto_awesome:"`, and Material Symbols icons appear on the `Run` button (`play_arrow`), the two expanders (`image`/`tune`), and the enhanced-prompt `st.info` (`auto_awesome`).

## Commands

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

```bash
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format
uv run ruff format --check .     # Check formatting only
uv run ty check .                # Type check
uv run pytest                    # Run all tests
uv run pytest tests/test_streamlit_app.py  # Run a single test file
```

Ruff is configured in `pyproject.toml` (`[tool.ruff.lint]`: rule sets `E`, `F`, `I`, `UP`, `B`, `C4`, `SIM`); line length (`E501`) is delegated to the formatter, not linted.

## Gotchas

### mflux / FLUX.2 Klein

- **Two model classes: `Flux2Klein` (text-to-image) and `Flux2KleinEdit` (multi-image editing).** Both created via `model_config=ModelConfig.flux2_klein_4b()`. No repo ID strings or token parameters.
- **`Flux2Klein.generate_image()` takes `image_path` (singular, `Path | str | None`) for img2img.** `Flux2KleinEdit.generate_image()` takes `image_paths` (plural, `list[Path | str] | None`) for multi-image editing.
- **`generate_image()` returns a `GeneratedImage` wrapper, not a PIL Image.** Access the PIL Image via `.image`.
- **MLX manages device placement automatically.** Apple Silicon unified memory is used directly.
- **Progress callbacks use `model.callbacks.register()`.** Register an object with a `call_in_loop(self, t, seed, prompt, latents, config, time_steps)` method. `CallbackRegistry` has no `unregister()` — callers must pop the instance from `model.callbacks.in_loop` themselves, or reporters accumulate on `@st.cache_resource`-cached models and fire on every subsequent Run. mflux invokes `call_in_loop` entirely by keyword, so the six parameter names must match exactly — the unused ones (`seed`, `prompt`, `latents`, `time_steps`) can't be renamed or dropped.
- **The guidance parameter is named `guidance`**, not `guidance_scale`.
- **FLUX.2 Klein does not support negative prompts.**
- **Quality mode uses different defaults than Fast mode.** Quality (base model): 50 steps, CFG 4.0. Fast (distilled model): 4 steps, CFG 1.0. The underlying model variants are named distilled and base; `"Fast"`/`"Quality"` are the internal `MODE_DEFAULTS`/`MODELS`/`EDIT_MODELS` keys, surfaced in the UI as `Distilled (4 steps)`/`Base (50 steps)` via the `MODE_LABELS` display mapping.

### mlx-vlm / SmolVLM

- **Use `mlx_vlm.load()` to get `(model, processor)` and `mlx_vlm.utils.load_config()` for config.** Config is required by `apply_chat_template`.
- **`mlx_vlm.generate()` handles tokenization and decoding internally.** Access result via `result.text`. SmolVLM appends `<end_of_utterance>` tokens that must be stripped from output.
- **`apply_chat_template` takes `num_images` instead of embedding image tokens in messages.** Pass images as a flat list to the `image` parameter of `generate()`.
- **`temperature > 0` implies sampling.** Use `temperature=0.0` for greedy decoding.
- **Use `max_tokens`** to control output length.
- **mlx-vlm's type hints are loose.** `generate()` annotates `image` as `str | list[str]` (no `Optional`, no PIL) though it accepts `None`/PIL Images at runtime — hence the `# ty: ignore[invalid-argument-type]` on the `image=` argument. `apply_chat_template` is typed as a broad union but returns a `str` for SmolVLM, so its result is wrapped in `cast(str, ...)`.

### General

- **Apple Silicon is the primary target.** The app uses MLX (mflux + mlx-vlm) which requires Apple Silicon or Linux CUDA. CPU-only and Windows are not supported.
- **All models share memory via MLX unified memory.** FLUX.2 Klein Distilled + Base (txt2img and edit variants) + SmolVLM in bfloat16. All loaded lazily via `@st.cache_resource`.
- **Streamlit widget state cannot be modified after instantiation.** Use `on_click` callbacks to set `st.session_state` keys for widgets, not direct assignment after the widget is created.
- **The output `st.empty()` slot (`result_slot`) is re-rendered on every run.** The bottom render block always rewrites it, overwriting anything drawn into it mid-run (e.g. the progress `st.status`), so transient messages like a generation error must go in the controls column to persist.
