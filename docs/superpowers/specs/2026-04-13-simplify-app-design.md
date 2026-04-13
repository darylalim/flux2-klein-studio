# Simplify FLUX.2 Klein Pipeline — Design Spec

**Date:** 2026-04-13
**Status:** Approved for implementation planning

## Goal

Simplify both the UI and the code of `streamlit_app.py` without losing core functionality. Priorities, in order:

1. Minimize clicks to get from prompt to result.
2. Reduce visual noise — fewer controls visible at once.
3. Clearer hierarchy — primary actions (prompt, Run) stand out; secondary controls recede.

## Scope

### In scope

- Restructure the UI into two explicit task modes: **Generate** (text-to-image) and **Edit** (multi-image editing).
- Rename the speed/quality selector from Distilled/Base jargon to **Fast / Quality**.
- Place the Generate/Edit pill and the Fast/Quality pill side-by-side on a single row at the top.
- Remove example buttons and all supporting code/assets.
- Remove the "Enhance Prompt" button and its conditional "Enhanced Prompt" text area; keep only the auto-enhance checkbox (default off).
- Strip decorative labels ("Task", "Quality", "Prompt", "Input images") that restate widget purpose.
- Rename "Auto-enhance prompt" → "Enhance prompt", "Advanced Settings" → "Settings".
- Change page title from "AI Image Studio" to "FLUX.2 Klein Pipeline"; remove the "Powered by…" caption.
- Split the UI into two rendering branches driven by task mode.

### Out of scope (unchanged)

- Both model variants remain (renamed, but both kept).
- VLM prompt upsampling remains; only the explicit button path is removed.
- Advanced Settings expander contents (seed, randomize, width/height, guidance, steps) are unchanged.
- Per-step progress bar is unchanged.
- `infer()`, `upsample_prompt()`, `_dimensions_from_images()`, `_get_vlm()`, and the four cached model getters are unchanged internally.
- Single-file architecture — everything remains in `streamlit_app.py`. No helper extraction, no new modules.

## UI Design

### Top-to-bottom control order (both modes)

1. Title: `FLUX.2 Klein Pipeline`
2. Two pills side-by-side in `st.columns(2)`:
   - Task pill: `Generate` / `Edit` (default: `Generate`)
   - Quality pill: `Fast` / `Quality` (default: `Fast`)
   - Both pills use `label_visibility="collapsed"`.
3. Prompt area:
   - **Generate mode:** full-width `st.text_area` with placeholder `Describe the image…`.
   - **Edit mode:** prompt area (left) and file uploader (right) in `st.columns(2)`. Prompt placeholder: `Describe the edit…`.
   - Both use `label_visibility="collapsed"`.
4. Checkbox: `Enhance prompt` (default unchecked).
5. Expander: `Settings` (seed, randomize seed, width, height, guidance scale, inference steps).
6. Primary button: `Run`.
7. Auto-enhanced prompt info banner (when applicable).
8. Result image and seed caption (when a result exists).

No examples section. No separate "Enhance Prompt" button. No conditional "Enhanced Prompt" text area.

### Mode-specific behaviors

- **Switching task mode clears uploads.** If the user switches from Edit to Generate, any uploaded images are discarded; re-entering Edit mode shows an empty uploader. This is the natural consequence of conditionally rendering the uploader and is preferred to hidden persistence.
- **Run is disabled in Edit mode with no images.** The Run button is rendered with `disabled=True` when `task_mode == "Edit"` and no images are uploaded. In Generate mode, Run is always enabled (subject to a non-empty prompt is not enforced — same as today).

## Code Changes

### Remove

- `Example` TypedDict (currently streamlit_app.py:56–59)
- `EXAMPLES` list (streamlit_app.py:62–108)
- `_select_example()` callback (streamlit_app.py:272–279)
- Examples column block at the bottom of the UI (streamlit_app.py:454–464)
- "Enhance Prompt" button and conditional "Enhanced Prompt" text area (streamlit_app.py:350–362)
- `already_enhanced` parameter from `_resolve_prompt()`; the signature becomes `_resolve_prompt(prompt, image_list, auto_enhance)` returning `(prompt, was_enhanced)`.
- `_clear_enhancement()` helper; the single remaining caller (currently streamlit_app.py:344, triggered when the prompt text changes) is replaced with a direct `st.session_state.pop("auto_enhanced_prompt", None)`.
- All `example_images` session-state handling and the fallback in the image-collection block.
- `examples/` directory and all three bundled `.webp` files.
- `from typing import TypedDict` import.

### Rename

- `MODE_DEFAULTS` keys: `"Distilled (4 steps)"` → `"Fast"`, `"Base (50 steps)"` → `"Quality"`.
- Propagate the new keys through `MODELS`, `EDIT_MODELS`, all `mode=...` defaults, and all `st.pills` options.

### Add

- Session-state key `task_mode` with values `"Generate"` / `"Edit"`, default `"Generate"`.
- Top row using `col_task, col_quality = st.columns(2)` with an `st.pills` in each.
- Single branching point: `if task_mode == "Generate":` renders the full-width prompt; the `else` branch renders the two-column prompt + uploader.
- Mode-specific prompt placeholders (`Describe the image…` / `Describe the edit…`).
- `label_visibility="collapsed"` on both pills, the prompt `text_area`, and the file uploader.

### Rename UI strings

- Page title → `FLUX.2 Klein Pipeline`; remove the caption.
- Checkbox label → `Enhance prompt` (default `value=False`).
- Expander label → `Settings`.

### Unchanged

- `infer()`, `upsample_prompt()`, `_dimensions_from_images()`.
- `_get_vlm()` and the four `_get_model_*()` / `_get_edit_model_*()` cached getters.
- Per-step progress bar behavior.
- Seed, width, height, guidance, steps sliders (internals and the earlier session-state warning fix stay in place).
- Result rendering.

## Testing Impact

### Tests to update (`tests/test_streamlit_app.py`)

- `TestConstants` — update `MODE_DEFAULTS` key assertions to `"Fast"` / `"Quality"`.
- `TestModelLoading` — update any mode-string keys used to look up model getters.
- `TestInfer` — flip all `mode=...` kwargs to the new names.
- `TestResolvePrompt` — remove the `already_enhanced` parameter from tests; drop any case that exercised it.
- `TestClearEnhancement` — rewrite to match the reduced helper, or delete if the helper is inlined.
- `TestStreamlitApp` — update `AppTest` interactions that reference removed widget keys (`enhanced_prompt_area`, example buttons, old mode labels).

### Tests to delete

- `TestExamples` — `EXAMPLES` no longer exists.

### Tests that stay as-is

- `TestDimensionsFromImages`
- `TestVLMInit`
- `TestUpsamplePrompt`

### New tests (recommended)

- Generate mode does not render the file uploader; Edit mode does (via `AppTest`).
- `task_mode` defaults to `"Generate"` on first load.
- Run button is disabled in Edit mode when no images are uploaded; enabled once an image is added.

## Verification Plan

After implementation:

1. `uv run ruff check .` and `uv run ruff format --check .`
2. `uv run ty check .`
3. `uv run pytest`
4. `uv run streamlit run streamlit_app.py` and manually exercise:
   - Generate mode: prompt → Run, with and without `Enhance prompt` checked.
   - Edit mode: upload one or more images, confirm dimensions auto-match aspect ratio, run with and without enhancement.
   - Toggle Fast ↔ Quality and confirm CFG + steps update.
   - Seed randomization, advanced-settings overrides.
   - Per-step progress bar advances during inference.

## Non-Goals

- No refactor into multiple modules or helper files; the single-file layout is intentional.
- No removal of either model variant.
- No removal of the VLM prompt upsampler.
- No changes to `infer()`, `upsample_prompt()`, `_dimensions_from_images()`, `_get_vlm()`, or any cached model getter.
- No new features beyond the explicit Generate/Edit split.
