# Simplify FLUX.2 Klein Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify `streamlit_app.py` by splitting the UI into explicit Generate/Edit task modes, renaming Distilled/Base to Fast/Quality, and removing example buttons and the "Enhance Prompt" button path.

**Architecture:** Keep the single-file layout. Introduce a top-level task-mode pill (`Generate`/`Edit`) that controls which rendering branch executes — Generate shows a full-width prompt, Edit shows prompt + uploader side-by-side. Rename `MODE_DEFAULTS` keys from `"Distilled (4 steps)"`/`"Base (50 steps)"` to `"Fast"`/`"Quality"`. Remove the example-buttons feature and the explicit "Enhance Prompt" button, keeping only the auto-enhance checkbox (default off).

**Tech Stack:** Python 3.12+, Streamlit, mflux, mlx-vlm, Apple Silicon (MLX), Ruff, ty, pytest.

---

## File Structure

- **Modify:** `streamlit_app.py` — all UI and helper changes.
- **Modify:** `tests/test_streamlit_app.py` — update mode-key assertions, delete `TestExamples`, adjust `TestResolvePrompt`, remove/rewrite `TestClearEnhancement`, update `TestStreamlitApp`, add new tests for task-mode branching and disabled-Run behavior.
- **Delete:** `examples/bird.webp`, `examples/cat.webp`, `examples/person.webp`, and the empty `examples/` directory.

Reference spec: `docs/superpowers/specs/2026-04-13-simplify-app-design.md`.

---

## Task 1: Rename `MODE_DEFAULTS` keys to `Fast` / `Quality`

**Files:**
- Modify: `streamlit_app.py` — `MODE_DEFAULTS`, `MODELS`, `EDIT_MODELS`, `infer()` default, all `mode` string references, the `st.pills` in the UI.
- Modify: `tests/test_streamlit_app.py` — `TestConstants`, `TestModelLoading`, `TestInfer`, `TestStreamlitApp` — any test that names the old mode keys.

- [ ] **Step 1: Update test assertions to use new mode keys**

In `tests/test_streamlit_app.py`, find every occurrence of the literal strings `"Distilled (4 steps)"` and `"Base (50 steps)"` and replace them with `"Fast"` and `"Quality"` respectively. Use the Grep tool to find them first, then update each occurrence.

- [ ] **Step 2: Run tests to confirm they fail**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: Multiple failures — tests assert `"Fast"` / `"Quality"` but `MODE_DEFAULTS` still contains `"Distilled (4 steps)"` / `"Base (50 steps)"`.

- [ ] **Step 3: Update `MODE_DEFAULTS` in `streamlit_app.py`**

Replace the current definition:

```python
MODE_DEFAULTS = {
    "Distilled (4 steps)": {"steps": 4, "cfg": 1.0},
    "Base (50 steps)": {"steps": 50, "cfg": 4.0},
}
```

with:

```python
MODE_DEFAULTS = {
    "Fast": {"steps": 4, "cfg": 1.0},
    "Quality": {"steps": 50, "cfg": 4.0},
}
```

- [ ] **Step 4: Update `MODELS` and `EDIT_MODELS` dicts**

Replace:

```python
MODELS = {
    "Distilled (4 steps)": _get_model_distilled,
    "Base (50 steps)": _get_model_base,
}
```

with:

```python
MODELS = {
    "Fast": _get_model_distilled,
    "Quality": _get_model_base,
}
```

And replace:

```python
EDIT_MODELS = {
    "Distilled (4 steps)": _get_edit_model_distilled,
    "Base (50 steps)": _get_edit_model_base,
}
```

with:

```python
EDIT_MODELS = {
    "Fast": _get_edit_model_distilled,
    "Quality": _get_edit_model_base,
}
```

- [ ] **Step 5: Update `infer()` default parameter**

Change:

```python
def infer(
    prompt,
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=None,
    num_inference_steps=None,
    mode="Distilled (4 steps)",
    image_list=None,
    progress_callback=None,
):
```

to use `mode="Fast"`:

```python
def infer(
    prompt,
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=None,
    num_inference_steps=None,
    mode="Fast",
    image_list=None,
    progress_callback=None,
):
```

- [ ] **Step 6: Update the UI `st.pills` call and its fallback**

Replace:

```python
    mode = st.pills(
        "Mode",
        options=["Distilled (4 steps)", "Base (50 steps)"],
        default="Distilled (4 steps)",
        key="mode_pills",
    )

    if mode is None:
        mode = "Distilled (4 steps)"
```

with:

```python
    mode = st.pills(
        "Mode",
        options=["Fast", "Quality"],
        default="Fast",
        key="mode_pills",
    )

    if mode is None:
        mode = "Fast"
```

- [ ] **Step 7: Update `_distilled_defaults` reference if present**

There may be a leftover reference `_distilled_defaults = MODE_DEFAULTS["Distilled (4 steps)"]` in the Advanced Settings block. The earlier slider fix already removed this, but double-check — if present, delete it. It is no longer used.

- [ ] **Step 8: Run tests to confirm they now pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests pass (or fail only in `TestExamples` / example-related tests, which we will address in Task 2).

- [ ] **Step 9: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "refactor: rename MODE_DEFAULTS keys from Distilled/Base to Fast/Quality"
```

---

## Task 2: Remove example buttons, `EXAMPLES` list, and bundled images

**Files:**
- Modify: `streamlit_app.py` — remove `Example` TypedDict, `EXAMPLES`, `_select_example`, example-button block, `example_images` session-state handling, `TypedDict` import.
- Modify: `tests/test_streamlit_app.py` — delete `TestExamples`.
- Delete: `examples/bird.webp`, `examples/cat.webp`, `examples/person.webp`, and the `examples/` directory.

- [ ] **Step 1: Delete `TestExamples` from the test file**

In `tests/test_streamlit_app.py`, find the `class TestExamples:` block (near line 1036) and delete the entire class and its methods.

- [ ] **Step 2: Run tests to confirm they still pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All remaining tests pass. `TestExamples` is no longer listed.

- [ ] **Step 3: Remove `Example` TypedDict and `EXAMPLES` list from `streamlit_app.py`**

Delete the entire block from the `class Example(TypedDict):` declaration through the closing bracket of the `EXAMPLES: list[Example] = [...]` list. Also delete the `TypedDict` import — change `from typing import TypedDict` (if it is the only `typing` import, remove the line; otherwise remove just `TypedDict` from the import).

- [ ] **Step 4: Remove `_select_example()` callback inside `if __name__ == "__main__"`**

Delete the inner `def _select_example(example):` block (four lines plus the `_clear_enhancement()` call).

- [ ] **Step 5: Remove the example-buttons block at the bottom of the UI**

Delete the final UI block — `st.divider()`, `st.subheader("Examples")`, the `example_cols = st.columns(len(EXAMPLES))` loop, and all `st.button(...)` calls inside it.

- [ ] **Step 6: Remove `example_images` session-state handling**

Find the `elif "example_images" in st.session_state:` branch in the image-collection block and delete it. Also delete the `if "example_images" in st.session_state and not uploaded_files:` image-preview block (the `st.image(st.session_state.example_images, width=150)` call). Also delete the `st.session_state.pop("example_images", None)` call in the `if uploaded_files:` branch.

After this step, no reference to `example_images` should remain anywhere in `streamlit_app.py`.

- [ ] **Step 7: Remove `_clear_enhancement()`'s and `last_prompt`'s `example_images` pop**

In the block:

```python
    if prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt
        _clear_enhancement()
        st.session_state.pop("example_images", None)
```

Remove only the final `st.session_state.pop("example_images", None)` line. Leave `_clear_enhancement()` for now — it is addressed in Task 3.

- [ ] **Step 8: Delete bundled example images and the directory**

Run:

```bash
rm examples/bird.webp examples/cat.webp examples/person.webp
rmdir examples
```

- [ ] **Step 9: Run the full check suite**

Run: `uv run ruff check . && uv run ty check . && uv run pytest`
Expected: All pass.

- [ ] **Step 10: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py examples
git commit -m "refactor: remove example buttons and bundled example images"
```

---

## Task 3: Remove "Enhance Prompt" button and simplify helpers

**Files:**
- Modify: `streamlit_app.py` — remove button UI and conditional enhanced-prompt text area, simplify `_resolve_prompt()` signature, remove `_clear_enhancement()`.
- Modify: `tests/test_streamlit_app.py` — update `TestResolvePrompt` to drop `already_enhanced`, delete `TestClearEnhancement`.

- [ ] **Step 1: Update `TestResolvePrompt` to use the new signature**

In `tests/test_streamlit_app.py`, update every call to `_resolve_prompt(...)` so it passes exactly three positional args: `prompt`, `image_list`, `auto_enhance` (the `already_enhanced` parameter is being removed). Delete any test method whose sole purpose was to verify the `already_enhanced=True` early-return path.

- [ ] **Step 2: Delete `TestClearEnhancement`**

Remove the `class TestClearEnhancement:` block entirely — `_clear_enhancement()` is being deleted in this task.

- [ ] **Step 3: Run tests to confirm they fail**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: Failures in `TestResolvePrompt` tests (signature mismatch) — the code still defines the old signature.

- [ ] **Step 4: Simplify `_resolve_prompt()` in `streamlit_app.py`**

Replace:

```python
def _resolve_prompt(prompt, image_list, auto_enhance, already_enhanced):
    """Resolve the final prompt, optionally auto-enhancing via the VLM."""
    if auto_enhance and not already_enhanced:
        return upsample_prompt(prompt, image_list=image_list), True
    return prompt, False
```

with:

```python
def _resolve_prompt(prompt, image_list, auto_enhance):
    """Resolve the final prompt, optionally auto-enhancing via the VLM."""
    if auto_enhance:
        return upsample_prompt(prompt, image_list=image_list), True
    return prompt, False
```

- [ ] **Step 5: Remove `_clear_enhancement()` function**

Delete the entire:

```python
def _clear_enhancement():
    """Remove all enhancement-related session state."""
    for key in ("enhanced_prompt", "enhanced_prompt_area", "auto_enhanced_prompt"):
        st.session_state.pop(key, None)
```

- [ ] **Step 6: Replace the remaining caller of `_clear_enhancement()`**

In the block:

```python
    if prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt
        _clear_enhancement()
```

Replace `_clear_enhancement()` with a direct pop:

```python
    if prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt
        st.session_state.pop("auto_enhanced_prompt", None)
```

- [ ] **Step 7: Remove the "Enhance Prompt" button and conditional text area**

Delete this block in the UI:

```python
    if st.button("Enhance Prompt"):
        with st.spinner("Enhancing prompt..."):
            enhanced = upsample_prompt(prompt, image_list=image_list)
        st.session_state.enhanced_prompt = enhanced

    if "enhanced_prompt" in st.session_state:
        final_prompt = st.text_area(
            "Enhanced Prompt",
            value=st.session_state.enhanced_prompt,
            key="enhanced_prompt_area",
        )
    else:
        final_prompt = prompt
```

Replace it with the single assignment:

```python
    final_prompt = prompt
```

- [ ] **Step 8: Update the `Run` handler to use the new `_resolve_prompt` signature**

In the `if st.button("Run", type="primary"):` block, replace:

```python
    if st.button("Run", type="primary"):
        st.session_state.pop("auto_enhanced_prompt", None)
        already_enhanced = "enhanced_prompt" in st.session_state
        run_prompt, was_auto_enhanced = _resolve_prompt(
            final_prompt, image_list, auto_enhance, already_enhanced
        )
```

with:

```python
    if st.button("Run", type="primary"):
        st.session_state.pop("auto_enhanced_prompt", None)
        run_prompt, was_auto_enhanced = _resolve_prompt(
            final_prompt, image_list, auto_enhance
        )
```

- [ ] **Step 9: Run the full check suite**

Run: `uv run ruff check . && uv run ty check . && uv run pytest`
Expected: All pass.

- [ ] **Step 10: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "refactor: remove Enhance Prompt button and simplify resolve_prompt"
```

---

## Task 4: Rename UI strings (title, caption, checkbox, expander)

**Files:**
- Modify: `streamlit_app.py` — `st.set_page_config`, `st.title`, `st.caption`, checkbox label, expander label.
- Modify: `tests/test_streamlit_app.py` — update `TestStreamlitApp` assertions that reference old strings.

- [ ] **Step 1: Search for test references to old strings**

Run: grep through `tests/test_streamlit_app.py` for `"AI Image Studio"`, `"Auto-enhance prompt"`, `"Advanced Settings"`. Note each occurrence.

- [ ] **Step 2: Update those test references to the new strings**

Replace `"AI Image Studio"` → `"FLUX.2 Klein Pipeline"`, `"Auto-enhance prompt"` → `"Enhance prompt"`, `"Advanced Settings"` → `"Settings"` in the test file.

- [ ] **Step 3: Update page title and remove caption**

Replace:

```python
    st.set_page_config(page_title="AI Image Studio", layout="centered")

    st.title("AI Image Studio")
    st.caption("Powered by [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)")
```

with:

```python
    st.set_page_config(page_title="FLUX.2 Klein Pipeline", layout="centered")

    st.title("FLUX.2 Klein Pipeline")
```

- [ ] **Step 4: Update the auto-enhance checkbox label**

Replace:

```python
        auto_enhance = st.checkbox(
            "Auto-enhance prompt",
            value=False,
            help="Automatically enhance the prompt using the VLM before generating",
            key="auto_enhance_checkbox",
        )
```

with:

```python
        auto_enhance = st.checkbox(
            "Enhance prompt",
            value=False,
            help="Enhance the prompt using the VLM before generating",
            key="auto_enhance_checkbox",
        )
```

- [ ] **Step 5: Update the expander label**

Replace:

```python
    with st.expander("Advanced Settings"):
```

with:

```python
    with st.expander("Settings"):
```

- [ ] **Step 6: Run the full check suite**

Run: `uv run ruff check . && uv run ty check . && uv run pytest`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "ui: rename title, checkbox label, and Settings expander"
```

---

## Task 5: Add task-mode pills (Generate/Edit, Fast/Quality) on one row with collapsed labels

**Files:**
- Modify: `streamlit_app.py` — add task-mode pills, restructure the top row, collapse labels on all pills.
- Modify: `tests/test_streamlit_app.py` — add a test that `task_mode` defaults to `"Generate"` on first load.

- [ ] **Step 1: Add the task-mode default-on-first-load test**

In `tests/test_streamlit_app.py`, inside `TestStreamlitApp`, add a new test method that uses `AppTest` to run the app and asserts that the first `st.pills` widget (task pill) has default value `"Generate"`.

Example (adjust helper usage to match existing patterns in the file):

```python
def test_task_mode_defaults_to_generate(self):
    from streamlit.testing.v1 import AppTest

    mock_model = _make_mock_model()
    mock_vlm = _make_mock_vlm()
    with _reload_app(mock_model, mock_vlm=mock_vlm):
        at = AppTest.from_file("streamlit_app.py").run(timeout=10)
        assert at.pills(key="task_pills").value == "Generate"
```

- [ ] **Step 2: Run the new test to confirm it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestStreamlitApp::test_task_mode_defaults_to_generate -v`
Expected: FAIL — `task_pills` key does not exist yet.

- [ ] **Step 3: Replace the single mode-pill row with a two-column row**

Find the current mode-pill block:

```python
    mode = st.pills(
        "Mode",
        options=["Fast", "Quality"],
        default="Fast",
        key="mode_pills",
    )

    if mode is None:
        mode = "Fast"
```

Replace it with two pills side-by-side, both with `label_visibility="collapsed"`, and add the task-mode pill first:

```python
    col_task, col_quality = st.columns(2)
    with col_task:
        task_mode = st.pills(
            "Task",
            options=["Generate", "Edit"],
            default="Generate",
            key="task_pills",
            label_visibility="collapsed",
        )
    with col_quality:
        mode = st.pills(
            "Quality",
            options=["Fast", "Quality"],
            default="Fast",
            key="mode_pills",
            label_visibility="collapsed",
        )

    if task_mode is None:
        task_mode = "Generate"
    if mode is None:
        mode = "Fast"
```

- [ ] **Step 4: Run the new test and the full suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: `test_task_mode_defaults_to_generate` passes, all others still pass.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "ui: add Generate/Edit task pill alongside Fast/Quality"
```

---

## Task 6: Branch UI by task mode — Generate vs Edit rendering paths

**Files:**
- Modify: `streamlit_app.py` — split prompt + uploader rendering into two branches, add mode-specific placeholders, collapse prompt/uploader labels, disable Run in Edit mode with no images.
- Modify: `tests/test_streamlit_app.py` — add tests that Generate mode does not render the uploader, Edit mode does, and Run is disabled in Edit mode with no images.

- [ ] **Step 1: Add a test that the uploader is absent in Generate mode**

In `tests/test_streamlit_app.py`, inside `TestStreamlitApp`, add:

```python
def test_generate_mode_hides_uploader(self):
    from streamlit.testing.v1 import AppTest

    mock_model = _make_mock_model()
    mock_vlm = _make_mock_vlm()
    with _reload_app(mock_model, mock_vlm=mock_vlm):
        at = AppTest.from_file("streamlit_app.py").run(timeout=10)
        # Task pill defaults to Generate — no file_uploader should be rendered
        assert len(at.get("file_uploader")) == 0
```

- [ ] **Step 2: Add a test that the uploader appears in Edit mode**

```python
def test_edit_mode_shows_uploader(self):
    from streamlit.testing.v1 import AppTest

    mock_model = _make_mock_model()
    mock_vlm = _make_mock_vlm()
    with _reload_app(mock_model, mock_vlm=mock_vlm):
        at = AppTest.from_file("streamlit_app.py").run(timeout=10)
        at.pills(key="task_pills").set_value("Edit").run(timeout=10)
        assert len(at.get("file_uploader")) == 1
```

- [ ] **Step 3: Add a test that Run is disabled in Edit mode with no images**

```python
def test_run_disabled_in_edit_mode_without_images(self):
    from streamlit.testing.v1 import AppTest

    mock_model = _make_mock_model()
    mock_vlm = _make_mock_vlm()
    with _reload_app(mock_model, mock_vlm=mock_vlm):
        at = AppTest.from_file("streamlit_app.py").run(timeout=10)
        at.pills(key="task_pills").set_value("Edit").run(timeout=10)
        run_button = [b for b in at.button if b.label == "Run"][0]
        assert run_button.disabled is True
```

- [ ] **Step 4: Run the three new tests to confirm they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestStreamlitApp -v`
Expected: The three new tests fail.

- [ ] **Step 5: Split the prompt/uploader block into two rendering branches**

Find the current block:

```python
    col_prompt, col_images = st.columns(2)
    with col_prompt:
        prompt = st.text_area(
            "Prompt", placeholder="Enter your prompt", key="prompt_input", height=160
        )
        auto_enhance = st.checkbox(
            "Enhance prompt",
            value=False,
            help="Enhance the prompt using the VLM before generating",
            key="auto_enhance_checkbox",
        )
    with col_images:
        uploaded_files = st.file_uploader(
            "Input images (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
```

Replace with two rendering branches — Generate mode: full-width prompt, no uploader; Edit mode: prompt + uploader side-by-side. Both branches render the `Enhance prompt` checkbox below the prompt area.

```python
    if task_mode == "Generate":
        prompt = st.text_area(
            "Prompt",
            placeholder="Describe the image…",
            key="prompt_input",
            height=160,
            label_visibility="collapsed",
        )
        uploaded_files = None
    else:
        col_prompt, col_images = st.columns(2)
        with col_prompt:
            prompt = st.text_area(
                "Prompt",
                placeholder="Describe the edit…",
                key="prompt_input",
                height=160,
                label_visibility="collapsed",
            )
        with col_images:
            uploaded_files = st.file_uploader(
                "Input images",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

    auto_enhance = st.checkbox(
        "Enhance prompt",
        value=False,
        help="Enhance the prompt using the VLM before generating",
        key="auto_enhance_checkbox",
    )
```

- [ ] **Step 6: Verify the image-collection block is clean**

Confirm that the image-collection block reads:

```python
    image_list = None
    if uploaded_files:
        image_list = [Image.open(f) for f in uploaded_files]
```

with no `elif "example_images" in st.session_state:` branch. Task 2 should have removed that fallback; if any trace remains, delete it now.

Note on upload clearing: Streamlit's `file_uploader` widget is rendered only inside the Edit branch. When the user switches to Generate mode, the widget leaves the tree, and because we assign no `key` to it, switching back to Edit creates a fresh (empty) uploader. This satisfies the spec's "switching task mode clears uploads" requirement without explicit session-state pops.

- [ ] **Step 7: Disable the Run button in Edit mode with no images**

Replace:

```python
    if st.button("Run", type="primary"):
```

with:

```python
    run_disabled = task_mode == "Edit" and not image_list
    if st.button("Run", type="primary", disabled=run_disabled):
```

- [ ] **Step 8: Run the new tests and the full suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests pass, including the three added in Steps 1–3.

- [ ] **Step 9: Run formatters and type checker**

Run: `uv run ruff check . && uv run ruff format . && uv run ty check .`
Expected: All pass.

- [ ] **Step 10: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "ui: split UI into Generate and Edit task-mode branches"
```

---

## Task 7: Manual smoke test

**Files:** none — manual verification only.

- [ ] **Step 1: Start the app**

Run: `uv run streamlit run streamlit_app.py`
Open the browser tab Streamlit prints.

- [ ] **Step 2: Verify Generate mode (default)**

- Title reads `FLUX.2 Klein Pipeline`; no caption underneath.
- Two pills on one row — Task (`Generate` / `Edit`) on the left, Quality (`Fast` / `Quality`) on the right.
- Prompt box takes the full width, placeholder says `Describe the image…`.
- `Enhance prompt` checkbox is present and unchecked by default.
- `Settings` expander is collapsed by default; expanding shows seed, randomize, width, height, guidance, steps.
- Type a prompt, click **Run** — a result appears, progress bar advances then clears.

- [ ] **Step 3: Verify Edit mode**

- Click the **Edit** pill.
- Prompt box now occupies the left half; a file uploader occupies the right half. Prompt placeholder says `Describe the edit…`.
- **Run** button is disabled.
- Upload an image — width/height sliders in Settings auto-match the aspect ratio; **Run** becomes enabled.
- Type a prompt, click **Run** — result appears.

- [ ] **Step 4: Verify `Enhance prompt` path**

- Check the `Enhance prompt` box in either mode, type a prompt, click **Run**.
- After generation, an info banner shows the enhanced prompt the VLM produced.

- [ ] **Step 5: Verify Quality (Base) mode switches defaults**

- Click the **Quality** pill. Open **Settings** — guidance scale and steps now reflect Base defaults (cfg 4.0, steps 50).

- [ ] **Step 6: Verify no stale example section**

- Confirm nothing below the result section — no `Examples` heading, no sample-prompt buttons.

- [ ] **Step 7: Stop the app**

Press Ctrl+C in the terminal.

- [ ] **Step 8: If all smoke-test points pass, there is nothing to commit — the plan is complete.**

If any smoke-test point fails, stop and diagnose the specific failure before committing further changes.

---

## Verification Summary

Final checks before declaring the plan complete:

- `uv run ruff check .` — clean
- `uv run ruff format --check .` — clean
- `uv run ty check .` — clean
- `uv run pytest` — all green
- Manual smoke test (Task 7) — all points pass
