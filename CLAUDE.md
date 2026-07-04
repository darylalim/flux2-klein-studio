# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX.2 Klein Studio is a single-file Streamlit application for generating and editing images using Black Forest Labs [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) on Apple Silicon with MLX. Built on [mflux](https://github.com/filipstrand/mflux) (diffusion) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (VLM). Unified generation and editing: text-to-image by default, switching to multi-image editing automatically when input images are supplied. Two speed/quality modes, shown in the UI as Distilled (4 steps) and Base (50 steps) (internally `Fast`/`Quality`). Optional vision-aware prompt upsampling via [SmolVLM-500M-Instruct](https://huggingface.co/mlx-community/SmolVLM-500M-Instruct-bf16) — the VLM can see uploaded images when enhancing editing prompts.

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

Application code lives in `streamlit_app.py`, structured in four sections (with the test suite in `tests/`, bundled example images in `examples/`, repo-shared Claude Code hooks in `.claude/`, the CI workflow in `.github/workflows/`, and the MIT `LICENSE` — mirrored by `license = "MIT"` in `pyproject.toml` and credited alongside the upstream FLUX.2/SmolVLM (Apache-2.0) and mflux/mlx-vlm (MIT) licenses in `README.md`'s License section, with `tests/test_license.py` locking all three declarations against drift; `README.md` mirrors this file's Overview/Setup/CI/Hooks content for a public audience — keep the two in sync):

1. **Model initialization** — Two model classes from mflux: `Flux2Klein` for text-to-image and `Flux2KleinEdit` for multi-image editing. MLX manages Apple Silicon unified memory automatically. Four `@st.cache_resource`-cached getters: `_get_model_distilled()`, `_get_model_base()`, `_get_edit_model_distilled()`, `_get_edit_model_base()`. `MODELS` maps mode names to txt2img getters; `EDIT_MODELS` maps mode names to edit getters. `MODE_DEFAULTS` holds per-mode defaults (`"Fast"`: 4 steps, CFG 1.0; `"Quality"`: 50 steps, CFG 4.0).
2. **Prompt upsampling** — `_get_vlm()` loads SmolVLM-500M-Instruct-bf16 via `mlx_vlm.load()`, cached with `@st.cache_resource`, returns a `(model, processor, config)` triple. Two system prompts: `UPSAMPLE_PROMPT_TEXT_ONLY` (text-to-image, capped at 120 words) and `UPSAMPLE_PROMPT_WITH_IMAGES` (image editing, concrete language, preserve unchanged elements). `upsample_prompt(prompt, image_list: list | None = None)` selects the system prompt based on whether images are provided, formats messages via `mlx_vlm.prompt_utils.apply_chat_template`, generates via `mlx_vlm.generate()` (sampling, not greedy: `temperature=0.7`, `top_p=0.9`, `max_tokens=150`, for varied enhancements), and strips `<end_of_utterance>` tokens from the output. Loaded lazily on first use. `_resolve_prompt(prompt, image_list, auto_enhance)` wraps the auto-enhance decision: enhances only when `auto_enhance` is true, returns `(prompt, was_enhanced)` tuple. Enhanced results are memoized in a bounded per-session `st.session_state._enhance_cache` (keyed on `(prompt, image set)`, oldest entry evicted once it exceeds 32) so re-Running an unchanged request does not re-invoke the VLM; the `auto_enhanced_prompt` key behind the enhanced-prompt `st.info` banner is dropped whenever the prompt text or the image set changes (and on each Run).
3. **Inference** — `infer()` takes prompt, seed, dimensions (256–1024px), mode, optional `image_list`, and optional `progress_callback`. Defaults resolve from `MODE_DEFAULTS[mode]`. Selects `Flux2KleinEdit` from `EDIT_MODELS` when `image_list` is provided, otherwise `Flux2Klein` from `MODELS`. Calls `model.generate_image()` which returns a `GeneratedImage`; the PIL Image is extracted via `.image`. When `progress_callback` is provided, registers a `_ProgressReporter` via `model.callbacks.register()` for per-step progress reporting, then removes it from `model.callbacks.in_loop` in a `finally` block so reporters do not accumulate on cached models across Runs. `_dimensions_from_images()` calculates output dimensions from the first uploaded image's aspect ratio (larger side 1024, rounds to 32, clamps 256–1024).
4. **UI** — Behind `if __name__ == "__main__"` (`layout="wide"`, modeled on the FLUX.2 [klein] Gradio space): a lone `st.title` header (`FLUX.2 Klein Studio`), no description caption. Two-column studio layout via `st.columns([2, 3])` — **left** holds all controls, **right** the output. Broken out:
    - **Output slot (right)** — one reserved `st.empty()` slot, `result_slot`; during a run it renders a per-step `st.progress` bar inside an `st.status` within a bordered `st.container`, and otherwise shows the result image or a native bordered placeholder container with theme-aware `:gray[:material/image: …]` text (no raw HTML, no emoji).
    - **Prompt form** — a prompt `st.text_input` (placeholder `Enter your prompt`, key `prompt_input`) + an inline `Run` `st.form_submit_button` (`play_arrow`), wrapped in a borderless `st.form("prompt_form", border=False)` so pressing Enter in the prompt box submits the run, with `st.container(horizontal=True)` inside the form (the input stretches, the button hugs its icon+label content so it never wraps). The Run button is always enabled (editing is implicit), but an empty request (no prompt and no image) is guarded with a warning instead of running inference.
    - **Input image(s)** — an `Input image(s) (optional)` expander (key `input_expander`, `icon=":material/image:"`, `on_change="rerun"`) wrapping the file uploader (`accept_multiple_files=True`, label collapsed) — there is **no Task toggle**, uploading any image routes `infer()` to `Flux2KleinEdit` automatically. Corrupt uploads (`OSError`) raise an `st.warning`. A `Clear example images` button lives here.
    - **Mode** — a `Mode` `st.segmented_control` (key `mode_radio`, `required=True` so the active segment can't be deselected — otherwise a deselect returns `None`, blanks the control, and silently resets the steps/guidance sliders); options `Distilled (4 steps)`/`Base (50 steps)` from `MODE_LABEL_LIST`, mapped back to internal `Fast`/`Quality` via `LABEL_TO_MODE`.
    - **Advanced settings** — an expander (collapsed by default, `icon=":material/tune:"`) holding a `Prompt upsampling` toggle (key `auto_enhance_toggle`) + `st.caption`, a `Randomize seed` toggle + a `Seed` `st.number_input` (disabled while Randomize is on), Width/Height sliders, and Number of inference steps/Guidance scale sliders. Width/height auto-update to match the first input image's aspect ratio (uploaded or example); guidance scale and steps update when the mode changes.
    - **Examples** — clickable `EXAMPLE_PROMPTS` buttons (truncated labels via `_truncate`, full prompt on hover) fill the prompt box via `_set_example_prompt` (sets `st.session_state.prompt_input`, clears any example images).
    - **Editing examples** — rows from `EDIT_EXAMPLES` (`(prompt, [image paths])` tuples, verbatim from the Gradio space), each a prompt button + thumbnail `st.image`; clicking calls `_load_edit_example` to set the prompt and store the bundled image paths in `st.session_state.example_images`. Because `st.file_uploader` cannot be set programmatically, `image_list` falls back to `example_images` (opened with `Image.open`) when no manual upload is present, so editing examples route through `Flux2KleinEdit` exactly like uploads; a manual upload clears `example_images`. `_load_edit_example` opens the (keyed) uploader expander by setting `st.session_state.input_expander = True` (later user toggles respected) and increments `uploader_nonce` so the nonce-cycled uploader key (`uploader_{uploader_nonce}`) renders a fresh, empty uploader — otherwise a stale manual upload would survive the rerun and the manual-upload-overrides-example rule would immediately pop the just-loaded example images. Example images are bundled under `examples/` (`woman1.webp`, `cat_window.webp`, `bird.webp`), resolved via `_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"`.
    - **Errors** — an `infer()` failure sets the `st.status` to `state="error"` and shows an `st.error` (`error`) in the controls column (so it survives the output slot being re-rendered) rather than crashing the page.
    - **Theming** — native via `.streamlit/config.toml` (matched `[theme.light]`/`[theme.dark]` palettes, Inter font, `baseRadius`, plus a dark-mode `linkColor = "#9d86ff"` override — dark links otherwise inherit `primaryColor` (`#7457FF`) at only 4.14:1, under WCAG AA, so the lighter purple restores 6.6:1 while keeping the brand hue; light-mode links already pass at 5.2:1 so need no override). `TestThemeConfig` asserts these link/text/button contrasts meet WCAG AA (`ratio ≥ 4.5:1`) and that the dark `linkColor` differs from `primaryColor` — it locks the thresholds, not the specific 4.14/6.6/5.2 figures (those live only in the `config.toml` comment). No custom CSS — the in-app appearance switcher stays available.
    - **Icons & spinners** — `st.set_page_config` sets `page_icon=":material/auto_awesome:"`; Material Symbols icons appear on the `Run` button (`play_arrow`), the two expanders (`image`/`tune`), the enhanced-prompt `st.info` (`auto_awesome`), every `st.warning` (`warning`), and the generation-failure `st.error` (`error`). The five `@st.cache_resource` getters carry `show_spinner` labels so the long first-run model load explains itself, and when Prompt upsampling is on the pre-generation VLM call runs inside `st.spinner("Enhancing prompt…")` (a `contextlib.nullcontext()` when off).

## Commands

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

When editing Streamlit UI code in `streamlit_app.py`, invoke the `developing-with-streamlit` skill first — it loads reference docs version-matched to the Streamlit release installed in `.venv`. Note: run its discovery script with `python3` (bare `python` is not on PATH on this machine).

```bash
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format
uv run ruff format --check .     # Check formatting only
uv run ty check .                # Type check
uv run pytest                    # Run all tests
uv run pytest tests/test_streamlit_app.py  # Run the app test file
uv run pytest tests/test_hooks.py          # Run the hooks test file
uv run pytest tests/test_ci.py             # Run the CI-workflow test file
uv run pytest tests/test_secrets.py        # Run the secret-leak guard test file
uv run pytest tests/test_license.py        # Run the license-consistency test file
```

Ruff is configured in `pyproject.toml` (`[tool.ruff.lint]`: rule sets `E`, `F`, `I`, `UP`, `B`, `C4`, `SIM`); line length (`E501`) is delegated to the formatter, not linted.

## Claude Code hooks

`.claude/` ships repo-shared hooks (wired in `.claude/settings.json`, one bash script each under `.claude/hooks/`; the four tool-event hooks parse their JSON payload with `jq`, while the Stop hook `run-tests.sh` takes no payload and uses no `jq`) that enforce the commands above automatically. Hooks are snapshotted at session start — run `/hooks` to review/activate after changing them. `tests/test_hooks.py` locks their behavior (event wiring, matcher coverage, path routing, deny/allow, exit codes).

- **`ruff-format.sh`** (`PostToolUse`) — `ruff format` + `ruff check --fix` on each edited in-project `.py`, scoped to that file. **Non-blocking** (always exits 0): it corrects silently and never gates on leftover lint.
- **`ty-check.sh`** (`PostToolUse`) — `ty check .` after an in-project `.py` edit; **exits 2** on failure so it is surfaced back for a fix. The header is exit-code-agnostic (`ty check failed`) so an environmental failure isn't mislabeled a type error.
- **`mark-tests-pending.sh`** (`PostToolUse`) + **`run-tests.sh`** (`Stop`) — the test suite is gated **once per turn**, not per edit: the PostToolUse hook drops a `.claude/.tests-pending` marker when an edit touches `streamlit_app.py`, `tests/`, or the load-bearing `.streamlit/config.toml` (whose WCAG contrast `TestThemeConfig` asserts); the Stop hook consumes the marker and runs `uv run pytest`, **exits 2** on failure. Clearing the marker before running is what keeps the Stop loop safe. The marker is gitignored.
- **`guard-paths.sh`** (`PreToolUse`) — denies Edit/Write/MultiEdit to `.env`/`.env.*`, `secrets.toml`, and `uv.lock` (mutate only via `uv`) with a `permissionDecision: "deny"` JSON decision. Matching is **case-insensitive** (macOS's filesystem is), secret-free templates (`.env.example`/`.sample`/`.template`/`.dist`) are allowed through, and it **fails closed** — if `jq` is missing it denies rather than silently allowing. It is a best-effort guard on the Edit/Write/MultiEdit tools only: a **`Bash` command that writes these paths is not intercepted** (matchers match tool names, and Bash is also how `uv` legitimately rewrites `uv.lock`), so it is a convenience backstop, not a security boundary. The CI-enforced counterpart is `tests/test_secrets.py`, which scans `git ls-files` on every push/PR and fails the build if any tracked file contains a recognizable secret (HF or GitHub token, PEM private key, AWS key) or if `.env`/a non-template `.env.*`/`secrets.toml` is ever committed — the net that catches what the edit-time guard and `.gitignore` miss.

Matchers only see tool *names* (`Edit|Write|MultiEdit`), so each script does its own path/extension filtering internally, and only for files inside `${CLAUDE_PROJECT_DIR}` — a ruff hook fired on a Markdown or out-of-repo edit must no-op, not run. If `jq` is absent the payload-parsing hooks degrade (the PostToolUse hooks no-op; only the guard is loud, denying) — `run-tests.sh` is unaffected, since it parses no payload. `.claude/settings.local.json` (personal permission overrides) and `.claude/.tests-pending` are gitignored; `settings.json` is tracked non-executable (`100644` — Claude reads it, never execs it) while the `hooks/` scripts are tracked executable (`100755`, or a clone runs them with the wrong mode).

## Continuous integration

`.github/workflows/ci.yml` is the public mirror of the hooks: on every push to `main` and every pull request it runs the same four gates (ruff format/lint, ty, pytest) via the pinned toolchain (`astral-sh/setup-uv` with caching, then `uv sync` + `uv run <tool>`), so CI and the local hooks can't disagree on a rule. It runs on a `macos-latest` (Apple Silicon) runner — mandatory, not a preference: `uv.lock` resolves `mlx` to a CUDA-only build on Linux (`sys_platform == 'linux'`), so the suite can't even import on a GPU-less ubuntu runner; macOS is also the app's real target and is free for public repos. No model weights download (tests mock the getters), so CI needs no HF token. `.python-version` pins the interpreter to 3.12 so local uv and CI resolve the same runtime. `tests/test_ci.py` locks the workflow's triggers, the macOS-for-test/typecheck rule, the four-gate set (including that the lint gate stays failing, not `--fix`), uv usage, least-privilege permissions, per-job timeouts, and the version pin — it parses `ci.yml` with PyYAML (a dev dependency), normalizing the YAML-1.1 `on:`→`True` key.

## Gotchas

### mflux / FLUX.2 Klein

- **Two model classes: `Flux2Klein` (text-to-image) and `Flux2KleinEdit` (multi-image editing).** The distilled getters build them with `model_config=ModelConfig.flux2_klein_4b()`, the base getters with `ModelConfig.flux2_klein_base_4b()`. No repo ID strings or token parameters.
- **`Flux2Klein.generate_image()` takes `image_path` (singular, `Path | str | None`) for img2img.** `Flux2KleinEdit.generate_image()` takes `image_paths` (plural, `list[Path | str] | None`) for multi-image editing.
- **`generate_image()` returns a `GeneratedImage` wrapper, not a PIL Image.** Access the PIL Image via `.image`.
- **MLX manages device placement automatically.** Apple Silicon unified memory is used directly.
- **Progress callbacks use `model.callbacks.register()`.** Register an object with a `call_in_loop(self, t, seed, prompt, latents, config, time_steps)` method. `CallbackRegistry` has no `unregister()` — callers must remove the instance from `model.callbacks.in_loop` themselves (`in_loop.remove(reporter)`, matched by value — not `pop`, which takes an index), or reporters accumulate on `@st.cache_resource`-cached models and fire on every subsequent Run. mflux invokes `call_in_loop` entirely by keyword, so the six parameter names must match exactly — the unused ones (`seed`, `prompt`, `latents`, `time_steps`) can't be renamed or dropped.
- **The guidance parameter is named `guidance`**, not `guidance_scale`.
- **FLUX.2 Klein does not support negative prompts.**
- **Quality and Fast use different step/CFG defaults** — see Architecture §1 for the numbers. The non-obvious part is the naming inversion: the internal keys `"Fast"`/`"Quality"` (used by `MODE_DEFAULTS`/`MODELS`/`EDIT_MODELS`) surface in the UI as `Distilled (4 steps)`/`Base (50 steps)` via `MODE_LABELS`, so **Fast = the distilled variant, Quality = the base variant**.

### mlx-vlm / SmolVLM

- **Use `mlx_vlm.load()` to get `(model, processor)` and `mlx_vlm.utils.load_config()` for config.** Config is required by `apply_chat_template`.
- **`mlx_vlm.generate()` handles tokenization and decoding internally.** Access result via `result.text`. SmolVLM appends `<end_of_utterance>` tokens that must be stripped from output.
- **`apply_chat_template` takes `num_images` instead of embedding image tokens in messages.** Pass images as a flat list to the `image` parameter of `generate()`.
- **`temperature > 0` implies sampling.** `temperature=0.0` gives greedy decoding, but this app deliberately samples for prompt enhancement (`temperature=0.7`, `top_p=0.9` — see Architecture §2), so treat the greedy note as background, not the shipped setting.
- **Use `max_tokens`** to control output length.
- **mlx-vlm's type hints are loose.** `generate()` annotates `image` as `str | list[str]` (no `Optional`, no PIL) though it accepts `None`/PIL Images at runtime — hence the `# ty: ignore[invalid-argument-type]` on the `image=` argument. `apply_chat_template` is typed as a broad union but returns a `str` for SmolVLM, so its result is wrapped in `cast(str, ...)`.

### General

- **Apple Silicon is the primary target.** The app uses MLX (mflux + mlx-vlm) which requires Apple Silicon or Linux CUDA. CPU-only and Windows are not supported.
- **All models share memory via MLX unified memory.** FLUX.2 Klein Distilled + Base (txt2img and edit variants) + SmolVLM in bfloat16. All loaded lazily via `@st.cache_resource`.
- **Streamlit widget state cannot be modified after instantiation.** Use `on_click` callbacks to set `st.session_state` keys for widgets, not direct assignment after the widget is created.
- **The prompt input lives in a form (`prompt_form`), so typed edits reach Python only on submit** (Enter or the Run button). Prompt changes pushed via session state by example-button callbacks are *not* gated and land on their own rerun. Caveat for tests: `AppTest` does not model form gating — `set_value()` on a form widget is flushed into the next `run()` unconditionally — so a test of non-submit prompt-change behavior must use the session-state path (an example click), or it asserts behavior production doesn't have.
- **`st.file_uploader` cannot be set or reset programmatically — cycle its key instead.** The uploader retains files across reruns, so `_load_edit_example` bumps `uploader_nonce` to render a fresh, empty uploader (old widget state is garbage-collected). Keyed expanders are the opposite: `st.expander(key=...)` registers with writes allowed, so callbacks may set `st.session_state.input_expander` directly.
- **The output `st.empty()` slot (`result_slot`) is re-rendered on every run.** The bottom render block always rewrites it, overwriting anything drawn into it mid-run (e.g. the progress `st.status`), so transient messages like a generation error must go in the controls column to persist.
- **Slider-key seeding must stay above the `Advanced settings` expander.** The mode-change block (guidance/steps) and image-change block (width/height) write those slider *keys* into `st.session_state`; assigning a widget key after the widget is instantiated raises `StreamlitAPIException`, so the seeding must run before the expander that renders the sliders.
- **Never gate the `Advanced settings` expander body to skip when collapsed** (e.g. `on_change="rerun"` + `.open`). Its four slider return values (width, height, steps, guidance) feed `infer()` on every run, so a skipped body leaves them unbound → `NameError`. `test_advanced_sliders_render_while_collapsed` guards this.
- **The `if __name__ == "__main__"` guard is a deliberate testability seam.** A plain `import streamlit_app` (the unit tests) leaves `__name__ == "streamlit_app"` and skips the UI, so the helpers import without a Streamlit runtime; `AppTest.from_file()` and `streamlit run` exec the module as `"__main__"` and do render it. `test_ui_not_executed_on_import` asserts the seam.
