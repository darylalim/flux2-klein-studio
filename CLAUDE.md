# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX.2 Klein Studio is a single-file Streamlit application for generating and editing images using Black Forest Labs [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) on Apple Silicon with MLX. Built on [mflux](https://github.com/filipstrand/mflux) (diffusion) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (VLM). Text-to-image by default, switching to multi-image editing automatically when input images are supplied — there is no mode toggle.

One model only — the distilled 4B pre-quantized to 8-bit ([mlx-community/flux2-klein-4b-8bit](https://huggingface.co/mlx-community/flux2-klein-4b-8bit)), ~8.6GB versus ~16GB for bf16. The 50-step base variant has no pre-quantized build under `mlx-community` (third-party mflux-format 8-bit builds do exist on the Hub) and is not offered. Optional vision-aware prompt upsampling via [Qwen3-VL-2B-Instruct](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-8bit) — the VLM can see uploaded images when enhancing editing prompts.

`README.md` mirrors this file's Overview/Setup/CI/Releases/Hooks content for a public audience (plus a Usage section and a light/dark screenshot hero) — keep the two in sync.

## Setup

```bash
uv sync                                    # install
uv run streamlit run streamlit_app.py      # run
uv add <pkg>                               # add a runtime dependency
uv add --dev <pkg>                         # add a dev tool -> [dependency-groups] dev
```

Requires Apple Silicon (M1+) on macOS 14+ (the pinned mlx publishes no `macosx_13` wheel) and Python 3.12+. Commit `pyproject.toml` and `uv.lock` together — CI runs `uv sync --locked` and fails on a stale lock.

## Definition of done

```bash
uv run ruff format --check . && uv run ruff check . && uv run ty check . && uv run pytest
```

All four green, plus `README.md` re-synced if mirrored content moved. The `Stop` hook runs these automatically — except on a turn whose only writes went through `Bash` (see Hooks).

## Do not touch

| Rule | What breaks if ignored | Guard |
|---|---|---|
| Don't add `ruff check --fix` to the per-edit hook | `--fix` deletes an import added in one edit and first used several edits later; run before the formatter it also strands the blank line the import left behind. Verified: `import contextlib\n\nx = 1\n` came out as `\nx = 1\n`, failing CI while the hook exited 0 silently | `TestRuffHandlesTheHookedExtensions` |
| Don't case-insensitize `ruff-format.sh`'s globs | ruff picks its parser from the extension, and JSON is valid Python expression syntax — on macOS's case-insensitive filesystem `ruff format Notebook.IPYNB` opens the real notebook, parses its JSON as a dict literal and rewrites it as Python (verified) | `TestRuffHandlesTheHookedExtensions` |
| Don't register ruff/ty/pytest as separate `Stop` hooks | Same-event hooks run in parallel with no ordering guarantee; whichever reached `rm -f "$marker"` first would delete it out from under the other. Fold them into `run-tests.sh` | — |
| Don't add a `${CLAUDE_PROJECT_DIR}` containment check to `guard-paths.sh` | An out-of-repo `.env` is still a secrets file git cannot restore, so containment could only ever remove protection | `tests/test_hooks.py` |
| Don't hand-edit `uv.lock` | Mutate only via `uv`; a hand-edit desyncs from `pyproject.toml` and fails CI's `--locked` | `guard-paths.sh` |
| Don't delete `tests/__init__.py` | The suite fails at collection — see Testing conventions | — |
| Don't gate the `Advanced settings` expander body | Its four slider return values feed `infer()` on every run, so a skipped body leaves them unbound → `NameError` | `test_advanced_sliders_render_while_collapsed` |
| Don't move slider-key seeding below the expander | Assigning a widget key after the widget is instantiated raises `StreamlitAPIException` | — |
| Don't rename `call_in_loop`'s six parameters | mflux invokes it entirely by keyword, unused params included | `tests/test_smoke.py` |
| Don't "fix" `image_paths` to accept only `str` paths | It takes PIL Images at runtime and the UI depends on it; the mocked suite cannot see this | `test_edit_accepts_the_pil_images_the_ui_builds` (`-m smoke`) |
| Don't reword `UPSAMPLE_PROMPT_*` without re-measuring | Two rewrites measured worse — see Gotchas | — |
| Don't split tag creation from the publish | GitHub raises no workflow run for refs pushed with the default `GITHUB_TOKEN` → a tag, no release, and an all-green run | `TestTagsAreNeverPushedByGit` |
| Don't match a hyphen in the prerelease predicate | uv writes PEP 440 canonical versions (`0.9.0rc1`), so a hyphen branch never fires and an RC publishes as `Latest` | `test_prerelease_predicate_matches_the_versions_uv_actually_writes` |
| Don't interpolate `${{ }}` into a `run:` script | The GitHub Actions command-injection vector — pass values through `env:` and reference `"$VAR"` | `tests/test_workflows.py` |
| Don't switch CI to an ubuntu runner | `uv.lock` resolves `mlx` to a CUDA-only build on Linux; the suite can't even import | `tests/test_ci.py` |
| Don't bump `setup-uv` past `@v7` casually | v7 is the last floating major tag it publishes; above it you own every bump | `test_actions_pin_a_floating_major_tag` |
| Don't loosen the uv pin from `==` | setup-uv strips `==` but cannot parse a PEP 440 comma range, and silently falls back to latest | `tests/test_ci.py` |

## Architecture

| Path | Contents |
|---|---|
| `streamlit_app.py` | the entire application |
| `tests/` | the suite (see Testing conventions) |
| `examples/` | bundled edit-example images (`woman1.webp`, `cat_window.webp`, `bird.webp`) |
| `docs/` | README screenshots |
| `.claude/` | repo-shared Claude Code hooks |
| `.github/workflows/` | CI quality gates + release |
| `.streamlit/config.toml` | Streamlit config — deliberately carries no `[theme]` |
| `LICENSE` | MIT — mirrored by `license = "MIT"` in `pyproject.toml` and README's License section, all three locked by `tests/test_license.py` |

`streamlit_app.py` is structured in four sections:

1. **Model initialization** — `Flux2Klein` (text-to-image) and `Flux2KleinEdit` (multi-image editing) from mflux, behind two `@st.cache_resource` getters, `_get_model()` and `_get_edit_model()`. Both are built with `model_path=MODEL_REPO` *and* `model_config=ModelConfig.flux2_klein_4b()`; the rationale for passing both is in the comment above `_get_model()`. MLX manages Apple Silicon unified memory automatically. `DEFAULT_STEPS` (4) and `DEFAULT_GUIDANCE` (1.0) are the distilled variant's settings — they seed the sliders and back-fill `infer()`'s optional args.

2. **Prompt upsampling** — `_get_vlm()` loads Qwen3-VL-2B-Instruct-8bit via `mlx_vlm.load()`, cached, returning a `(model, processor, config)` triple. Two system prompts: `UPSAMPLE_PROMPT_TEXT_ONLY` (text-to-image, capped at 120 words) and `UPSAMPLE_PROMPT_WITH_IMAGES` (concrete language, preserve unchanged elements); `upsample_prompt(prompt, image_list=None)` selects between them on whether images were supplied. It formats messages via `apply_chat_template` and generates via `mlx_vlm.generate()` — the app samples rather than decoding greedily, and every sampling parameter carries its own justification in the comments inside that call. Images are downscaled to `VLM_MAX_IMAGE_SIZE` (768px longest edge) by `_vlm_images()`. `_resolve_prompt(prompt, image_list, auto_enhance)` wraps the decision and returns `(prompt, was_enhanced)`. Results are memoized in a bounded per-session `st.session_state._enhance_cache` (keyed on `(prompt, image set)`, oldest evicted past 32) so re-Running an unchanged request doesn't re-invoke the VLM; the `auto_enhanced_prompt` key behind the enhanced-prompt banner is dropped whenever the prompt text or image set changes, and on each Run.

3. **Inference** — `infer(prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, image_list=None, progress_callback=None)`. There is no `mode` parameter: `image_list` alone selects `_get_edit_model()` over `_get_model()`. `randomize_seed=True` replaces the caller's `seed`; unset `guidance_scale`/`num_inference_steps` fall back to the `DEFAULT_*` constants. `model.generate_image()` returns a `GeneratedImage` whose PIL image is `.image`. When a `progress_callback` is given, a `_ProgressReporter` is registered via `model.callbacks.register()` and removed again in a `finally` block — mandatory, see Gotchas. `_dimensions_from_images()` derives output dimensions from the first input image's aspect ratio (larger side 1024, rounds to 32, clamps 256–1024).

4. **UI** — behind `if __name__ == "__main__"` (`layout="wide"`, modeled on the FLUX.2 [klein] Gradio space): a lone `st.title` header and a two-column studio via `st.columns([2, 3])`, left for all controls, right for the output. Read the file for the widget inventory; these are the invariants it won't tell you:
   - **No Task toggle.** Uploading any image routes `infer()` to `Flux2KleinEdit` automatically. The Run button is always enabled, but an empty request (no prompt, no image) is guarded with a warning instead of running inference.
   - **Example images bypass the uploader.** `st.file_uploader` cannot be set programmatically, so `image_list` falls back to `st.session_state.example_images` (opened with `Image.open`) when there is no manual upload, and a manual upload clears it. `_load_edit_example` opens the keyed `input_expander` and bumps `uploader_nonce` so the nonce-cycled uploader key renders a fresh, empty uploader — otherwise a stale manual upload survives the rerun and the manual-overrides-example rule immediately pops the images just loaded.
   - **Errors belong in the controls column.** The bottom render block rewrites the output `st.empty()` slot (`result_slot`) on every run, overwriting anything drawn there mid-run, so a generation error placed in the output slot would vanish.
   - **Slider-key seeding stays above the `Advanced settings` expander** — the `setdefault` block for all four slider keys and the image-change width/height writes both assign widget keys, which must happen before the widgets render.
   - **The theme is Streamlit's own.** `.streamlit/config.toml` declares no `[theme]` section at all, so the stock light and dark themes apply and the in-app appearance switcher picks between them. That absence is the setting: adding back *any* `[theme]` key makes it a custom theme, and a custom theme without both `[theme.light]` and `[theme.dark]` locks the app to a single mode — even a lone `font` costs the switcher. `TestThemeConfig` locks the absence; the palette's contrast is Streamlit's contract, not this repo's. No custom CSS either.

## Commands

```bash
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format
uv run ruff format --check .     # Check formatting only
uv run ty check .                # Type check
uv run pytest                    # Run all tests
uv run pytest tests/test_ci.py   # Run one file
uv run pytest -m smoke           # Opt-in: real weights, real generation
```

Test files: `test_streamlit_app.py`, `test_hooks.py`, `test_ci.py`, `test_workflows.py`, `test_secrets.py`, `test_license.py`, `test_readme.py`, `test_smoke.py`.

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff. When editing Streamlit UI code in `streamlit_app.py`, invoke the `developing-with-streamlit` skill first — it loads reference docs version-matched to the Streamlit release in `.venv`. Run its discovery script with `python3`; bare `python` is not on PATH on this machine.

Ruff is configured in `pyproject.toml` (`[tool.ruff.lint]`: rule sets `E`, `F`, `I`, `UP`, `B`, `C4`, `SIM`); line length (`E501`) is delegated to the formatter, not linted. Since ruff 0.16 `ruff format` also formats Python fenced inside Markdown, so CI's `ruff format --check .` covers `README.md` and `CLAUDE.md` too (`ruff check` still ignores Markdown entirely). The set of fence tags it reformats is wider than `python` alone — `py`, `{python}` and `pycon` count as well, while an untagged fence is left alone — and `ruff-format.sh` routes `.md` to match, so a Python block in a doc is auto-fixed locally instead of failing only in CI. The `ruff>=0.16` floor in `[dependency-groups]` is what holds this together.

## Testing conventions

- **`tests/__init__.py` is empty but required.** `pyproject.toml` declares no `[build-system]`, so this is a uv *virtual* project and `streamlit_app` is never installed into `.venv` — a bare `import streamlit_app` works only because `tests/` being a package puts the repo root on `sys.path`. Any new test subdirectory needs one too. There is no `conftest.py`.
- **Tests do not mock the getters.** `_reload_app()` patches the *upstream* symbols (mflux's `Flux2Klein`/`Flux2KleinEdit`, `mlx_vlm.load`, `mlx_vlm.utils.load_config`) **and** swaps `streamlit.cache_resource` for `_passthrough_cache_resource`, then `importlib.reload`s the module; individual tests afterwards patch the app-namespace name (`streamlit_app.Flux2Klein`). That reload is the only cache reset — nothing calls `.clear()` — so skipping it leaks the first mock into every later test. The passthrough exists because the app uses `@st.cache_resource(show_spinner=…)` and a bare `lambda f: f` chokes on the factory form.
- **`AppTest.from_file()` resolves a relative script path against the calling test file**, not the working directory, so `tests/test_streamlit_app.py` passes the absolute `_APP_PATH` (`_REPO_ROOT / "streamlit_app.py"`); a bare `"streamlit_app.py"` resolves to `tests/streamlit_app.py` and fails every `AppTest` test at once.
- **`AppTest` does not model form gating.** The prompt input lives in `prompt_form`, so in production typed edits reach Python only on submit (Enter or the Run button), while prompt changes pushed via session state by example-button callbacks land on their own rerun. But `set_value()` on a form widget is flushed into the next `run()` unconditionally — so a test of non-submit prompt behavior must use the session-state path (an example click), or it asserts behavior production does not have.
- **The `if __name__ == "__main__"` guard is a deliberate testability seam.** A plain `import streamlit_app` leaves `__name__ == "streamlit_app"` and skips the UI, so the helpers import without a Streamlit runtime; `AppTest.from_file()` and `streamlit run` exec the module as `"__main__"` and do render it. `test_ui_not_executed_on_import` asserts the seam.
- **`-m smoke` is opt-in** (`addopts` deselects it) and is the only thing that touches the real mflux/mlx-vlm surface. No GitHub runner can host it, so it is a documented manual gate — run it before a release.

## Claude Code hooks

`.claude/` ships repo-shared hooks, wired in `.claude/settings.json`, one bash script each under `.claude/hooks/`. Hooks are snapshotted at session start — run `/hooks` to review/activate after changing them. `tests/test_hooks.py` locks their behavior (event wiring, matcher coverage, path routing, deny/allow, exit codes, gate ordering).

**The split between events is the design, not an accident.** `PostToolUse` fires once per `Edit`/`Write`/`MultiEdit`, so only cheap, file-scoped, side-effect-free work belongs there; every whole-project gate lives in the `Stop` hook, which fires once per turn. Running `ty check .` per edit re-derived one whole-tree fact N times a turn *and* surfaced a transient `exit 2` on every intermediate edit of a multi-file refactor. `TestHookWiring::test_no_whole_project_gate_runs_per_edit` greps the wired PostToolUse scripts and fails if a whole-project command reappears in one. `TestHookWiring::test_scripts_are_bound_to_the_correct_events` is the one assertion that catches a hook *disappearing* — every other test iterates whatever `settings.json` lists, so a removed entry is invisible to them.

- **`ruff-format.sh`** (`PostToolUse`) — `ruff format` on each edited in-project `.py`, `.pyi`, `.md` or `.ipynb`, the set CI's `ruff format --check .` walks, so nothing is auto-fixed only in CI. Scoped to the edited file, formatting only, always exits 0. The lint-fix pass deliberately does not live here, and every glob is lowercase deliberately — see Do not touch for both.
- **`mark-tests-pending.sh`** (`PostToolUse`) + **`run-tests.sh`** (`Stop`) — the PostToolUse hook drops a `.claude/.tests-pending` marker when an edit touches something the suite asserts on; the Stop hook consumes it and runs `ty check .` (fail-fast) then `uv run pytest`, **exiting 2** on failure. The `ty` header is exit-code-agnostic (`ty check failed`) so an environmental failure isn't mislabeled a type error. `ruff format .` runs **unconditionally, above the marker gate** — pure layout, idempotent, incapable of losing work, and the only thing covering a file written by a `Bash` heredoc or redirect. `ruff check --fix .` runs **below the gate**, because it deletes code tree-wide including in files the turn never touched: above the gate it would rewrite code on a turn where `ty` and `pytest` never run, leaving nothing to validate what it did. A second `ruff format .` follows the `--fix`. The hook is wired with a 180s timeout in `settings.json`; the gates cost ~14s today, so a suite approaching that budget needs the timeout raised, not the tests trimmed.

  The covered set is **derived, not transcribed**: the hook asks `git check-ignore` and arms unless git ignores the path. Since `tests/test_secrets.py` scans `git ls-files`, every tracked file is already suite-relevant — the covered set is not a curated union of modules, it is simply "not gitignored", which makes the rule fail-**closed**: an unrecognized path arms the suite rather than skipping it (`git check-ignore` exits 1 for "not ignored" and 128 for "no repo", and both fall through to arm). Clearing the marker before running is what keeps the Stop loop safe. The marker is gitignored.
- **`guard-paths.sh`** (`PreToolUse`) — denies Edit/Write/MultiEdit to `.env`/`.env.*`, `secrets.toml`, and `uv.lock` with a `permissionDecision: "deny"` JSON decision. Matching is **case-insensitive** (macOS's filesystem is) and on **basename alone**. Secret-free templates (`.env.example`/`.sample`/`.template`/`.dist`) are allowed through. Missing `jq` **degrades it rather than disabling it**: the path is extracted with bash string operations instead, and the decisions match the `jq` path for any payload that yields a path — a payload that yields *no* path diverges deliberately, the bash branch denying (fail-closed) where the `jq` branch allows. It is a best-effort guard on the Edit/Write/MultiEdit tools only: a **`Bash` command that writes these paths is not intercepted** (matchers match tool names, and Bash is also how `uv` legitimately rewrites `uv.lock`), so it is a convenience backstop, not a security boundary. The CI-enforced counterpart is `tests/test_secrets.py`, which scans `git ls-files` on every push/PR and fails the build if any tracked file contains a recognizable secret (HF or GitHub token, PEM private key, AWS key) or if `.env`/a non-template `.env.*`/`secrets.toml` is ever committed.

**A turn whose only writes went through `Bash` runs almost no gates.** A heredoc or redirect fires no PostToolUse hook, so nothing arms the marker and `run-tests.sh` returns before `ruff check --fix`, `ty` and `pytest` — leaving only the unconditional `ruff format .`. Run `uv run ty check . && uv run pytest` by hand on such a turn, or make the edit with `Write`/`Edit` instead.

Matchers only see tool *names* (`Edit|Write|MultiEdit`), so each script does its own path filtering internally, and only for files inside `${CLAUDE_PROJECT_DIR}` — an out-of-repo edit must no-op whatever its extension. `guard-paths.sh` is the exception and matches on basename alone, so it denies a protected path anywhere. If `jq` is absent the PostToolUse hooks no-op, while `guard-paths.sh` keeps working and `run-tests.sh` is unaffected (it parses no payload). `.claude/settings.local.json` and `.claude/.tests-pending` are gitignored; `settings.json` is tracked non-executable (`100644` — Claude reads it, never execs it) while the `hooks/` scripts are tracked executable (`100755`, or a clone runs them with the wrong mode).

## Continuous integration

`.github/workflows/ci.yml` is the public mirror of the hooks: on every push to `main` and every pull request it runs the same four gates (ruff format/lint, ty, pytest) via the pinned toolchain (`astral-sh/setup-uv`, then `uv sync --locked` + `uv run <tool>`), so CI and the local hooks can't disagree on a rule.

- **Runner: `macos-latest`** (Apple Silicon) — mandatory, not a preference. `uv.lock` resolves `mlx` to a CUDA-only build on Linux (`sys_platform == 'linux'`), so the suite can't even import on a GPU-less ubuntu runner; macOS is also the app's real target and is free for public repos.
- **`uv sync --locked`** fails the build on a stale `uv.lock` (e.g. a version bump that forgot `uv lock`) instead of silently re-resolving.
- **Action pinning: floating major tags** (`@v7` across `actions/checkout`, `astral-sh/setup-uv`, `actions/setup-python`), never a SHA, an exact version or a branch — the repo runs no dependabot, and a pin with nothing to bump it rots silently, while a major tag keeps taking patches by itself. `setup-uv@v7` looks stale next to a `v10.x` release and is not — v7 is the last floating major tag it publishes (see Do not touch). `TestCIWorkflow::test_actions_pin_a_floating_major_tag` locks the shape.
- **Version pins:** `.python-version` pins the interpreter to 3.12 so local uv and CI resolve the same runtime, and `[tool.uv] required-version = "==0.12.5"` pins **uv itself** — `astral-sh/setup-uv` reads that key when its `version:` input is empty, so one declaration covers local and CI. Without it setup-uv installs latest, and a uv whose resolution-marker normalization differs from the one that wrote `uv.lock` fails `uv sync --locked` on a lockfile nobody edited. Bump the pin and re-run `uv lock` together.
- **`enable-cache` is deliberately not set** on setup-uv: it defaults to `auto`, which resolves true on GitHub-hosted runners anyway, and the cache it produced measured ~0 MB because setup-uv's `prune-cache` default strips exactly the pre-built wheels every locked package ships. Leaving it unset also means a future setup-uv bump inherits that version's cache behavior. Don't add it.
- **No model weights download** — tests patch the upstream mflux/mlx-vlm symbols (see Testing conventions), so CI needs no HF token.
- **Permissions are scoped per job:** the workflow grants `contents: read` and only `release` re-declares `contents: write` — a job-level block *replaces* the workflow grant rather than merging with it, so the gate job that runs the test suite never holds a write-scoped token.

`tests/test_ci.py` locks the triggers, the macOS-for-test/typecheck rule, the four-gate set (including that the lint gate stays failing, not `--fix`), uv usage, least-privilege permissions, per-job timeouts and the version pin, parsing `ci.yml` with PyYAML.

## Releases

Cut by bumping `version` in `pyproject.toml` and pushing to `main` — nothing else. `uv version --bump <major|minor|patch>` does the bump *and* re-locks in one step (uv 0.12.5 re-locks by default; `--frozen` is the opt-out), which is the whole reason to prefer it over hand-editing: a forgotten `uv lock` is exactly the mistake CI's `--locked` exists to catch. Run `uv run pytest -m smoke` **before** bumping.

`ci.yml`'s `release` job does the rest: `needs: ci` gates it on all four quality gates, and `if: github.event_name == 'push' && github.ref == 'refs/heads/main'` keeps pull requests from publishing. It publishes only when the version `pyproject.toml` declares has no matching tag on the remote.

- **Detection is a tag-existence check** (`git ls-remote --exit-code --tags origin refs/tags/v$version`), not a `HEAD~1` diff of `pyproject.toml`. A diff misses a bump that arrived in an earlier commit of a multi-commit push or behind a merge, and has no parent at all on a root commit. The check is a pure function of (declared version, remote tags) — invariant under merges, squashes and force-pushes, and idempotent on re-runs. `ls-remote --exit-code` exits 0 when the tag exists and 2 when it does not, and the script treats *any other* code as an error rather than reading it as "unreleased"; the call is written `... || status=$?` because Actions runs `run:` under `bash -e`, where a bare exit-2 would abort the step before the status could be read.
- **Publishing is one `gh release create "v$VERSION" --target "$SHA" --generate-notes` call**, which creates the tag **and** the release together — there is no window where a tag exists without a release. `--target` pins both to the exact commit the gates passed.
- **Never split the tag creation from the publish.** GitHub starts no workflow run for events raised by the default `GITHUB_TOKEN` (only `workflow_dispatch` and `repository_dispatch` are exempt), so a tag-triggered publisher is a silent no-op, not a red build.
- **Pre-releases** get `--prerelease` so they never become `Latest`, predicated on `case "$VERSION" in *[!0-9.]*)` — "not purely digits and dots", **not** a hyphen match (see Do not touch). `TestCIReleaseJob::test_prerelease_predicate_matches_the_versions_uv_actually_writes` extracts the pattern from the YAML and runs it under bash.

`.github/release.yml` buckets the release notes by PR label; `tests/test_ci.py::TestReleaseNotesConfig` locks it. `tests/test_workflows.py` locks the invariants that must hold across *every* workflow rather than one: no `run:` script may interpolate a `${{ }}` expression directly, and no `run:` script may create or push a git ref. It carries non-vacuity guards, including one asserting a publisher still exists, so deleting the release path can't make the no-git-tag rule pass vacuously.

## Gotchas

### mflux / FLUX.2 Klein

- **Two model classes: `Flux2Klein` (text-to-image) and `Flux2KleinEdit` (multi-image editing).** Both getters build the distilled variant with `model_config=ModelConfig.flux2_klein_4b()`. mflux also exposes `ModelConfig.flux2_klein_base_4b()`, which this app never calls — the base 50-step mode was removed. The weights come from a repo ID string passed as `model_path`; there is no HF token parameter.
- **`Flux2Klein.generate_image()` takes `image_path` (singular, `Path | str | None`) for img2img; `Flux2KleinEdit.generate_image()` takes `image_paths` (plural, `list[Path | str] | None`).** **Despite that annotation `image_paths` accepts PIL Images at runtime**, which the app depends on: both UI branches build `image_list` with `Image.open(...)` (uploads and bundled examples alike) and `infer()` passes it straight through. The mocked suite cannot see this — `tests/test_smoke.py::test_edit_accepts_the_pil_images_the_ui_builds` is what verifies it, against real weights.
- **`generate_image()` returns a `GeneratedImage` wrapper, not a PIL Image.** Access the PIL Image via `.image`.
- **Progress callbacks use `model.callbacks.register()`.** Register an object with a `call_in_loop(self, t, seed, prompt, latents, config, time_steps)` method. `CallbackRegistry` has no `unregister()` — callers must remove the instance from `model.callbacks.in_loop` themselves (`in_loop.remove(reporter)`, matched by value — not `pop`, which takes an index), or reporters accumulate on `@st.cache_resource`-cached models and fire on every subsequent Run. mflux invokes `call_in_loop` entirely by keyword, so the six parameter names must match exactly — the unused ones (`seed`, `prompt`, `latents`, `time_steps`) can't be renamed or dropped.
- **`model_path` selects the weights; `model_config` supplies the architecture** (`transformer_overrides`/`text_encoder_overrides`, which mflux does not read from the repo). The app's explicit `model_config` is currently redundant — both `__init__`s already default to `ModelConfig.flux2_klein_4b()` — and is passed to keep the repo/architecture pairing visible; it becomes load-bearing if `MODEL_REPO` ever points at a non-4b build.
- **Pre-quantized weights are detected from safetensors metadata, not from the path** (mflux's own `quantization_level` key), and the freshly-built modules are quantized *before* `model.update()`, because a `U32`-packed tensor cannot be loaded into a `bf16` `Linear`. Passing `quantize=8` alongside an already-8-bit repo is a harmless no-op, so the app does not pass it.
- **The VAE is effectively unshrunk in a "quantized" repo.** `nn.quantize` is applied with the predicate `hasattr(module, "to_quantized")`, which `Conv2d` does not satisfy, so every convolution stays bf16; only the 8 `nn.Linear` attention projections in the encoder/decoder mid-block quantize — 8 packed tensors against 258 bf16 ones, against 77 of 261 for the transformer.
- **The guidance parameter is named `guidance`**, not `guidance_scale`.
- **FLUX.2 Klein does not support negative prompts.**

### mlx-vlm / Qwen3-VL

- **Use `mlx_vlm.load()` to get `(model, processor)` and `mlx_vlm.utils.load_config()` for config.** Config is required by `apply_chat_template`. `_get_vlm()` calls `load_config()` separately on purpose — `model.config.to_dict()` is not a substitute. It looks redundant — `apply_chat_template` only reads `config["model_type"]`, which both carry — but the dicts differ materially: 64 keys against 20, with `architectures`, `bos_token_id` and 50 others absent from `to_dict()`, and `vision_config`/`text_config` differing. The saving is one cached path resolution per session; the risk is silently shrinking the triple.
- **Qwen3-VL's processor does not cap input resolution, so the app must.** `preprocessor_config.json` has `max_pixels: None` and `size.longest_edge: 16777216`, unlike SmolVLM's 512px tiling — the measurements behind the 768px choice are on the `VLM_MAX_IMAGE_SIZE` constant. `_vlm_images()` thumbnails **copies**; copies because `PIL.Image.thumbnail()` resizes in place and `infer()` still needs the originals for mflux.
- **Qwen3-VL-2B handles single-image edits well and multi-image compositional edits poorly, and prompt wording does not fix it.** Two rewrites of the shipped prompt were measured and **both came out worse** — adding "begin with a verb" biased it toward `Remove`/`Replace` (on a single-image "make it snowy" it produced "Remove the cat and the green shutters"), and adding "never describe what the images show" *raised* the describe rate, the usual small-model negation failure. Treat this as a model-capacity ceiling, not a prompt bug; re-measure before rewording.
- **`mlx_vlm.generate()` handles tokenization and decoding internally.** Access the result via `result.text`. Qwen3-VL's `<|im_end|>` is a real stop id (`generation_config.json` `eos_token_id`), consumed before detokenization, so — unlike SmolVLM's `<end_of_utterance>` — no stop-string stripping is needed. Note `temperature > 0` implies sampling: `0.0` would give greedy decoding, but this app deliberately samples for prompt enhancement, so treat the greedy note as background rather than the shipped setting.
- **`apply_chat_template` takes `num_images`** instead of embedding image tokens in messages. Pass images as a flat list to the `image` parameter of `generate()`.
- **mlx-vlm's type hints are loose.** `generate()` annotates `image` as `str | list[str] | None` — still no PIL, though it accepts PIL Images at runtime; ty cannot flag that here only because `upsample_prompt`'s `image_list` is an untyped `list`. `apply_chat_template` is typed as a broad union but returns a `str` for Qwen3-VL, so its result is wrapped in `cast(str, ...)`.

### General

- **Apple Silicon is the primary target.** The app uses MLX (mflux + mlx-vlm), which requires Apple Silicon or Linux CUDA. CPU-only and Windows are not supported.
- **All models share MLX unified memory, but not with each other**: `_get_model()` and `_get_edit_model()` are separate `@st.cache_resource` entries holding separate weight copies (they do share the single HF download), and Qwen3-VL-2B-8bit is a third. **Weight size badly understates the real footprint** — with only the txt2img model loaded, `phys_footprint` measured 24 GB after one 1024×1024 run against 8.6GB of weights. The excess is MLX's Metal buffer cache plus activations — reclaimable rather than leaked, with `mx.clear_cache()` as the lever. Budget from measurement, not from safetensors sizes.
- **Streamlit widget state cannot be modified after instantiation.** Use `on_click` callbacks to set `st.session_state` keys for widgets, not direct assignment after the widget is created. Keyed expanders are the exception — `st.expander(key=...)` registers with writes allowed, so callbacks may set `st.session_state.input_expander` directly.
- **The working tree carries an untracked `.env`, but the app reads no environment variables** and needs no HF token — don't wire configuration through it.
- **`docs/screenshot-{light,dark}.png` are the README hero and are captured by hand** (run the app wide, one shot per appearance mode). `tests/test_readme.py` locks only that the README's local images exist and are git-tracked, never that they match the UI — so any change to the controls column means re-shooting both. **They are stale today** on two counts: they still show a "Mode: Distilled / Base" toggle the app no longer has, and they were shot against the retired custom purple palette rather than Streamlit's default themes.
- **Commit style:** Conventional Commits (including the local `harden(...)` type, and `feat!:` for breaking changes) with a body that states the evidence, on a `<type>/<slug>` topic branch merged with `--no-ff`. There is no pre-commit config — the `.claude/` hooks are the only local automation.
