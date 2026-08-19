"""Tests for the Claude Code hooks under .claude/.

The hooks are plain bash + jq scripts wired up in ``.claude/settings.json``, so
this suite drives them the way Claude Code does: it feeds the tool payload on
stdin and asserts on the exit code and — for the guard — the JSON decision on
stdout. A stubbed ``uv`` on ``PATH`` lets the routing of the format/type hooks be
checked (which file types trigger the toolchain, which no-op) without paying for
a real ruff/ty run. The test suite itself is gated at end of turn: a PostToolUse
hook (mark-tests-pending.sh) drops a marker and the Stop hook (run-tests.sh)
consumes it, so those two are tested via the marker rather than a per-edit run.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HOOKS_DIR = _REPO_ROOT / ".claude" / "hooks"
_SETTINGS = _HOOKS_DIR.parent / "settings.json"

# The subprocess-driven tests need bash to run the scripts and jq (which the
# parsing hooks shell out to); skip them cleanly on an environment that lacks
# either rather than fail on an infrastructure gap. The wiring tests below are
# pure JSON/filesystem checks and run regardless.
_requires_shell = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="hook scripts require bash and jq",
)

_HOOK_PREFIX = "${CLAUDE_PROJECT_DIR}/.claude/hooks/"


def _run_hook(script, file_path, *, project_dir, env=None, tool_name="Edit"):
    """Invoke a tool-event hook exactly like Claude Code: the payload on stdin.

    ``CLAUDE_PROJECT_DIR`` is always set (the scripts reference it); ``env``
    extends/overrides the child environment (e.g. a stubbed ``PATH``). Returns
    the CompletedProcess so callers can assert on returncode/stdout/stderr.
    """
    payload = json.dumps(
        {"tool_name": tool_name, "tool_input": {"file_path": str(file_path)}}
    )
    return _exec(script, payload, project_dir=project_dir, env=env)


def _run_stop_hook(script, *, project_dir, env=None):
    """Invoke a Stop hook: no tool payload (Stop receives none)."""
    return _exec(script, "", project_dir=project_dir, env=env)


def _exec(script, stdin, *, project_dir, env=None):
    child_env = {**os.environ, "CLAUDE_PROJECT_DIR": str(project_dir)}
    if env:
        child_env.update(env)
    return subprocess.run(
        ["bash", str(_HOOKS_DIR / script)],
        input=stdin,
        capture_output=True,
        text=True,
        env=child_env,
        timeout=30,
    )


def _uv_stub_env(tmp_path, *, exit_code=0):
    """Put a fake ``uv`` on ``PATH`` that records its args and exits ``exit_code``.

    Returns ``(env, log_path)``. The stub appends each invocation's args to the
    log, so a test can assert *whether* and *how* the real toolchain would be
    invoked without actually running ruff/ty/pytest.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = bin_dir / "uv.log"
    log.write_text("")
    stub = bin_dir / "uv"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$STUB_UV_LOG"\n'
        'exit "${STUB_UV_EXIT:-0}"\n'
    )
    stub.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "STUB_UV_LOG": str(log),
        "STUB_UV_EXIT": str(exit_code),
    }
    return env, log


def _no_jq_env(tmp_path):
    """A PATH with bash/cat/basename but NOT jq, to exercise the fail-closed path."""
    bin_dir = tmp_path / "nojq-bin"
    bin_dir.mkdir(exist_ok=True)
    for tool in ("bash", "cat", "basename"):
        src = shutil.which(tool)
        if src and not (bin_dir / tool).exists():
            (bin_dir / tool).symlink_to(src)
    return {"PATH": str(bin_dir)}


def _project(tmp_path):
    """A throwaway project dir with a .claude/ subdir (for the marker)."""
    proj = tmp_path / "proj"
    (proj / ".claude").mkdir(parents=True, exist_ok=True)
    return proj


def _commands_by_event():
    """{event: {script basename, ...}} across settings.json."""
    settings = json.loads(_SETTINGS.read_text())
    return {
        event: {
            Path(hook["command"]).name for group in groups for hook in group["hooks"]
        }
        for event, groups in settings["hooks"].items()
    }


def _matchers_for(event):
    settings = json.loads(_SETTINGS.read_text())
    return [group.get("matcher") for group in settings["hooks"][event]]


def _all_commands():
    settings = json.loads(_SETTINGS.read_text())
    return [
        hook["command"]
        for groups in settings["hooks"].values()
        for group in groups
        for hook in group["hooks"]
    ]


class TestHookWiring:
    """settings.json must reference real, executable scripts, bound to events."""

    def test_settings_is_valid_json_with_expected_events(self):
        settings = json.loads(_SETTINGS.read_text())
        assert {"PreToolUse", "PostToolUse", "Stop"} <= set(settings["hooks"])

    def test_scripts_are_bound_to_the_correct_events(self):
        by_event = _commands_by_event()
        assert by_event["PreToolUse"] == {"guard-paths.sh"}
        assert by_event["PostToolUse"] == {
            "ruff-format.sh",
            "ty-check.sh",
            "mark-tests-pending.sh",
        }
        assert by_event["Stop"] == {"run-tests.sh"}

    def test_commands_use_the_project_dir_prefix(self):
        # Catches a prefix/variable typo like ${CLAUDE_PROJ_DIR}.
        for cmd in _all_commands():
            assert cmd.startswith(_HOOK_PREFIX), cmd

    def test_referenced_scripts_exist_and_are_executable(self):
        commands = _all_commands()
        assert commands, "no hook commands configured"
        for cmd in commands:
            script = _HOOKS_DIR / Path(cmd).name
            assert script.exists(), f"{cmd} -> missing {script}"
            assert os.access(script, os.X_OK), f"{script} is not executable"

    @pytest.mark.parametrize("event", ["PreToolUse", "PostToolUse"])
    def test_tool_matchers_cover_edit_write_multiedit(self, event):
        # A matcher typo (or a dropped MultiEdit) would silently disable the
        # hooks on those tools while the suite stayed green — assert all three.
        matchers = _matchers_for(event)
        assert matchers, f"{event} has no matcher groups"
        assert all(
            {"Edit", "Write", "MultiEdit"} <= set(m.split("|")) for m in matchers
        ), f"{event} matchers miss a tool: {matchers}"


@_requires_shell
class TestGuardPaths:
    """PreToolUse guard: deny writes to protected files, allow everything else."""

    @pytest.mark.parametrize(
        "name",
        [".env", ".env.local", ".env.production", "uv.lock", "secrets.toml"],
    )
    def test_protected_files_are_denied(self, name, tmp_path):
        result = _run_hook("guard-paths.sh", tmp_path / name, project_dir=tmp_path)
        # The decision rides in the JSON, not the exit code.
        assert result.returncode == 0
        decision = json.loads(result.stdout)["hookSpecificOutput"]
        assert decision["hookEventName"] == "PreToolUse"
        assert decision["permissionDecision"] == "deny"
        assert name in decision["permissionDecisionReason"]

    @pytest.mark.parametrize("name", ["UV.LOCK", ".ENV", ".Env.Local"])
    def test_matching_is_case_insensitive(self, name, tmp_path):
        # macOS's default filesystem is case-insensitive, so a mis-cased path
        # opens the real protected file and must still be denied.
        result = _run_hook("guard-paths.sh", tmp_path / name, project_dir=tmp_path)
        decision = json.loads(result.stdout)["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"

    @pytest.mark.parametrize(
        "name", [".env.example", ".env.sample", ".env.template", ".env.dist"]
    )
    def test_secret_free_templates_are_allowed(self, name, tmp_path):
        result = _run_hook("guard-paths.sh", tmp_path / name, project_dir=tmp_path)
        assert result.returncode == 0
        assert result.stdout.strip() == ""  # not denied

    @pytest.mark.parametrize(
        "name",
        # "notes.env" proves only true dotenv files are blocked, not any *.env.
        ["streamlit_app.py", "tests/test_app.py", "notes.env", "README.md"],
    )
    def test_ordinary_files_are_allowed(self, name, tmp_path):
        result = _run_hook("guard-paths.sh", tmp_path / name, project_dir=tmp_path)
        assert result.returncode == 0
        assert result.stdout.strip() == ""  # no decision -> default permission flow

    def test_missing_file_path_is_a_noop(self, tmp_path):
        # A tool call with no file_path (e.g. Bash) is defended by jq's `// empty`.
        result = _exec(
            "guard-paths.sh",
            json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}}),
            project_dir=tmp_path,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_fails_closed_when_jq_is_missing(self, tmp_path):
        # Without jq the guard cannot parse the payload, so it must DENY rather
        # than silently allow a protected write through.
        result = _run_hook(
            "guard-paths.sh",
            tmp_path / "anything.txt",
            project_dir=tmp_path,
            env=_no_jq_env(tmp_path),
        )
        decision = json.loads(result.stdout)["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"
        assert "jq" in decision["permissionDecisionReason"]


@_requires_shell
class TestRuffFormatHook:
    """PostToolUse: format edited in-project files ruff handles, else no-op."""

    # `ruff check` handles .pyi and .ipynb as well as .py, so all three get the
    # lint-fix pass; only Markdown is format-only (see the test below).
    @pytest.mark.parametrize("name", ["mod.py", "stub.pyi", "nb.ipynb"])
    def test_formats_and_lint_fixes_python_scoped_to_that_file(self, name, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        target = proj / name
        target.write_text("x=1\n")
        result = _run_hook("ruff-format.sh", target, project_dir=proj, env=env)
        assert result.returncode == 0
        logged = log.read_text()
        assert "ruff format" in logged
        assert "ruff check --fix" in logged
        assert str(target) in logged  # scoped to the edited file, not the repo

    def test_formats_markdown_without_lint_fixing_it(self, tmp_path):
        # `ruff format` (>=0.16) formats Python fenced in Markdown and CI checks
        # the repo root, so docs are in scope here or the two would disagree.
        # `ruff check` has no Markdown support, so the lint-fix pass must not run.
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        md = proj / "README.md"
        md.write_text("# hi\n")
        result = _run_hook("ruff-format.sh", md, project_dir=proj, env=env)
        assert result.returncode == 0
        logged = log.read_text()
        assert "ruff format" in logged
        assert str(md) in logged  # scoped to the edited file, not the repo
        assert "ruff check" not in logged

    @pytest.mark.parametrize(
        "name",
        # The mis-cased entries are the point: ruff picks its parser from the
        # extension it is handed and parses anything unrecognized as Python
        # source. On a case-insensitive filesystem a mis-cased path opens the
        # real file, so handing ruff a NB.IPYNB rewrites the notebook's JSON as
        # formatted Python and destroys it. These must no-op, not reach ruff.
        [
            "notes.txt",
            "settings.json",
            "pyproject.toml",
            "README.MD",
            "NB.IPYNB",
            "STUB.PYI",
        ],
    )
    def test_skips_file_ruff_does_not_format(self, name, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        other = proj / name
        other.write_text("# hi\n")
        result = _run_hook("ruff-format.sh", other, project_dir=proj, env=env)
        assert result.returncode == 0
        assert log.read_text() == ""  # uv never invoked

    @pytest.mark.parametrize(
        "name", ["ghost.py", "ghost.md", "ghost.pyi", "ghost.ipynb"]
    )
    def test_skips_nonexistent_file(self, name, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        result = _run_hook("ruff-format.sh", proj / name, project_dir=proj, env=env)
        assert result.returncode == 0
        assert log.read_text() == ""

    @pytest.mark.parametrize(
        "name", ["outside.py", "outside.md", "outside.pyi", "outside.ipynb"]
    )
    def test_skips_file_outside_the_project(self, name, tmp_path):
        # An edit to a formattable file outside the repo must not be reformatted.
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        outside = tmp_path / name
        outside.write_text("x=1\n")
        result = _run_hook("ruff-format.sh", outside, project_dir=proj, env=env)
        assert result.returncode == 0
        assert log.read_text() == ""

    def test_is_non_blocking(self, tmp_path):
        # Even if ruff "fails", the formatter hook must never block (exit 0).
        proj = _project(tmp_path)
        env, _ = _uv_stub_env(tmp_path, exit_code=1)
        py = proj / "mod.py"
        py.write_text("x=1\n")
        result = _run_hook("ruff-format.sh", py, project_dir=proj, env=env)
        assert result.returncode == 0


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs the project's ruff")
class TestRuffHandlesTheHookedExtensions:
    """The real ruff must reformat every extension ruff-format.sh routes to it.

    The hook tests above drive a stubbed ``uv``: they lock *that* ruff is
    invoked, not that ruff does anything once it is. If a future ruff dropped
    Markdown or notebook formatting, those tests would stay green while the hook
    silently did nothing and CI's ``ruff format --check .`` diverged from it all
    over again. This runs the project's actual ruff once per routed extension to
    pin the assumption the hook rests on. (``pyproject.toml`` declares the
    matching ``ruff>=0.16`` floor.)
    """

    @staticmethod
    def _ruff_format(target):
        subprocess.run(
            ["uv", "run", "ruff", "format", "--no-cache", str(target)],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )

    @pytest.mark.parametrize(
        ("name", "source", "expected"),
        [
            # Markdown is why the hook was widened: ruff >=0.16 formats Python
            # fenced inside it, and CI's `ruff format --check .` walks the root.
            ("doc.md", "# t\n\n```python\nx  =  1\n```\n", "x = 1"),
            ("mod.py", "x  =  1\n", "x = 1"),
            ("stub.pyi", "def f(x:int)->int: ...\n", "def f(x: int) -> int: ..."),
        ],
    )
    def test_ruff_reformats(self, name, source, expected, tmp_path):
        target = tmp_path / name
        target.write_text(source)
        self._ruff_format(target)
        assert expected in target.read_text()

    def test_ruff_reformats_notebook_cells_in_place(self, tmp_path):
        # .ipynb is JSON, so assert on the cell source *and* that the envelope
        # still parses: ruff handed a notebook under an extension it does not
        # recognize rewrites the JSON as a Python dict literal, which would
        # leave the expected substring present but the notebook unreadable.
        nb = tmp_path / "nb.ipynb"
        nb.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": ["x  =  1\n"],
                        }
                    ],
                    "metadata": {
                        "kernelspec": {
                            "display_name": "Python 3",
                            "language": "python",
                            "name": "python3",
                        },
                        "language_info": {"name": "python", "version": "3.12.0"},
                    },
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            )
        )
        self._ruff_format(nb)
        loaded = json.loads(nb.read_text())  # still a notebook, not Python
        assert "".join(loaded["cells"][0]["source"]).strip() == "x = 1"


@_requires_shell
class TestTyCheckHook:
    """PostToolUse: type-check on in-project Python edits; exit 2 on failure."""

    def test_type_checks_after_python_edit(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        result = _run_hook("ty-check.sh", proj / "mod.py", project_dir=proj, env=env)
        assert result.returncode == 0
        assert "ty check" in log.read_text()

    def test_skips_non_python_file(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        result = _run_hook("ty-check.sh", proj / "README.md", project_dir=proj, env=env)
        assert result.returncode == 0
        assert log.read_text() == ""

    def test_skips_file_outside_the_project(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        outside = tmp_path / "outside.py"
        result = _run_hook("ty-check.sh", outside, project_dir=proj, env=env)
        assert result.returncode == 0
        assert log.read_text() == ""

    def test_failure_blocks_with_exit_2_and_neutral_wording(self, tmp_path):
        proj = _project(tmp_path)
        env, _ = _uv_stub_env(tmp_path, exit_code=1)
        result = _run_hook("ty-check.sh", proj / "mod.py", project_dir=proj, env=env)
        assert result.returncode == 2  # surfaced back to Claude as an error
        # Exit-code-agnostic wording: an env failure isn't mislabeled a type error.
        assert "ty check failed" in result.stderr


@_requires_shell
class TestMarkTestsPendingHook:
    """PostToolUse: drop the end-of-turn marker for suite-relevant edits."""

    def _marker(self, proj):
        return proj / ".claude" / ".tests-pending"

    # One entry per test module that reads a repo file: if the suite asserts
    # on it, editing it must arm the end-of-turn pytest run. Drift here is
    # silent -- CI still fails, but the local loop goes quiet.
    @pytest.mark.parametrize(
        "rel",
        [
            "streamlit_app.py",
            "tests/test_app.py",
            ".streamlit/config.toml",
            ".github/workflows/ci.yml",
            ".github/workflows/release.yml",
            ".github/release.yml",
            ".claude/settings.json",
            ".claude/hooks/ruff-format.sh",
            "README.md",
            "docs/screenshot-light.png",
            "examples/woman1.webp",
            "pyproject.toml",
            "LICENSE",
            ".python-version",
        ],
    )
    def test_marks_pending_for_covered_files(self, rel, tmp_path):
        proj = _project(tmp_path)
        result = _run_hook("mark-tests-pending.sh", proj / rel, project_dir=proj)
        assert result.returncode == 0
        assert self._marker(proj).exists()

    @pytest.mark.parametrize(
        "rel",
        [
            # No test reads CLAUDE.md -- it is covered by `ruff format
            # --check`, which is ruff-format.sh's job, not pytest's. Listing
            # it here keeps that exclusion deliberate rather than an oversight.
            "CLAUDE.md",
            # Personal permission overrides; gitignored and unasserted.
            ".claude/settings.local.json",
            "notes.txt",
            ".gitignore",
        ],
    )
    def test_no_marker_for_unrelated_edit(self, rel, tmp_path):
        proj = _project(tmp_path)
        result = _run_hook("mark-tests-pending.sh", proj / rel, project_dir=proj)
        assert result.returncode == 0
        assert not self._marker(proj).exists()

    def test_no_marker_for_file_outside_project(self, tmp_path):
        proj = _project(tmp_path)
        outside = tmp_path / "streamlit_app.py"
        result = _run_hook("mark-tests-pending.sh", outside, project_dir=proj)
        assert result.returncode == 0
        assert not self._marker(proj).exists()


@_requires_shell
class TestRunTestsHook:
    """Stop hook: run the suite once per turn iff a marker is pending."""

    def _marker(self, proj):
        return proj / ".claude" / ".tests-pending"

    def test_runs_pytest_and_consumes_marker_when_pending(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        self._marker(proj).write_text("")
        result = _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        assert result.returncode == 0
        assert "pytest" in log.read_text()
        assert not self._marker(proj).exists()  # consumed -> loop-safe

    def test_noop_when_not_pending(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        result = _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        assert result.returncode == 0
        assert log.read_text() == ""  # pytest never invoked

    def test_failure_blocks_with_exit_2_and_clears_marker(self, tmp_path):
        proj = _project(tmp_path)
        env, _ = _uv_stub_env(tmp_path, exit_code=1)
        self._marker(proj).write_text("")
        result = _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        assert result.returncode == 2
        assert "pytest run failed" in result.stderr
        # Marker cleared even on failure, so a no-edit follow-up turn won't loop.
        assert not self._marker(proj).exists()
