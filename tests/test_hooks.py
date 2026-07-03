"""Tests for the Claude Code hooks under .claude/.

The hooks are plain bash + jq scripts wired up in ``.claude/settings.json``, so
this suite drives them the way Claude Code does: it feeds the PreToolUse /
PostToolUse tool payload on stdin and asserts on the exit code and — for the
guard — the JSON decision on stdout. A stubbed ``uv`` on ``PATH`` lets the
routing of the PostToolUse hooks be checked (which file types trigger the
toolchain, which no-op) without paying for a real ruff/ty/pytest run.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_HOOKS_DIR = Path(__file__).resolve().parent.parent / ".claude" / "hooks"
_SETTINGS = _HOOKS_DIR.parent / "settings.json"

# The subprocess-driven tests need bash to run the scripts and jq (which every
# hook shells out to); skip them cleanly on an environment that lacks either
# rather than fail on an infrastructure gap. The wiring tests below are pure
# JSON/filesystem checks and run regardless.
_requires_shell = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="hook scripts require bash and jq",
)


def _run_hook(script, file_path, *, project_dir, env=None, tool_name="Edit"):
    """Invoke a hook exactly like Claude Code: the tool payload on stdin.

    ``CLAUDE_PROJECT_DIR`` is always set (the scripts reference it); ``env``
    extends/overrides the child environment (e.g. a stubbed ``PATH``). Returns
    the CompletedProcess so callers can assert on returncode/stdout/stderr.
    """
    payload = json.dumps(
        {"tool_name": tool_name, "tool_input": {"file_path": str(file_path)}}
    )
    child_env = {**os.environ, "CLAUDE_PROJECT_DIR": str(project_dir)}
    if env:
        child_env.update(env)
    return subprocess.run(
        ["bash", str(_HOOKS_DIR / script)],
        input=payload,
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


def _configured_commands():
    """Every hook command string across all events in settings.json."""
    settings = json.loads(_SETTINGS.read_text())
    return [
        hook["command"]
        for event in settings["hooks"].values()
        for group in event
        for hook in group["hooks"]
    ]


class TestHookWiring:
    """settings.json must reference real, executable hook scripts."""

    def test_settings_is_valid_json_with_hooks(self):
        settings = json.loads(_SETTINGS.read_text())
        assert "hooks" in settings
        assert {"PreToolUse", "PostToolUse"} <= set(settings["hooks"])

    def test_expected_hooks_are_wired(self):
        names = {Path(cmd).name for cmd in _configured_commands()}
        assert names == {
            "guard-paths.sh",
            "ruff-format.sh",
            "ty-check.sh",
            "pytest.sh",
        }

    def test_referenced_scripts_exist_and_are_executable(self):
        commands = _configured_commands()
        assert commands, "no hook commands configured"
        for cmd in commands:
            # Commands look like "${CLAUDE_PROJECT_DIR}/.claude/hooks/foo.sh".
            script = _HOOKS_DIR / Path(cmd).name
            assert script.exists(), f"{cmd} -> missing {script}"
            assert os.access(script, os.X_OK), f"{script} is not executable"

    def test_guard_runs_before_edits_writes(self):
        settings = json.loads(_SETTINGS.read_text())
        matchers = {group["matcher"] for group in settings["hooks"]["PreToolUse"]}
        # The guard must fire on the file-mutating tools.
        assert any("Edit" in m and "Write" in m for m in matchers), (
            f"guard matcher does not cover Edit/Write: {matchers}"
        )


@_requires_shell
class TestGuardPaths:
    """PreToolUse guard: deny writes to protected files, allow everything else."""

    @pytest.mark.parametrize(
        "name", [".env", ".env.local", ".env.production", "uv.lock"]
    )
    def test_protected_files_are_denied(self, name, tmp_path):
        result = _run_hook("guard-paths.sh", tmp_path / name, project_dir=tmp_path)
        # The decision rides in the JSON, not the exit code.
        assert result.returncode == 0
        decision = json.loads(result.stdout)["hookSpecificOutput"]
        assert decision["hookEventName"] == "PreToolUse"
        assert decision["permissionDecision"] == "deny"
        assert name in decision["permissionDecisionReason"]

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
        payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}})
        result = subprocess.run(
            ["bash", str(_HOOKS_DIR / "guard-paths.sh")],
            input=payload,
            capture_output=True,
            text=True,
            env={**os.environ, "CLAUDE_PROJECT_DIR": str(tmp_path)},
            timeout=30,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""


@_requires_shell
class TestRuffFormatHook:
    """PostToolUse: format + lint-fix edited Python files, no-op otherwise."""

    def test_formats_python_file(self, tmp_path):
        env, log = _uv_stub_env(tmp_path)
        py = tmp_path / "mod.py"
        py.write_text("x=1\n")
        result = _run_hook("ruff-format.sh", py, project_dir=tmp_path, env=env)
        assert result.returncode == 0
        logged = log.read_text()
        assert "ruff format" in logged
        assert "ruff check --fix" in logged

    def test_skips_non_python_file(self, tmp_path):
        env, log = _uv_stub_env(tmp_path)
        md = tmp_path / "README.md"
        md.write_text("# hi\n")
        result = _run_hook("ruff-format.sh", md, project_dir=tmp_path, env=env)
        assert result.returncode == 0
        assert log.read_text() == ""  # uv never invoked

    def test_skips_nonexistent_python_file(self, tmp_path):
        # The `-f` guard means a path that no longer exists is left alone.
        env, log = _uv_stub_env(tmp_path)
        result = _run_hook(
            "ruff-format.sh", tmp_path / "ghost.py", project_dir=tmp_path, env=env
        )
        assert result.returncode == 0
        assert log.read_text() == ""

    def test_is_non_blocking(self, tmp_path):
        # Even if ruff "fails", the formatter hook must never block (exit 0).
        env, _ = _uv_stub_env(tmp_path, exit_code=1)
        py = tmp_path / "mod.py"
        py.write_text("x=1\n")
        result = _run_hook("ruff-format.sh", py, project_dir=tmp_path, env=env)
        assert result.returncode == 0


@_requires_shell
class TestTyCheckHook:
    """PostToolUse: type-check on Python edits; block (exit 2) on type errors."""

    def test_type_checks_after_python_edit(self, tmp_path):
        env, log = _uv_stub_env(tmp_path)
        result = _run_hook(
            "ty-check.sh", tmp_path / "mod.py", project_dir=tmp_path, env=env
        )
        assert result.returncode == 0
        assert "ty check" in log.read_text()

    def test_skips_non_python_file(self, tmp_path):
        env, log = _uv_stub_env(tmp_path)
        result = _run_hook(
            "ty-check.sh", tmp_path / "README.md", project_dir=tmp_path, env=env
        )
        assert result.returncode == 0
        assert log.read_text() == ""

    def test_type_errors_block_with_exit_2(self, tmp_path):
        env, _ = _uv_stub_env(tmp_path, exit_code=1)
        result = _run_hook(
            "ty-check.sh", tmp_path / "mod.py", project_dir=tmp_path, env=env
        )
        assert result.returncode == 2  # surfaced back to Claude as an error
        assert "ty reported type errors" in result.stderr


@_requires_shell
class TestPytestHook:
    """PostToolUse: run the suite on app/test edits; block (exit 2) on failures."""

    @pytest.mark.parametrize("rel", ["streamlit_app.py", "tests/test_app.py"])
    def test_runs_on_app_and_test_edits(self, rel, tmp_path):
        env, log = _uv_stub_env(tmp_path)
        result = _run_hook("pytest.sh", tmp_path / rel, project_dir=tmp_path, env=env)
        assert result.returncode == 0
        assert "pytest" in log.read_text()

    def test_skips_unrelated_edits(self, tmp_path):
        env, log = _uv_stub_env(tmp_path)
        result = _run_hook(
            "pytest.sh", tmp_path / "README.md", project_dir=tmp_path, env=env
        )
        assert result.returncode == 0
        assert log.read_text() == ""

    def test_failures_block_with_exit_2(self, tmp_path):
        env, _ = _uv_stub_env(tmp_path, exit_code=1)
        result = _run_hook(
            "pytest.sh", tmp_path / "streamlit_app.py", project_dir=tmp_path, env=env
        )
        assert result.returncode == 2
        assert "pytest failures" in result.stderr
