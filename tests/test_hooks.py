"""Tests for the Claude Code hooks under .claude/.

The hooks are plain bash + jq scripts wired up in ``.claude/settings.json``, so
this suite drives them the way Claude Code does: it feeds the tool payload on
stdin and asserts on the exit code and — for the guard — the JSON decision on
stdout. A stubbed ``uv`` on ``PATH`` lets the routing be checked (which file
types trigger the toolchain, which no-op, in what order) without paying for a
real ruff/ty/pytest run.

Two rules keep this file from growing back into a transcription of the scripts:

  * A test must be able to fail for a reason the script's own source does not
    already make obvious. Reading a ``case`` statement back as one parametrized
    case per arm cannot detect drift — it moves in lockstep with the thing it
    mirrors. Prefer asserting against external truth (real ruff, real git) or
    against a cross-script invariant no single file can express.
  * Exit codes are the contract with Claude Code (2 surfaces stderr back as an
    error to fix, 0 is silent), so every deliberate choice of one over the other
    is pinned.
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
_requires_git = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="mark-tests-pending.sh derives its filter from git",
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
        timeout=60,
    )


def _uv_stub_env(tmp_path, *, exit_code=0, fail_on=None):
    """Put a fake ``uv`` on ``PATH`` that records its args and exits ``exit_code``.

    Returns ``(env, log_path)``. The stub appends each invocation's args to the
    log, so a test can assert *whether*, *how* and *in what order* the real
    toolchain would be invoked without actually running ruff/ty/pytest.

    ``fail_on`` makes only the matching subcommand fail. run-tests.sh chains
    several gates, so a blanket non-zero exit would always trip the first one
    and leave the later ones untestable.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = bin_dir / "uv.log"
    log.write_text("")
    stub = bin_dir / "uv"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$STUB_UV_LOG"\n'
        'if [[ -n "${STUB_UV_FAIL_ON:-}" && "$*" == *"$STUB_UV_FAIL_ON"* ]]; then\n'
        "  exit 1\n"
        "fi\n"
        'exit "${STUB_UV_EXIT:-0}"\n'
    )
    stub.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "STUB_UV_LOG": str(log),
        "STUB_UV_EXIT": str(exit_code),
        "STUB_UV_FAIL_ON": fail_on or "",
    }
    return env, log


def _real_ruff(*args):
    """Run the project's actual ruff (never the stub), always cache-free."""
    return subprocess.run(
        ["uv", "run", "ruff", args[0], "--no-cache", *args[1:]],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


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


def _git_project(tmp_path):
    """A throwaway project that is a real git repo with the repo's own ignores.

    mark-tests-pending.sh asks ``git check-ignore`` rather than consulting a
    hardcoded list, so its tests must ask real git too — that externality is the
    entire point of the rewrite.
    """
    proj = _project(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=proj, check=True, capture_output=True, timeout=30
    )
    (proj / ".gitignore").write_text(
        "\n".join(
            [
                ".venv/",
                "__pycache__/",
                ".env",
                ".env.*",
                "!.env.example",
                "secrets.toml",
                ".claude/settings.local.json",
                ".claude/.tests-pending",
                "",
            ]
        )
    )
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


def _commands_for(event):
    settings = json.loads(_SETTINGS.read_text())
    return [
        hook["command"] for group in settings["hooks"][event] for hook in group["hooks"]
    ]


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

    def test_scripts_are_bound_to_the_correct_events(self):
        """Every hook that exists is actually wired, at the right event.

        This is the one assertion that can catch a hook *disappearing*. The
        others all iterate whatever settings.json happens to list, so a removed
        entry is invisible to them: delete the entire "Stop" block and the rest
        of this file stays green while ruff, ty and pytest go silently dead.
        It is not the transcription this file avoids elsewhere — those mirrored
        a list inside the same script under test. This pins the config against
        the scripts on disk, the exact mirror image of
        test_referenced_scripts_exist_and_are_executable below.
        """
        by_event = _commands_by_event()
        assert by_event["PreToolUse"] == {"guard-paths.sh"}
        # Cheap and file-scoped only — see test_no_whole_project_gate_runs_per_edit.
        assert by_event["PostToolUse"] == {"ruff-format.sh", "mark-tests-pending.sh"}
        assert by_event["Stop"] == {"run-tests.sh"}

    def test_commands_use_the_project_dir_prefix(self):
        # Catches a prefix/variable typo like ${CLAUDE_PROJ_DIR}, which would
        # disable the whole system with no error anywhere.
        for cmd in _all_commands():
            assert cmd.startswith(_HOOK_PREFIX), cmd

    def test_referenced_scripts_exist_and_are_executable(self):
        # A lost +x bit on a fresh clone silently disables a hook.
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

    def test_no_whole_project_gate_runs_per_edit(self):
        """The invariant this wiring exists to hold, checked against the scripts.

        PostToolUse fires once per Edit/Write/MultiEdit, so a whole-project
        command there re-derives one fact N times a turn and surfaces a
        transient failure on every intermediate edit of a multi-file refactor.
        Every such gate belongs in the Stop hook. Unlike reading settings.json
        back as set literals, this can fail for a reason the config alone does
        not show: it greps what the wired scripts actually run.
        """
        whole_project = ("pytest", "ty check", "ruff check --fix .", "ruff format .")
        for cmd in _commands_for("PostToolUse"):
            source = (_HOOKS_DIR / Path(cmd).name).read_text()
            body = "\n".join(
                line
                for line in source.splitlines()
                if not line.lstrip().startswith("#")
            )
            for needle in whole_project:
                assert needle not in body, (
                    f"{Path(cmd).name} runs '{needle}' per edit; it belongs in the Stop hook"
                )


@_requires_shell
class TestGuardPaths:
    """PreToolUse guard: deny writes to protected files, allow everything else.

    One parametrized case per *case-statement arm*, not per filename: the arms
    are `.env` (literal), `.env.*` (glob) and the literal names.
    """

    @pytest.mark.parametrize("name", [".env", ".env.local", "uv.lock", "secrets.toml"])
    def test_protected_files_are_denied(self, name, tmp_path):
        result = _run_hook("guard-paths.sh", tmp_path / name, project_dir=tmp_path)
        # The decision rides in the JSON, not the exit code.
        assert result.returncode == 0
        decision = json.loads(result.stdout)["hookSpecificOutput"]
        assert decision["hookEventName"] == "PreToolUse"
        assert decision["permissionDecision"] == "deny"
        assert name in decision["permissionDecisionReason"]

    def test_matching_is_case_insensitive(self, tmp_path):
        # macOS's default filesystem is case-insensitive, so a mis-cased path
        # opens the real protected file and must still be denied.
        result = _run_hook("guard-paths.sh", tmp_path / "UV.LOCK", project_dir=tmp_path)
        decision = json.loads(result.stdout)["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"

    def test_secret_free_template_is_allowed(self, tmp_path):
        # The template arm must be reached before the .env.* deny arm.
        result = _run_hook(
            "guard-paths.sh", tmp_path / ".env.example", project_dir=tmp_path
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""  # not denied

    def test_ordinary_file_is_allowed(self, tmp_path):
        # "notes.env" proves only true dotenv files are blocked, not any *.env.
        result = _run_hook(
            "guard-paths.sh", tmp_path / "notes.env", project_dir=tmp_path
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""  # no decision -> default permission flow

    def test_fails_closed_when_jq_is_missing(self, tmp_path):
        # An inverted default worth pinning: without jq the guard cannot parse
        # the payload, so it must DENY rather than silently allow a protected
        # write. The natural refactor (`command -v jq || exit 0`, which every
        # other hook does) flips this to permit.
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
    """PostToolUse: format the edited in-project file, else no-op."""

    @pytest.mark.parametrize("name", ["mod.py", "doc.md", "nb.ipynb"])
    def test_formats_scoped_to_that_file_without_lint_fixing(self, name, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        target = proj / name
        target.write_text("x=1\n")
        result = _run_hook("ruff-format.sh", target, project_dir=proj, env=env)
        assert result.returncode == 0
        logged = log.read_text()
        assert "ruff format" in logged
        assert str(target) in logged  # scoped to the edited file, not the repo
        # `ruff check --fix` deletes code and must not race a refactor: an
        # import added in one edit and first used several edits later is unused
        # in between. It belongs in the Stop hook, which also runs the formatter
        # after it so the blank line an F401 fix leaves behind gets cleaned up.
        assert "ruff check" not in logged

    @pytest.mark.parametrize(
        "name",
        # settings.json is the load-bearing correctly-cased case: JSON is valid
        # Python expression syntax, so ruff handed it does not error, it
        # silently rewrites the file as formatted Python. The mis-cased entries
        # are the same hazard reached through macOS's case-insensitive
        # filesystem, where NB.IPYNB opens the real notebook.
        ["settings.json", "notes.txt", "README.MD", "NB.IPYNB", "STUB.PYI"],
    )
    def test_skips_what_ruff_would_destroy(self, name, tmp_path):
        # Ruff picks its parser from the extension it is handed and parses
        # anything unrecognized as Python source. On a case-insensitive
        # filesystem a mis-cased path opens the real file, so handing ruff a
        # NB.IPYNB rewrites the notebook's JSON as formatted Python and destroys
        # it. These must no-op, not reach ruff. (Verified, not theoretical —
        # see TestRuffHandlesTheHookedExtensions for the other half.)
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        other = proj / name
        other.write_text("# hi\n")
        result = _run_hook("ruff-format.sh", other, project_dir=proj, env=env)
        assert result.returncode == 0
        assert log.read_text() == ""  # uv never invoked

    def test_skips_nonexistent_file(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        result = _run_hook(
            "ruff-format.sh", proj / "ghost.py", project_dir=proj, env=env
        )
        assert result.returncode == 0
        assert log.read_text() == ""

    def test_skips_file_outside_the_project(self, tmp_path):
        # An edit to a formattable file outside the repo must not be reformatted.
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        outside = tmp_path / "outside.py"
        outside.write_text("x=1\n")
        result = _run_hook("ruff-format.sh", outside, project_dir=proj, env=env)
        assert result.returncode == 0
        assert log.read_text() == ""

    def test_is_non_blocking(self, tmp_path):
        # Even if ruff "fails", the formatter hook must never block (exit 0) —
        # the deliberate opposite of the Stop hook's exit 2.
        proj = _project(tmp_path)
        env, _ = _uv_stub_env(tmp_path, exit_code=1)
        py = proj / "mod.py"
        py.write_text("x=1\n")
        result = _run_hook("ruff-format.sh", py, project_dir=proj, env=env)
        assert result.returncode == 0


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs the project's ruff")
class TestRuffHandlesTheHookedExtensions:
    """The real ruff must behave as ruff-format.sh and run-tests.sh assume.

    Every other test here drives a stubbed ``uv``: they lock *that* ruff is
    invoked, not that ruff does anything once it is. If a future ruff dropped
    Markdown or notebook formatting, those would stay green while the hook
    silently did nothing and CI's ``ruff format --check .`` diverged from it all
    over again. These are the only tests in the file that exercise something the
    repo does not control, which is why they survive every trim.
    """

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
        _real_ruff("format", str(target))
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
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            )
        )
        _real_ruff("format", str(nb))
        loaded = json.loads(nb.read_text())  # still a notebook, not Python
        assert "".join(loaded["cells"][0]["source"]).strip() == "x = 1"

    def test_lint_fix_must_run_before_format(self, tmp_path):
        """The verified bug the ruff split exists to prevent, against real ruff.

        `ruff check --fix` removing an unused import strands the blank line that
        followed it, and `ruff format --check` — CI's gate — then rejects the
        file. Formatter last is clean; formatter first is the red build the old
        PostToolUse hook caused silently (it discarded output and exited 0).
        """
        target = tmp_path / "mod.py"
        source = "import contextlib\n\nx = 1\n"
        fix = ("check", "--fix", "--select", "F401", str(target))
        fmt = ("format", str(target))

        def rejected_by_ci():
            return _real_ruff("format", "--check", str(target)).returncode != 0

        target.write_text(source)  # wrong order: formatter first
        _real_ruff(*fmt)
        _real_ruff(*fix)
        assert rejected_by_ci(), (
            "format-then-fix no longer strands a blank line; the Stop hook's "
            "ordering comment and this guard can be revisited"
        )

        target.write_text(source)  # shipped order: formatter last
        _real_ruff(*fix)
        _real_ruff(*fmt)
        assert not rejected_by_ci()


@_requires_shell
@_requires_git
class TestMarkTestsPendingHook:
    """PostToolUse: drop the end-of-turn marker for suite-relevant edits.

    The covered set is derived from ``git check-ignore``, so these assert
    against real git rather than against a transcription of the script.
    """

    def _marker(self, proj):
        return proj / ".claude" / ".tests-pending"

    @pytest.mark.parametrize(
        "rel",
        [
            "streamlit_app.py",
            # A tracked path in a directory the old thirteen globs never
            # listed. Verified against the old script: this did not arm, so the
            # local loop went silently dark on it. (A new module under tests/
            # *did* arm, via the old `*/tests/*` glob — the fail-open case was
            # new directories, not new files in known ones.)
            "assets/logo.svg",
            # Tracked, so test_secrets.py scans it — the old list wrongly
            # excluded it on the claim that "no test reads CLAUDE.md".
            "CLAUDE.md",
        ],
    )
    def test_arms_for_tracked_paths(self, rel, tmp_path):
        proj = _git_project(tmp_path)
        result = _run_hook("mark-tests-pending.sh", proj / rel, project_dir=proj)
        assert result.returncode == 0
        assert self._marker(proj).exists()

    @pytest.mark.parametrize(
        "rel",
        [
            ".claude/settings.local.json",  # personal overrides, gitignored
            ".env",  # never tracked; test_secrets.py cannot see it
            "__pycache__/streamlit_app.cpython-312.pyc",
        ],
    )
    def test_no_marker_for_gitignored_path(self, rel, tmp_path):
        proj = _git_project(tmp_path)
        result = _run_hook("mark-tests-pending.sh", proj / rel, project_dir=proj)
        assert result.returncode == 0
        assert not self._marker(proj).exists()

    def test_arms_when_git_cannot_answer(self, tmp_path):
        # Fail-closed. `git check-ignore` exits 128 outside a repo; an
        # unanswerable question must arm the suite, not skip it. The natural
        # refactor (treating any non-zero as "skip") inverts this silently.
        proj = _project(tmp_path)  # deliberately NOT a git repo
        result = _run_hook(
            "mark-tests-pending.sh", proj / "streamlit_app.py", project_dir=proj
        )
        assert result.returncode == 0
        assert self._marker(proj).exists()

    def test_no_marker_for_git_internals(self, tmp_path):
        proj = _git_project(tmp_path)
        result = _run_hook(
            "mark-tests-pending.sh", proj / ".git" / "config", project_dir=proj
        )
        assert result.returncode == 0
        assert not self._marker(proj).exists()

    def test_no_marker_for_file_outside_project(self, tmp_path):
        proj = _git_project(tmp_path)
        outside = tmp_path / "streamlit_app.py"
        result = _run_hook("mark-tests-pending.sh", outside, project_dir=proj)
        assert result.returncode == 0
        assert not self._marker(proj).exists()


@_requires_shell
class TestRunTestsHook:
    """Stop hook: ruff every turn; ty + pytest once per turn iff a marker is pending."""

    def _marker(self, proj):
        return proj / ".claude" / ".tests-pending"

    def test_runs_every_gate_when_pending(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        self._marker(proj).write_text("")
        result = _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        assert result.returncode == 0
        logged = log.read_text()
        for gate in ("ruff check --fix .", "ruff format .", "ty check .", "pytest"):
            assert gate in logged, f"{gate} not run"
        assert not self._marker(proj).exists()  # consumed -> loop-safe

    def test_a_format_pass_always_follows_the_lint_fix(self, tmp_path):
        # Load-bearing order: `ruff check --fix` strands a blank line where it
        # removed an unused import, so a formatter pass must come AFTER it or
        # CI's `ruff format --check .` rejects what this hook just wrote. Note
        # `rindex`: an unconditional format runs before the marker gate too, so
        # the assertion is that a format follows the fix, not that none precedes.
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        self._marker(proj).write_text("")
        _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        logged = log.read_text()
        assert logged.rindex("ruff format .") > logged.index("ruff check --fix .")

    def test_formats_but_skips_the_expensive_gates_when_not_pending(self, tmp_path):
        # ruff is unconditional (~0.06s, non-blocking) and also covers files
        # written by a Bash heredoc, which fire no PostToolUse hook at all.
        # ty + pytest cost ~14s together and stay behind the marker.
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path)
        result = _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        assert result.returncode == 0
        logged = log.read_text()
        assert "ruff format ." in logged
        # `--fix` deletes code and must not run where nothing validates it: on a
        # turn with no marker, ty and pytest never execute, so a tree-wide fix
        # would rewrite unrelated files with no gate behind it.
        assert "ruff check --fix" not in logged
        assert "pytest" not in logged
        assert "ty check" not in logged

    def test_ty_failure_blocks_with_exit_2_and_clears_marker(self, tmp_path):
        proj = _project(tmp_path)
        env, log = _uv_stub_env(tmp_path, fail_on="ty check")
        self._marker(proj).write_text("")
        result = _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        assert result.returncode == 2  # surfaced back to Claude as an error
        # Exit-code-agnostic wording: an env failure isn't mislabeled a type error.
        assert "ty check failed" in result.stderr
        assert "pytest" not in log.read_text()  # fails fast, before the 14s suite
        assert not self._marker(proj).exists()

    def test_pytest_failure_blocks_with_exit_2_and_clears_marker(self, tmp_path):
        proj = _project(tmp_path)
        env, _ = _uv_stub_env(tmp_path, fail_on="pytest")
        self._marker(proj).write_text("")
        result = _run_stop_hook("run-tests.sh", project_dir=proj, env=env)
        assert result.returncode == 2
        assert "pytest run failed" in result.stderr
        # Marker cleared even on failure, so a no-edit follow-up turn won't loop.
        assert not self._marker(proj).exists()
