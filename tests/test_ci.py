"""Contract tests for the GitHub Actions CI workflow and the Python pin.

CI (``.github/workflows/ci.yml``) is the public mirror of the local ``.claude``
quality hooks: it runs the same four gates (ruff format, ruff lint, ty, pytest)
on every push to ``main`` and every pull request, so a contributor who hasn't
installed the hooks still can't merge red. These tests lock the parts that are
easy to break silently:

  * the runner is macOS, not Linux — ``uv.lock`` resolves ``mlx`` to a
    CUDA-only build on Linux (``sys_platform == 'linux'``), so the suite can't
    even import on a GPU-less ubuntu runner. A well-meaning "switch to ubuntu
    for speed" must fail here, not in a red run.
  * CI runs all four gates the hooks enforce — dropping one would let an
    unformatted / untyped / failing change through review.
  * ``.python-version`` pins the runtime and stays consistent with
    ``requires-python``, so local uv and CI resolve the same interpreter.

YAML note: PyYAML follows YAML 1.1, which parses the bare mapping key ``on:`` as
the boolean ``True``. ``_load_workflow`` normalizes that back so the trigger
table is reachable as ``workflow["on"]``.
"""

import tomllib
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_PYTHON_VERSION_FILE = _REPO_ROOT / ".python-version"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# The four quality gates the .claude hooks enforce locally; CI must run the same
# set so an un-hooked contributor can't merge past them.
_REQUIRED_GATES = (
    "ruff format --check",
    "ruff check",
    "ty check",
    "pytest",
)


def _load_workflow():
    """Return the parsed ci.yml with the YAML-1.1 ``on:``→True key normalized."""
    with _CI_WORKFLOW.open() as fh:
        data = yaml.safe_load(fh)
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _run_commands(job):
    """Join every ``run:`` script body in a job's steps into one string."""
    return "\n".join(step["run"] for step in job["steps"] if "run" in step)


def _all_run_commands(workflow):
    """Every ``run:`` command across every job, joined into one string."""
    return "\n".join(_run_commands(job) for job in workflow["jobs"].values())


class TestCIWorkflow:
    """Contract for ``.github/workflows/ci.yml`` — the public mirror of the hooks."""

    def test_workflow_file_exists_and_parses(self):
        assert _CI_WORKFLOW.is_file()
        assert _load_workflow()  # valid, non-empty YAML

    def test_triggers_on_push_to_main_and_pull_request(self):
        # PR coverage is the whole point: contributors without the local hooks
        # get gated at review time. Losing either trigger silently drops that.
        triggers = _load_workflow()["on"]
        assert "pull_request" in triggers
        assert "main" in triggers["push"]["branches"]

    def test_test_and_typecheck_jobs_run_on_macos(self):
        # Load-bearing: uv.lock resolves mlx to a CUDA-only build on Linux, so
        # any job that imports the app — pytest or `ty check` — must run on
        # macOS or the module can't load. (A pure-ruff lint job needs no deps
        # and could legitimately run on Linux, so gate on the command, not the
        # job count.)
        gate_seen = False
        for name, job in _load_workflow()["jobs"].items():
            commands = _run_commands(job)
            if "pytest" in commands or "ty check" in commands:
                gate_seen = True
                assert job["runs-on"].startswith("macos"), (
                    f"job '{name}' runs the test/type gate off macOS: {job['runs-on']}"
                )
        # If a future edit renamed both invocations the loop would assert
        # nothing; fail loudly rather than pass vacuously.
        assert gate_seen, "no job runs the test/type gate; macOS rule unverified"

    def test_runs_all_four_quality_gates(self):
        # CI must enforce exactly what the .claude hooks do; a dropped gate
        # would let unformatted / untyped / failing code through review.
        commands = _all_run_commands(_load_workflow())
        for gate in _REQUIRED_GATES:
            assert gate in commands, f"CI is missing the `{gate}` gate"
        # The lint gate must FAIL on violations, not auto-fix them: "ruff check"
        # is a substring of "ruff check --fix", so pin the failing form. (The
        # local ruff-format.sh hook uses --fix intentionally; CI must not.)
        assert "ruff check --fix" not in commands, (
            "CI lint gate must fail on violations, not auto-fix them"
        )

    def test_installs_pinned_toolchain_via_uv(self):
        # The gates run the pinned toolchain from uv.lock (`uv run <tool>`), so
        # CI and the hooks can't disagree on a lint/type rule. That needs a
        # synced env: setup-uv + `uv sync`.
        workflow = _load_workflow()
        assert "uv sync" in _all_run_commands(workflow)
        uses = [
            step.get("uses", "")
            for job in workflow["jobs"].values()
            for step in job["steps"]
        ]
        assert any(action.startswith("astral-sh/setup-uv") for action in uses)

    def test_syncs_against_locked_lockfile(self):
        # `uv sync --locked` fails if uv.lock is stale (e.g. a version bump or
        # dependency change that forgot `uv lock`), so a drifted lockfile can't
        # slip through CI silently — plain `uv sync` would just re-resolve.
        assert "uv sync --locked" in _all_run_commands(_load_workflow()), (
            "CI must sync against the committed lockfile (`uv sync --locked`)"
        )

    def test_grants_no_write_permissions(self):
        # The workflow only reads source and runs lint/type/test; GITHUB_TOKEN
        # must not carry write scope. An explicit read-only block replaces the
        # repo-default (which can be read/write).
        permissions = _load_workflow().get("permissions")
        assert permissions, "workflow must declare an explicit permissions block"
        assert all(scope in ("read", "none") for scope in permissions.values()), (
            f"permissions grant more than read: {permissions}"
        )
        assert permissions.get("contents") == "read"  # checkout needs read

    def test_jobs_cap_their_runtime(self):
        # Every job must set an explicit timeout so a hang (e.g. a stalled
        # `uv sync`) fails fast instead of holding a runner for the 6-hour
        # default.
        for name, job in _load_workflow()["jobs"].items():
            assert isinstance(job.get("timeout-minutes"), int), (
                f"job '{name}' sets no timeout-minutes cap"
            )


class TestPythonVersionPin:
    """``.python-version`` keeps local uv and CI on the same interpreter."""

    def test_python_version_file_pins_312(self):
        assert _PYTHON_VERSION_FILE.is_file()
        assert _PYTHON_VERSION_FILE.read_text().strip() == "3.12"

    def test_pin_satisfies_requires_python(self):
        # The pin must not drift below pyproject's floor, or CI would run an
        # interpreter the project declares unsupported.
        with _PYPROJECT.open("rb") as fh:
            requires = tomllib.load(fh)["project"]["requires-python"]
        assert requires == ">=3.12"
        pin = tuple(int(p) for p in _PYTHON_VERSION_FILE.read_text().strip().split("."))
        floor = tuple(int(p) for p in requires.removeprefix(">=").split("."))
        assert pin >= floor, f"pin {pin} < requires-python {requires}"
