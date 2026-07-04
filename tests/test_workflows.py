"""Cross-workflow invariants — properties that must hold for *every* GitHub
Actions workflow, not just one.

``tests/test_ci.py`` and ``tests/test_release.py`` each lock a single workflow's
unique contract (CI's four gates and macOS pin; the release version gate). This
file locks what they must *share*, so a workflow added later can't silently skip
a repo-wide guarantee.

Injection safety is the one that bites hardest: an untrusted-ish input — a tag
name, a branch ref, a commit message — interpolated as a ``${{ ... }}``
expression directly inside a ``run:`` script is the classic GitHub Actions
command-injection vector (the expression is substituted into the shell *before*
it runs, so a crafted value executes). The safe pattern is to pass the value
through ``env:`` and reference ``"$VAR"``. ``release.yml`` is already checked in
test_release.py; this sweep extends the same guarantee to ``ci.yml`` — whose
``run:`` scripts nothing else checks — and to every future workflow.
"""

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"


def _workflow_files():
    """Every workflow file in ``.github/workflows/`` (.yml and .yaml)."""
    return sorted(_WORKFLOWS_DIR.glob("*.yml")) + sorted(_WORKFLOWS_DIR.glob("*.yaml"))


def _run_scripts():
    """(file, step-label, run-body) for every ``run:`` step across all workflows."""
    scripts = []
    for path in _workflow_files():
        with path.open() as fh:
            data = yaml.safe_load(fh)
        for job_name, job in (data.get("jobs") or {}).items():
            for step in job.get("steps", []):
                if "run" in step:
                    scripts.append((path.name, step.get("name", job_name), step["run"]))
    return scripts


class TestWorkflowInjectionSafety:
    """No ``run:`` script in any workflow may interpolate a ``${{ }}`` expression."""

    def test_workflows_are_present(self):
        # Non-vacuity: if the glob finds nothing (e.g. a moved directory) the
        # sweep below would pass trivially and stop guarding anything.
        assert _workflow_files(), f"no workflow files found under {_WORKFLOWS_DIR}"

    def test_no_run_script_interpolates_an_expression(self):
        scripts = _run_scripts()
        # Non-vacuity again: at least one workflow must actually have a run: step,
        # or a refactor that moved all logic into actions would make this pass
        # without checking anything.
        assert scripts, "no run: scripts found; the injection sweep would be vacuous"
        offenders = [f"{wf} :: {label}" for wf, label, body in scripts if "${{" in body]
        assert not offenders, (
            "run: scripts interpolate a ${{ }} expression directly, which is a "
            "command-injection risk — pass the value via env: and reference "
            '"$VAR" instead. Offending steps: ' + "; ".join(offenders)
        )
