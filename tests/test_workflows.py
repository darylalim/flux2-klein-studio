"""Cross-workflow invariants — properties that must hold for *every* GitHub
Actions workflow, not just one.

``tests/test_ci.py`` locks ``ci.yml``'s own contract (the four gates, the macOS
pin, the release job). This file locks what every workflow must share, so one
added later can't silently skip a repo-wide guarantee.

Injection safety is the one that bites hardest: an untrusted-ish input — a tag
name, a branch ref, a commit message — interpolated as a ``${{ ... }}``
expression directly inside a ``run:`` script is the classic GitHub Actions
command-injection vector (the expression is substituted into the shell *before*
it runs, so a crafted value executes). The safe pattern is to pass the value
through ``env:`` and reference ``"$VAR"``.

The second invariant encodes a GitHub behavior that fails *silently* rather than
loudly, which is what makes it worth a test: events raised by the default
``GITHUB_TOKEN`` do not start new workflow runs. A workflow that pushes a tag
and expects a tag-triggered workflow to publish it produces a tag and no
release, with every job green. The only correct shape is to create the tag and
the release together, in one ``gh release create --target <sha>`` call — which
is why ``ci.yml`` publishes inline instead of handing off. This repo used to
have exactly the hand-off shape it now forbids (a tag-triggered
``release.yml``); it worked only because a *human* pushed the tag.
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
            # An empty or comments-only .yml parses to None; `or {}` keeps a
            # placeholder/disabled workflow from crashing the sweep with AttributeError.
            data = yaml.safe_load(fh) or {}
        for job_name, job in (data.get("jobs") or {}).items():
            for step in job.get("steps", []):
                if "run" in step:
                    scripts.append((path.name, step.get("name", job_name), step["run"]))
    return scripts


def _code(body):
    """Strip comment lines so substring assertions match code, not prose.

    ci.yml's publish step explains *why* it does not push a tag, quoting the
    commands this sweep forbids. Matching raw text would flag the explanation.
    """
    return "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("#")
    )


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


class TestTagsAreNeverPushedByGit:
    """A tag pushed with the default ``GITHUB_TOKEN`` triggers nothing.

    GitHub's rule: "events triggered by the ``GITHUB_TOKEN`` will not create a
    new workflow run", with only ``workflow_dispatch`` and
    ``repository_dispatch`` exempted. It keys on the *token*, not the verb, so
    ``git push origin vX.Y.Z``, ``gh api .../git/refs`` and a ``create`` event
    are all equally suppressed.

    The consequence is a silent failure, not a red build: the tag appears, no
    release follows, every job is green. Creating the tag as part of
    ``gh release create --target <sha>`` sidesteps it entirely and is atomic
    besides — there is no window where a tag exists without its release.
    """

    def test_no_run_script_pushes_a_tag(self):
        scripts = _run_scripts()
        assert scripts, "no run: scripts found; this sweep would be vacuous"
        offenders = [
            f"{wf} :: {label}"
            for wf, label, body in scripts
            if any(cmd in _code(body) for cmd in ("git tag", "git push"))
        ]
        assert not offenders, (
            "run: scripts create or push a git ref. A ref pushed with the "
            "default GITHUB_TOKEN starts no workflow run, so a tag made this "
            "way is a dead end — create it via `gh release create --target "
            "<sha>` in the job that publishes. Offending steps: " + "; ".join(offenders)
        )

    def test_something_actually_publishes(self):
        # Non-vacuity with teeth: the assertion above is satisfied by a repo
        # that publishes nothing at all. Pin that a publisher still exists, so
        # deleting the release path can't quietly make this file pass harder.
        bodies = "\n".join(_code(body) for _, _, body in _run_scripts())
        assert "gh release create" in bodies, (
            "no workflow publishes a release; the no-git-tag rule above is "
            "vacuously satisfied"
        )
