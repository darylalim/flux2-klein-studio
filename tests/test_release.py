"""Contract tests for the tag-triggered release workflow.

``.github/workflows/release.yml`` publishes a GitHub release when a ``vX.Y.Z``
tag is pushed. It is the release-side mirror of the repo's "lock consistency,
don't hope for it" discipline (``uv sync --locked``, ``tests/test_license.py``):
before publishing, it asserts the tag names the same version ``pyproject.toml``
declares, so a mislabeled tag fails the build instead of shipping a wrong
release. These tests lock the parts that are easy to break silently:

  * it fires on ``v*`` tags only — not on every push to ``main`` (that is
    ci.yml's job); a release firing on branch pushes would spam releases.
  * the version-consistency gate stays present — dropping it would let a tag
    whose version disagrees with pyproject.toml publish anyway.
  * ``run:`` scripts never interpolate a ``${{ }}`` expression directly (the
    tag name flows through ``env:`` and is referenced as ``"$TAG"``), so a
    crafted tag can't inject shell.
  * least-privilege ``contents: write`` (releases need it) and nothing broader.

YAML note: PyYAML (YAML 1.1) parses the bare mapping key ``on:`` as ``True``;
``_load_workflow`` normalizes it back so triggers are reachable as
``workflow["on"]`` — same treatment as tests/test_ci.py.
"""

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yml"
_RELEASE_NOTES_CONFIG = _REPO_ROOT / ".github" / "release.yml"
_PYTHON_VERSION_FILE = _REPO_ROOT / ".python-version"


def _load_workflow():
    """Return the parsed release.yml with the YAML-1.1 ``on:``→True key fixed."""
    with _RELEASE_WORKFLOW.open() as fh:
        data = yaml.safe_load(fh)
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _all_steps(workflow):
    """Every step across every job."""
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def _all_run_commands(workflow):
    """Every ``run:`` script body, joined into one string."""
    return "\n".join(step["run"] for step in _all_steps(workflow) if "run" in step)


class TestReleaseWorkflow:
    """Contract for ``.github/workflows/release.yml``."""

    def test_workflow_file_exists_and_parses(self):
        assert _RELEASE_WORKFLOW.is_file()
        assert _load_workflow()  # valid, non-empty YAML

    def test_triggers_only_on_version_tags(self):
        # Releases are cut by pushing a vX.Y.Z tag; the trigger must be tag-based
        # and scoped to the v-prefix, or unrelated tags would publish releases.
        triggers = _load_workflow()["on"]
        assert "push" in triggers, "release must trigger on a push of a tag"
        tags = triggers["push"].get("tags")
        assert tags and any(pattern.startswith("v") for pattern in tags), (
            f"release must trigger on v* tags, got tags={tags}"
        )

    def test_does_not_trigger_on_branch_push_or_pr(self):
        # A release must NOT fire on every push to main or on PRs — that is
        # ci.yml's contract. Sharing that trigger here would publish a release
        # (or attempt to) on ordinary commits.
        triggers = _load_workflow()["on"]
        assert "branches" not in triggers.get("push", {}), (
            "release must not trigger on branch pushes, only on tags"
        )
        assert "pull_request" not in triggers

    def test_grants_only_contents_write(self):
        # Publishing a release needs write to contents (releases live under the
        # contents scope) and nothing more. An explicit block replaces the
        # repo-default (which can be broader).
        permissions = _load_workflow().get("permissions")
        assert permissions, "workflow must declare an explicit permissions block"
        assert permissions.get("contents") == "write", (
            "release must grant contents: write to publish"
        )
        assert all(
            scope in ("read", "write", "none") for scope in permissions.values()
        ), f"permissions grant more than write: {permissions}"

    def test_asserts_tag_matches_pyproject_version(self):
        # The enforced consistency gate: the published tag must name the same
        # version the code declares. Losing this would let a mislabeled tag ship.
        commands = _all_run_commands(_load_workflow())
        assert "tomllib" in commands and "pyproject" in commands, (
            "release must read the declared version from pyproject.toml"
        )
        assert "github.ref_name" in _RELEASE_WORKFLOW.read_text(), (
            "release must capture the pushed tag name to compare against it"
        )
        # The tag carries a leading `v` (v0.6.5) but pyproject declares the bare
        # version (0.6.5); the check must strip it, or *every* release would fail
        # the gate on a spurious "v0.6.5 != 0.6.5". Locks the normalization.
        assert "TAG#v" in commands, (
            "release must strip the leading 'v' from the tag before comparing"
        )
        # The gate must fire on INEQUALITY (mismatch fails). Locking the operator
        # catches an accidental inversion to `==` — which would publish mismatched
        # tags and reject matching ones while leaving every other token in this
        # test satisfied, so none of the checks above would notice.
        assert "!=" in commands, (
            "the version check must gate on inequality (tag != pyproject version)"
        )
        assert "exit 1" in commands, (
            "a tag/version mismatch must fail the build (exit 1), not warn"
        )

    def test_publishes_with_gh_release_create(self):
        commands = _all_run_commands(_load_workflow())
        assert "gh release create" in commands

    def test_publishes_auto_generated_notes(self):
        # From the second release on, notes are diffed against the previous tag;
        # dropping --generate-notes would silently ship note-less releases.
        commands = _all_run_commands(_load_workflow())
        assert "--generate-notes" in commands, (
            "release must publish with auto-generated notes"
        )

    def test_publish_is_idempotent(self):
        # A re-run or a re-pushed tag must not fail on "release already exists";
        # the publish step guards on `gh release view` first.
        commands = _all_run_commands(_load_workflow())
        assert "gh release view" in commands, (
            "publish must be idempotent (check existence before creating)"
        )

    def test_reuses_pinned_python(self):
        # Reading the version needs tomllib (3.11+); the job pins the interpreter
        # via the repo's .python-version rather than trusting the runner image.
        setup_steps = [
            step
            for step in _all_steps(_load_workflow())
            if str(step.get("uses", "")).startswith("actions/setup-python")
        ]
        assert setup_steps, "release must set up a known Python for tomllib"
        assert any(
            step.get("with", {}).get("python-version-file") == ".python-version"
            for step in setup_steps
        ), "release must pin Python via the repo's .python-version"

    def test_run_scripts_are_injection_safe(self):
        # Untrusted-ish input (the tag name) must reach the shell only through
        # env vars, never interpolated as `${{ ... }}` inside a run: script.
        for step in _all_steps(_load_workflow()):
            if "run" in step:
                assert "${{" not in step["run"], (
                    f"run script interpolates a ${{{{ }}}} expression directly "
                    f"(injection risk); pass it via env: instead:\n{step['run']}"
                )

    def test_job_caps_runtime(self):
        # Every job sets an explicit timeout so a hang fails fast instead of
        # holding a runner for GitHub's 6-hour default.
        for name, job in _load_workflow()["jobs"].items():
            assert isinstance(job.get("timeout-minutes"), int), (
                f"job '{name}' sets no timeout-minutes cap"
            )


class TestReleaseNotesConfig:
    """Contract for ``.github/release.yml`` (auto-generated release notes)."""

    def test_config_exists_and_parses(self):
        assert _RELEASE_NOTES_CONFIG.is_file()
        with _RELEASE_NOTES_CONFIG.open() as fh:
            data = yaml.safe_load(fh)
        assert "changelog" in data

    def test_has_catch_all_category(self):
        # A "*" category guarantees no change is silently dropped from the notes.
        with _RELEASE_NOTES_CONFIG.open() as fh:
            data = yaml.safe_load(fh)
        categories = data["changelog"].get("categories", [])
        labels = [label for cat in categories for label in cat.get("labels", [])]
        assert "*" in labels, "notes config needs a '*' catch-all category"
