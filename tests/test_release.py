"""Contract tests for the tag-triggered release workflow.

``.github/workflows/release.yml`` publishes a GitHub release when a version tag
(``vMAJOR.MINOR.PATCH``) is pushed. It is the release-side mirror of the repo's
"lock consistency, don't hope for it" discipline (``uv sync --locked``,
``tests/test_license.py``): before publishing, it asserts the tag names the same
version ``pyproject.toml`` declares, so a mislabeled tag fails the build instead
of shipping a wrong release. These tests lock the parts that are easy to break
silently:

  * it fires on version-shaped ``v*.*.*`` tags only — not on every push to
    ``main`` (that is ci.yml's job), and not on an unrelated v-prefixed tag.
  * the version-consistency gate stays present and fails on a mismatch.
  * only ``contents: write`` is granted — no additional scope.
  * ``run:`` scripts never interpolate a ``${{ }}`` expression directly (the
    tag name flows through ``env:`` and is referenced as ``"$TAG"``).

YAML note: PyYAML (YAML 1.1) parses the bare mapping key ``on:`` as ``True``;
``_load_workflow`` normalizes it back so triggers are reachable as
``workflow["on"]`` — same treatment as tests/test_ci.py.
"""

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yml"
_RELEASE_NOTES_CONFIG = _REPO_ROOT / ".github" / "release.yml"


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


def _step_run(workflow, name_contains):
    """The ``run`` body of the first step whose name contains ``name_contains``."""
    needle = name_contains.lower()
    for step in _all_steps(workflow):
        if "run" in step and needle in str(step.get("name", "")).lower():
            return step["run"]
    return None


class TestReleaseWorkflow:
    """Contract for ``.github/workflows/release.yml``."""

    def test_workflow_file_exists_and_parses(self):
        assert _RELEASE_WORKFLOW.is_file()
        assert _load_workflow()  # valid, non-empty YAML

    def test_triggers_only_on_version_tags(self):
        # Releases are cut by pushing a vMAJOR.MINOR.PATCH tag; the trigger must
        # be tag-based and version-SHAPED, not a bare `v*` — otherwise an
        # unrelated v-prefixed tag (`v2`, `viewer-checkpoint`) fires a spurious
        # run that then dies in the version gate.
        triggers = _load_workflow()["on"]
        assert "push" in triggers, "release must trigger on a push of a tag"
        tags = triggers["push"].get("tags")
        assert tags, f"release must trigger on version tags, got tags={tags}"
        assert all(p.startswith("v") for p in tags), (
            f"tag patterns must be v-prefixed: {tags}"
        )
        # Require the MAJOR.MINOR.PATCH dot shape so a loosening back to a bare
        # `v*` (which matches any v-prefixed tag) is caught here.
        assert all(p.count(".") >= 2 for p in tags), (
            f"tag patterns must be version-shaped (v*.*.*), not a bare v*: {tags}"
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

    def test_serializes_same_tag_runs(self):
        # A concurrency group keyed on the tag stops a re-pushed tag from racing
        # two publishes into an "already exists" failure. cancel-in-progress must
        # stay false — cancelling a half-finished publish is worse than waiting.
        concurrency = _load_workflow().get("concurrency")
        assert isinstance(concurrency, dict), (
            "release must declare a concurrency group to serialize same-tag runs"
        )
        assert concurrency.get("cancel-in-progress") is False, (
            "release must NOT cancel in-progress runs (would abort a publish mid-flight)"
        )

    def test_grants_only_contents_write(self):
        # Publishing a release needs write to contents (releases live under the
        # contents scope) and nothing more.
        permissions = _load_workflow().get("permissions")
        assert isinstance(permissions, dict), (
            "workflow must declare an explicit permissions block (a mapping, not "
            "a bare 'write-all'/'read-all' string)"
        )
        assert permissions.get("contents") == "write", (
            "release must grant contents: write to publish"
        )
        # Assert on the KEY SET, not the values: every valid Actions permission
        # value is already one of read/write/none, so a values-only check is a
        # tautology that can never fail. What must be locked is that no
        # *additional* scope (id-token, packages, actions, ...) is granted — a
        # token escalation releasing doesn't need.
        assert set(permissions) == {"contents"}, (
            f"release must grant only 'contents', not {sorted(permissions)}"
        )

    def test_asserts_tag_matches_pyproject_version(self):
        # The enforced consistency gate: the published tag must name the same
        # version the code declares. Scope the checks to the gate step's OWN run
        # body (not the joined commands) so an `exit 1` in an unrelated step
        # can't satisfy them.
        gate = _step_run(_load_workflow(), "Assert tag matches")
        assert gate, "release must have an 'Assert tag matches ...' step"
        assert "tomllib" in gate and "pyproject" in gate, (
            "the gate must read the declared version from pyproject.toml"
        )
        assert "github.ref_name" in _RELEASE_WORKFLOW.read_text(), (
            "release must capture the pushed tag name to compare against it"
        )
        # The tag carries a leading `v` (v0.6.5) but pyproject declares the bare
        # version (0.6.5); the gate must strip it, or *every* release would fail
        # on a spurious "v0.6.5 != 0.6.5".
        assert "TAG#v" in gate, (
            "the gate must strip the leading 'v' from the tag before comparing"
        )
        # Fire on INEQUALITY and exit — locking both the operator AND that the
        # `exit 1` lives in THIS step catches an inversion to `==` or a neutered
        # log-only branch (which a global substring search over all steps misses).
        assert "!=" in gate and "exit 1" in gate, (
            "the gate must fail (exit 1) on a tag != pyproject-version mismatch"
        )

    def test_cleans_up_orphan_tag_on_mismatch(self):
        # On a version mismatch the gate deletes the just-pushed tag so it does
        # not linger on the remote pointing at un-released code.
        gate = _step_run(_load_workflow(), "Assert tag matches")
        assert gate and "push --delete" in gate, (
            "the gate must delete the orphan tag when the version check fails"
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

    def test_marks_prerelease_tags_as_prerelease(self):
        # A pre-release tag (v1.2.0-rc1) must publish with --prerelease so it
        # never becomes the repo's "Latest" release.
        commands = _all_run_commands(_load_workflow())
        assert "--prerelease" in commands, (
            "publish must mark pre-release tags with --prerelease"
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

    def test_excludes_bots_by_their_actual_login(self):
        # exclude.authors matches a PR author's exact login; bot logins carry a
        # `[bot]` suffix, so bare `dependabot`/`github-actions` would match
        # nothing. Lock the suffixed form so the exclusion actually fires.
        with _RELEASE_NOTES_CONFIG.open() as fh:
            data = yaml.safe_load(fh)
        authors = data["changelog"].get("exclude", {}).get("authors", [])
        for bot in authors:
            assert not bot.endswith("]") or bot.endswith("[bot]"), (
                f"bot author {bot!r} looks malformed"
            )
        assert not any(a in ("dependabot", "github-actions") for a in authors), (
            "bot authors must use their real login handle (e.g. 'dependabot[bot]'), "
            "not the bare name, or the exclusion never matches"
        )
