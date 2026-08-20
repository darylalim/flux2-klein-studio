"""Contract tests for the GitHub Actions CI workflow and the Python pin.

CI (``.github/workflows/ci.yml``) is the public mirror of the local ``.claude``
quality hooks: it runs the same four gates (ruff format, ruff lint, ty, pytest)
on every push to ``main`` and every pull request, so a contributor who hasn't
installed the hooks still can't merge red. Since the release job moved into this
same workflow it is also the *only* publisher, so these tests now cover both
halves. The parts that are easy to break silently:

  * the runner is macOS, not Linux — ``uv.lock`` resolves ``mlx`` to a
    CUDA-only build on Linux (``sys_platform == 'linux'``), so the suite can't
    even import on a GPU-less ubuntu runner. A well-meaning "switch to ubuntu
    for speed" must fail here, not in a red run.
  * CI runs all four gates the hooks enforce — dropping one would let an
    unformatted / untyped / failing change through review *and*, now that
    ``release`` declares ``needs: ci``, auto-publish it.
  * no gate step can be neutered with ``if:``, ``continue-on-error:`` or a
    shell short-circuit. This mattered less when a green CI only decorated a
    badge; it gates a publish now.
  * exactly one job holds ``contents: write``. A job-level ``permissions`` block
    *replaces* the workflow grant rather than merging with it, so a stray block
    on the gate job would hand a write-scoped token to the step that runs
    third-party test code.
  * the release job creates its tag and its release in one ``gh release
    create`` call. Splitting them is the one "simplification" that fails
    silently rather than loudly — see ``TestCIReleaseJob``.
  * ``.python-version`` pins the runtime and stays consistent with
    ``requires-python``, so local uv and CI resolve the same interpreter.

YAML note: PyYAML follows YAML 1.1, which parses the bare mapping key ``on:`` as
the boolean ``True``. ``_load_workflow`` normalizes that back so the trigger
table is reachable as ``workflow["on"]``.
"""

import re
import subprocess
import tomllib
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_RELEASE_NOTES_CONFIG = _REPO_ROOT / ".github" / "release.yml"
_PYTHON_VERSION_FILE = _REPO_ROOT / ".python-version"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# The two jobs ci.yml defines. Named here because several assertions have to
# distinguish "the gate job" from "the publishing job" — a substring match over
# every run: body in the file would let release-job text satisfy a gate check.
_GATE_JOB = "ci"
_RELEASE_JOB = "release"

# The four quality gates the .claude hooks enforce locally; CI must run the same
# set so an un-hooked contributor can't merge past them. Pinned as the full
# `uv run <tool> .` invocation, not a bare tool name: dropping `uv run` would
# silently swap the pinned toolchain for whatever the runner image ships, and
# narrowing `.` to a single path would shrink the checked tree. Both stayed
# green when the gates were only matched on the tool name.
_REQUIRED_GATES = (
    "uv run ruff format --check .",
    "uv run ruff check .",
    "uv run ty check .",
    "uv run pytest",
)


def _load_workflow():
    """Return the parsed ci.yml with the YAML-1.1 ``on:``→True key normalized."""
    with _CI_WORKFLOW.open() as fh:
        data = yaml.safe_load(fh)
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _job(name):
    """Return one job by name, with a readable failure if it was renamed."""
    jobs = _load_workflow()["jobs"]
    assert name in jobs, f"ci.yml defines no '{name}' job; found {sorted(jobs)}"
    return jobs[name]


def _code(text):
    """Strip comment lines so substring assertions match code, not prose.

    The run: bodies in ci.yml explain themselves at length, and several of those
    comments quote the very commands these tests forbid (the publish step's
    comment mentions the ``git push --delete`` the old flow needed). Matching
    raw text would trip on the explanation instead of the code.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _run_commands(job):
    """Join every ``run:`` script body in a job's steps into one string."""
    return "\n".join(step["run"] for step in job["steps"] if "run" in step)


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

    def test_declares_no_privileged_trigger(self):
        # `pull_request_target` runs fork-authored code against a write-scoped
        # token with access to secrets. It is the one trigger that would turn
        # this workflow — which now publishes releases — into a supply-chain
        # hole, and nothing else in the repo forbids it.
        triggers = _load_workflow()["on"]
        assert set(triggers) <= {"push", "pull_request"}, (
            f"ci.yml declares an unexpected trigger: {sorted(triggers)}"
        )

    def test_actions_pin_a_floating_major_tag(self):
        # Unguarded until now: `astral-sh/setup-uv@main` passed, because the
        # only assertion touching an action ref was a bare startswith on its
        # name. A branch ref moves under you between runs; a major tag only
        # receives patches.
        #
        # This repo deliberately tracks floating majors rather than SHA- or
        # exact-pinning, because it runs no dependabot and an exact pin with
        # nothing to bump it rots silently. Note `setup-uv@v7` is the newest
        # ref that can be written this way at all — setup-uv stopped publishing
        # floating majors after v7 — so if this assertion ever has to be
        # relaxed, that is the reason, and it should be relaxed deliberately.
        seen = 0
        for name, job in _load_workflow()["jobs"].items():
            for step in job["steps"]:
                ref = step.get("uses")
                if not ref:
                    continue
                seen += 1
                assert re.fullmatch(r"[\w.-]+/[\w.-]+@v\d+", ref), (
                    f"job '{name}' uses {ref!r}; actions must pin a floating "
                    "major tag (owner/repo@vN), not a branch, a bare name, or "
                    "an exact version that nothing will bump"
                )
        assert seen, "no `uses:` steps found; the pinning rule is unverified"

    def test_test_and_typecheck_jobs_run_on_macos(self):
        # Load-bearing: uv.lock resolves mlx to a CUDA-only build on Linux, so
        # any job that imports the app — pytest or `ty check` — must run on
        # macOS or the module can't load. (The release job needs neither and
        # runs on Linux, so gate on the command, not the job count.)
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
        # would let unformatted / untyped / failing code through review — and
        # now past `needs: ci` into a published release. Scoped to the gate job
        # so text in the release job can't satisfy a gate substring.
        commands = _code(_run_commands(_job(_GATE_JOB)))
        for gate in _REQUIRED_GATES:
            assert gate in commands, f"CI is missing the `{gate}` gate"
        # The lint gate must FAIL on violations, not auto-fix them: "ruff check"
        # is a substring of "ruff check --fix", so pin the failing form. (The
        # local ruff-format.sh hook uses --fix intentionally; CI must not.)
        assert "ruff check --fix" not in commands, (
            "CI lint gate must fail on violations, not auto-fix them"
        )

    def test_no_gate_step_can_be_neutered(self):
        # Every one of these kept the suite green before this test existed:
        # `if: false` on a step, a job-level `continue-on-error: true`, and
        # `|| true` appended to a gate. Each turns the job green while running
        # nothing — and the release job trusts `needs: ci` to mean the gates
        # actually passed.
        job = _job(_GATE_JOB)
        assert not job.get("continue-on-error"), (
            "the gate job sets continue-on-error; a failing gate would report green"
        )
        for step in job["steps"]:
            label = step.get("name", step.get("uses", "?"))
            assert "if" not in step, f"gate step '{label}' is conditional"
            assert not step.get("continue-on-error"), (
                f"gate step '{label}' sets continue-on-error"
            )
            body = _code(step.get("run", ""))
            for short_circuit in ("|| true", "|| exit 0", "; true"):
                assert short_circuit not in body, (
                    f"gate step '{label}' swallows its exit code with `{short_circuit}`"
                )

    def test_checks_out_the_source(self):
        # Unguarded until now: deleting the checkout step left every test green
        # while the gates ran against an empty workspace.
        for name in (_GATE_JOB, _RELEASE_JOB):
            uses = [step.get("uses", "") for step in _job(name)["steps"]]
            assert any(u.startswith("actions/checkout") for u in uses), (
                f"job '{name}' never checks out the repository"
            )

    def test_installs_pinned_toolchain_via_uv(self):
        # The gates run the pinned toolchain from uv.lock (`uv run <tool>`), so
        # CI and the hooks can't disagree on a lint/type rule. That needs a
        # synced env: setup-uv + `uv sync`. (`_REQUIRED_GATES` pins the `uv run`
        # prefix itself; this covers the install half.)
        assert "uv sync" in _code(_run_commands(_job(_GATE_JOB)))
        uses = [step.get("uses", "") for step in _job(_GATE_JOB)["steps"]]
        assert any(action.startswith("astral-sh/setup-uv") for action in uses)

    def test_syncs_against_locked_lockfile(self):
        # `uv sync --locked` fails if uv.lock is stale (e.g. a version bump or
        # dependency change that forgot `uv lock`), so a drifted lockfile can't
        # slip through CI silently — plain `uv sync` would just re-resolve.
        # This is the *only* place it can be caught: every local hook
        # invocation in run-tests.sh is a bare `uv run`, which self-heals the
        # lock on disk without telling anyone.
        assert "uv sync --locked" in _code(_run_commands(_job(_GATE_JOB))), (
            "CI must sync against the committed lockfile (`uv sync --locked`)"
        )

    def test_workflow_default_permissions_are_read_only(self):
        # The workflow-level grant is what every job inherits unless it declares
        # its own. An explicit read-only block replaces the repo default (which
        # can be read/write).
        permissions = _load_workflow().get("permissions")
        assert permissions, "workflow must declare an explicit permissions block"
        assert all(scope in ("read", "none") for scope in permissions.values()), (
            f"workflow-level permissions grant more than read: {permissions}"
        )
        assert permissions.get("contents") == "read"  # checkout needs read

    def test_only_the_release_job_holds_write_scope(self):
        # A job-level `permissions:` block REPLACES the workflow grant rather
        # than merging with it, so a block on the gate job silently re-elevates
        # the token that runs third-party test code — and stayed green while
        # only the workflow-level key was checked. Assert on the key SET, not
        # just the values: a values-only check passes for any key added later.
        for name, job in _load_workflow()["jobs"].items():
            permissions = job.get("permissions")
            if name == _RELEASE_JOB:
                assert permissions, f"job '{name}' must scope its own permissions"
                assert set(permissions) == {"contents"}, (
                    f"the release job grants more than contents: {permissions}"
                )
                assert permissions["contents"] == "write", (
                    "publishing a release needs contents: write"
                )
            else:
                assert permissions is None, (
                    f"job '{name}' declares its own permissions block, which "
                    "replaces the workflow's read-only grant; only the release "
                    "job may do that"
                )

    def test_jobs_cap_their_runtime(self):
        # Every job must set an explicit timeout so a hang (e.g. a stalled
        # `uv sync`) fails fast instead of holding a runner for the 6-hour
        # default.
        for name, job in _load_workflow()["jobs"].items():
            assert isinstance(job.get("timeout-minutes"), int), (
                f"job '{name}' sets no timeout-minutes cap"
            )


class TestCIReleaseJob:
    """Contract for the ``release`` job — the repo's only publisher.

    It replaced a separate tag-triggered ``release.yml``. The move is what makes
    a release a consequence of the gates instead of a parallel event: two of the
    first three releases published *before* their own CI run finished, because
    nothing connected a pushed tag to a passing build.
    """

    def test_release_is_gated_on_the_quality_job(self):
        needs = _job(_RELEASE_JOB)["needs"]
        needs = [needs] if isinstance(needs, str) else needs
        assert _GATE_JOB in needs, (
            "the release job must declare `needs: ci`, or it can publish a "
            "commit the gates never passed"
        )

    def test_release_only_fires_on_pushes_to_main(self):
        condition = _job(_RELEASE_JOB).get("if", "")
        assert "push" in condition, "pull requests must never publish a release"
        assert "refs/heads/main" in condition, (
            "the release job must be pinned to main, not any pushed branch"
        )

    def test_tag_and_release_are_created_by_the_same_call(self):
        # THE load-bearing assertion in this class. GitHub does not start a
        # workflow run for events raised by the default GITHUB_TOKEN, so
        # "push a tag here, let a tag-triggered workflow publish" lands the tag
        # and never publishes anything — a silent no-op, not a red build. The
        # only safe shape is one `gh release create --target <sha>` that makes
        # the tag and the release together.
        body = _code(_run_commands(_job(_RELEASE_JOB)))
        assert "gh release create" in body, (
            "the release job must publish with `gh release create`"
        )
        assert "--target" in body, (
            "`gh release create --target <sha>` pins the release to the commit "
            "the gates passed and creates the tag atomically"
        )
        for tagging in ("git tag", "git push"):
            assert tagging not in body, (
                f"the release job runs `{tagging}`: a tag pushed with the "
                "default GITHUB_TOKEN triggers no workflow, so a tag created "
                "outside `gh release create` is a dead end"
            )

    def test_release_is_idempotent_on_an_already_released_version(self):
        # The trigger is "pyproject declares a version with no tag yet". Re-runs,
        # re-pushes and no-op commits must all be no-ops, and the check must be
        # against remote tags rather than a HEAD~1 diff (which misses a bump
        # buried in a multi-commit push and has no parent on a root commit).
        body = _code(_run_commands(_job(_RELEASE_JOB)))
        assert "ls-remote" in body, (
            "release detection must ask the remote which versions are tagged"
        )
        assert "HEAD~" not in body, (
            "a HEAD~1 diff misses a bump that arrived in an earlier commit of "
            "the push; check tag existence instead"
        )
        # Grepping for "ls-remote" alone guards nothing about the *decision* it
        # feeds. Pin the mapping (exit 0 = tagged = skip, exit 2 = publish) and
        # the gate on the publish step, which is what actually stops a re-run
        # from cutting a second release.
        assert 'echo "publish=false" >> "$GITHUB_OUTPUT"' in body
        assert 'echo "publish=true" >> "$GITHUB_OUTPUT"' in body
        detect = body[body.index("ls-remote") :]
        skip, publish = (
            detect.index('publish=false" >> "$GITHUB_OUTPUT'),
            detect.index('publish=true" >> "$GITHUB_OUTPUT'),
        )
        assert detect.index("0)") < skip < detect.index("2)") < publish, (
            "the ls-remote exit-code branches are wired backwards: 0 means the "
            "tag exists (skip) and 2 means it does not (publish)"
        )
        publish_step = next(
            s for s in _job(_RELEASE_JOB)["steps"] if s.get("name") == "Publish"
        )
        assert publish_step.get("if") == "steps.detect.outputs.publish == 'true'", (
            "the publish step must be gated on the detect step's verdict, or "
            "every push to main re-publishes"
        )

    def test_release_generates_notes(self):
        assert "--generate-notes" in _code(_run_commands(_job(_RELEASE_JOB)))

    def test_prerelease_predicate_matches_the_versions_uv_actually_writes(self):
        # Executed, not grepped. The predicate this replaced was `case $VERSION
        # in *-*)`, carried over from a flow where a *human* typed the tag. The
        # version now comes from pyproject.toml, and uv writes PEP 440's
        # canonical spelling: `uv version 0.9.0-rc1` stores `0.9.0rc1`. The
        # hyphen match therefore never fired, and an RC would have published as
        # the repo's Latest release — with `assert "--prerelease" in body`
        # staying green the whole time, because the dead branch still contained
        # the string. Run the real predicate against real version strings.
        body = _code(_run_commands(_job(_RELEASE_JOB)))
        start = body.index('case "$VERSION" in')
        pattern = body[start : body.index("esac", start)]
        script = f'VERSION="$1"\nprerelease=""\n{pattern}esac\necho "$prerelease"'
        for version, expected in (
            ("0.8.0", ""),  # a normal release must NOT be marked prerelease
            ("0.8.0rc1", "--prerelease"),  # what `uv version --bump rc` writes
            ("1.0.0a2", "--prerelease"),  # what `uv version 1.0.0-alpha.2` writes
            ("0.8.0-rc1", "--prerelease"),  # the hand-edited spelling
            ("0.8.0.dev1", "--prerelease"),
        ):
            result = subprocess.run(
                ["bash", "-c", script, "bash", version],
                capture_output=True,
                text=True,
                check=True,
            )
            assert result.stdout.strip() == expected, (
                f"version {version!r} yields {result.stdout.strip()!r}, "
                f"expected {expected!r}"
            )

    def test_release_validates_the_version_before_using_it_as_a_ref(self):
        # The version is read out of pyproject.toml and then interpolated into a
        # ref name. Shape-check it first. Assert the version-specific failure —
        # a bare "::error::" probe passes with the whole shape check deleted,
        # because the ls-remote failure branch emits one too.
        body = _code(_run_commands(_job(_RELEASE_JOB)))
        assert "is not MAJOR.MINOR.PATCH" in body, (
            "the release job must fail loudly on a malformed version rather "
            "than create a garbage ref"
        )
        assert "[0-9]*.[0-9]*.[0-9]*" in body, "the shape check itself is gone"

    def test_release_pins_python_via_the_version_file(self):
        # It reads the version with tomllib, so it needs a guaranteed 3.11+
        # interpreter rather than whatever the runner image ships.
        for step in _job(_RELEASE_JOB)["steps"]:
            if step.get("uses", "").startswith("actions/setup-python"):
                assert step.get("with", {}).get("python-version-file") == (
                    ".python-version"
                ), "the release job must pin Python from .python-version"
                return
        raise AssertionError("the release job never sets up Python")


class TestUvVersionPin:
    """``[tool.uv] required-version`` keeps local uv and CI on the same uv.

    Without it, ``astral-sh/setup-uv`` installs whatever uv is latest, and a uv
    whose resolution-marker normalization differs from the one that wrote
    ``uv.lock`` fails ``uv sync --locked`` on a lockfile nobody edited.
    """

    def test_pyproject_pins_uv_exactly(self):
        with _PYPROJECT.open("rb") as fh:
            required = (
                tomllib.load(fh).get("tool", {}).get("uv", {}).get("required-version")
            )
        assert required, "pyproject must declare [tool.uv] required-version"
        # Must be `==`: setup-uv strips that prefix and installs the literal
        # version, but cannot parse a PEP 440 comma range (">=0.12,<0.13") and
        # would silently fall back to installing latest.
        assert required.startswith("=="), (
            f"required-version must be an == pin for setup-uv to resolve it, got {required!r}"
        )

    def test_ci_does_not_override_the_pyproject_pin(self):
        # setup-uv reads required-version only when its `version:` input is
        # empty. A hardcoded version here would shadow pyproject and give the
        # pin two sources of truth that can drift apart.
        seen = False
        for job in _load_workflow()["jobs"].values():
            for step in job["steps"]:
                if step.get("uses", "").startswith("astral-sh/setup-uv"):
                    seen = True
                    assert not step.get("with", {}).get("version"), (
                        "ci.yml pins uv directly; the pin belongs in pyproject's "
                        "[tool.uv] required-version so local and CI share one source"
                    )
        # Non-vacuity: with no setup-uv step the loop asserts nothing.
        assert seen, "no setup-uv step found; the pin rule is unverified"


class TestSmokeTestsStayOptIn:
    """CI's `uv run pytest` must never select the real-weights smoke suite.

    ``tests/test_smoke.py`` loads ~8.6GB of weights and generates for real. It is
    the only thing that exercises the live mflux/mlx-vlm surface (everything else
    mocks the model classes), but CI has neither the weights nor the budget, so
    the default run has to deselect it. Losing the ``addopts`` line would make
    every CI run try to download the model.
    """

    def _pytest_config(self):
        with _PYPROJECT.open("rb") as fh:
            return (
                tomllib.load(fh)
                .get("tool", {})
                .get("pytest", {})
                .get("ini_options", {})
            )

    def test_default_run_deselects_smoke(self):
        addopts = self._pytest_config().get("addopts", "")
        assert "not smoke" in addopts, (
            "pyproject's [tool.pytest.ini_options] addopts must deselect the "
            f"smoke marker so CI does not download weights, got {addopts!r}"
        )

    def test_smoke_marker_is_registered(self):
        # An unregistered marker is a warning, not an error, so a typo'd
        # `@pytest.mark.smoke` would silently select nothing.
        markers = self._pytest_config().get("markers", [])
        assert any(m.startswith("smoke:") for m in markers), (
            f"the smoke marker must be declared in [tool.pytest.ini_options], got {markers}"
        )

    def test_ci_runs_pytest_without_overriding_the_marker(self):
        # `-m smoke` on CI's pytest invocation would defeat the addopts default,
        # since pytest honours the last -m it is given. Substring matching, not
        # token matching: `-msmoke` (no space) and `--override-ini=addopts=`
        # both evade a whitespace-split check and would make CI attempt the
        # ~8.6GB download.
        seen = False
        for job in _load_workflow()["jobs"].values():
            for step in job["steps"]:
                run = _code(step.get("run", ""))
                if "pytest" in run:
                    seen = True
                    for override in ("-m", "--override-ini"):
                        assert override not in run, (
                            f"ci.yml must not pass {override} to pytest, got {run!r}"
                        )
        assert seen, "no pytest invocation found; the marker rule is unverified"


class TestReleaseNotesConfig:
    """Contract for ``.github/release.yml`` (auto-generated release notes).

    Consumed by the release job's ``gh release create --generate-notes``. It
    outlived the ``release.yml`` *workflow* it shipped alongside, so its tests
    moved here rather than being deleted with ``tests/test_release.py``.
    """

    def _config(self):
        with _RELEASE_NOTES_CONFIG.open() as fh:
            return yaml.safe_load(fh)

    def test_config_exists_and_parses(self):
        assert _RELEASE_NOTES_CONFIG.is_file()
        assert "changelog" in self._config()

    def test_has_catch_all_category(self):
        # A "*" category guarantees no change is silently dropped from the notes.
        categories = self._config()["changelog"].get("categories", [])
        labels = [label for cat in categories for label in cat.get("labels", [])]
        assert "*" in labels, "notes config needs a '*' catch-all category"

    def test_excludes_bots_by_their_actual_login(self):
        # exclude.authors matches a PR author's exact login; bot logins carry a
        # `[bot]` suffix, so bare `dependabot`/`github-actions` would match
        # nothing. Lock the suffixed form so the exclusion actually fires.
        authors = self._config()["changelog"].get("exclude", {}).get("authors", [])
        for bot in authors:
            assert not bot.endswith("]") or bot.endswith("[bot]"), (
                f"bot author {bot!r} looks malformed"
            )
        assert not any(a in ("dependabot", "github-actions") for a in authors), (
            "bot authors must use their real login handle (e.g. 'dependabot[bot]'), "
            "not the bare name, or the exclusion never matches"
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
