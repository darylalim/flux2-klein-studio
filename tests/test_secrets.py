"""Guard tests: no real secret may reach a git-tracked file.

Context: the CLAUDE.md audit found a live Hugging Face access token sitting in a
local ``.env``. It was correctly gitignored and blocked by ``guard-paths.sh``,
so it never reached the repo — but "never reached the repo" was luck plus one
convenience hook, not something CI could prove. These tests turn that into an
enforced invariant:

  * no tracked file may contain a recognizable secret (HF token, PEM private
    key, AWS access-key id);
  * the secret-bearing files themselves (``.env`` / non-template ``.env.*``,
    ``secrets.toml``) must never be tracked.

They scan ``git ls-files`` (so a gitignored local ``.env`` is invisible here,
exactly as it should be) and run in CI on every push/PR, catching an accidental
``git add`` before it becomes a published leak. The detector is self-tested
below so a broken regex can't let the guard pass vacuously.

Note: the sample secrets in the detector self-test are built by concatenation so
no contiguous secret literal exists in this file — which is itself scanned by
``test_no_secret_in_tracked_files``.
"""

import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ``git ls-files`` is the source of truth for "in the repo": it excludes the
# gitignored local .env and every other untracked file, so this guard sees only
# what a push would actually publish.
_requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="secret scan needs git to list tracked files"
)

# High-signal secret shapes: fixed prefixes + fixed lengths keep false positives
# near zero, so a match is almost certainly a real leak. Add new shapes here.
_SECRET_PATTERNS = (
    # Hugging Face user access token: "hf_" + 34 base62 chars — the exact shape
    # the audit found in a local .env, and the one we most need to keep out.
    ("hugging-face-token", re.compile(r"hf_[A-Za-z0-9]{34,}")),
    # PEM private-key block header (RSA / EC / OPENSSH / DSA / PGP or unlabelled).
    ("private-key-block", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")),
    # AWS access-key id: fixed "AKIA" prefix + 16 upper/digits.
    ("aws-access-key-id", re.compile(r"AKIA[0-9A-Z]{16}")),
)

# Mirror guard-paths.sh: real dotenv files hold secrets and must stay untracked,
# but committed *templates* (which carry no secrets) are allowed.
_ALLOWED_ENV_TEMPLATES = frozenset(
    {".env.example", ".env.sample", ".env.template", ".env.dist"}
)


def _tracked_files():
    """Every git-tracked path, repo-relative and POSIX-separated (git's format)."""
    out = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def _read_text_or_none(rel_path):
    """Text of a tracked file, or None if it is binary (undecodable) or gone."""
    try:
        return (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


@_requires_git
class TestNoSecretsInRepo:
    """No recognizable secret — and no secret-bearing file — may be tracked."""

    def test_no_secret_in_tracked_files(self):
        tracked = _tracked_files()
        assert tracked, "git ls-files returned nothing; scan would pass vacuously"

        scanned = 0
        offenders = []
        for rel_path in tracked:
            text = _read_text_or_none(rel_path)
            if text is None:  # binary (e.g. examples/*.webp) — nothing to scan
                continue
            scanned += 1
            for name, pattern in _SECRET_PATTERNS:
                if pattern.search(text):
                    offenders.append(f"{rel_path}: {name}")
        assert scanned, "no tracked text files were scanned; guard is vacuous"
        assert not offenders, (
            "secret-shaped strings found in tracked files: " + "; ".join(offenders)
        )

    def test_secret_files_are_not_tracked(self):
        offenders = []
        for rel_path in _tracked_files():
            base = PurePosixPath(rel_path).name
            if (
                base == "secrets.toml"
                or base == ".env"
                or (base.startswith(".env.") and base not in _ALLOWED_ENV_TEMPLATES)
            ):
                offenders.append(rel_path)
        assert not offenders, f"secret-bearing files must never be tracked: {offenders}"


class TestSecretDetector:
    """The scan is only as good as its patterns, so prove they fire (and don't
    fire on lookalikes) — a broken regex must not let the guard pass vacuously."""

    def test_patterns_flag_known_secret_shapes(self):
        # Built by concatenation: no contiguous secret literal lands in this
        # file, which test_no_secret_in_tracked_files also scans.
        hf = "hf_" + "a1B2c3D4e5" * 3 + "F6g7"  # "hf_" + 34 chars
        assert len(hf) == 3 + 34
        pem = "-----BEGIN " + "OPENSSH PRIVATE KEY" + "-----"
        aws = "AKIA" + "ABCDEFGH12345678"  # "AKIA" + 16 chars
        for sample in (hf, pem, aws):
            assert any(rx.search(sample) for _, rx in _SECRET_PATTERNS), sample

    def test_patterns_ignore_benign_lookalikes(self):
        # Strings that legitimately live in the repo (doc URLs, the max_tokens
        # kwarg, a test name) must not trip the guard, or it cries wolf.
        benign = (
            "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
            "max_tokens=150",
            "test_strips_end_of_utterance_token",
        )
        for text in benign:
            assert not any(rx.search(text) for _, rx in _SECRET_PATTERNS), text
