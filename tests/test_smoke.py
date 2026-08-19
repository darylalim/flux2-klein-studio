"""Opt-in smoke tests: real weights, real generation, no mocks.

Every other test in this suite patches ``Flux2Klein``/``Flux2KleinEdit`` with a
MagicMock, so an mflux API break sails through the whole green suite. These are
the only tests that exercise the real stack end to end, which makes them the
pre-release gate — and also why they are deselected by default (see
``[tool.pytest.ini_options]`` in ``pyproject.toml``): they need ~8.6GB of weights
on disk and Apple Silicon, neither of which CI has.

    uv run pytest -m smoke            # all of them
    uv run pytest -m smoke -k text    # just text-to-image

Weights come from the HF cache; the first run downloads them.
"""

import pytest
from PIL import Image

pytestmark = pytest.mark.smoke

# Smallest the app allows, so a smoke run stays cheap. Both are multiples of 32.
_W = _H = 256


@pytest.fixture(scope="module")
def app():
    """The real module, unmocked. Module-scoped so weights load once."""
    import streamlit_app

    return streamlit_app


def _is_real_image(image, size):
    """A PIL image of the requested size that is not a uniform blank canvas."""
    assert isinstance(image, Image.Image), f"not a PIL Image: {type(image)}"
    assert image.size == size, f"{image.size} != {size}"
    low, high = image.convert("L").getextrema()
    assert low != high, "image is a single flat colour — generation produced nothing"


def test_text_to_image_generates_a_real_image(app):
    image, seed = app.infer(
        "a red cube on a white table, studio lighting",
        seed=1234,
        width=_W,
        height=_H,
    )
    _is_real_image(image, (_W, _H))
    assert seed == 1234  # not randomized unless asked


def test_edit_accepts_the_pil_images_the_ui_builds(app):
    """The editing path's real contract, which the mocks cannot check.

    Both UI branches build ``image_list`` with ``Image.open(...)`` — PIL Images,
    not paths — and ``infer()`` hands that straight to mflux's ``image_paths``,
    which is typed ``list[Path | str]``. Whether mflux tolerates PIL objects
    there is invisible to a MagicMock and load-bearing in production, so drive
    it exactly the way the UI does rather than passing paths.
    """
    image_list = [
        Image.open(app._EXAMPLES_DIR / name)
        for name in ("woman1.webp", "cat_window.webp")
    ]
    image, _ = app.infer(
        "the person from image 1 is petting the cat from image 2",
        seed=7,
        width=_W,
        height=_H,
        image_list=image_list,
    )
    _is_real_image(image, (_W, _H))


def test_progress_callback_fires_and_is_deregistered(app):
    """mflux's CallbackRegistry has no unregister(), so infer() cleans up itself.

    The models are ``@st.cache_resource``-cached, so a reporter left behind
    fires on every later Run. Assert both halves against the real registry: the
    callback is actually invoked, and ``in_loop`` is empty afterwards.
    """
    seen = []
    image, _ = app.infer(
        "a blue sphere on grey concrete",
        seed=3,
        width=_W,
        height=_H,
        progress_callback=lambda step, total: seen.append((step, total)),
    )
    _is_real_image(image, (_W, _H))
    assert seen, "progress_callback never fired"
    assert seen[-1][0] <= seen[-1][1], f"step exceeded total: {seen[-1]}"
    assert seen[-1][1] == app.DEFAULT_STEPS
    assert app._get_model().callbacks.in_loop == [], "reporter left registered"


def test_prompt_upsampling_returns_usable_text(app):
    """Qwen3-VL is the third real dependency and equally mocked elsewhere."""
    enhanced = app.upsample_prompt("a cat")
    assert isinstance(enhanced, str)
    assert enhanced.strip()
    # Qwen3-VL's <|im_end|> is a real stop id, consumed before detokenization.
    assert "<|im_end|>" not in enhanced
