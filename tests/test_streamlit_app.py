import contextlib
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image


class _MockGeneratedImage:
    """Mock mflux GeneratedImage with a .image attribute."""

    def __init__(self, image=None):
        self.image = image or Image.new("RGB", (64, 64))


def _make_mock_model():
    """Create a mock mflux model that returns a dummy GeneratedImage.

    `callbacks.in_loop` is a real list and `callbacks.register` appends to it,
    mirroring mflux's CallbackRegistry so the reporter-cleanup path in
    `infer()` is actually exercised.
    """
    model = MagicMock()
    model.generate_image.return_value = _MockGeneratedImage()
    callbacks = MagicMock()
    callbacks.in_loop = []
    callbacks.register.side_effect = callbacks.in_loop.append
    model.callbacks = callbacks
    return model


class _MockGenerationResult:
    """Mock mlx-vlm GenerationResult with a .text attribute."""

    def __init__(self, text="enhanced prompt"):
        self.text = text


def _make_mock_vlm():
    """Create a mock VLM (model, processor, config) triple."""
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_config = MagicMock()
    return mock_model, mock_processor, mock_config


def _reload_app(mock_model, *, mock_edit_model=None, mock_vlm=None):
    """Reload app module with mocked heavy dependencies and passthrough cache."""
    with (
        patch(
            "mflux.models.flux2.variants.Flux2Klein", return_value=mock_model
        ) as mock_cls,
        patch(
            "mflux.models.flux2.variants.Flux2KleinEdit",
            return_value=mock_edit_model or mock_model,
        ) as mock_edit_cls,
        patch("mflux.models.common.config.ModelConfig") as _mock_model_config,
        patch("mlx_vlm.load") as mock_load,
        patch("mlx_vlm.generate") as _mock_generate,
        patch("mlx_vlm.prompt_utils.apply_chat_template") as _mock_chat,
        patch("mlx_vlm.utils.load_config") as mock_load_config,
        patch("streamlit.cache_resource", lambda f: f),
    ):
        if mock_vlm is not None:
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_load_config.return_value = mock_vlm_config

        import streamlit_app

        importlib.reload(streamlit_app)
        return streamlit_app, mock_cls, mock_edit_cls


class TestConstants:
    def test_max_seed(self):
        import streamlit_app

        assert streamlit_app.MAX_SEED == 2_147_483_647

    def test_max_image_size(self):
        import streamlit_app

        assert streamlit_app.MAX_IMAGE_SIZE == 1024

    def test_vlm_model_id(self):
        import streamlit_app

        assert streamlit_app.VLM_MODEL_ID == "mlx-community/SmolVLM-500M-Instruct-bf16"

    def test_mode_defaults(self):
        import streamlit_app

        assert streamlit_app.MODE_DEFAULTS == {
            "Fast": {"steps": 4, "cfg": 1.0},
            "Quality": {"steps": 50, "cfg": 4.0},
        }

    def test_models_maps_to_getters(self):
        import streamlit_app

        assert streamlit_app.MODELS["Fast"] is streamlit_app._get_model_distilled
        assert streamlit_app.MODELS["Quality"] is streamlit_app._get_model_base

    def test_edit_models_maps_to_getters(self):
        import streamlit_app

        assert (
            streamlit_app.EDIT_MODELS["Fast"] is streamlit_app._get_edit_model_distilled
        )
        assert (
            streamlit_app.EDIT_MODELS["Quality"] is streamlit_app._get_edit_model_base
        )

    def test_mode_defaults_keys_match_models(self):
        import streamlit_app

        assert set(streamlit_app.MODE_DEFAULTS) == set(streamlit_app.MODELS)

    def test_mode_defaults_keys_match_edit_models(self):
        import streamlit_app

        assert set(streamlit_app.MODE_DEFAULTS) == set(streamlit_app.EDIT_MODELS)


class TestModelLoading:
    def test_distilled_model_created_with_correct_config(self):
        mock_model = _make_mock_model()
        streamlit_app, mock_cls, _ = _reload_app(mock_model)
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model) as mock_klein,
            patch("streamlit_app.ModelConfig") as mock_config,
        ):
            mock_config.flux2_klein_4b.return_value = "distilled_config"
            streamlit_app._get_model_distilled()
            mock_klein.assert_called_once_with(model_config="distilled_config")

    def test_base_model_created_with_correct_config(self):
        mock_model = _make_mock_model()
        streamlit_app, mock_cls, _ = _reload_app(mock_model)
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model) as mock_klein,
            patch("streamlit_app.ModelConfig") as mock_config,
        ):
            mock_config.flux2_klein_base_4b.return_value = "base_config"
            streamlit_app._get_model_base()
            mock_klein.assert_called_once_with(model_config="base_config")

    def test_edit_distilled_model_created_with_correct_config(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with (
            patch("streamlit_app.Flux2KleinEdit", return_value=mock_model) as mock_edit,
            patch("streamlit_app.ModelConfig") as mock_config,
        ):
            mock_config.flux2_klein_4b.return_value = "distilled_config"
            streamlit_app._get_edit_model_distilled()
            mock_edit.assert_called_once_with(model_config="distilled_config")

    def test_edit_base_model_created_with_correct_config(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with (
            patch("streamlit_app.Flux2KleinEdit", return_value=mock_model) as mock_edit,
            patch("streamlit_app.ModelConfig") as mock_config,
        ):
            mock_config.flux2_klein_base_4b.return_value = "base_config"
            streamlit_app._get_edit_model_base()
            mock_edit.assert_called_once_with(model_config="base_config")


class TestInfer:
    def test_returns_image_and_seed(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            image, seed = streamlit_app.infer("a cat", seed=42)
            assert isinstance(image, Image.Image)
            assert seed == 42

    def test_forwards_args_to_model(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer(
                "a cat",
                seed=123,
                width=768,
                height=512,
                guidance_scale=3.0,
                num_inference_steps=20,
            )
            mock_model.generate_image.assert_called_once_with(
                seed=123,
                prompt="a cat",
                num_inference_steps=20,
                width=768,
                height=512,
                guidance=3.0,
            )

    def test_fixed_seed(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            _, seed = streamlit_app.infer("a cat", seed=99, randomize_seed=False)
            assert seed == 99

    def test_randomized_seed(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            _, seed = streamlit_app.infer("a cat", seed=42, randomize_seed=True)
            assert 0 <= seed <= streamlit_app.MAX_SEED

    def test_default_params(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat")
            mock_model.generate_image.assert_called_once_with(
                seed=42,
                prompt="a cat",
                num_inference_steps=4,
                width=1024,
                height=1024,
                guidance=1.0,
            )

    def test_empty_prompt(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            image, seed = streamlit_app.infer("", seed=42)
            assert isinstance(image, Image.Image)
            mock_model.generate_image.assert_called_once_with(
                seed=42,
                prompt="",
                num_inference_steps=4,
                width=1024,
                height=1024,
                guidance=1.0,
            )

    def test_mode_selects_base_defaults(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat", mode="Quality")
            mock_model.generate_image.assert_called_once_with(
                seed=42,
                prompt="a cat",
                num_inference_steps=50,
                width=1024,
                height=1024,
                guidance=4.0,
            )

    def test_explicit_params_override_mode_defaults(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer(
                "a cat",
                mode="Quality",
                guidance_scale=2.0,
                num_inference_steps=10,
            )
            mock_model.generate_image.assert_called_once_with(
                seed=42,
                prompt="a cat",
                num_inference_steps=10,
                width=1024,
                height=1024,
                guidance=2.0,
            )

    def test_partial_override_steps_only(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer(
                "a cat",
                mode="Quality",
                num_inference_steps=10,
            )
            call_kwargs = mock_model.generate_image.call_args[1]
            assert call_kwargs["num_inference_steps"] == 10
            assert call_kwargs["guidance"] == 4.0

    def test_image_list_uses_edit_model(self):
        mock_model = _make_mock_model()
        mock_edit_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model, mock_edit_model=mock_edit_model)
        images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model),
            patch("streamlit_app.Flux2KleinEdit", return_value=mock_edit_model),
        ):
            streamlit_app.infer("edit this", image_list=images)
            call_kwargs = mock_edit_model.generate_image.call_args[1]
            assert call_kwargs["image_paths"] is images
            mock_model.generate_image.assert_not_called()

    def test_no_images_uses_txt2img_model(self):
        mock_model = _make_mock_model()
        mock_edit_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model, mock_edit_model=mock_edit_model)
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model),
            patch("streamlit_app.Flux2KleinEdit", return_value=mock_edit_model),
        ):
            streamlit_app.infer("a cat")
            mock_model.generate_image.assert_called_once()
            assert "image_paths" not in mock_model.generate_image.call_args[1]
            mock_edit_model.generate_image.assert_not_called()

    def test_progress_callback_registered(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer("a cat", progress_callback=callback)
            mock_model.callbacks.register.assert_called_once()

    def test_no_callback_when_progress_callback_none(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat")
            mock_model.callbacks.register.assert_not_called()

    def test_progress_callback_invoked_with_step_and_total(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", num_inference_steps=4, progress_callback=callback
            )
            registered = mock_model.callbacks.register.call_args[0][0]
            mock_config = MagicMock()
            mock_config.num_inference_steps = 4
            registered.call_in_loop(0, 42, "a cat", None, mock_config, None)
            callback.assert_called_once_with(1, 4)

    def test_progress_callback_step_counts_across_steps(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", num_inference_steps=4, progress_callback=callback
            )
            registered = mock_model.callbacks.register.call_args[0][0]
            mock_config = MagicMock()
            mock_config.num_inference_steps = 4
            for step in range(4):
                registered.call_in_loop(step, 42, "a cat", None, mock_config, None)
            assert callback.call_count == 4
            callback.assert_any_call(1, 4)
            callback.assert_any_call(2, 4)
            callback.assert_any_call(3, 4)
            callback.assert_any_call(4, 4)

    def test_progress_callback_with_base_mode(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            callback = MagicMock()
            streamlit_app.infer("a cat", mode="Quality", progress_callback=callback)
            registered = mock_model.callbacks.register.call_args[0][0]
            mock_config = MagicMock()
            mock_config.num_inference_steps = 50
            registered.call_in_loop(0, 42, "a cat", None, mock_config, None)
            callback.assert_called_once_with(1, 50)

    def test_progress_callback_with_image_list(self):
        mock_model = _make_mock_model()
        mock_edit_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model, mock_edit_model=mock_edit_model)
        images = [Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model),
            patch("streamlit_app.Flux2KleinEdit", return_value=mock_edit_model),
        ):
            callback = MagicMock()
            streamlit_app.infer(
                "edit this", image_list=images, progress_callback=callback
            )
            call_kwargs = mock_edit_model.generate_image.call_args[1]
            assert call_kwargs["image_paths"] is images
            mock_edit_model.callbacks.register.assert_called_once()

    def test_progress_reporter_removed_after_infer(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            streamlit_app.infer("a cat", progress_callback=MagicMock())
            assert mock_model.callbacks.in_loop == []

    def test_reporters_do_not_accumulate_across_runs(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            for _ in range(3):
                streamlit_app.infer("a cat", progress_callback=MagicMock())
            assert mock_model.callbacks.in_loop == []
            assert mock_model.callbacks.register.call_count == 3

    def test_progress_reporter_removed_on_generate_failure(self):
        mock_model = _make_mock_model()
        mock_model.generate_image.side_effect = RuntimeError("boom")
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.Flux2Klein", return_value=mock_model):
            try:
                streamlit_app.infer("a cat", progress_callback=MagicMock())
            except RuntimeError:
                pass
            assert mock_model.callbacks.in_loop == []


class TestDimensionsFromImages:
    def test_square_image(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (800, 800))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_landscape_image(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1600, 800))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 512

    def test_portrait_image(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (800, 1600))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 512
        assert h == 1024

    def test_rounds_to_multiple_of_32(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1000, 700))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w % 32 == 0
        assert h % 32 == 0

    def test_clamps_min_to_256(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (3000, 500))]
        _, h = streamlit_app._dimensions_from_images(images)
        assert h == 256

    def test_zero_height_returns_default(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (100, 0))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_zero_width_returns_default(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (0, 100))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_uses_first_image_only(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1600, 800)), Image.new("RGB", (800, 1600))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 512

    def test_4_3_aspect_ratio(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1200, 900))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 768

    def test_16_9_aspect_ratio(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (1920, 1080))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 576

    def test_portrait_3_4_aspect_ratio(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (900, 1200))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 768
        assert h == 1024

    def test_extreme_panoramic_clamps_height(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (5000, 500))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 256

    def test_extreme_tall_clamps_width(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (500, 5000))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 256
        assert h == 1024


class TestVLMInit:
    def test_vlm_loads_correct_model(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            streamlit_app._get_vlm()
            mock_load.assert_called_once_with(
                "mlx-community/SmolVLM-500M-Instruct-bf16"
            )
            mock_lc.assert_called_once_with("mlx-community/SmolVLM-500M-Instruct-bf16")

    def test_vlm_returns_triple(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            result = streamlit_app._get_vlm()
            assert result == (mock_vlm_model, mock_vlm_processor, mock_vlm_config)


EXPECTED_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for FLUX.2 by Black Forest Labs. "
    "Rewrite user prompts to be more descriptive while strictly preserving "
    "their core subject and intent. Keep the enhanced prompt under 120 "
    "words.\n\n"
    "Guidelines:\n"
    "- Add concrete visual specifics: textures, materials, lighting, "
    "shadows, and spatial relationships.\n"
    "- Put ALL text that should appear in the image in quotation marks "
    "(signs, labels, screens, etc.) - without quotes, the model generates "
    "gibberish.\n\n"
    "Output only the revised prompt and nothing else."
)

EXPECTED_SYSTEM_PROMPT_WITH_IMAGES = (
    "You are an image-editing expert. Convert the user's editing request "
    "into one concise instruction (50-80 words, ~30 for brief requests).\n\n"
    "Rules:\n"
    "- Single instruction only, no commentary\n"
    "- Use clear, analytical language (avoid vague words like "
    '"whimsical" or "cascading")\n'
    "- Specify what changes AND what stays the same (face, lighting, "
    "composition)\n"
    "- Turn negatives into positives "
    '("don\'t change X" becomes "keep X")\n'
    '- Make abstractions concrete ("futuristic" becomes '
    '"glowing cyan neon, metallic panels")\n\n'
    "Output only the final instruction in plain text and nothing else."
)


class TestUpsamplePrompt:
    def test_chat_message_format_text_only(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat")
            mock_chat.assert_called_once_with(
                mock_vlm_processor,
                mock_vlm_config,
                [
                    {"role": "system", "content": EXPECTED_SYSTEM_PROMPT},
                    {"role": "user", "content": "a cat"},
                ],
                num_images=0,
            )

    def test_chat_message_format_with_images(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("make it blue", image_list=images)
            mock_chat.assert_called_once_with(
                mock_vlm_processor,
                mock_vlm_config,
                [
                    {"role": "system", "content": EXPECTED_SYSTEM_PROMPT_WITH_IMAGES},
                    {"role": "user", "content": "make it blue"},
                ],
                num_images=2,
            )

    def test_images_passed_to_generate(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        images = [Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("edit", image_list=images)
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["image"] is images

    def test_no_images_passed_for_text_only(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat")
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["image"] is None

    def test_generation_kwargs(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat")
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["max_tokens"] == 150
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["top_p"] == 0.9

    def test_extracts_and_strips_output(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("  A majestic feline  ")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "A majestic feline"

    def test_strips_end_of_utterance_token(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult(
                "A majestic feline<end_of_utterance>"
            )
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "A majestic feline"

    def test_strips_end_of_utterance_token_mid_text(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult(
                "A majestic<end_of_utterance> feline"
            )
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "A majestic feline"

    def test_empty_output_returns_original(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"

    def test_whitespace_only_output_returns_original(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("   ")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"

    def test_exception_returns_original(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
            patch("streamlit_app.st") as mock_st,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.side_effect = RuntimeError("OOM")
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"
            mock_st.warning.assert_called_once_with(
                "Prompt enhancement failed. Using original prompt."
            )

    def test_empty_image_list_uses_text_only_path(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            streamlit_app.upsample_prompt("a cat", image_list=[])
            mock_chat.assert_called_once_with(
                mock_vlm_processor,
                mock_vlm_config,
                [
                    {"role": "system", "content": EXPECTED_SYSTEM_PROMPT},
                    {"role": "user", "content": "a cat"},
                ],
                num_images=0,
            )
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["image"] is None


class TestResolvePrompt:
    def test_returns_original_when_auto_enhance_off(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        result, was_enhanced = streamlit_app._resolve_prompt(
            "a cat", None, auto_enhance=False
        )
        assert result == "a cat"
        assert was_enhanced is False

    def test_enhances_when_auto_enhance_on(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            result, was_enhanced = streamlit_app._resolve_prompt(
                "a cat", None, auto_enhance=True
            )
            assert result == "enhanced prompt"
            assert was_enhanced is True

    def test_enhances_with_images(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        images = [Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.return_value = _MockGenerationResult("enhanced prompt")
            result, was_enhanced = streamlit_app._resolve_prompt(
                "edit this", images, auto_enhance=True
            )
            assert was_enhanced is True
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["image"] is images

    def test_falls_back_on_vlm_error(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.load_vlm") as mock_load,
            patch("streamlit_app.load_config") as mock_lc,
            patch("streamlit_app.apply_chat_template") as mock_chat,
            patch("streamlit_app.vlm_generate") as mock_gen,
            patch("streamlit_app.st"),
        ):
            mock_vlm_model, mock_vlm_processor, mock_vlm_config = mock_vlm
            mock_load.return_value = (mock_vlm_model, mock_vlm_processor)
            mock_lc.return_value = mock_vlm_config
            mock_chat.return_value = "formatted prompt"
            mock_gen.side_effect = RuntimeError("OOM")
            result, was_enhanced = streamlit_app._resolve_prompt(
                "a cat", None, auto_enhance=True
            )
            assert result == "a cat"
            assert was_enhanced is False

    def test_returns_not_enhanced_when_vlm_output_matches_input(self):
        mock_model = _make_mock_model()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _, _ = _reload_app(mock_model, mock_vlm=mock_vlm)
        with patch("streamlit_app.upsample_prompt") as mock_upsample:
            mock_upsample.return_value = "a cat"
            result, was_enhanced = streamlit_app._resolve_prompt(
                "a cat", None, auto_enhance=True
            )
            assert result == "a cat"
            assert was_enhanced is False


class TestStreamlitApp:
    def test_get_model_distilled_uses_cache_resource(self):
        """Verify _get_model_distilled is decorated with @st.cache_resource."""
        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.flux2.variants.Flux2KleinEdit"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_model_distilled, "clear")

    def test_get_model_base_uses_cache_resource(self):
        """Verify _get_model_base is decorated with @st.cache_resource."""
        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.flux2.variants.Flux2KleinEdit"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_model_base, "clear")

    def test_get_edit_model_distilled_uses_cache_resource(self):
        """Verify _get_edit_model_distilled is decorated with @st.cache_resource."""
        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.flux2.variants.Flux2KleinEdit"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_edit_model_distilled, "clear")

    def test_get_edit_model_base_uses_cache_resource(self):
        """Verify _get_edit_model_base is decorated with @st.cache_resource."""
        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.flux2.variants.Flux2KleinEdit"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_edit_model_base, "clear")

    def test_get_vlm_uses_cache_resource(self):
        """Verify _get_vlm is decorated with @st.cache_resource."""
        with (
            patch("mflux.models.flux2.variants.Flux2Klein"),
            patch("mflux.models.flux2.variants.Flux2KleinEdit"),
            patch("mflux.models.common.config.ModelConfig"),
            patch("mlx_vlm.load"),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config"),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_vlm, "clear")

    def test_ui_not_executed_on_import(self):
        mock_model = _make_mock_model()
        with (
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.button") as mock_button,
        ):
            _reload_app(mock_model)
            mock_markdown.assert_not_called()
            mock_text_input.assert_not_called()
            mock_button.assert_not_called()

    def test_mode_defaults_to_distilled(self):
        from streamlit.testing.v1 import AppTest

        mock_model = _make_mock_model()
        mock_vlm_model, mock_vlm_processor, mock_vlm_config = _make_mock_vlm()
        with (
            patch("mflux.models.flux2.variants.Flux2Klein", return_value=mock_model),
            patch(
                "mflux.models.flux2.variants.Flux2KleinEdit", return_value=mock_model
            ),
            patch("mflux.models.common.config.ModelConfig"),
            patch(
                "mlx_vlm.load",
                return_value=(mock_vlm_model, mock_vlm_processor),
            ),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config", return_value=mock_vlm_config),
            patch("streamlit.cache_resource", lambda f: f),
        ):
            at = AppTest.from_file("streamlit_app.py").run(timeout=10)
            assert at.radio(key="mode_radio").value == "Distilled (4 steps)"

    def test_uploader_always_present(self):
        from streamlit.testing.v1 import AppTest

        mock_model = _make_mock_model()
        mock_vlm_model, mock_vlm_processor, mock_vlm_config = _make_mock_vlm()
        with (
            patch("mflux.models.flux2.variants.Flux2Klein", return_value=mock_model),
            patch(
                "mflux.models.flux2.variants.Flux2KleinEdit", return_value=mock_model
            ),
            patch("mflux.models.common.config.ModelConfig"),
            patch(
                "mlx_vlm.load",
                return_value=(mock_vlm_model, mock_vlm_processor),
            ),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config", return_value=mock_vlm_config),
            patch("streamlit.cache_resource", lambda f: f),
        ):
            at = AppTest.from_file("streamlit_app.py").run(timeout=10)
            # Unified layout: the optional uploader is always rendered
            assert len(at.get("file_uploader")) == 1

    def test_run_button_present_and_enabled(self):
        from streamlit.testing.v1 import AppTest

        mock_model = _make_mock_model()
        mock_vlm_model, mock_vlm_processor, mock_vlm_config = _make_mock_vlm()
        with (
            patch("mflux.models.flux2.variants.Flux2Klein", return_value=mock_model),
            patch(
                "mflux.models.flux2.variants.Flux2KleinEdit", return_value=mock_model
            ),
            patch("mflux.models.common.config.ModelConfig"),
            patch(
                "mlx_vlm.load",
                return_value=(mock_vlm_model, mock_vlm_processor),
            ),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config", return_value=mock_vlm_config),
            patch("streamlit.cache_resource", lambda f: f),
        ):
            at = AppTest.from_file("streamlit_app.py").run(timeout=10)
            run_buttons = [b for b in at.button if b.label == "Run"]
            assert len(run_buttons) == 1
            # Editing is implicit now, so Run is never disabled
            assert run_buttons[0].disabled is False

    def test_prompt_uses_enter_prompt_placeholder(self):
        from streamlit.testing.v1 import AppTest

        mock_model = _make_mock_model()
        mock_vlm_model, mock_vlm_processor, mock_vlm_config = _make_mock_vlm()
        with (
            patch("mflux.models.flux2.variants.Flux2Klein", return_value=mock_model),
            patch(
                "mflux.models.flux2.variants.Flux2KleinEdit", return_value=mock_model
            ),
            patch("mflux.models.common.config.ModelConfig"),
            patch(
                "mlx_vlm.load",
                return_value=(mock_vlm_model, mock_vlm_processor),
            ),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config", return_value=mock_vlm_config),
            patch("streamlit.cache_resource", lambda f: f),
        ):
            at = AppTest.from_file("streamlit_app.py").run(timeout=10)
            assert at.text_input(key="prompt_input").placeholder == "Enter your prompt"

    def test_mode_change_updates_steps_and_guidance(self):
        from streamlit.testing.v1 import AppTest

        mock_model = _make_mock_model()
        mock_vlm_model, mock_vlm_processor, mock_vlm_config = _make_mock_vlm()
        with (
            patch("mflux.models.flux2.variants.Flux2Klein", return_value=mock_model),
            patch(
                "mflux.models.flux2.variants.Flux2KleinEdit", return_value=mock_model
            ),
            patch("mflux.models.common.config.ModelConfig"),
            patch(
                "mlx_vlm.load",
                return_value=(mock_vlm_model, mock_vlm_processor),
            ),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config", return_value=mock_vlm_config),
            patch("streamlit.cache_resource", lambda f: f),
        ):
            at = AppTest.from_file("streamlit_app.py").run(timeout=10)
            assert at.slider(key="steps_slider").value == 4
            assert at.slider(key="guidance_scale_slider").value == 1.0
            at.radio(key="mode_radio").set_value("Base (50 steps)").run(timeout=10)
            assert at.slider(key="steps_slider").value == 50
            assert at.slider(key="guidance_scale_slider").value == 4.0

    def test_example_buttons_render(self):
        from streamlit.testing.v1 import AppTest

        mock_model = _make_mock_model()
        mock_vlm_model, mock_vlm_processor, mock_vlm_config = _make_mock_vlm()
        with (
            patch("mflux.models.flux2.variants.Flux2Klein", return_value=mock_model),
            patch(
                "mflux.models.flux2.variants.Flux2KleinEdit", return_value=mock_model
            ),
            patch("mflux.models.common.config.ModelConfig"),
            patch(
                "mlx_vlm.load",
                return_value=(mock_vlm_model, mock_vlm_processor),
            ),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config", return_value=mock_vlm_config),
            patch("streamlit.cache_resource", lambda f: f),
        ):
            at = AppTest.from_file("streamlit_app.py").run(timeout=10)
            # Run button + 4 example buttons
            assert len(at.button) >= 5
            assert at.button(key="example_2").label  # capybara text-to-image example

    def test_clicking_example_fills_prompt(self):
        from streamlit.testing.v1 import AppTest

        mock_model = _make_mock_model()
        mock_vlm_model, mock_vlm_processor, mock_vlm_config = _make_mock_vlm()
        with (
            patch("mflux.models.flux2.variants.Flux2Klein", return_value=mock_model),
            patch(
                "mflux.models.flux2.variants.Flux2KleinEdit", return_value=mock_model
            ),
            patch("mflux.models.common.config.ModelConfig"),
            patch(
                "mlx_vlm.load",
                return_value=(mock_vlm_model, mock_vlm_processor),
            ),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config", return_value=mock_vlm_config),
            patch("streamlit.cache_resource", lambda f: f),
        ):
            at = AppTest.from_file("streamlit_app.py").run(timeout=10)
            example = at.button(key="example_2")  # capybara prompt
            example.click().run(timeout=10)
            # Button labels are truncated; clicking sets the full prompt
            value = at.text_input(key="prompt_input").value or ""
            assert "capybara" in value
            assert value.endswith("close up photo")

    def test_edit_example_loads_prompt_and_images(self):
        from streamlit.testing.v1 import AppTest

        mock_model = _make_mock_model()
        mock_vlm_model, mock_vlm_processor, mock_vlm_config = _make_mock_vlm()
        with (
            patch("mflux.models.flux2.variants.Flux2Klein", return_value=mock_model),
            patch(
                "mflux.models.flux2.variants.Flux2KleinEdit", return_value=mock_model
            ),
            patch("mflux.models.common.config.ModelConfig"),
            patch(
                "mlx_vlm.load",
                return_value=(mock_vlm_model, mock_vlm_processor),
            ),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config", return_value=mock_vlm_config),
            patch("streamlit.cache_resource", lambda f: f),
        ):
            at = AppTest.from_file("streamlit_app.py").run(timeout=10)
            edit_example = at.button(key="edit_example_0")
            edit_example.click().run(timeout=10)
            assert "petting" in (at.text_input(key="prompt_input").value or "")
            assert len(at.session_state["example_images"]) == 3


class _FakeSessionState(dict):
    # Minimal st.session_state stand-in: attribute access + item access + pop.
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


@contextlib.contextmanager
def _app_test():
    # Build an AppTest for streamlit_app.py with the heavy deps mocked out.
    mock_model = _make_mock_model()
    mock_vlm_model, mock_vlm_processor, mock_vlm_config = _make_mock_vlm()
    with (
        patch("mflux.models.flux2.variants.Flux2Klein", return_value=mock_model),
        patch("mflux.models.flux2.variants.Flux2KleinEdit", return_value=mock_model),
        patch("mflux.models.common.config.ModelConfig"),
        patch("mlx_vlm.load", return_value=(mock_vlm_model, mock_vlm_processor)),
        patch("mlx_vlm.generate"),
        patch("mlx_vlm.prompt_utils.apply_chat_template"),
        patch("mlx_vlm.utils.load_config", return_value=mock_vlm_config),
        patch("streamlit.cache_resource", lambda f: f),
    ):
        from streamlit.testing.v1 import AppTest

        yield AppTest.from_file("streamlit_app.py")


class TestExamplesAndLabels:
    def test_mode_labels_and_inverse_mapping(self):
        import streamlit_app

        assert streamlit_app.MODE_LABELS == {
            "Fast": "Distilled (4 steps)",
            "Quality": "Base (50 steps)",
        }
        assert streamlit_app.LABEL_TO_MODE == {
            "Distilled (4 steps)": "Fast",
            "Base (50 steps)": "Quality",
        }
        # A label exists for every mode and the mapping round-trips.
        assert set(streamlit_app.MODE_LABELS) == set(streamlit_app.MODE_DEFAULTS)
        assert all(
            streamlit_app.MODE_LABELS[streamlit_app.LABEL_TO_MODE[label]] == label
            for label in streamlit_app.LABEL_TO_MODE
        )

    def test_example_prompts_shape(self):
        import streamlit_app

        assert len(streamlit_app.EXAMPLE_PROMPTS) == 4
        assert all(
            isinstance(p, str) and p.strip() for p in streamlit_app.EXAMPLE_PROMPTS
        )
        assert any("capybara" in p for p in streamlit_app.EXAMPLE_PROMPTS)

    def test_edit_examples_shape_and_files_exist(self):
        import streamlit_app

        assert streamlit_app._EXAMPLES_DIR.is_dir()
        assert streamlit_app._EXAMPLES_DIR.name == "examples"
        assert len(streamlit_app.EDIT_EXAMPLES) == 1
        for prompt, images in streamlit_app.EDIT_EXAMPLES:
            assert isinstance(prompt, str) and prompt.strip()
            assert isinstance(images, list) and len(images) == 3
            for image_path in images:
                assert Path(image_path).is_file()

    def test_truncate(self):
        import streamlit_app

        truncate = streamlit_app._truncate
        assert truncate("hi") == "hi"
        assert truncate("x" * 70) == "x" * 70
        long_label = truncate("y" * 71)
        assert long_label.endswith("…")
        assert len(long_label) == 71
        # The slice is rstrip'd before the ellipsis is appended.
        assert truncate("a" * 68 + "  zzzz") == "a" * 68 + "…"
        assert truncate("abcdef", length=3) == "abc…"


class TestExampleCallbacks:
    def test_set_example_prompt_clears_example_images(self):
        import streamlit_app

        fake = _FakeSessionState(example_images=["x"])
        with patch.object(streamlit_app, "st") as mock_st:
            mock_st.session_state = fake
            streamlit_app._set_example_prompt("a cat")
        assert fake["prompt_input"] == "a cat"
        assert "example_images" not in fake

    def test_load_edit_example_sets_prompt_and_copies_images(self):
        import streamlit_app

        src = ["p1", "p2"]
        fake = _FakeSessionState()
        with patch.object(streamlit_app, "st") as mock_st:
            mock_st.session_state = fake
            streamlit_app._load_edit_example("edit me", src)
        assert fake["prompt_input"] == "edit me"
        assert fake["example_images"] == ["p1", "p2"]
        # Stored as a copy so mutating session_state cannot corrupt EDIT_EXAMPLES.
        assert fake["example_images"] is not src

    def test_clear_example_images(self):
        import streamlit_app

        fake = _FakeSessionState(example_images=["x"], prompt_input="keep")
        with patch.object(streamlit_app, "st") as mock_st:
            mock_st.session_state = fake
            streamlit_app._clear_example_images()
            assert "example_images" not in fake
            assert fake["prompt_input"] == "keep"
            streamlit_app._clear_example_images()  # idempotent, must not raise


class TestMoreCoreLogic:
    def test_randomized_seed_forwarded_to_model(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with (
            patch("streamlit_app.Flux2Klein", return_value=mock_model),
            patch("streamlit_app.random.randint", return_value=777),
        ):
            _, seed = streamlit_app.infer("a cat", seed=42, randomize_seed=True)
            assert seed == 777
            assert mock_model.generate_image.call_args[1]["seed"] == 777

    def test_aspect_landing_on_min_floor(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        # aspect 4 -> short side rounds to exactly 256 (the new floor)
        assert streamlit_app._dimensions_from_images(
            [Image.new("RGB", (2048, 512))]
        ) == (
            1024,
            256,
        )
        assert streamlit_app._dimensions_from_images(
            [Image.new("RGB", (512, 2048))]
        ) == (
            256,
            1024,
        )

    def test_long_side_equals_max_image_size(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        for size in [(1600, 800), (800, 1600), (1200, 900), (1920, 1080)]:
            w, h = streamlit_app._dimensions_from_images([Image.new("RGB", size)])
            assert max(w, h) == streamlit_app.MAX_IMAGE_SIZE


class TestUIWidgets:
    def test_dimension_sliders_range(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            for key in ("width_slider", "height_slider"):
                slider = at.slider(key=key)
                assert slider.min == 256
                assert slider.max == 1024
                assert slider.step == 32

    def test_guidance_slider_uses_g_format(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            assert at.slider(key="guidance_scale_slider").proto.format == "%g"

    def test_seed_is_number_input(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            assert len(at.number_input) == 1
            seed = at.number_input[0]
            assert seed.label == "Seed"
            assert seed.min == 0
            assert seed.max == 2_147_483_647
            assert seed.value == 0

    def test_two_examples_sections_render(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            assert sum(1 for m in at.markdown if m.value == "**Examples**") == 2
            assert at.button(key="edit_example_0").label

    def test_t2i_example_clears_loaded_edit_images(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            at.button(key="edit_example_0").click().run(timeout=10)
            assert "example_images" in at.session_state
            at.button(key="example_2").click().run(timeout=10)
            assert "example_images" not in at.session_state
            assert "capybara" in (at.text_input(key="prompt_input").value or "")

    def test_clear_example_images_button(self):
        with _app_test() as app:
            at = app.run(timeout=10)
            at.button(key="edit_example_0").click().run(timeout=10)
            assert any(b.label == "Clear example images" for b in at.button)
            next(b for b in at.button if b.label == "Clear example images").click().run(
                timeout=10
            )
            assert "example_images" not in at.session_state
            assert not any(b.label == "Clear example images" for b in at.button)

    def test_example_run_routes_through_edit_model(self):
        from streamlit.testing.v1 import AppTest

        mock_txt2img = _make_mock_model()
        mock_edit = _make_mock_model()
        mock_vlm_model, mock_vlm_processor, mock_vlm_config = _make_mock_vlm()
        with (
            patch("mflux.models.flux2.variants.Flux2Klein", return_value=mock_txt2img),
            patch("mflux.models.flux2.variants.Flux2KleinEdit", return_value=mock_edit),
            patch("mflux.models.common.config.ModelConfig"),
            patch(
                "mlx_vlm.load",
                return_value=(mock_vlm_model, mock_vlm_processor),
            ),
            patch("mlx_vlm.generate"),
            patch("mlx_vlm.prompt_utils.apply_chat_template"),
            patch("mlx_vlm.utils.load_config", return_value=mock_vlm_config),
            patch("streamlit.cache_resource", lambda f: f),
        ):
            at = AppTest.from_file("streamlit_app.py").run(timeout=10)
            # Load the editing example, then Run — loaded images must route to edit.
            at.button(key="edit_example_0").click().run(timeout=10)
            next(b for b in at.button if b.label == "Run").click().run(timeout=10)
            assert mock_edit.generate_image.called
            assert not mock_txt2img.generate_image.called
