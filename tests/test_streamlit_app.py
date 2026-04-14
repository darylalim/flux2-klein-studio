import importlib
from unittest.mock import MagicMock, patch

from PIL import Image


class _MockGeneratedImage:
    """Mock mflux GeneratedImage with a .image attribute."""

    def __init__(self, image=None):
        self.image = image or Image.new("RGB", (64, 64))


def _make_mock_model():
    """Create a mock mflux model that returns a dummy GeneratedImage."""
    model = MagicMock()
    model.generate_image.return_value = _MockGeneratedImage()
    model.callbacks = MagicMock()
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

        assert streamlit_app.MAX_IMAGE_SIZE == 1440

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

        assert (
            streamlit_app.MODELS["Fast"]
            is streamlit_app._get_model_distilled
        )
        assert streamlit_app.MODELS["Quality"] is streamlit_app._get_model_base

    def test_edit_models_maps_to_getters(self):
        import streamlit_app

        assert (
            streamlit_app.EDIT_MODELS["Fast"]
            is streamlit_app._get_edit_model_distilled
        )
        assert (
            streamlit_app.EDIT_MODELS["Quality"]
            is streamlit_app._get_edit_model_base
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
            streamlit_app.infer(
                "a cat", mode="Quality", progress_callback=callback
            )
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

    def test_clamps_min_to_512(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (3000, 500))]
        _, h = streamlit_app._dimensions_from_images(images)
        assert h >= 512

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
        assert h == 512

    def test_extreme_tall_clamps_width(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        images = [Image.new("RGB", (500, 5000))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 512
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
            "a cat", None, auto_enhance=False, already_enhanced=False
        )
        assert result == "a cat"
        assert was_enhanced is False

    def test_returns_original_when_already_enhanced(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        result, was_enhanced = streamlit_app._resolve_prompt(
            "a cat", None, auto_enhance=True, already_enhanced=True
        )
        assert result == "a cat"
        assert was_enhanced is False

    def test_enhances_when_auto_enhance_on_and_not_already_enhanced(self):
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
                "a cat", None, auto_enhance=True, already_enhanced=False
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
                "edit this", images, auto_enhance=True, already_enhanced=False
            )
            assert was_enhanced is True
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["image"] is images

    def test_both_flags_false(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        result, was_enhanced = streamlit_app._resolve_prompt(
            "a cat", None, auto_enhance=False, already_enhanced=True
        )
        assert result == "a cat"
        assert was_enhanced is False

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
                "a cat", None, auto_enhance=True, already_enhanced=False
            )
            assert result == "a cat"
            assert was_enhanced is True


class TestClearEnhancement:
    def test_clears_all_enhancement_keys(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.st") as mock_st:
            mock_st.session_state = {
                "enhanced_prompt": "foo",
                "enhanced_prompt_area": "bar",
                "auto_enhanced_prompt": "baz",
                "other_key": "keep",
            }
            streamlit_app._clear_enhancement()
            assert "enhanced_prompt" not in mock_st.session_state
            assert "enhanced_prompt_area" not in mock_st.session_state
            assert "auto_enhanced_prompt" not in mock_st.session_state
            assert mock_st.session_state["other_key"] == "keep"

    def test_ignores_missing_keys(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        with patch("streamlit_app.st") as mock_st:
            mock_st.session_state = {"other_key": "keep"}
            streamlit_app._clear_enhancement()
            assert mock_st.session_state == {"other_key": "keep"}


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


class TestExamples:
    def test_examples_list_structure(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        assert isinstance(streamlit_app.EXAMPLES, list)
        assert len(streamlit_app.EXAMPLES) == 5
        for example in streamlit_app.EXAMPLES:
            assert "label" in example
            assert "prompt" in example
            assert "images" in example

    def test_text_only_examples_have_no_images(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        for example in streamlit_app.EXAMPLES[:4]:
            assert example["images"] is None

    def test_image_example_has_valid_paths(self):
        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        image_example = streamlit_app.EXAMPLES[4]
        assert image_example["images"] is not None
        assert len(image_example["images"]) == 3

    def test_bundled_images_are_valid(self):
        import os

        mock_model = _make_mock_model()
        streamlit_app, _, _ = _reload_app(mock_model)
        image_example = streamlit_app.EXAMPLES[4]
        for path in image_example["images"]:
            assert os.path.exists(path), f"Missing: {path}"
            img = Image.open(path)
            assert img.size[0] > 0 and img.size[1] > 0
