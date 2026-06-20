import random
from pathlib import Path

import streamlit as st
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as load_vlm
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

MAX_SEED = 2_147_483_647
MAX_IMAGE_SIZE = 1024

VLM_MODEL_ID = "mlx-community/SmolVLM-500M-Instruct-bf16"

MODE_DEFAULTS = {
    "Fast": {"steps": 4, "cfg": 1.0},
    "Quality": {"steps": 50, "cfg": 4.0},
}

# UI display labels for the speed/quality modes (match the FLUX.2 [klein] Gradio
# space). Internal keys stay "Fast"/"Quality"; only the visible text changes.
MODE_LABELS = {
    "Fast": "Distilled (4 steps)",
    "Quality": "Base (50 steps)",
}
LABEL_TO_MODE = {label: mode for mode, label in MODE_LABELS.items()}

EXAMPLE_PROMPTS = [
    "Create a vase on a table in living room, the color of the vase is a gradient of color, starting with #02eb3c color and finishing with #edfa3c. The flowers inside the vase have the color #ff0088",
    "Photorealistic infographic showing the complete Berlin TV Tower (Fernsehturm) from ground base to antenna tip, full vertical view with entire structure visible including concrete shaft, metallic sphere, and antenna spire. Slight upward perspective angle looking up toward the iconic sphere, perfectly centered on clean white background. Left side labels with thin horizontal connector lines: the text '368m' in extra large bold dark grey numerals (#2D3748) positioned at exactly the antenna tip with 'TOTAL HEIGHT' in small caps below. The text '207m' in extra large bold with 'TELECAFÉ' in small caps below, with connector line touching the sphere precisely at the window level. Right side label with horizontal connector line touching the sphere's equator: the text '32m' in extra large bold dark grey numerals with 'SPHERE DIAMETER' in small caps below. Bottom section arranged in three balanced columns: Left - Large text '986' in extra bold dark grey with 'STEPS' in caps below. Center - 'BERLIN TV TOWER' in bold caps with 'FERNSEHTURM' in lighter weight below. Right - 'INAUGURATED' in bold caps with 'OCTOBER 3, 1969' below. All typography in modern sans-serif font (such as Inter or Helvetica), color #2D3748, clean minimal technical diagram style. Horizontal connector lines are thin, precise, and clearly visible, touching the tower structure at exact corresponding measurement points. Professional architectural elevation drawing aesthetic with dynamic low angle perspective creating sense of height and grandeur, poster-ready infographic design with perfect visual hierarchy.",
    "Soaking wet capybara taking shelter under a banana leaf in the rainy jungle, close up photo",
    "A kawaii die-cut sticker of a chubby orange cat, featuring big sparkly eyes and a happy smile with paws raised in greeting and a heart-shaped pink nose. The design should have smooth rounded lines with black outlines and soft gradient shading with pink cheeks.",
]

_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

# Editing examples as (prompt, [image paths]) from the FLUX.2 [klein] space.
EDIT_EXAMPLES = [
    (
        "The person from image 1 is petting the cat from image 2, the bird from image 3 is next to them",
        [
            str(_EXAMPLES_DIR / "woman1.webp"),
            str(_EXAMPLES_DIR / "cat_window.webp"),
            str(_EXAMPLES_DIR / "bird.webp"),
        ],
    ),
]


@st.cache_resource
def _get_model_distilled():
    return Flux2Klein(model_config=ModelConfig.flux2_klein_4b())


@st.cache_resource
def _get_model_base():
    return Flux2Klein(model_config=ModelConfig.flux2_klein_base_4b())


MODELS = {
    "Fast": _get_model_distilled,
    "Quality": _get_model_base,
}


@st.cache_resource
def _get_edit_model_distilled():
    return Flux2KleinEdit(model_config=ModelConfig.flux2_klein_4b())


@st.cache_resource
def _get_edit_model_base():
    return Flux2KleinEdit(model_config=ModelConfig.flux2_klein_base_4b())


EDIT_MODELS = {
    "Fast": _get_edit_model_distilled,
    "Quality": _get_edit_model_base,
}


@st.cache_resource
def _get_vlm():
    model, processor = load_vlm(VLM_MODEL_ID)
    config = load_config(VLM_MODEL_ID)
    return model, processor, config


UPSAMPLE_PROMPT_TEXT_ONLY = (
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

UPSAMPLE_PROMPT_WITH_IMAGES = (
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


def upsample_prompt(prompt, image_list=None):
    try:
        model, processor, config = _get_vlm()
        system_prompt = (
            UPSAMPLE_PROMPT_WITH_IMAGES if image_list else UPSAMPLE_PROMPT_TEXT_ONLY
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = apply_chat_template(
            processor,
            config,
            messages,
            num_images=len(image_list) if image_list else 0,
        )
        result = vlm_generate(
            model,
            processor,
            formatted_prompt,  # ty: ignore[invalid-argument-type]  # apply_chat_template returns str at runtime
            image=image_list if image_list else None,  # ty: ignore[invalid-argument-type]  # accepts PIL Images at runtime
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
        )
        enhanced = result.text.replace("<end_of_utterance>", "").strip()
        return enhanced or prompt
    except Exception:
        st.warning("Prompt enhancement failed. Using original prompt.")
        return prompt


def _resolve_prompt(prompt, image_list, auto_enhance):
    """Resolve the final prompt, optionally auto-enhancing via the VLM.

    Returns was_enhanced=False when the VLM call fails or produces output
    identical to the input, so the caller can avoid showing a misleading
    "Enhanced prompt" banner.
    """
    if not auto_enhance:
        return prompt, False
    enhanced = upsample_prompt(prompt, image_list=image_list)
    if enhanced == prompt:
        return prompt, False
    return enhanced, True


def _set_example_prompt(example):
    """Fill the prompt box from a text-to-image example (clears any example images)."""
    st.session_state.prompt_input = example
    st.session_state.pop("example_images", None)


def _load_edit_example(prompt, images):
    """Load an editing example: its prompt plus its bundled input images."""
    st.session_state.prompt_input = prompt
    st.session_state.example_images = list(images)
    # Reveal the input panel once (not on every later rerun — see the expander).
    st.session_state.expand_input_once = True


def _clear_example_images():
    st.session_state.pop("example_images", None)


def _truncate(text, length=70):
    return text if len(text) <= length else text[:length].rstrip() + "…"


class _ProgressReporter:
    def __init__(self, callback):
        self._callback = callback

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        self._callback(t + 1, config.num_inference_steps)


def _dimensions_from_images(image_list):
    """Calculate output dimensions matching the aspect ratio of the first input image."""
    w, h = image_list[0].size
    if w == 0 or h == 0:
        return 1024, 1024
    aspect = w / h
    if aspect >= 1:
        new_w = 1024
        new_h = round(1024 / aspect / 32) * 32
    else:
        new_h = 1024
        new_w = round(1024 * aspect / 32) * 32
    return max(256, min(MAX_IMAGE_SIZE, new_w)), max(256, min(MAX_IMAGE_SIZE, new_h))


def infer(
    prompt,
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=None,
    num_inference_steps=None,
    mode="Fast",
    image_list=None,
    progress_callback=None,
):
    defaults = MODE_DEFAULTS[mode]
    if guidance_scale is None:
        guidance_scale = defaults["cfg"]
    if num_inference_steps is None:
        num_inference_steps = defaults["steps"]

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    model = EDIT_MODELS[mode]() if image_list else MODELS[mode]()

    reporter = None
    if progress_callback is not None:
        reporter = _ProgressReporter(progress_callback)
        model.callbacks.register(reporter)

    try:
        if image_list:
            image = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                guidance=guidance_scale,
                image_paths=image_list,
            )
        else:
            image = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                guidance=guidance_scale,
            )
    finally:
        if reporter is not None:
            model.callbacks.in_loop.remove(reporter)

    return image.image, seed


if __name__ == "__main__":
    st.set_page_config(page_title="FLUX.2 Klein Studio", layout="wide")

    st.title("FLUX.2 Klein Studio")
    st.markdown(
        "FLUX.2 [Klein] is a fast, unified image generation and editing model "
        "designed for fast inference. "
        "[[model]](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) · "
        "[[blog]](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)"
    )

    col_controls, col_output = st.columns([2, 3], gap="large")

    # Reserve output slots in the right column (filled after inference / on rerun)
    with col_output:
        progress_slot = st.empty()
        result_slot = st.empty()

    with col_controls:
        # Prompt + Run, inline at the top (like the Gradio space)
        col_prompt, col_run = st.columns([4, 1])
        with col_prompt:
            prompt = st.text_input(
                "Prompt",
                placeholder="Enter your prompt",
                key="prompt_input",
                label_visibility="collapsed",
            )
        with col_run:
            run_clicked = st.button("Run", type="primary", width="stretch")

        # Optional input images — uploading any switches to editing automatically.
        # Auto-open once when an example loads, then respect the user's toggle.
        _has_example_images = bool(st.session_state.get("example_images"))
        with st.expander(
            "Input image(s) (optional)",
            expanded=st.session_state.pop("expand_input_once", False),
        ):
            uploaded_files = st.file_uploader(
                "Input images",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if not uploaded_files and _has_example_images:
                st.caption("Loaded example images:")
                st.image(st.session_state.example_images, width=80)
                st.button("Clear example images", on_click=_clear_example_images)

        # Mode (speed/quality). Display labels map back to internal MODE_DEFAULTS keys.
        mode_label = st.radio(
            "Mode",
            options=list(MODE_LABELS.values()),
            index=0,
            key="mode_radio",
            horizontal=True,
        )
        mode = LABEL_TO_MODE[mode_label]

        if mode != st.session_state.get("prev_mode"):
            st.session_state.prev_mode = mode
            defaults = MODE_DEFAULTS[mode]
            st.session_state.guidance_scale_slider = defaults["cfg"]
            st.session_state.steps_slider = defaults["steps"]

        image_list = None
        if uploaded_files:
            # A manual upload overrides any loaded example images
            st.session_state.pop("example_images", None)
            image_list = [Image.open(f) for f in uploaded_files]
            _image_key = tuple((f.name, f.file_id) for f in uploaded_files)
        elif st.session_state.get("example_images"):
            example_images = st.session_state.example_images
            try:
                image_list = [Image.open(p) for p in example_images]
                _image_key = tuple(example_images)
            except OSError:
                st.warning("Could not load the example images.")
                st.session_state.pop("example_images", None)
                _image_key = ()
        else:
            _image_key = ()
        if _image_key != st.session_state.get("prev_images", ()):
            st.session_state.prev_images = _image_key
            # An enhanced prompt is tied to its image set; drop it when that changes.
            st.session_state.pop("auto_enhanced_prompt", None)
            if image_list:
                _w, _h = _dimensions_from_images(image_list)
                st.session_state.width_slider = _w
                st.session_state.height_slider = _h
            else:
                st.session_state.width_slider = 1024
                st.session_state.height_slider = 1024

        if "last_prompt" not in st.session_state:
            st.session_state.last_prompt = ""

        if prompt != st.session_state.last_prompt:
            st.session_state.last_prompt = prompt
            st.session_state.pop("auto_enhanced_prompt", None)

        final_prompt = prompt

        st.session_state.setdefault("width_slider", 1024)
        st.session_state.setdefault("height_slider", 1024)

        with st.expander("Advanced Settings", expanded=False):
            auto_enhance = st.checkbox(
                "Prompt Upsampling",
                value=False,
                key="auto_enhance_checkbox",
            )
            st.caption("Automatically enhance the prompt using a VLM")

            randomize_seed = st.checkbox("Randomize seed", value=True)
            seed_val = st.number_input(
                "Seed",
                min_value=0,
                max_value=MAX_SEED,
                value=0,
                step=1,
                disabled=randomize_seed,
            )

            col_w, col_h = st.columns(2)
            with col_w:
                width = st.slider(
                    "Width",
                    min_value=256,
                    max_value=MAX_IMAGE_SIZE,
                    step=32,
                    key="width_slider",
                )
            with col_h:
                height = st.slider(
                    "Height",
                    min_value=256,
                    max_value=MAX_IMAGE_SIZE,
                    step=32,
                    key="height_slider",
                )

            col_steps, col_guidance = st.columns(2)
            with col_steps:
                num_inference_steps = st.slider(
                    "Number of inference steps",
                    min_value=1,
                    max_value=100,
                    step=1,
                    key="steps_slider",
                )
            with col_guidance:
                guidance_scale = st.slider(
                    "Guidance scale",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.1,
                    format="%g",
                    key="guidance_scale_slider",
                )

        st.markdown("**Examples**")
        _ex_cols = st.columns(2) + st.columns(2)
        for _i, (_col, _example) in enumerate(
            zip(_ex_cols, EXAMPLE_PROMPTS, strict=False)
        ):
            with _col:
                st.button(
                    _truncate(_example),
                    key=f"example_{_i}",
                    on_click=_set_example_prompt,
                    args=(_example,),
                    width="stretch",
                    help=_example,
                )

        st.markdown("**Examples**")
        for _i, (_ex_prompt, _ex_imgs) in enumerate(EDIT_EXAMPLES):
            _col_prompt, _col_imgs = st.columns([3, 2])
            with _col_prompt:
                st.button(
                    _truncate(_ex_prompt),
                    key=f"edit_example_{_i}",
                    on_click=_load_edit_example,
                    args=(_ex_prompt, _ex_imgs),
                    width="stretch",
                    help=_ex_prompt,
                )
            with _col_imgs:
                st.image(_ex_imgs, width=56)

        if run_clicked:
            st.session_state.pop("auto_enhanced_prompt", None)

            cache = st.session_state.setdefault("_enhance_cache", {})
            cache_key = (final_prompt, _image_key) if auto_enhance else None

            if cache_key is not None and cache_key in cache:
                run_prompt, was_auto_enhanced = cache[cache_key], True
            else:
                run_prompt, was_auto_enhanced = _resolve_prompt(
                    final_prompt, image_list, auto_enhance
                )
                if was_auto_enhanced and cache_key is not None:
                    cache[cache_key] = run_prompt

            if was_auto_enhanced:
                st.session_state.auto_enhanced_prompt = run_prompt

            progress_bar = progress_slot.progress(0, text="Starting...")

            def _update_progress(step, total):
                progress_bar.progress(step / total, text=f"Step {step}/{total}")

            image, used_seed = infer(
                run_prompt,
                seed_val,
                randomize_seed,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                mode=mode,
                image_list=image_list,
                progress_callback=_update_progress,
            )
            progress_slot.empty()
            st.session_state.result_image = image
            st.session_state.result_seed = used_seed

        if "auto_enhanced_prompt" in st.session_state:
            st.info(f"Enhanced prompt: {st.session_state.auto_enhanced_prompt}")

    # Output (right column): result image, or a placeholder canvas
    if "result_image" in st.session_state:
        with result_slot.container():
            st.image(st.session_state.result_image, width="stretch")
            if st.session_state.result_seed is not None:
                st.caption(f"Seed: {st.session_state.result_seed}")
    else:
        result_slot.markdown(
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:420px;border:1px solid #333;border-radius:8px;color:#888;'>"
            "🖼️ Your image will appear here</div>",
            unsafe_allow_html=True,
        )
