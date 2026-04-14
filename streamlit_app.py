import random

import streamlit as st
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as load_vlm
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

MAX_SEED = 2_147_483_647
MAX_IMAGE_SIZE = 1440

VLM_MODEL_ID = "mlx-community/SmolVLM-500M-Instruct-bf16"

MODE_DEFAULTS = {
    "Fast": {"steps": 4, "cfg": 1.0},
    "Quality": {"steps": 50, "cfg": 4.0},
}


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
            formatted_prompt,  # type: ignore[arg-type]  # apply_chat_template returns str at runtime
            image=image_list if image_list else None,  # type: ignore[arg-type]  # accepts PIL Images at runtime
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
    """Resolve the final prompt, optionally auto-enhancing via the VLM."""
    if auto_enhance:
        return upsample_prompt(prompt, image_list=image_list), True
    return prompt, False


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
    return max(512, min(MAX_IMAGE_SIZE, new_w)), max(512, min(MAX_IMAGE_SIZE, new_h))


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

    if image_list:
        model = EDIT_MODELS[mode]()
    else:
        model = MODELS[mode]()

    if progress_callback is not None:

        class _ProgressReporter:
            def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
                progress_callback(t + 1, config.num_inference_steps)

        model.callbacks.register(_ProgressReporter())

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

    return image.image, seed



if __name__ == "__main__":
    st.set_page_config(page_title="FLUX.2 Klein Pipeline", layout="centered")

    st.title("FLUX.2 Klein Pipeline")

    col_task, col_quality = st.columns(2)
    with col_task:
        task_mode = st.pills(
            "Task",
            options=["Generate", "Edit"],
            default="Generate",
            key="task_pills",
            label_visibility="collapsed",
        )
    with col_quality:
        mode = st.pills(
            "Quality",
            options=["Fast", "Quality"],
            default="Fast",
            key="mode_pills",
            label_visibility="collapsed",
        )

    if task_mode is None:
        task_mode = "Generate"
    if mode is None:
        mode = "Fast"

    if mode != st.session_state.get("prev_mode"):
        st.session_state.prev_mode = mode
        defaults = MODE_DEFAULTS[mode]
        st.session_state.guidance_scale_slider = defaults["cfg"]
        st.session_state.steps_slider = defaults["steps"]

    col_prompt, col_images = st.columns(2)
    with col_prompt:
        prompt = st.text_area(
            "Prompt", placeholder="Enter your prompt", key="prompt_input", height=160
        )
        auto_enhance = st.checkbox(
            "Enhance prompt",
            value=False,
            help="Enhance the prompt using the VLM before generating",
            key="auto_enhance_checkbox",
        )
    with col_images:
        uploaded_files = st.file_uploader(
            "Input images (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )

    image_list = None
    if uploaded_files:
        image_list = [Image.open(f) for f in uploaded_files]

    _image_key = (
        tuple((f.name, f.file_id) for f in uploaded_files)
        if uploaded_files
        else tuple(id(img) for img in image_list)
        if image_list
        else ()
    )
    if _image_key != st.session_state.get("prev_images", ()):
        st.session_state.prev_images = _image_key
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

    with st.expander("Settings"):
        seed_val = st.slider(
            "Seed",
            min_value=0,
            max_value=MAX_SEED,
            value=0,
            step=1,
        )

        randomize_seed = st.checkbox("Randomize seed", value=True)

        col1, col2 = st.columns(2)
        with col1:
            width = st.slider(
                "Width",
                min_value=512,
                max_value=MAX_IMAGE_SIZE,
                step=32,
                key="width_slider",
            )
        with col2:
            height = st.slider(
                "Height",
                min_value=512,
                max_value=MAX_IMAGE_SIZE,
                step=32,
                key="height_slider",
            )

        col3, col4 = st.columns(2)
        with col3:
            guidance_scale = st.slider(
                "Guidance scale",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                key="guidance_scale_slider",
            )
        with col4:
            num_inference_steps = st.slider(
                "Number of inference steps",
                min_value=1,
                max_value=100,
                step=1,
                key="steps_slider",
            )

    if st.button("Run", type="primary"):
        st.session_state.pop("auto_enhanced_prompt", None)
        run_prompt, was_auto_enhanced = _resolve_prompt(
            final_prompt, image_list, auto_enhance
        )
        if was_auto_enhanced:
            st.session_state.auto_enhanced_prompt = run_prompt

        progress_bar = st.progress(0, text="Starting...")

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
        progress_bar.empty()
        st.session_state.result_image = image
        st.session_state.result_seed = used_seed if randomize_seed else None

    if "auto_enhanced_prompt" in st.session_state:
        st.info(f"Enhanced prompt: {st.session_state.auto_enhanced_prompt}")

    if "result_image" in st.session_state:
        st.image(st.session_state.result_image)
        if st.session_state.result_seed is not None:
            st.caption(f"Seed: {st.session_state.result_seed}")

