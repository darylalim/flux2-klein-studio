import contextlib
import random
from pathlib import Path
from typing import cast

import streamlit as st
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as load_vlm
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

APP_TITLE = "FLUX.2 Klein Studio"

MAX_SEED = 2_147_483_647
MAX_IMAGE_SIZE = 1024

# Fixed height for the idle/generating output frame so it doesn't collapse
# between states (the result image then sizes to its own content).
OUTPUT_FRAME_HEIGHT = 420

VLM_MODEL_ID = "mlx-community/SmolVLM-500M-Instruct-bf16"

# The app ships exactly one model: the distilled FLUX.2 Klein 4B, pre-quantized
# to 8-bit by mflux itself (its safetensors carry mflux's own
# quantization_level=8 metadata), so loading needs no local quantization pass.
# 8.6GB on disk against 16GB for the bf16 original. The base (50-step) variant
# has no pre-quantized build on the Hub, which is why the app no longer offers a
# mode switch.
MODEL_REPO = "mlx-community/flux2-klein-4b-8bit"

# The distilled variant is guidance-free and converges in 4 steps. These seed the
# steps/guidance sliders, which stay adjustable.
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 1.0

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


# model_path picks the weights; model_config is still required alongside it
# because mflux builds the architecture from its transformer/text-encoder
# overrides, not from anything in the repo.
@st.cache_resource(show_spinner="Loading FLUX.2 Klein (8-bit)…")
def _get_model():
    return Flux2Klein(model_path=MODEL_REPO, model_config=ModelConfig.flux2_klein_4b())


@st.cache_resource(show_spinner="Loading FLUX.2 Klein Edit (8-bit)…")
def _get_edit_model():
    return Flux2KleinEdit(
        model_path=MODEL_REPO, model_config=ModelConfig.flux2_klein_4b()
    )


@st.cache_resource(show_spinner="Loading SmolVLM prompt enhancer…")
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


def upsample_prompt(prompt, image_list: list | None = None):
    try:
        model, processor, config = _get_vlm()
        system_prompt = (
            UPSAMPLE_PROMPT_WITH_IMAGES if image_list else UPSAMPLE_PROMPT_TEXT_ONLY
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        # apply_chat_template returns a str at runtime for SmolVLM, though the
        # stub types it as a broader union.
        formatted_prompt = cast(
            str,
            apply_chat_template(
                processor,
                config,
                messages,
                num_images=len(image_list) if image_list else 0,
            ),
        )
        result = vlm_generate(
            model,
            processor,
            formatted_prompt,
            image=image_list if image_list else None,  # ty: ignore[invalid-argument-type]  # mlx_vlm types image as str|list[str] (no Optional) but accepts None/PIL Images at runtime
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
        )
        enhanced = result.text.replace("<end_of_utterance>", "").strip()
        return enhanced or prompt
    except Exception:
        st.warning(
            "Prompt enhancement failed. Using original prompt.",
            icon=":material/warning:",
        )
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
    # Cycle the uploader key so a fresh, empty uploader renders — otherwise a
    # stale manual upload survives the rerun and silently overrides the example.
    st.session_state.uploader_nonce = st.session_state.get("uploader_nonce", 0) + 1
    # Open the input panel via the keyed expander's state; later user toggles
    # are respected because the widget syncs this key on every change.
    st.session_state.input_expander = True


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
    image_list=None,
    progress_callback=None,
):
    if guidance_scale is None:
        guidance_scale = DEFAULT_GUIDANCE
    if num_inference_steps is None:
        num_inference_steps = DEFAULT_STEPS

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    model = _get_edit_model() if image_list else _get_model()

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


# The UI lives under this guard as a testability seam, not ceremony. Plain
# `import streamlit_app` (the unit tests) leaves __name__ == "streamlit_app", so
# the UI is skipped and the helpers above import without a Streamlit runtime;
# AppTest.from_file() and `streamlit run` exec the module as "__main__", so the
# UI runs. Dropping the guard would fire the whole UI on every plain import.
if __name__ == "__main__":
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":material/auto_awesome:",
        layout="wide",
    )

    st.title(APP_TITLE)

    col_controls, col_output = st.columns([2, 3], gap="large")

    # Reserve the output slot in the right column (filled after inference / on
    # rerun). Progress and the final image share this one slot so the live
    # status renders inside the same frame the image lands in.
    with col_output:
        result_slot = st.empty()

    with col_controls:
        # Prompt + Run, inline at the top (like the Gradio space). A borderless
        # form makes Enter in the prompt box submit the run; the horizontal
        # container lets the input stretch while the button hugs its icon+label
        # content, so the Run label never gets crammed into a narrow column and
        # wraps character-by-character.
        with (
            st.form("prompt_form", border=False),
            st.container(horizontal=True, vertical_alignment="bottom"),
        ):
            prompt = st.text_input(
                "Prompt",
                placeholder="Enter your prompt",
                key="prompt_input",
                label_visibility="collapsed",
                width="stretch",
            )
            run_clicked = st.form_submit_button(
                "Run",
                type="primary",
                icon=":material/play_arrow:",
            )

        # Optional input images — uploading any switches to editing automatically.
        # The keyed expander mirrors its open/closed state in session state, so
        # loading an example opens it programmatically (see _load_edit_example)
        # while the user's own toggling is respected on later reruns.
        _has_example_images = bool(st.session_state.get("example_images"))
        with st.expander(
            "Input image(s) (optional)",
            icon=":material/image:",
            key="input_expander",
            on_change="rerun",
        ):
            uploaded_files = st.file_uploader(
                "Input images",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key=f"uploader_{st.session_state.get('uploader_nonce', 0)}",
            )
            if not uploaded_files and _has_example_images:
                st.caption("Loaded example images:")
                # Unreadable paths would crash this preview before the load guard
                # below runs; that guard warns and clears them.
                with contextlib.suppress(OSError):
                    st.image(st.session_state.example_images, width=80)
                st.button("Clear example images", on_click=_clear_example_images)

        image_list = None
        if uploaded_files:
            # A manual upload overrides any loaded example images
            st.session_state.pop("example_images", None)
            try:
                image_list = [Image.open(f) for f in uploaded_files]
                _image_key = tuple((f.name, f.file_id) for f in uploaded_files)
            except OSError:
                st.warning(
                    "Could not load one or more uploaded images.",
                    icon=":material/warning:",
                )
                _image_key = ()
        elif st.session_state.get("example_images"):
            example_images = st.session_state.example_images
            try:
                image_list = [Image.open(p) for p in example_images]
                _image_key = tuple(example_images)
            except OSError:
                st.warning(
                    "Could not load the example images.",
                    icon=":material/warning:",
                )
                st.session_state.pop("example_images", None)
                _image_key = ()
        else:
            _image_key = ()
        if _image_key != st.session_state.get("prev_images", ()):
            st.session_state.prev_images = _image_key
            # An enhanced prompt is tied to its image set; drop it when that changes.
            st.session_state.pop("auto_enhanced_prompt", None)
            # Match the sliders to a new input image, but leave a manual size
            # untouched when the image set is cleared. Writes the width/height
            # slider keys, so it must stay above the Advanced settings expander
            # that instantiates those sliders.
            if image_list:
                _w, _h = _dimensions_from_images(image_list)
                st.session_state.width_slider = _w
                st.session_state.height_slider = _h

        if "last_prompt" not in st.session_state:
            st.session_state.last_prompt = ""

        # The prompt lives in a form, so typed edits only arrive here on
        # submit (where the Run branch drops the banner anyway); this check
        # catches prompt changes pushed via session state by example clicks.
        if prompt != st.session_state.last_prompt:
            st.session_state.last_prompt = prompt
            st.session_state.pop("auto_enhanced_prompt", None)

        final_prompt = prompt

        st.session_state.setdefault("width_slider", 1024)
        st.session_state.setdefault("height_slider", 1024)
        st.session_state.setdefault("steps_slider", DEFAULT_STEPS)
        st.session_state.setdefault("guidance_scale_slider", DEFAULT_GUIDANCE)

        # The four sliders below are instantiated on every run — two invariants
        # to preserve:
        #  1. Keys seeded above (width/height on image change, plus the
        #     steps/guidance defaults) are written before this line, so they land
        #     before the widgets exist. Keep those blocks above this expander.
        #  2. Do NOT gate this body with on_change="rerun" + `.open` to skip it
        #     when collapsed: the slider return values feed infer() below, so they
        #     must be assigned every run — a collapsed, un-run body leaves them
        #     unset (NameError at generate time).
        with st.expander("Advanced settings", icon=":material/tune:", expanded=False):
            auto_enhance = st.toggle(
                "Prompt upsampling",
                value=False,
                key="auto_enhance_toggle",
            )
            st.caption("Automatically enhance the prompt using a VLM")

            randomize_seed = st.toggle("Randomize seed", value=True)
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
        _ex_cols = st.columns(2)
        for _i, _example in enumerate(EXAMPLE_PROMPTS):
            with _ex_cols[_i % 2]:
                st.button(
                    _truncate(_example),
                    key=f"example_{_i}",
                    on_click=_set_example_prompt,
                    args=(_example,),
                    width="stretch",
                    help=_example,
                )

        st.markdown("**Editing examples**")
        for _i, (_ex_prompt, _ex_imgs) in enumerate(EDIT_EXAMPLES):
            _col_prompt, _col_imgs = st.columns([3, 2], vertical_alignment="center")
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

        if run_clicked and not final_prompt.strip() and not image_list:
            # The Run button is always enabled, so guard the empty request here
            # rather than run the VLM and a full diffusion pass on nothing.
            st.warning(
                "Enter a prompt or add an input image.",
                icon=":material/warning:",
            )
        elif run_clicked:
            st.session_state.pop("auto_enhanced_prompt", None)

            cache = st.session_state.setdefault("_enhance_cache", {})
            cache_key = (final_prompt, _image_key) if auto_enhance else None

            if cache_key is not None and cache_key in cache:
                run_prompt, was_auto_enhanced = cache[cache_key], True
            else:
                # The VLM call takes seconds (plus a first-use model load), and
                # it runs before the status frame opens — show a spinner so an
                # enhanced Run isn't silent. Skipped when upsampling is off.
                _enhance_ctx = (
                    st.spinner("Enhancing prompt…")
                    if auto_enhance
                    else contextlib.nullcontext()
                )
                with _enhance_ctx:
                    run_prompt, was_auto_enhanced = _resolve_prompt(
                        final_prompt, image_list, auto_enhance
                    )
                if was_auto_enhanced and cache_key is not None:
                    cache[cache_key] = run_prompt
                    # Bound the per-session cache so a long editing session
                    # can't accumulate enhanced prompts without limit.
                    if len(cache) > 32:
                        cache.pop(next(iter(cache)))

            if was_auto_enhanced:
                st.session_state.auto_enhanced_prompt = run_prompt

            generation_error = None
            with (
                result_slot.container(border=True, height=OUTPUT_FRAME_HEIGHT),
                st.status("Generating image…", expanded=True) as status,
            ):
                progress_bar = st.progress(0, text="Starting…")

                def _update_progress(step, total):
                    progress_bar.progress(step / total, text=f"Step {step}/{total}")

                try:
                    image, used_seed = infer(
                        run_prompt,
                        seed_val,
                        randomize_seed,
                        width,
                        height,
                        guidance_scale,
                        num_inference_steps,
                        image_list=image_list,
                        progress_callback=_update_progress,
                    )
                except Exception as exc:
                    status.update(label="Generation failed", state="error")
                    generation_error = str(exc)
                else:
                    status.update(label="Image generated", state="complete")
                    st.session_state.result_image = image
                    st.session_state.result_seed = used_seed

            # Surface a failure in the controls column, where it survives the
            # bottom block re-rendering result_slot (which overwrites the status).
            if generation_error is not None:
                st.error(
                    f"Image generation failed: {generation_error}",
                    icon=":material/error:",
                )

        if auto_enhance and "auto_enhanced_prompt" in st.session_state:
            st.info(
                f"Enhanced prompt: {st.session_state.auto_enhanced_prompt}",
                icon=":material/auto_awesome:",
            )

    # Output (right column): result image, or a placeholder canvas. Both states
    # share a bordered container so the output frame stays consistent, and the
    # placeholder uses native theme-aware centering instead of raw-HTML colors.
    if "result_image" in st.session_state:
        with result_slot.container(border=True, horizontal_alignment="center"):
            st.image(st.session_state.result_image, width="stretch")
            if st.session_state.result_seed is not None:
                st.caption(f"Seed: {st.session_state.result_seed}")
    else:
        with result_slot.container(
            border=True,
            height=OUTPUT_FRAME_HEIGHT,
            horizontal_alignment="center",
            vertical_alignment="center",
        ):
            st.markdown(":gray[:material/image: Your image will appear here]")
