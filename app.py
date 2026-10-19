"""Streamlit UI for dental caries detection. Run with: streamlit run app.py

UI only: model, drawing and storage logic live in the caries_detector package.
"""

import hashlib
import io

import streamlit as st

from caries_detector import (
    InvalidImageError,
    ModelNotFoundError,
    draw_detections,
    load_model,
    predict,
    prepare_image,
    save_result,
    warm_up,
)
from caries_detector import config

st.set_page_config(page_title="Dental Caries Detection", layout="wide")

st.markdown(
    f"""
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] {{ border-color: #bbf7d0; }}
    [data-testid="stMetric"] {{ background: {config.LIGHT_GREEN}; border-radius: 0.5rem; padding: 0.5rem 0.75rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def footer() -> None:
    st.divider()
    st.caption(config.DISCLAIMER)


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    model = load_model(config.MODEL_PATH)
    warm_up(model)
    return model


@st.cache_data(show_spinner=False)
def list_sample_images() -> list[str]:
    folder = config.SAMPLE_IMAGES_DIR
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower().lstrip(".") in config.IMAGE_EXTENSIONS)


def nearest_threshold_row(conf: float) -> int:
    return min(range(len(config.THRESHOLD_TABLE)), key=lambda i: abs(config.THRESHOLD_TABLE[i]["conf"] - conf))


def render_sidebar() -> float:
    with st.sidebar:
        st.header("Settings")
        conf = st.slider(
            "Confidence threshold",
            min_value=config.CONF_MIN,
            max_value=config.CONF_MAX,
            value=config.DEFAULT_CONF,
            step=config.CONF_STEP,
        )
        st.caption("Lower = catches more cavities, more false alarms. Higher = fewer false alarms, more missed cavities.")

        with st.expander("Model info"):
            for key, value in config.MODEL_INFO.items():
                st.markdown(f"**{key}:** {value}")
            st.markdown("**Test set** (285 images of unseen people)")
            st.markdown(" · ".join(f"{k} {v:.3f}" for k, v in config.TEST_METRICS.items()))

            st.markdown("**Precision and recall by threshold** (test set)")
            highlight = nearest_threshold_row(conf)
            rows = [
                {"Conf": f"{r['conf']:.1f}", "Precision": f"{r['precision']:.3f}", "Recall": f"{r['recall']:.3f}"}
                for r in config.THRESHOLD_TABLE
            ]
            header = "| Conf | Precision | Recall |\n|---|---|---|\n"
            body = "\n".join(
                ("| **{Conf}** | **{Precision}** | **{Recall}** |" if i == highlight else "| {Conf} | {Precision} | {Recall} |").format(**r)
                for i, r in enumerate(rows)
            )
            st.markdown(header + body)
            st.caption(f"Bold row: closest to the current threshold ({conf:.2f}).")
    return conf


def collect_inputs() -> list[tuple[str, bytes]]:
    """Return (filename, raw bytes) for every uploaded file and the optional sample."""
    uploads = st.file_uploader(
        "Upload intraoral photos",
        type=list(config.IMAGE_EXTENSIONS),
        accept_multiple_files=True,
    )
    inputs = [(f.name, f.getvalue()) for f in uploads or []]

    samples = list_sample_images()
    if samples:
        choice = st.selectbox("Or try a sample image from the test set", ["None"] + samples)
        if choice != "None":
            try:
                inputs.append((choice, (config.SAMPLE_IMAGES_DIR / choice).read_bytes()))
            except OSError as exc:
                st.error(f"Could not read sample image {choice}: {exc}")
    return inputs


def save_once(key: str, annotated, name, detections, conf, inference_ms, size):
    """Save each (image, threshold) pair once per session, not on every rerun."""
    saved = st.session_state.setdefault("saved", {})
    paths = saved.get(key)
    if paths is None or not all(p.exists() for p in paths):
        paths = save_result(
            annotated,
            original_filename=name,
            detections=detections,
            conf=conf,
            inference_ms=inference_ms,
            image_size=size,
        )
        saved[key] = paths
    return paths


def render_result(model, name: str, data: bytes, conf: float) -> None:
    with st.container(border=True):
        st.subheader(name)
        try:
            image = prepare_image(io.BytesIO(data))
        except InvalidImageError as exc:
            st.error(f"{name}: {exc}")
            return

        detections, inference_ms = predict(model, image, conf)
        annotated = draw_detections(image, detections)

        left, right = st.columns(2)
        left.image(image, caption="Original", width="stretch")
        right.image(annotated, caption=f"Detections at threshold {conf:.2f}", width="stretch")

        m1, m2, m3 = st.columns(3)
        m1.metric("Suspected lesions", len(detections))
        m2.metric("Highest confidence", f"{detections[0].confidence:.2f}" if detections else "-")
        m3.metric("Inference time", f"{inference_ms:.0f} ms")

        if detections:
            st.dataframe(
                [
                    {
                        "#": i,
                        "Confidence": round(d.confidence, 3),
                        "x1": d.xyxy[0],
                        "y1": d.xyxy[1],
                        "x2": d.xyxy[2],
                        "y2": d.xyxy[3],
                        "Box size (px)": f"{d.width} x {d.height}",
                    }
                    for i, d in enumerate(detections, start=1)
                ],
                hide_index=True,
            )
        else:
            st.info("No suspected lesions at this threshold.")

        key = f"{hashlib.sha1(data).hexdigest()}_{conf:.2f}"
        try:
            image_path, json_path = save_once(key, annotated, name, detections, conf, inference_ms, image.size)
        except OSError as exc:
            st.error(f"Could not save results: {exc}")
            return

        rel_dir = image_path.parent.relative_to(config.ROOT_DIR).as_posix()
        st.caption(f"Saved to `{rel_dir}/`: `{image_path.name}` and `{json_path.name}`")
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button(
            "Download annotated image",
            data=image_path.read_bytes(),
            file_name=image_path.name,
            mime="image/jpeg",
            key=f"img_{key}",
            on_click="ignore",
        )
        d2.download_button(
            "Download JSON",
            data=json_path.read_bytes(),
            file_name=json_path.name,
            mime="application/json",
            key=f"json_{key}",
            on_click="ignore",
        )


def main() -> None:
    st.title("Dental Caries Detection")
    st.markdown("Upload an intraoral photo to screen for tooth decay with a YOLO26 model")

    conf = render_sidebar()

    try:
        model = get_model()
    except ModelNotFoundError:
        st.error(
            "The model weights were not found. Put the trained weights file at "
            f"`{config.MODEL_PATH.relative_to(config.ROOT_DIR).as_posix()}` (relative to the repository root) "
            "and reload the page."
        )
        footer()
        st.stop()
    except Exception as exc:  # corrupt weights, incompatible Ultralytics version, ...
        st.error(f"The model could not be loaded: {exc}")
        footer()
        st.stop()

    inputs = collect_inputs()
    if not inputs:
        st.info("Upload one or more .jpg or .png photos to get started.")

    seen = set()
    for name, data in inputs:
        digest = hashlib.sha1(data).hexdigest()
        if digest in seen:  # same image uploaded twice, or also picked as a sample
            continue
        seen.add(digest)
        render_result(model, name, data, conf)

    footer()


main()
