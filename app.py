"""
REALITYSEAL-X Streamlit image-forensics prototype.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io

import streamlit as st
import config
from PIL import Image
from ai_image_detector import (
    AiImageDetectorError,
    AiImageDetectionResult,
    AttentionVisualization,
    analyze_ai_generated_image,
    generate_attention_visualization,
)
from cnn_detector import (
    CnnDetectionResult,
    CnnDetectorError,
    analyze_face_manipulation,
)
from ela_analyzer import ElaResult, analyze_image
from image_io import ImageLoadError, load_image_from_bytes
from metadata_analyzer import BasicMetadata, analyze_file
from report_generator import report_json, report_text
from result_analyzer import AnalysisSummary, build_analysis_summary
from vit_detector import (
    VitDetectionResult,
    VitDetectorError,
    analyze_vit_deepfake,
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1200px; padding-top: 1.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _contributor_explanation(components: dict) -> str:
    contributions = {
        "mean ELA difference": components["mean_term_0_to_1"] * components["weight_mean"],
        "standard deviation": components["std_term_0_to_1"] * components["weight_std"],
        "maximum ELA difference": components["max_term_0_to_1"] * components["weight_max"],
        "share of pixels above the hot-pixel threshold": (
            components["hot_percent_term_0_to_1"] * components["weight_hot"]
        ),
        "99th-percentile residual": components.get("p99_term_0_to_1", 0)
        * components.get("weight_p99", 0),
        "local block contrast": components.get("contrast_term_0_to_1", 0)
        * components.get("weight_contrast", 0),
        "outlier blocks": components.get("outlier_blocks_term_0_to_1", 0)
        * components.get("weight_outlier_blocks", 0),
    }
    ranked = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
    top = ", ".join(f"{name} ({value:.3f})" for name, value in ranked[:3])
    return (
        "The heuristic combines seven image-comparison measurements (see config.py). "
        f"Largest contributions in this run: {top}."
    )


def _render_ela_results(ela: ElaResult, meta: BasicMetadata) -> None:
    st.success("Analysis complete.")
    st.info(ela.source_mode_note)

    score = ela.suspicion.score
    is_png = (meta.format or "").upper() == "PNG"
    st.markdown("#### Result")
    with st.container(border=True):
        result_col, score_col = st.columns([2, 1])
        with result_col:
            if is_png:
                st.info(
                    "**NOT ASSESSED** — PNG-to-JPEG conversion results cannot "
                    "determine whether a PNG was edited."
                )
            elif score <= config.LOW_SCORE_MAX:
                st.success("**NOT FLAGGED** — no strong suspicious ELA pattern found.")
            elif score <= config.MODERATE_SCORE_MAX:
                st.warning(
                    "**FLAGGED AS SUSPICIOUS** — an unusual ELA pattern was found; "
                    "review is recommended."
                )
            else:
                st.error(
                    "**FLAGGED AS HIGHLY SUSPICIOUS** — a strong unusual ELA pattern "
                    "was found."
                )
            st.caption(
                "A flagged pattern can also come from recompression or camera processing. "
                "This test alone cannot prove whether an image was edited."
            )
        with score_col:
            score_label = "JPEG conversion difference" if is_png else "ELA contrast score"
            st.metric(
                score_label,
                f"{score}/100",
                help=(
                    "Heuristic strength of ELA differences, not the percentage chance "
                    "that the image was edited."
                ),
            )
            st.progress(score / 100.0)
    st.caption(
        "The score is a heuristic signal, not a calibrated probability or proof of editing."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Original (with highlighted area, if any)")
        st.image(ela.overlay_image, width="stretch")
        st.caption(
            "A red box appears only when one region stands out from the rest. "
            "No box means this test did not find a compact hotspot."
        )
    with col_b:
        st.markdown("#### ELA heatmap")
        st.image(ela.ela_heatmap, width="stretch")
        st.caption(
            "Warmer / brighter = larger difference after re-saving as JPEG. "
            "Colours are stretched so small patches stay visible."
        )

    st.markdown("#### Amplified ELA residual (RGB)")
    st.image(ela.ela_rgb, width="stretch")
    st.caption(
        "Brighter regions indicate larger differences between the decoded "
        "original pixels and a JPEG recompressed at the selected quality."
    )

    st.markdown("#### ELA statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Mean ELA difference", f"{ela.statistics.mean_difference:.3f}")
    s2.metric("Maximum ELA difference", f"{ela.statistics.max_difference:.3f}")
    s3.metric("Standard deviation", f"{ela.statistics.std_difference:.3f}")
    s4.metric(
        "Pixels above threshold",
        f"{ela.statistics.percent_pixels_above_threshold:.2f}%",
    )
    t1, t2, t3 = st.columns(3)
    t1.metric("99th percentile ELA", f"{ela.statistics.p99_difference:.3f}")
    t2.metric("Local contrast (hot/typical block)", f"{ela.statistics.block_contrast_ratio:.2f}×")
    t3.metric("Outlier blocks", f"{ela.statistics.outlier_block_percent:.2f}%")
    st.caption(
        f"Hot-pixel threshold = {ela.statistics.hot_pixel_threshold:.1f} on the raw "
        f"0–255 difference scale (before amplification). Pixel count = "
        f"{ela.statistics.pixel_count}."
    )

    st.markdown("#### ELA contrast score details")
    st.write(ela.suspicion.interpretation)
    st.write(_contributor_explanation(ela.suspicion.components))
    with st.expander("Scoring formula and components (inspectable)"):
        st.write(ela.suspicion.formula_note)
        st.json(ela.suspicion.components)
        st.caption(
            "Thresholds and caps are named in config.py. They need calibration using "
            "a larger labelled dataset and must not be described as scientifically proven."
        )

    st.markdown("#### Basic metadata and file integrity")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Format", meta.format or "Unknown")
    m2.metric("Width × height", f"{meta.width} × {meta.height}")
    m3.metric("File size", f"{meta.file_size_bytes:,} bytes")
    m4.metric("EXIF tags", str(len(meta.exif)))
    st.code(meta.sha256, language=None)
    st.caption(
        "SHA-256 identifies the exact uploaded file and can help check whether that "
        "file changes. It does not establish whether the image is authentic."
    )
    if meta.exif:
        st.json(meta.exif)
    else:
        st.write(
            "No EXIF metadata was readable in this file. Missing EXIF is not proof "
            "of manipulation."
        )

    st.markdown("#### Downloadable analysis report")
    text_report = report_text(meta, ela)
    json_report = report_json(meta, ela)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download text report",
            data=text_report,
            file_name="realityseal_x_ela_report.txt",
            mime="text/plain",
        )
    with d2:
        st.download_button(
            "Download JSON report",
            data=json_report,
            file_name="realityseal_x_ela_report.json",
            mime="application/json",
        )
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with st.expander("Limitations", expanded=True):
        for item in config.LIMITATIONS:
            st.markdown(f"- {item}")

    with st.expander("What this prototype does not implement"):
        st.markdown(
            "- Deep-learning detectors, C2PA/content credentials, or full provenance graphs\n"
            "- Copy-move specialised matching, video analysis, or batch forensic casework\n"
            "- Cloud APIs, login, databases, or legally certified forensic reporting"
        )


def _render_cnn_status(image: Image.Image | None) -> None:
    st.subheader("CNN facial manipulation detection")
    if image is None:
        st.info("Upload an image to run this detector.")
        return

    st.caption(
        "Free, public EfficientNet-B2 checkpoint. It downloads only when you click "
        "run; inference stays on this computer. OpenCV detects frontal faces locally."
    )
    if st.button("Run CNN face manipulation analysis", type="primary"):
        try:
            with st.spinner("Detecting face and running EfficientNet-B2 on CPU…"):
                st.session_state.cnn_result = analyze_face_manipulation(image)
            st.session_state.cnn_error = None
        except CnnDetectorError as exc:
            st.session_state.cnn_result = None
            st.session_state.cnn_error = str(exc)

    if st.session_state.cnn_error:
        st.error(f"CNN analysis failed: {st.session_state.cnn_error}")

    result: CnnDetectionResult | None = st.session_state.cnn_result
    if result is None:
        st.info("Not run yet. The checkpoint is downloaded only after you start analysis.")
        return
    st.caption(f"Model: {result.model_name}")
    if result.status == "not_applicable":
        st.info(result.explanation)
        return

    outcome_col, score_col = st.columns(2)
    outcome_col.metric("Face manipulation result", result.prediction or "Unavailable")
    score_col.metric(
        "Predicted-class model score",
        f"{result.confidence:.1%}" if result.confidence is not None else "Unavailable",
        help="Raw model output, not a calibrated probability.",
    )
    st.image(result.face_crop, caption="Face crop analyzed by the CNN", width="content")
    st.write(result.explanation)


def _render_vit_status(image: Image.Image | None) -> None:
    st.subheader("Vision Transformer deepfake detection")
    if image is None:
        st.info("Upload an image to run this detector.")
        return

    st.caption(
        "Free Apache-2.0 public ViT checkpoint. It downloads only when you click "
        "run; inference stays on this computer and uses the model's published processor."
    )
    if st.button("Run ViT deepfake analysis", type="primary"):
        try:
            with st.spinner("Loading ViT and analyzing image on CPU…"):
                st.session_state.vit_result = analyze_vit_deepfake(image)
            st.session_state.vit_error = None
        except VitDetectorError as exc:
            st.session_state.vit_result = None
            st.session_state.vit_error = str(exc)

    if st.session_state.vit_error:
        st.error(f"ViT analysis failed: {st.session_state.vit_error}")

    result: VitDetectionResult | None = st.session_state.vit_result
    if result is None:
        st.info("Not run yet. The checkpoint is downloaded only after you start analysis.")
        return

    st.caption(f"Model: {result.model_name}")
    outcome_col, score_col = st.columns(2)
    outcome_col.metric("Deepfake result", result.prediction)
    score_col.metric(
        "Predicted-class model score",
        f"{result.confidence:.1%}",
        help="Raw model output, not a calibrated probability.",
    )
    st.metric("Manipulated class score", f"{result.manipulated_score:.1%}")
    st.write(result.explanation)


def _render_ai_generated_tab(image: Image.Image | None) -> None:
    st.subheader("AI-generated image detection")
    st.caption(
        "A general-image classifier, independent from the face-manipulation detectors."
    )
    if image is None:
        st.info("Upload an image to run this detector.")
        return

    st.caption(
        "The Community Forensics checkpoint (~87 MB) is fetched on the first "
        "explicit run and cached locally. No dataset is downloaded."
    )
    if st.button("Run AI-generated image analysis", type="primary"):
        try:
            with st.spinner("Loading model and analyzing image on CPU…"):
                st.session_state.ai_image_result = analyze_ai_generated_image(image)
            st.session_state.ai_image_error = None
        except AiImageDetectorError as exc:
            st.session_state.ai_image_result = None
            st.session_state.ai_image_error = str(exc)

    if st.session_state.ai_image_error:
        st.error(f"AI-generated image analysis failed: {st.session_state.ai_image_error}")

    result: AiImageDetectionResult | None = st.session_state.ai_image_result
    if result is None:
        return

    st.markdown("#### Independent model result")
    class_col, confidence_col, generated_score_col = st.columns(3)
    class_col.metric("Predicted class", result.predicted_class)
    confidence_col.metric(
        "Predicted-class model score",
        f"{result.predicted_class_score:.1%}",
        help="Raw model output for the selected class; not a calibrated probability.",
    )
    generated_score_col.metric(
        "AI-generated model score",
        f"{result.ai_generated_score:.1%}",
        help="Sigmoid model score; not a calibrated probability or proof of origin.",
    )
    st.caption(f"Model: {result.model_name}")
    st.write(result.explanation)
    st.caption(result.input_note)

    if st.button("Generate ViT attention visualization"):
        try:
            with st.spinner("Generating model attention visualization…"):
                st.session_state.ai_attention = generate_attention_visualization(image)
            st.session_state.ai_attention_error = None
        except AiImageDetectorError as exc:
            st.session_state.ai_attention = None
            st.session_state.ai_attention_error = str(exc)

    if st.session_state.ai_attention_error:
        st.error(f"Attention visualization failed: {st.session_state.ai_attention_error}")
    attention: AttentionVisualization | None = st.session_state.ai_attention
    if attention is not None:
        st.markdown("#### ViT attention (supporting indicator)")
        st.image(attention.overlay, width="stretch")
        st.caption(attention.explanation)


def _render_combined_dashboard(
    image: Image.Image | None,
    ela: ElaResult | None,
    meta: BasicMetadata | None,
) -> None:
    st.subheader("Combined analysis dashboard")
    if image is None:
        st.info("Upload an image to view its independent analysis results.")
        return

    cnn_result: CnnDetectionResult | None = st.session_state.cnn_result
    vit_result: VitDetectionResult | None = st.session_state.vit_result
    summary: AnalysisSummary = build_analysis_summary(
        ela=ela,
        image_format=meta.format if meta is not None else image.format,
        cnn=cnn_result,
        vit=vit_result,
        ai_image=st.session_state.ai_image_result,
        cnn_error=st.session_state.cnn_error,
        vit_error=st.session_state.vit_error,
        ai_error=st.session_state.ai_image_error,
    )

    st.markdown("#### Independent-results summary")
    if summary.label in {
        "Suspicious indication detected",
        "Different models produced conflicting results",
    }:
        st.warning(f"**{summary.label}** — {summary.explanation}")
    elif summary.label == "Analysis incomplete":
        st.info(f"**{summary.label}** — {summary.explanation}")
    else:
        st.success(f"**{summary.label}** — {summary.explanation}")
    if summary.suspicious_indications:
        for indication in summary.suspicious_indications:
            st.markdown(f"- {indication}")
    st.caption(
        "No authenticity probability is calculated. ELA residual analysis and "
        "AI-generated-image classification are different tasks; their scores are "
        "kept separate and are not averaged."
    )

    st.markdown("#### Original image")
    st.image(image.convert("RGB"), width="stretch")

    st.markdown("#### Layer results")
    ela_col, cnn_col, vit_col, ai_col = st.columns(4)
    with ela_col:
        st.markdown("**ELA**")
        if ela is None:
            st.info("Not run")
        elif ((meta.format if meta is not None else image.format) or "").upper() == "PNG":
            st.info("Not assessed — PNG-to-JPEG conversion only")
        else:
            st.metric("Heuristic score", f"{ela.suspicion.score}/100")
            st.write(ela.suspicion.indicator)
    with cnn_col:
        st.markdown("**CNN face manipulation**")
        if cnn_result is None and st.session_state.cnn_error:
            st.error("Analysis failed")
            st.caption(st.session_state.cnn_error)
        elif cnn_result is None:
            st.info("Not run")
        elif cnn_result.status == "not_applicable":
            st.info("Not applicable — no face detected")
        else:
            st.write(cnn_result.prediction)
            if cnn_result.confidence is not None:
                st.metric("Model score", f"{cnn_result.confidence:.1%}")
    with vit_col:
        st.markdown("**ViT face deepfake**")
        if vit_result is None and st.session_state.vit_error:
            st.error("Analysis failed")
            st.caption(st.session_state.vit_error)
        elif vit_result is None:
            st.info("Not run")
        else:
            st.write(vit_result.prediction)
            st.metric("Model score", f"{vit_result.confidence:.1%}")
    with ai_col:
        st.markdown("**AI-generated image**")
        ai_result: AiImageDetectionResult | None = st.session_state.ai_image_result
        if ai_result is not None:
            st.write(ai_result.predicted_class)
            st.metric(
                "Predicted-class model score",
                f"{ai_result.predicted_class_score:.1%}",
            )
            st.metric("AI-generated model score", f"{ai_result.ai_generated_score:.1%}")
        elif st.session_state.ai_image_error:
            st.error("Analysis failed")
            st.caption(st.session_state.ai_image_error)
        else:
            st.info("Not run")

    if ela is not None:
        st.markdown("#### ELA heatmap")
        st.image(ela.ela_heatmap, width="stretch")
        st.caption(
            "ELA colors represent JPEG-recompression residual differences; they "
            "are not a model confidence visualization."
        )
    attention: AttentionVisualization | None = st.session_state.ai_attention
    if attention is not None:
        st.markdown("#### AI detector attention visualization")
        st.image(attention.overlay, width="stretch")
        st.caption(attention.explanation)

    st.markdown("#### Metadata and file integrity")
    if meta is None:
        st.info("Run ELA Analysis to extract metadata and SHA-256.")
    else:
        metadata_col, hash_col = st.columns([1, 2])
        with metadata_col:
            st.write(f"**Format:** {meta.format or 'Unknown'}")
            st.write(f"**Dimensions:** {meta.width} × {meta.height}")
            st.write(f"**File size:** {meta.file_size_bytes:,} bytes")
            st.write(f"**EXIF tags:** {len(meta.exif)}")
            if meta.exif:
                st.json(meta.exif)
        with hash_col:
            st.markdown("**SHA-256**")
            st.code(meta.sha256, language=None)
            st.caption("The hash identifies these exact file bytes; it does not prove authenticity.")

    if summary.unavailable_models:
        st.caption("Unavailable: " + "; ".join(summary.unavailable_models))
    if summary.not_assessed_layers:
        st.caption("Not assessed: " + "; ".join(summary.not_assessed_layers))
    if summary.not_run_layers:
        st.caption("Not run: " + "; ".join(summary.not_run_layers))
    if summary.failed_layers:
        st.caption("Failed: " + "; ".join(summary.failed_layers))


def main() -> None:
    st.set_page_config(
        page_title=f"{config.PROJECT_NAME} — Image Forensics",
        page_icon="🛡️",
        layout="wide",
    )
    _inject_css()

    st.title(config.PROJECT_NAME)
    st.subheader(config.PROJECT_SUBTITLE)
    st.caption(
        f"{config.PROJECT_LAYER}. Models are free public checkpoints, fetched "
        "only when you run their independent analysis tabs; inference is local on CPU."
    )

    with st.sidebar:
        st.header("ELA settings")
        st.caption(
            "These values are demonstration defaults. Change them here for the current "
            "session; persistent defaults live in config.py."
        )
        jpeg_quality = st.slider(
            "JPEG recompress quality",
            min_value=50,
            max_value=95,
            value=int(config.JPEG_RECOMPRESS_QUALITY),
            help="Quality used when writing the temporary JPEG copy.",
        )
        amplification = st.slider(
            "ELA amplification scale",
            min_value=5.0,
            max_value=30.0,
            value=float(config.ELA_AMPLIFICATION_SCALE),
            step=1.0,
        )
        hot_threshold = st.slider(
            "Hot-pixel threshold (raw difference)",
            min_value=5.0,
            max_value=60.0,
            value=float(config.HOT_PIXEL_THRESHOLD),
            step=1.0,
        )
        st.divider()
        st.markdown(f"**Max upload size:** {config.MAX_UPLOAD_MB} MB")
        st.markdown("**Allowed types:** JPG, JPEG, PNG")
        st.caption("The original uploaded file is never overwritten.")

    uploaded = st.file_uploader(
        "Upload a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if "last_name" not in st.session_state:
        st.session_state.last_name = None
        st.session_state.result = None
        st.session_state.meta = None
        st.session_state.error = None
        st.session_state.original_preview = None
        st.session_state.ai_image_result = None
        st.session_state.ai_image_error = None
        st.session_state.ai_attention = None
        st.session_state.ai_attention_error = None
        st.session_state.cnn_result = None
        st.session_state.cnn_error = None
        st.session_state.vit_result = None
        st.session_state.vit_error = None
    if "ai_image_result" not in st.session_state:
        st.session_state.ai_image_result = None
        st.session_state.ai_image_error = None
    if "ai_attention" not in st.session_state:
        st.session_state.ai_attention = None
        st.session_state.ai_attention_error = None
    if "cnn_result" not in st.session_state:
        st.session_state.cnn_result = None
        st.session_state.cnn_error = None
    if "vit_result" not in st.session_state:
        st.session_state.vit_result = None
        st.session_state.vit_error = None

    if uploaded is None:
        st.info("Upload an image, then click **Analyse Image**.")
        st.session_state.result = None
        st.session_state.meta = None
        st.session_state.error = None
        st.session_state.original_preview = None
        st.session_state.last_name = None
        st.session_state.loaded_image = None
        st.session_state.ai_image_result = None
        st.session_state.ai_image_error = None
        st.session_state.ai_attention = None
        st.session_state.ai_attention_error = None
        st.session_state.cnn_result = None
        st.session_state.cnn_error = None
        st.session_state.vit_result = None
        st.session_state.vit_error = None
    else:
        if uploaded.name != st.session_state.last_name:
            st.session_state.result = None
            st.session_state.meta = None
            st.session_state.error = None
            st.session_state.original_preview = None
            st.session_state.last_name = uploaded.name
            st.session_state.ai_image_result = None
            st.session_state.ai_image_error = None
            st.session_state.ai_attention = None
            st.session_state.ai_attention_error = None
            st.session_state.cnn_result = None
            st.session_state.cnn_error = None
            st.session_state.vit_result = None
            st.session_state.vit_error = None

        file_bytes = uploaded.getvalue()
        try:
            image = load_image_from_bytes(file_bytes, uploaded.name)
            preview = image.convert("RGB")
            st.session_state.original_preview = preview
            st.session_state.loaded_image = image
            st.session_state.file_bytes = file_bytes
        except ImageLoadError as exc:
            st.error(str(exc))
            st.stop()

        st.markdown("#### Uploaded original")
        st.image(st.session_state.original_preview, width="stretch")

        analyse = st.button("Analyse Image", type="primary")
        if analyse:
            try:
                with st.spinner("Running JPEG-recompression ELA…"):
                    ela = analyze_image(
                        image,
                        jpeg_quality=jpeg_quality,
                        amplification_scale=amplification,
                        hot_pixel_threshold=hot_threshold,
                        original_format=image.format,
                    )
                    meta = analyze_file(uploaded.name, file_bytes, image)
                st.session_state.result = ela
                st.session_state.meta = meta
                st.session_state.error = None
            except Exception as exc:
                st.session_state.result = None
                st.session_state.meta = None
                st.session_state.error = f"Analysis failed: {exc}"

    if st.session_state.error:
        st.error(st.session_state.error)

    ela = st.session_state.result
    meta = st.session_state.meta
    ela_tab, cnn_tab, vit_tab, ai_tab, dashboard_tab = st.tabs(
        [
            "ELA Analysis",
            "CNN Deepfake Detection",
            "Vision Transformer Detection",
            "AI-Generated Image Detection",
            "Combined Analysis Dashboard",
        ]
    )
    with ela_tab:
        if ela is None or meta is None:
            with st.expander("Limitations", expanded=True):
                for item in config.LIMITATIONS:
                    st.markdown(f"- {item}")
        else:
            _render_ela_results(ela, meta)
    with cnn_tab:
        _render_cnn_status(st.session_state.loaded_image)
    with vit_tab:
        _render_vit_status(st.session_state.loaded_image)
    with ai_tab:
        _render_ai_generated_tab(st.session_state.loaded_image)
    with dashboard_tab:
        _render_combined_dashboard(
            image=st.session_state.loaded_image,
            ela=ela,
            meta=meta,
        )


if __name__ == "__main__":
    main()
