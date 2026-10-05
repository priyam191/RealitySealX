"""
REALITYSEAL-X Streamlit prototype — ELA layer only.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io

import streamlit as st
import config
from ela_analyzer import analyze_image
from image_io import ImageLoadError, load_image_from_bytes
from metadata_analyzer import analyze_file
from report_generator import report_json, report_text


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1200px; padding-top: 1.4rem; }
        .score-box {
            border: 1px solid #d0d7de;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            background: #f6f8fa;
        }
        .warn-box {
            border-left: 4px solid #d4a017;
            padding: 0.6rem 0.9rem;
            background: #fff8e8;
            margin: 0.6rem 0 1rem 0;
        }
        .verdict-card {
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin: 0.4rem 0 1rem 0;
            border: 1px solid #d0d7de;
        }
        .verdict-low { background: #eef8f0; border-color: #8fd19e; }
        .verdict-mid { background: #fff8e8; border-color: #e0c060; }
        .verdict-high { background: #fdeeee; border-color: #e08080; }
        .verdict-card h3 { margin: 0 0 0.35rem 0; }
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


def main() -> None:
    st.set_page_config(
        page_title=f"{config.PROJECT_NAME} — ELA Prototype",
        page_icon="🛡️",
        layout="wide",
    )
    _inject_css()

    st.title(config.PROJECT_NAME)
    st.subheader(config.PROJECT_SUBTITLE)
    st.caption(
        f"{config.PROJECT_LAYER}. Local, free, CPU-only demonstration. "
        "No cloud APIs, model weights, or GPU required."
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

    if uploaded is None:
        st.info("Upload an image, then click **Analyse Image**.")
        st.session_state.result = None
        st.session_state.meta = None
        st.session_state.error = None
        st.session_state.original_preview = None
        st.session_state.last_name = None
    else:
        if uploaded.name != st.session_state.last_name:
            st.session_state.result = None
            st.session_state.meta = None
            st.session_state.error = None
            st.session_state.original_preview = None
            st.session_state.last_name = uploaded.name

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
    if ela is None or meta is None:
        with st.expander("Limitations", expanded=True):
            for item in config.LIMITATIONS:
                st.markdown(f"- {item}")
        return

    st.success("Analysis complete. Start with the plain-language result, then inspect the heatmap.")
    st.info(ela.source_mode_note)

    score = ela.suspicion.score
    if (meta.format or "").upper() == "PNG":
        verdict_class = "verdict-mid"
    elif score <= config.LOW_SCORE_MAX:
        verdict_class = "verdict-low"
    elif score <= config.MODERATE_SCORE_MAX:
        verdict_class = "verdict-mid"
    else:
        verdict_class = "verdict-high"
    st.markdown("#### What this means (plain language)")
    st.markdown(
        f'<div class="verdict-card {verdict_class}">'
        f"<h3>{ela.suspicion.simple_headline}</h3>"
        f"<p><strong>Simple reading:</strong> {ela.suspicion.simple_status}</p>"
        f"<p><strong>Pattern score:</strong> {ela.suspicion.score}/100 "
        f"(strength of this test's signal, not the chance the image was edited) "
        f"&nbsp;|&nbsp; {ela.suspicion.indicator}</p>"
        f"<p>{ela.suspicion.simple_explanation}</p>"
        f"<p>{ela.suspicion.heatmap_hint}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="warn-box">{config.SCORE_WARNING}</div>',
        unsafe_allow_html=True,
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

    st.markdown("#### ELA Suspicion Score (technical)")
    st.markdown(
        f'<div class="score-box"><strong>ELA Suspicion Score: '
        f"{ela.suspicion.score}/100</strong><br>Indicator: {ela.suspicion.indicator}</div>",
        unsafe_allow_html=True,
    )
    st.progress(ela.suspicion.score / 100.0)
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


if __name__ == "__main__":
    main()
