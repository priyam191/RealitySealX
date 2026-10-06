# REALITYSEAL-X — ELA prototype

**Image Integrity and ELA-Based Tampering Analysis — Prototype**

This repository is a **local, free, CPU-only** demonstration of **one** layer of a larger proposed framework: **Error Level Analysis (ELA)** for still images. It does **not** implement deep-learning detection, C2PA provenance, copy-move search, video analysis, or a legally certified forensic workflow.

College demonstration target: **7 October 2026**.

## What is implemented

1. Upload a JPG, JPEG, or PNG in a Streamlit browser UI.
2. Preview the original image (pixels loaded in memory; the original file is not modified).
3. Recompress an RGB copy as JPEG (default quality 90), compute the absolute pixel difference, amplify it, and show an ELA heatmap.
4. Report mean / max / standard deviation of the difference map and the percentage of pixels above a configurable threshold.
5. Show a transparent **ELA contrast score (0–100)** and a presentation-aligned result: **Not flagged**, **Flagged as suspicious**, or **Flagged as highly suspicious**. This is a heuristic, not a calibrated probability.
6. Show basic metadata (format, size, EXIF if present) and a **SHA-256** digest of the uploaded bytes.
7. Download a text or JSON report. Pixel data are not written into the report.

## File structure

```
RealitySealX/
  app.py                      # Streamlit UI
  ela_analyzer.py             # ELA pipeline and heuristic score
  metadata_analyzer.py        # EXIF + SHA-256
  report_generator.py         # Text/JSON reports
  image_io.py                 # Upload validation and safe loading
  config.py                   # Limits, ELA settings, score caps (calibrate later)
  generate_sample_images.py   # Builds synthetic demo images
  requirements.txt
  README.md
  .streamlit/config.toml      # 10 MB upload cap
  sample_images/              # Created by the generator script
  outputs/                    # Optional local report folder
  tests/test_ela_and_hash.py
```

## Setup (Windows PowerShell)

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_sample_images.py
python -m pytest -q
streamlit run app.py
```

If script activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_sample_images.py
python -m pytest -q
streamlit run app.py
```

## How ELA is computed here

1. Load the upload and convert pixels to RGB (PNG transparency is flattened onto white).
2. Save a **temporary in-memory JPEG** at the chosen quality (default 90). The original file on disk is not rewritten.
3. Reload that JPEG.
4. Compute `|original_rgb − recompressed_rgb|`.
5. Take the per-pixel mean across RGB channels for statistics.
6. Multiply the residual by an amplification scale (default 20) and clip to 0–255 for display.
7. Colour a percentile-stretched magnitude map so a small pasted region stays visible.
8. Measure 16×16 block contrast so a local paste can raise the score even when the image-wide average stays low.
9. Show a plain-language verdict and an optional red box on the original.

For **PNG** uploads, the UI states clearly that analysis is **JPEG-recompression-based**, not a native PNG forensic test.

## ELA contrast score (heuristic)

Displayed as `XX/100` with a **Not flagged / Flagged as suspicious / Flagged as highly suspicious** result. Scores above the low-score cutoff are flagged; the higher band indicates a stronger unusual pattern.

```
score = 100 * (
    w_mean * min(mean / MEAN_CAP, 1) +
    w_std  * min(std  / STD_CAP,  1) +
    w_max  * min(max  / MAX_CAP,  1) +
    w_hot  * min(hot% / HOT_CAP,  1) +
    w_p99  * min(p99 / P99_CAP, 1) +
    w_contrast * min((block_contrast - 1) / CONTRAST_CAP, 1) +
    w_outlier  * min(outlier_block% / OUTLIER_CAP, 1)
)
```

Local contrast is weighted more than the image-wide average. Block contrast uses the 95th-percentile block residual and a minimum denominator, while unusual-block counting applies a minimum difference threshold; this avoids enormous ratios when most JPEG residuals are zero. The score summarizes seven image-comparison measurements. It measures the strength of an ELA pattern, not the probability an image was edited. PNG inputs are labeled **Not assessed** for suspicious editing because their displayed differences come from conversion to JPEG.

Weights, caps, and flagging cut-offs are in `config.py`. They are **demonstration defaults** and **need calibration on a labelled dataset**. Do not describe them as scientifically proven.

The UI explains that this is a heuristic signal, not a calibrated probability or proof of editing.

The number describes the strength of the pattern in this test. It is not the percentage chance that the image was edited; do not read it as “80% probability fake” or “definitely edited”.

## Testing guide

After `python generate_sample_images.py`, open each file in the app and click **Analyse Image**.

| File | Intended exercise | What to look for |
| --- | --- | --- |
| `sample_images/01_unedited.jpg` | Unedited JPEG saved once at quality 90 | Often a relatively even residual field. A **low** score is possible but **does not prove authenticity**. |
| `sample_images/02_spliced.jpg` | Synthetic splice: a never-compressed high-detail patch pasted onto a JPEG that was already saved | The pasted rectangle should stand out on the heatmap and produce a **higher** score than the unedited example. This is **illustrative of this synthetic construction**, not a general detection guarantee. |
| `sample_images/03_recompressed.jpg` | Extra JPEG generations of the unedited scene | Recompression **can** flatten or change ELA texture. Social-media-style processing is a known confounder. |
| `sample_images/04_source.png` | PNG analysed via JPEG recompression | Read the PNG explanation in the UI. Compare with `04_png_converted.jpg`. |

**Do not treat numbers in this README as experimental results.** Run the four cases yourself and record the scores, heatmaps, and metadata for your report. Synthetic images are easier to separate with ELA than many real camera photographs.

Also try:

- A camera JPEG from your phone (unedited export).
- The same photo after an obvious edit in any editor, re-saved as JPEG.
- A file renamed to `.jpg` that is not an image (should be rejected).

Unit tests (no UI):

```powershell
python -m pytest -q
```

## Limitations (also shown in the app)

- ELA is not a reliable standalone detector of all image manipulation.
- Recompression and social-media processing can affect ELA results.
- Copy-move edits may not be detected reliably.
- Different image regions can naturally have different compression characteristics.
- A high ELA score is not proof of manipulation.
- A low ELA score is not proof of authenticity.
- This prototype does not verify the complete origin or history of an image.
- Missing EXIF is not treated as proof of manipulation.
- SHA-256 identifies file bytes; it does not prove authenticity.

## Future work (not in this deadline build)

- Additional integrity layers (for example specialised copy-move, noise inconsistency, or learned detectors) with proper evaluation.
- Provenance recovery (C2PA / content credentials, edit history) when signed assets exist.
- Calibration of the ELA contrast score on a labelled dataset with reported error rates.
- Video and multi-image case workflows.

## Expert-panel one-minute summary

This demo shows a working **ELA recompression residual** pipeline with inspectable statistics, an **explicitly heuristic** 0–100 ELA contrast score, basic EXIF/hash context, and a downloadable report. The number combines seven ELA measurements; it is not a probability of fakery. Authenticity and full provenance remain **future layers**.
