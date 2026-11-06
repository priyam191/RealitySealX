# REALITYSEAL-X — ELA prototype

**Image Integrity and ELA-Based Tampering Analysis — Prototype**

This repository is a **local, CPU-capable** demonstration of ELA for still images, with explicitly unavailable CNN and facial-deepfake ViT status panels and a separate AI-generated-image ViT detector. It does **not** yet provide face-manipulation inference, C2PA provenance, copy-move search, video analysis, or a legally certified forensic workflow.

College demonstration target: **7 October 2026**.

## What is implemented

1. Upload a JPG, JPEG, or PNG in a Streamlit browser UI.
2. Preview the original image (pixels loaded in memory; the original file is not modified).
3. Recompress an RGB copy as JPEG (default quality 90), compute the absolute pixel difference, amplify it, and show an ELA heatmap.
4. Report mean / max / standard deviation of the difference map and the percentage of pixels above a configurable threshold.
5. Show a transparent **ELA contrast score (0–100)** and a presentation-aligned result: **Not flagged**, **Flagged as suspicious**, or **Flagged as highly suspicious**. This is a heuristic, not a calibrated probability.
6. Show basic metadata (format, size, EXIF if present) and a **SHA-256** digest of the uploaded bytes.
7. Download a text or JSON report. Pixel data are not written into the report.
8. View the CNN and ViT tabs, which clearly report that no face/deepfake analysis ran because compatible checkpoints are not configured.
9. Run the separate AI-generated-image detector explicitly. The first run downloads its model checkpoint; it does not download any datasets.

## File structure

```
RealitySealX/
  app.py                      # Streamlit UI
  ela_analyzer.py             # ELA pipeline and heuristic score
  cnn_detector.py             # Explicit unavailable status until a compatible checkpoint is configured
  vit_detector.py             # Explicit unavailable status until a compatible checkpoint is configured
  ai_image_detector.py        # Community Forensics general AI-generated-image inference
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
  tests/test_cnn_detector.py
  tests/test_vit_detector.py
  tests/test_ai_image_detector.py
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

## CNN facial-manipulation layer status

The **CNN Deepfake Detection** tab is currently a status-only module. It reports **Model unavailable** and emits no prediction, confidence, or face-detection result. This is intentional: no compatible checkpoint and matching inference implementation are shipped or configured, so the app must not produce a fabricated result.

Checkpoint research found an Xception model card tagged MIT that says it was trained on StyleGAN-generated faces and depends on a custom model implementation not included with the checkpoint files: [RamadhanZome/deepfake-xception](https://huggingface.co/RamadhanZome/deepfake-xception). Its own stated training domain is too narrow to present as a general facial-manipulation detector. The [FaceForensics++ authors' classification README](https://github.com/ondyari/FaceForensics/blob/master/classification/README.md) links to their Xception models and notes that these models were not fine-tuned for general compressed videos; the cited route does not establish a clear redistribution license. Neither candidate is configured by this project.

Before enabling CNN inference, select a checkpoint with documented architecture, preprocessing, class order, dataset provenance, and license, then evaluate it on a relevant held-out dataset. No model accuracy is claimed here. The current module has no additional dependencies and does not download weights or datasets.

## Vision Transformer deepfake layer status

The **Vision Transformer Detection** tab is currently a status-only module. It reports **Model unavailable** and emits no prediction or confidence until a suitable fine-tuned checkpoint and its preprocessing are verified. No ViT weights or datasets are downloaded by this project.

Checkpoint research identified [ashwin-raikar/vit-deepfake-detector-celebdfv2](https://huggingface.co/ashwin-raikar/vit-deepfake-detector-celebdfv2), whose model card describes a ViT-Base fine-tuned on Celeb-DF v2 and labels the checkpoint Apache-2.0. The card does not provide evaluation metrics or preprocessing details, and the repository has no image-processor configuration; the dataset's reuse terms also require separate verification. It is therefore not configured as a general-purpose detector. The [CommunityForensics ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) is trained for detecting AI-generated images, which is a separate task and must not be presented as facial-manipulation detection. No accuracy or generalization claim is made for either model.

## AI-generated-image detector

The **AI-Generated Image Detection** tab uses the ViT-Small model from [OwensLab/commfor-model-384](https://huggingface.co/OwensLab/commfor-model-384), based on the [Community Forensics CVPR 2025 work](https://arxiv.org/abs/2411.04125). The model card identifies the checkpoint as MIT-licensed; the model has about 21.8M parameters. The implementation uses the authors' model architecture and evaluation preprocessing: resize the shorter edge to 440 pixels, center-crop to 384×384, convert to RGB, then ImageNet-normalize. The checkpoint is downloaded from the pinned model revision only when the user explicitly runs this tab; it is approximately 87 MB and cached by Hugging Face Hub. Inference uses CPU.

The displayed AI-generated score is a sigmoid model output, not a calibrated probability or proof of origin. The predicted-class score is the model score for the selected class and is also not calibrated. The model is a research classifier and may not generalize to unseen generators, edits, or heavy image processing. Its center crop can omit image edges, and low-resolution images are upscaled. Explainability is not implemented for this model yet.

To avoid the first-use download in an offline environment, set `REALITYSEAL_AI_MODEL_PATH` to a local compatible `model.safetensors` checkpoint before starting Streamlit.

### Dataset and training notes

Community Forensics contains about 2.7M generated images from 4,803 generator models; its full dataset is about 1.1 TB. The dataset card also offers a smaller ~556K paired-image dataset (~278K generated and ~278K real), but it is licensed CC-BY-NC-SA-4.0 for non-commercial research. The complete dataset combines multiple source datasets with their own terms; review those terms before any training or redistribution. The recommended comprehensive evaluation set is also large. **This project downloads none of these datasets and performs no training.**

CIFAKE contains 120K images (60K real CIFAR-10 images and 60K Stable Diffusion 1.4 synthetic counterparts), useful for introductory experiments, but its 32×32 CIFAR-style images do not establish performance on contemporary, high-resolution image generators. Kaggle marks the license as “Other (specified in description)”; the dataset description says it is published under the same MIT license as CIFAR-10. Verify the current host record and source-image terms before use. CIFAKE alone is insufficient evidence of modern detector performance.

No training is included or performed. The official training repository supports distributed `torchrun` workflows; practical training requires GPUs and substantial storage/compute. The full dataset alone is about 1.1 TB, while the smaller research-only set may require up to ~600 GB including processed cache. No training budget or accuracy is claimed.

The new inference dependencies are PyTorch, torchvision, timm, safetensors, and Hugging Face Hub. They are declared in `requirements.txt`; the base ELA path does not load AI model weights unless this tab is explicitly run.

## Testing guide

After `python generate_sample_images.py`, open each file in the app and click **Analyse Image**.

| File                                | Intended exercise                                                                                | What to look for                                                                                                                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample_images/01_unedited.jpg`     | Unedited JPEG saved once at quality 90                                                           | Often a relatively even residual field. A **low** score is possible but **does not prove authenticity**.                                                                                                   |
| `sample_images/02_spliced.jpg`      | Synthetic splice: a never-compressed high-detail patch pasted onto a JPEG that was already saved | The pasted rectangle should stand out on the heatmap and produce a **higher** score than the unedited example. This is **illustrative of this synthetic construction**, not a general detection guarantee. |
| `sample_images/03_recompressed.jpg` | Extra JPEG generations of the unedited scene                                                     | Recompression **can** flatten or change ELA texture. Social-media-style processing is a known confounder.                                                                                                  |
| `sample_images/04_source.png`       | PNG analysed via JPEG recompression                                                              | Read the PNG explanation in the UI. Compare with `04_png_converted.jpg`.                                                                                                                                   |

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
Rajarshi Samaddar
