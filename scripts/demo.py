"""
demo.py — Interactive Gradio demo for MF-MSKA-Net.

Run locally:
    python scripts/demo.py --ckpt ./runs/exp1/mf_mska_best.keras

Share publicly (generates a temporary URL):
    python scripts/demo.py --ckpt ./runs/exp1/mf_mska_best.keras --share
"""

import os
import sys
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from scipy.ndimage import distance_transform_edt, gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model  import build_mf_mska_net
from src.losses import combined_loss, dice_coef, iou_metric
from src.data   import load_image, IMG_SIZE


# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="MF-MSKA-Net Gradio Demo")
    p.add_argument("--ckpt",  required=True,
                   help="Path to trained .keras checkpoint")
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio link")
    p.add_argument("--port",  type=int, default=7860)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def build_model(ckpt_path: str):
    model = build_mf_mska_net()
    model.compile(
        optimizer    = "adam",
        loss         = [combined_loss, combined_loss],
        loss_weights = [1.0, 0.4],
        metrics      = {"final_mask": [dice_coef, iou_metric]},
    )
    model.load_weights(ckpt_path)
    print(f"✅  Model loaded — {model.count_params():,} parameters")
    return model


# ─────────────────────────────────────────────────────────────────────────────
def segment_image(pil_img, threshold: float, enable_tta: bool, model):
    """Core inference function called by Gradio."""
    # ── Pre-process ───────────────────────────────────────────────────────
    gray  = np.array(pil_img.convert("L"))
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    batch   = resized[np.newaxis, ..., np.newaxis]           # (1,256,256,1)

    # ── Inference (optional TTA) ──────────────────────────────────────────
    if enable_tta:
        preds = []
        for flip in range(4):
            aug = batch.copy()
            if flip == 1: aug = aug[:, :, ::-1, :]
            if flip == 2: aug = aug[:, ::-1, :, :]
            if flip == 3: aug = aug[:, ::-1, ::-1, :]
            out  = model.predict(aug, verbose=0)
            pred = out[0][0, ..., 0]
            if flip == 1: pred = pred[:, ::-1]
            if flip == 2: pred = pred[::-1, :]
            if flip == 3: pred = pred[::-1, ::-1]
            preds.append(pred)
        prob_map = np.mean(preds, axis=0)
    else:
        out      = model.predict(batch, verbose=0)
        prob_map = out[0][0, ..., 0]

    coarse_map = model.predict(batch, verbose=0)[1][0, ..., 0]

    # ── SKA refined attention map ─────────────────────────────────────────
    pred_bin  = (prob_map > threshold).astype(np.float32)
    dt        = distance_transform_edt(1.0 - pred_bin)
    ska_attn  = gaussian_filter(
                    np.exp(-dt**2 / (2 * 15.0**2)), sigma=4.0)
    ska_attn  = np.clip(ska_attn, 0, 1)

    # ── FFT frequency spectrum ────────────────────────────────────────────
    fft  = np.fft.fftshift(np.fft.fft2(resized))
    mag  = np.log1p(np.abs(fft)).astype(np.float32)
    mn, mx = mag.min(), mag.max()
    mag  = (mag - mn) / (mx - mn + 1e-6)

    # ── Overlays ──────────────────────────────────────────────────────────
    rgb = np.stack([resized]*3, axis=-1)

    # Green overlay (tumour region)
    green_ov = rgb.copy()
    tm = pred_bin > 0
    green_ov[tm, 0] = 0;   green_ov[tm, 1] = 0.85; green_ov[tm, 2] = 0

    # Error-style overlay (no GT available in demo mode)
    # Skeleton: approximate centreline of prediction
    from skimage.morphology import skeletonize
    skel_bin = skeletonize(pred_bin.astype(bool))
    skel_ov  = rgb.copy()
    skel_ov[tm,       0] = 0;    skel_ov[tm,       1] = 0.75; skel_ov[tm,       2] = 0
    skel_ov[skel_bin, 0] = 0.9;  skel_ov[skel_bin, 1] = 0.1;  skel_ov[skel_bin, 2] = 0.1

    # ── Build 6-panel figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor("#0d1117")

    panels = [
        (resized,     "gray",    "Input MRI"),
        (mag,         "plasma",  "MSFE: Freq. Spectrum"),
        (coarse_map,  "gray",    "SKA Coarse Map"),
        (ska_attn,    "plasma",  "SKA Refined Attention $A^{ska}$"),
        (green_ov,    None,      "Predicted Tumour (green)"),
        (prob_map,    "hot",     "Probability Map"),
    ]
    for ax, (img, cmap, title) in zip(axes.flat, panels):
        if cmap:
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle("MF-MSKA-Net — Brain Tumour MRI Segmentation",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # ── Stats string ──────────────────────────────────────────────────────
    tumour_px   = int(pred_bin.sum())
    coverage    = 100.0 * pred_bin.mean()
    skel_px     = int(skel_bin.sum())
    max_prob    = float(prob_map.max())
    mean_prob   = float(prob_map[tm].mean()) if tm.any() else 0.0

    stats = (
        f"**Model:** MF-MSKA-Net (1.53 M params)  \n"
        f"**Threshold:** {threshold:.2f}  \n"
        f"**TTA:** {'4-flip' if enable_tta else 'disabled'}  \n\n"
        f"**Tumour pixels:** {tumour_px:,}  \n"
        f"**Coverage:** {coverage:.2f} %  \n"
        f"**Skeleton pixels:** {skel_px:,}  \n"
        f"**Max probability:** {max_prob:.4f}  \n"
        f"**Mean probability (tumour):** {mean_prob:.4f}  \n"
    )

    return fig, stats


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args  = get_args()
    model = build_model(args.ckpt)

    # Wrap with fixed model reference
    def _infer(pil_img, threshold, enable_tta):
        if pil_img is None:
            return None, "Please upload an MRI slice."
        return segment_image(pil_img, threshold, enable_tta, model)

    # ── Gradio UI ─────────────────────────────────────────────────────────
    with gr.Blocks(title="MF-MSKA-Net Demo",
                   theme=gr.themes.Base()) as demo:

        gr.Markdown(
            "# 🧠  MF-MSKA-Net — Brain Tumour MRI Segmentation\n"
            "**Paper:** *MF-MSKA-Net: A Multi-Scale Morphological Skeleton "
            "Attention Network with Hybrid Cross-Attention Fusion*  \n"
            "**Author:** Oussama El Othmani  \n"
            "**GitHub:** https://github.com/oussama123-ai/MF_MSKA_Net"
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(
                    type="pil", label="Upload Brain MRI Slice")
                threshold = gr.Slider(
                    0.1, 0.9, value=0.5, step=0.05,
                    label="Binarisation Threshold")
                enable_tta = gr.Checkbox(
                    value=False, label="Enable Test-Time Augmentation (4-flip)")
                run_btn = gr.Button("▶  Segment", variant="primary")

            with gr.Column(scale=2):
                out_plot  = gr.Plot(label="Segmentation Results")
                out_stats = gr.Markdown(label="Statistics")

        run_btn.click(
            fn=_infer,
            inputs=[img_input, threshold, enable_tta],
            outputs=[out_plot, out_stats],
        )

        gr.Markdown(
            "---\n"
            "**Components:** MSFE · SKA · HCAF  \n"
            "**Dataset:** BRISC 2025 (Kaggle)  \n"
            "**Test Dice:** 0.7912  |  **IoU:** 0.6546  |  **HD95:** 8.90 px  \n"
            "**Params:** 1.53 M  |  **FLOPs:** 15.35 GFLOPs"
        )

    demo.launch(
        server_port=args.port,
        share=args.share,
        debug=False,
    )


if __name__ == "__main__":
    main()
