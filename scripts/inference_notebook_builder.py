"""
inference_notebook_builder.py — generates MF_MSKA_Net_Inference.ipynb
Run: python scripts/inference_notebook_builder.py
"""
import nbformat as nbf, json, os

nb = nbf.v4.new_notebook()
cells = []

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells.append(md("""# MF-MSKA-Net — Inference Notebook
## Brain Tumour MRI Segmentation · BRISC 2025

> **Purpose:** Load a trained checkpoint and run segmentation on your own images.
> No training required — just provide a `.keras` checkpoint and an MRI slice.
"""))

cells.append(md("## 1 — Install & Imports"))
cells.append(code("""!pip install -q tensorflow opencv-python matplotlib \
                      scikit-image scipy gradio

import os, sys, json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# If running from the repo root
sys.path.insert(0, ".")
from src.model  import build_mf_mska_net
from src.losses import combined_loss, dice_coef, iou_metric
from src.data   import load_image, IMG_SIZE

print("All imports OK ✅")
"""))

cells.append(md("## 2 — Configuration"))
cells.append(code("""# ── Set your paths here ───────────────────────────────────────────────────
CKPT_PATH   = "./runs/exp1/mf_mska_best.keras"   # trained checkpoint
INPUT_IMAGE = "./sample_mri.png"                   # any grayscale MRI slice
THRESHOLD   = 0.5                                  # binarisation threshold
ENABLE_TTA  = True                                 # 4-flip test-time augmentation
# ──────────────────────────────────────────────────────────────────────────────

assert os.path.exists(CKPT_PATH),   f"Checkpoint not found: {CKPT_PATH}"
assert os.path.exists(INPUT_IMAGE), f"Image not found: {INPUT_IMAGE}"
print(f"Checkpoint : {CKPT_PATH}")
print(f"Input image: {INPUT_IMAGE}")
"""))

cells.append(md("## 3 — Load Model"))
cells.append(code("""model = build_mf_mska_net()
model.compile(
    optimizer    = "adam",
    loss         = [combined_loss, combined_loss],
    loss_weights = [1.0, 0.4],
    metrics      = {"final_mask": [dice_coef, iou_metric]},
)
model.load_weights(CKPT_PATH)
print(f"Parameters: {model.count_params():,}")
"""))

cells.append(md("## 4 — Inference"))
cells.append(code("""def predict(img_path, threshold=0.5, tta=False):
    img  = load_image(img_path)          # (256,256,1) float32
    batch = img[np.newaxis]

    if tta:
        preds = []
        for flip in range(4):
            aug = batch.copy()
            if flip==1: aug=aug[:,:,::-1,:]
            if flip==2: aug=aug[:,::-1,:,:]
            if flip==3: aug=aug[:,::-1,::-1,:]
            out  = model.predict(aug, verbose=0)
            pred = out[0][0,...,0]
            if flip==1: pred=pred[:,::-1]
            if flip==2: pred=pred[::-1,:]
            if flip==3: pred=pred[::-1,::-1]
            preds.append(pred)
        prob = np.mean(preds, axis=0)
    else:
        prob = model.predict(batch, verbose=0)[0][0,...,0]

    mask = (prob > threshold).astype(np.uint8)
    return img[...,0], prob, mask

img, prob, mask = predict(INPUT_IMAGE, threshold=THRESHOLD, tta=ENABLE_TTA)
print(f"Image shape : {img.shape}")
print(f"Prob range  : [{prob.min():.4f}, {prob.max():.4f}]")
print(f"Tumour px   : {mask.sum():,}  ({100*mask.mean():.2f}%)")
"""))

cells.append(md("## 5 — Visualise Results"))
cells.append(code("""fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# 5a. Input
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Input MRI", fontweight="bold")

# 5b. Probability map
im1 = axes[1].imshow(prob, cmap="hot", vmin=0, vmax=1)
axes[1].set_title("Probability Map", fontweight="bold")
fig.colorbar(im1, ax=axes[1], fraction=0.046)

# 5c. Binary mask
axes[2].imshow(mask, cmap="gray")
axes[2].set_title(f"Predicted Mask (τ={THRESHOLD})", fontweight="bold")

# 5d. Colour overlay
rgb = np.stack([img]*3, axis=-1)
ov  = rgb.copy()
tm  = mask > 0
ov[tm,0]=0; ov[tm,1]=0.85; ov[tm,2]=0
axes[3].imshow(np.clip(ov,0,1))
axes[3].set_title("Overlay (green = tumour)", fontweight="bold")

for ax in axes: ax.axis("off")
plt.suptitle("MF-MSKA-Net Segmentation Output", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

cells.append(md("## 6 — Save Outputs"))
cells.append(code("""os.makedirs("predictions", exist_ok=True)
stem = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]

cv2.imwrite(f"predictions/{stem}_mask.png",    mask*255)
cv2.imwrite(f"predictions/{stem}_overlay.png",
            (np.clip(ov,0,1)*255).astype(np.uint8)[:,:,::-1])

prob_u8 = (prob*255).astype(np.uint8)
heat    = cv2.applyColorMap(prob_u8, cv2.COLORMAP_HOT)
cv2.imwrite(f"predictions/{stem}_prob.png", heat)

print("Saved:")
print(f"  predictions/{stem}_mask.png")
print(f"  predictions/{stem}_overlay.png")
print(f"  predictions/{stem}_prob.png")
"""))

cells.append(md("## 7 — Launch Interactive Demo (optional)"))
cells.append(code("""# Uncomment to launch the Gradio demo
# !python scripts/demo.py --ckpt {CKPT_PATH} --share
"""))

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
    "language_info": {"name":"python","version":"3.10.0"}
}

out = "notebooks/MF_MSKA_Net_Inference.ipynb"
os.makedirs("notebooks", exist_ok=True)
with open(out,"w") as f:
    nbf.write(nb, f)
print(f"Notebook saved → {out}")
