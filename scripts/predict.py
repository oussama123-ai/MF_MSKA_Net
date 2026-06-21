"""
predict.py — Run MF-MSKA-Net inference on a single image or a directory.

Usage:
    # Single image → saves prediction mask and overlay PNG
    python scripts/predict.py \
        --input  /path/to/mri_slice.png \
        --ckpt   ./runs/exp1/mf_mska_best.keras \
        --output ./predictions

    # Directory of images
    python scripts/predict.py \
        --input  /path/to/images/ \
        --ckpt   ./runs/exp1/mf_mska_best.keras \
        --output ./predictions
"""

import os
import sys
import argparse
import glob

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model  import build_mf_mska_net
from src.losses import combined_loss, dice_coef, iou_metric
from src.data   import load_image, IMG_SIZE


def get_args():
    p = argparse.ArgumentParser(description="MF-MSKA-Net inference")
    p.add_argument("--input",  required=True,
                   help="Path to a single image file or a directory of images")
    p.add_argument("--ckpt",   required=True,
                   help="Path to saved .keras checkpoint")
    p.add_argument("--output", default="./predictions",
                   help="Directory to write output masks and overlays")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binarisation threshold (default 0.5)")
    p.add_argument("--tta", action="store_true",
                   help="Enable 4-flip test-time augmentation")
    return p.parse_args()


def predict_single(model, img_path: str, threshold: float = 0.5,
                   tta: bool = False) -> tuple:
    """
    Returns:
        img_orig   : (H, W) uint8 original image
        prob_map   : (H, W) float32 probability map
        pred_mask  : (H, W) uint8 binary mask {0, 255}
        overlay    : (H, W, 3) uint8 colour overlay
    """
    img_arr = load_image(img_path)          # (256, 256, 1) float32
    batch   = img_arr[np.newaxis]           # (1, 256, 256, 1)

    if tta:
        preds = []
        for flip in range(4):
            aug = batch.copy()
            if flip == 1: aug = aug[:, :, ::-1, :]
            if flip == 2: aug = aug[:, ::-1, :, :]
            if flip == 3: aug = aug[:, ::-1, ::-1, :]
            out  = model.predict(aug, verbose=0)
            pred = out[0] if isinstance(out, (list, tuple)) else out
            if flip == 1: pred = pred[:, :, ::-1, :]
            if flip == 2: pred = pred[:, ::-1, :, :]
            if flip == 3: pred = pred[:, ::-1, ::-1, :]
            preds.append(pred)
        prob_map = np.mean(preds, axis=0)[0, ..., 0]
    else:
        out      = model.predict(batch, verbose=0)
        prob_map = (out[0] if isinstance(out, (list, tuple)) else out)[0, ..., 0]

    pred_mask = (prob_map > threshold).astype(np.uint8) * 255

    # Colour overlay: green = tumour, grey = background
    img_gray = (img_arr[..., 0] * 255).astype(np.uint8)
    overlay  = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    tumour   = pred_mask > 0
    overlay[tumour, 0] = 0
    overlay[tumour, 1] = 200
    overlay[tumour, 2] = 0

    return img_gray, prob_map, pred_mask, overlay


def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    model = build_mf_mska_net()
    model.compile(
        optimizer    = "adam",
        loss         = [combined_loss, combined_loss],
        loss_weights = [1.0, 0.4],
        metrics      = {"final_mask": [dice_coef, iou_metric]},
    )
    model.load_weights(args.ckpt)
    print(f"Parameters: {model.count_params():,}")

    # Collect image paths
    if os.path.isdir(args.input):
        img_paths = sorted(
            glob.glob(os.path.join(args.input, "*.*")))
    else:
        img_paths = [args.input]

    print(f"\nProcessing {len(img_paths)} image(s) …")
    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        try:
            img_gray, prob_map, pred_mask, overlay = predict_single(
                model, img_path,
                threshold=args.threshold,
                tta=args.tta)
        except Exception as e:
            print(f"  [SKIP] {img_path}: {e}")
            continue

        # Save outputs
        cv2.imwrite(os.path.join(args.output, f"{stem}_mask.png"),    pred_mask)
        cv2.imwrite(os.path.join(args.output, f"{stem}_overlay.png"), overlay)

        # Save probability map as heat-map PNG
        prob_u8  = (prob_map * 255).astype(np.uint8)
        heat_map = cv2.applyColorMap(prob_u8, cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(args.output, f"{stem}_prob.png"), heat_map)

        coverage = 100.0 * (pred_mask > 0).mean()
        print(f"  {stem}: tumour coverage = {coverage:.2f}%")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
