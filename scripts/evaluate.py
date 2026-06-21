"""
evaluate.py — MF-MSKA-Net evaluation script.

Runs the full evaluation suite used in the paper:
    · Independent held-out test set (Dice, IoU, Precision, Recall, F1,
      Specificity, HD95)
    · 5-fold cross-validation (mean ± std)
    · Inference time profiling
    · Optional: ablation study

Usage:
    # Basic test-set evaluation
    python scripts/evaluate.py \
        --data_dir  /path/to/brisc2025/train \
        --ckpt_path ./runs/exp1/mf_mska_best.keras \
        --output_dir ./runs/exp1

    # Full evaluation including 5-fold CV
    python scripts/evaluate.py \
        --data_dir  /path/to/brisc2025/train \
        --ckpt_path ./runs/exp1/mf_mska_best.keras \
        --output_dir ./runs/exp1 \
        --run_cv

    # Full evaluation + ablation (slow — trains 4 variants)
    python scripts/evaluate.py \
        --data_dir  /path/to/brisc2025/train \
        --ckpt_path ./runs/exp1/mf_mska_best.keras \
        --output_dir ./runs/exp1 \
        --run_cv --run_ablation
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model  import build_mf_mska_net
from src.losses import combined_loss, dice_coef, iou_metric
from src.data   import (build_dataframe, split_dataset,
                        dual_output_generator, get_cv_folds,
                        load_image, load_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Evaluate MF-MSKA-Net")
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--ckpt_path",   required=True)
    p.add_argument("--output_dir",  default="./eval_results")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--train_frac",  type=float, default=0.70)
    p.add_argument("--val_frac",    type=float, default=0.10)
    p.add_argument("--n_folds",     type=int,   default=5)
    p.add_argument("--run_cv",      action="store_true",
                   help="Run 5-fold cross-validation")
    p.add_argument("--run_ablation",action="store_true",
                   help="Run ablation study (trains 4 variants)")
    p.add_argument("--ablation_epochs", type=int, default=50,
                   help="Epochs for each ablation variant")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def hausdorff95(pred_bin: np.ndarray, true_bin: np.ndarray) -> float:
    """95th-percentile Hausdorff distance between two binary maps."""
    if pred_bin.sum() == 0 or true_bin.sum() == 0:
        return float("nan")
    d1 = distance_transform_edt(~pred_bin.astype(bool))
    d2 = distance_transform_edt(~true_bin.astype(bool))
    return float(np.percentile(
        np.concatenate([d1[true_bin.astype(bool)],
                        d2[pred_bin.astype(bool)]]), 95))


def compute_metrics(y_true_flat: np.ndarray,
                    y_pred_flat: np.ndarray,
                    hd95_scores: list) -> dict:
    """Compute the full metric suite from flattened arrays."""
    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))

    eps       = 1e-9
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * TP / (2 * TP + FP + FN + eps)
    iou       = TP / (TP + FP + FN + eps)
    spec      = TN / (TN + FP + eps)
    hd95_mean = float(np.nanmean(hd95_scores)) if hd95_scores else float("nan")

    return {
        "Dice":        round(float(f1),        4),
        "IoU":         round(float(iou),       4),
        "Precision":   round(float(precision), 4),
        "Recall":      round(float(recall),    4),
        "F1":          round(float(f1),        4),
        "Specificity": round(float(spec),      4),
        "HD95":        round(hd95_mean,        2),
        "TP": int(TP), "FP": int(FP),
        "FN": int(FN), "TN": int(TN),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test-time augmentation (TTA)
# ─────────────────────────────────────────────────────────────────────────────

def predict_tta(model, img_batch: np.ndarray, n_aug: int = 4) -> np.ndarray:
    """Average predictions over 4 flips (identity + H-flip + V-flip + HV-flip)."""
    preds = []
    for flip in range(n_aug):
        aug = img_batch.copy()
        if flip == 1: aug = aug[:, :, ::-1, :]   # H-flip
        if flip == 2: aug = aug[:, ::-1, :, :]   # V-flip
        if flip == 3: aug = aug[:, ::-1, ::-1, :]# HV-flip
        out  = model.predict(aug, verbose=0)
        pred = out[0] if isinstance(out, (list, tuple)) else out
        if flip == 1: pred = pred[:, :, ::-1, :]
        if flip == 2: pred = pred[:, ::-1, :, :]
        if flip == 3: pred = pred[:, ::-1, ::-1, :]
        preds.append(pred)
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate on a split
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split(model, data_df, batch_size: int = 8,
                   desc: str = "Test") -> dict:
    """Run the full metric suite on a dataset split.

    Uses test-time augmentation and computes per-image HD95.
    """
    steps = max(1, len(data_df) // batch_size)
    gen   = dual_output_generator(data_df, batch_size, shuffle=False)

    all_true, all_pred, hd_scores = [], [], []

    for _ in range(steps):
        X_b, Y_b = next(gen)
        Y_b  = Y_b[0]
        pred = predict_tta(model, X_b)
        bp   = (pred > 0.5).astype(np.uint8)
        bt   = (Y_b   > 0.5).astype(np.uint8)
        all_true.append(bt.flatten())
        all_pred.append(bp.flatten())
        for b in range(len(bt)):
            hd_scores.append(hausdorff95(bp[b, ..., 0], bt[b, ..., 0]))

    y_t = np.concatenate(all_true)
    y_p = np.concatenate(all_pred)
    m   = compute_metrics(y_t, y_p, hd_scores)

    print(f"\n=== {desc} ===")
    for k, v in m.items():
        if k not in ("TP", "FP", "FN", "TN"):
            print(f"  {k:12s}: {v}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Inference time profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_inference(model, img_size: int = 256,
                      n_warmup: int = 10, n_measure: int = 50) -> dict:
    """Measure per-sample GPU inference time."""
    dummy = np.random.rand(1, img_size, img_size, 1).astype("float32")
    for _ in range(n_warmup):
        model.predict(dummy, verbose=0)
    t0 = time.perf_counter()
    for _ in range(n_measure):
        model.predict(dummy, verbose=0)
    t1 = time.perf_counter()
    ms = 1000.0 * (t1 - t0) / n_measure

    # FLOPs (optional — requires full TF installation)
    flops = None
    try:
        from tensorflow.python.framework.convert_to_constants import \
            convert_variables_to_constants_v2
        concrete = tf.function(model).get_concrete_function(
            tf.TensorSpec([1, img_size, img_size, 1], tf.float32))
        frozen   = convert_variables_to_constants_v2(concrete)
        run_meta = tf.compat.v1.RunMetadata()
        opts     = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        prof     = tf.compat.v1.profiler.profile(
                       frozen.graph, run_meta=run_meta, cmd="op", options=opts)
        flops    = round(prof.total_float_ops / 1e9, 3)
    except Exception:
        pass

    return {
        "inference_ms_per_sample": round(ms, 2),
        "total_params":            model.count_params(),
        "model_size_MB":           round(model.count_params() * 4 / 1e6, 2),
        "flops_GFLOPs":            flops if flops else "unavailable",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(df, n_folds: int, batch_size: int,
                         seed: int, epochs: int, output_dir: str) -> list:
    """Train and evaluate on n_folds cross-validation splits."""
    cv_results = []

    for fold, fold_train, fold_val in get_cv_folds(df, n_splits=n_folds, seed=seed):
        print(f"\n{'='*60}\n  FOLD {fold}/{n_folds}\n{'='*60}")

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed + fold)

        m = build_mf_mska_net()
        s = max(1, len(fold_train) // batch_size)
        v = max(1, len(fold_val)   // batch_size)

        m.compile(
            optimizer    = tf.keras.optimizers.Adam(1e-4),
            loss         = [combined_loss, combined_loss],
            loss_weights = [1.0, 0.4],
            metrics      = {"final_mask": [dice_coef, iou_metric]},
        )

        tg = dual_output_generator(fold_train, batch_size,
                                   shuffle=True, augment_data=True,
                                   seed=seed + fold)
        vg = dual_output_generator(fold_val, batch_size, shuffle=False)

        m.fit(tg, steps_per_epoch=s, validation_data=vg,
              validation_steps=v, epochs=epochs, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_final_mask_dice_coef",
                  patience=10, mode="max", restore_best_weights=True)])

        fold_metrics = evaluate_split(m, fold_val,
                                      batch_size=batch_size,
                                      desc=f"Fold {fold}")
        fold_metrics["fold"] = fold
        cv_results.append(fold_metrics)

    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# Ablation study
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(train_df, val_df, batch_size: int,
                 seed: int, epochs: int) -> dict:
    """Train and evaluate 4 ablation variants."""
    from src.model import (MSFE, SKAModule, HCAFBlock,
                           PatchExtractor, build_mf_mska_net)

    variants = {
        "Baseline":       dict(use_msfe=False, use_ska=False, use_hcaf=False),
        "+MSFE":          dict(use_msfe=True,  use_ska=False, use_hcaf=False),
        "+MSFE+SKA":      dict(use_msfe=True,  use_ska=True,  use_hcaf=False),
        "Full MF-MSKA-Net": None,   # built with build_mf_mska_net()
    }

    results = {}
    s = max(1, len(train_df) // batch_size)
    v = max(1, len(val_df)   // batch_size)

    for name, cfg in variants.items():
        print(f"\n{'='*60}\n  Ablation: {name}\n{'='*60}")
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)

        if cfg is None:
            model = build_mf_mska_net()
        else:
            model = _build_ablation_variant(**cfg)

        model.compile(
            optimizer    = tf.keras.optimizers.Adam(1e-4),
            loss         = [combined_loss, combined_loss],
            loss_weights = [1.0, 0.4],
            metrics      = {"final_mask": [dice_coef, iou_metric]},
        )
        tg = dual_output_generator(train_df, batch_size,
                                   shuffle=True, augment_data=True, seed=seed)
        vg = dual_output_generator(val_df, batch_size, shuffle=False)
        model.fit(tg, steps_per_epoch=s, validation_data=vg,
                  validation_steps=v, epochs=epochs, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_final_mask_dice_coef",
                      patience=10, mode="max", restore_best_weights=True)])

        m = evaluate_split(model, val_df, batch_size=batch_size, desc=name)
        results[name] = {"Dice": m["Dice"], "IoU": m["IoU"], "HD95": m["HD95"]}
        print(results[name])

    return results


def _build_ablation_variant(use_msfe: bool, use_ska: bool, use_hcaf: bool):
    """Build a partial MF-MSKA-Net with selected modules disabled."""
    from src.model import (MSFE, SKAModule, HCAFBlock,
                           PatchExtractor, build_mf_mska_net)

    IMG = 256; PATCH = 16; EMB = 192; NL = 4; NH = 6; MLP = 384
    DROP = 0.1; FREQ = 64
    grid = IMG // PATCH; n_pat = grid * grid; p_dim = PATCH * PATCH

    img_in = tf.keras.Input((IMG, IMG, 1))

    # MSFE or plain conv
    freq_feat = (MSFE(out_channels=FREQ)(img_in) if use_msfe
                 else tf.keras.layers.Conv2D(FREQ, 3, padding="same",
                                             activation="relu")(img_in))

    def enc(x, f, name):
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        return tf.keras.layers.BatchNormalization()(x)

    s1 = enc(img_in, 32, "e1")
    s2 = enc(tf.keras.layers.MaxPool2D(2)(s1), 32, "e2")
    s3 = enc(tf.keras.layers.MaxPool2D(2)(s2), 64, "e3")
    s3 = tf.keras.layers.Conv2D(64, 1, activation="relu")(
             tf.keras.layers.Concatenate()([s3,
                 tf.keras.layers.MaxPool2D(4)(freq_feat)]))
    s4 = enc(tf.keras.layers.MaxPool2D(2)(s3), 64, "e4")

    patches = PatchExtractor(PATCH)(img_in)
    flat    = tf.keras.layers.Reshape((n_pat, p_dim))(patches)
    proj    = tf.keras.layers.Dense(EMB)(flat)
    pos_emb = tf.keras.layers.Embedding(n_pat, EMB)(tf.keras.ops.arange(0, n_pat))
    x = proj + tf.expand_dims(pos_emb, 0)
    for _ in range(NL):
        n1   = tf.keras.layers.LayerNormalization(1e-6)(x)
        attn = tf.keras.layers.MultiHeadAttention(NH, EMB // NH, dropout=DROP)(n1, n1)
        x    = tf.keras.layers.Add()([x, attn])
        n2   = tf.keras.layers.LayerNormalization(1e-6)(x)
        ff   = tf.keras.layers.Dropout(DROP)(
               tf.keras.layers.Dense(MLP, activation="gelu")(n2))
        ff   = tf.keras.layers.Dense(EMB)(ff)
        x    = tf.keras.layers.Add()([x, ff])

    ska_in = tf.keras.layers.UpSampling2D(8)(s4)
    if use_ska:
        coarse_mask, skel_bias = SKAModule()(ska_in)
    else:
        coarse_mask = tf.keras.layers.Conv2D(1, 1, activation="sigmoid",
                                             dtype="float32")(
                          tf.keras.layers.Conv2D(32, 3, padding="same",
                                                 activation="relu")(ska_in))
        skel_bias = tf.keras.layers.Lambda(
                        lambda t: tf.zeros_like(t[..., :1]))(ska_in)

    gm = tf.keras.layers.Reshape((grid, grid, EMB))(x)
    gm_up4 = tf.keras.layers.Conv2D(64, 1, activation="relu")(
                 tf.keras.layers.UpSampling2D(2)(gm))

    if use_hcaf:
        tok4   = tf.keras.layers.Reshape((32*32, 64))(gm_up4)
        bias4  = tf.keras.layers.AveragePooling2D(8)(skel_bias)
        fused4 = HCAFBlock(64, num_heads=4)(s4, tok4, bias4)
        d4     = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(
                     tf.keras.layers.Add()([gm_up4, fused4]))
    else:
        d4 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(
                 tf.keras.layers.Concatenate()([gm_up4, s4]))

    d4 = tf.keras.layers.UpSampling2D(2)(tf.keras.layers.BatchNormalization()(d4))
    d3 = tf.keras.layers.UpSampling2D(2)(tf.keras.layers.BatchNormalization()(
             tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(
             tf.keras.layers.Concatenate()([d4, s3]))))
    d2 = tf.keras.layers.UpSampling2D(2)(tf.keras.layers.BatchNormalization()(
             tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(
             tf.keras.layers.Concatenate()([d3, s2]))))
    d1 = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(
             tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(
             tf.keras.layers.Concatenate()([d2, s1, freq_feat])))
    d1_fp32    = tf.keras.layers.Lambda(lambda t: tf.cast(t, tf.float32))(d1)
    final_mask = tf.keras.layers.Conv2D(1, 1, activation="sigmoid",
                                        name="final_mask", dtype="float32")(d1_fp32)
    return tf.keras.Model(inputs=img_in, outputs=[final_mask, coarse_mask])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset split ──────────────────────────────────────────────────────
    image_dir = os.path.join(args.data_dir, "images")
    mask_dir  = os.path.join(args.data_dir, "masks")
    df = build_dataframe(image_dir, mask_dir)
    train_df, val_df, test_df = split_dataset(
        df, train_frac=args.train_frac,
        val_frac=args.val_frac, seed=args.seed)

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.ckpt_path}")
    model = build_mf_mska_net()
    model.compile(
        optimizer    = tf.keras.optimizers.Adam(1e-4),
        loss         = [combined_loss, combined_loss],
        loss_weights = [1.0, 0.4],
        metrics      = {"final_mask": [dice_coef, iou_metric]},
    )
    model.load_weights(args.ckpt_path)

    # ── Test set evaluation ────────────────────────────────────────────────
    test_metrics = evaluate_split(model, test_df,
                                  batch_size=args.batch_size,
                                  desc="Independent Test")

    # ── Inference profiling ────────────────────────────────────────────────
    inference_info = profile_inference(model)
    print("\n=== Inference Profile ===")
    for k, v in inference_info.items():
        print(f"  {k}: {v}")

    # ── Optional: 5-fold CV ────────────────────────────────────────────────
    cv_summary = {}
    if args.run_cv:
        cv_results = run_cross_validation(
            df, n_folds=args.n_folds,
            batch_size=args.batch_size, seed=args.seed,
            epochs=50,
            output_dir=args.output_dir)
        import pandas as _pd
        cv_df = _pd.DataFrame(cv_results)
        cv_summary = {
            col: f"{cv_df[col].mean():.4f} ± {cv_df[col].std():.4f}"
            for col in ["Dice", "IoU", "HD95"]
            if col in cv_df.columns
        }
        print("\n=== 5-Fold CV Summary ===")
        for k, v in cv_summary.items():
            print(f"  {k}: {v}")

    # ── Optional: ablation ─────────────────────────────────────────────────
    ablation_results = {}
    if args.run_ablation:
        ablation_results = run_ablation(
            train_df, val_df,
            batch_size=args.batch_size,
            seed=args.seed,
            epochs=args.ablation_epochs)

    # ── Save all results ───────────────────────────────────────────────────
    output = {
        "test_metrics":      test_metrics,
        "inference_profile": inference_info,
        "cv_summary":        cv_summary,
        "ablation_results":  ablation_results,
    }
    out_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nAll results saved → {out_path}")


if __name__ == "__main__":
    main()
