"""
train.py — MF-MSKA-Net training script.

Usage:
    python scripts/train.py --data_dir /path/to/brisc2025/train \
                            --output_dir ./runs/exp1

The script:
    1. Sets deterministic seeds across Python / NumPy / TensorFlow
    2. Builds the 70 / 10 / 20 stratified split (seed = 42)
    3. Compiles and trains MF-MSKA-Net with the triple hybrid loss
    4. Saves the best checkpoint, training history, and final metrics JSON
"""

import os
import sys
import json
import time
import argparse
import platform

# Deterministic ops must be set BEFORE importing TensorFlow
os.environ["PYTHONHASHSEED"]       = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import tensorflow as tf

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model  import build_mf_mska_net
from src.losses import combined_loss, dice_coef, iou_metric
from src.data   import build_dataframe, split_dataset, dual_output_generator


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Train MF-MSKA-Net for brain tumour MRI segmentation")

    # Paths
    p.add_argument("--data_dir",    required=True,
                   help="Root directory that contains 'images/' and 'masks/' sub-folders")
    p.add_argument("--output_dir",  default="./runs/default",
                   help="Directory to save checkpoints, logs, and metrics")

    # Dataset
    p.add_argument("--train_frac",  type=float, default=0.70)
    p.add_argument("--val_frac",    type=float, default=0.10)
    p.add_argument("--seed",        type=int,   default=42)

    # Training hyper-parameters
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=12,
                   help="Early-stopping patience (epochs)")

    # Model hyper-parameters
    p.add_argument("--img_size",    type=int,   default=256)
    p.add_argument("--patch_size",  type=int,   default=16)
    p.add_argument("--emb_dim",     type=int,   default=192)
    p.add_argument("--num_layers",  type=int,   default=4)
    p.add_argument("--num_heads",   type=int,   default=6)
    p.add_argument("--mlp_dim",     type=int,   default=384)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--freq_dim",    type=int,   default=64)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule: cosine decay with linear warm-up
# ─────────────────────────────────────────────────────────────────────────────

def build_lr_schedule(
    initial_lr: float,
    total_steps: int,
    warmup_frac: float = 0.05,
):
    warmup_steps = int(warmup_frac * total_steps)
    decay_steps  = total_steps - warmup_steps
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=max(1, decay_steps),
        alpha=1e-6,
        warmup_target=initial_lr,
        warmup_steps=warmup_steps,
    )
    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Reproducibility ───────────────────────────────────────────────────
    set_seeds(args.seed)

    # ── Mixed precision ───────────────────────────────────────────────────
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # ── Hardware info ─────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    hw_info = {
        "python":       platform.python_version(),
        "tensorflow":   tf.__version__,
        "platform":     platform.platform(),
        "gpu_count":    len(gpus),
        "gpu_names":    [g.name for g in gpus],
        "seed":         args.seed,
    }
    print("\n=== Hardware ===")
    print(json.dumps(hw_info, indent=2))

    # ── Output directory ──────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path    = os.path.join(args.output_dir, "mf_mska_best.keras")
    history_path = os.path.join(args.output_dir, "history.json")
    metrics_path = os.path.join(args.output_dir, "final_metrics.json")
    split_path   = os.path.join(args.output_dir, "split_indices.json")

    # ── Dataset ───────────────────────────────────────────────────────────
    image_dir = os.path.join(args.data_dir, "images")
    mask_dir  = os.path.join(args.data_dir, "masks")
    df = build_dataframe(image_dir, mask_dir)
    print(f"\nTotal samples: {len(df)}")

    train_df, val_df, test_df = split_dataset(
        df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        save_path=split_path,
    )

    steps_per_epoch  = max(1, len(train_df) // args.batch_size)
    validation_steps = max(1, len(val_df)   // args.batch_size)
    total_steps      = args.epochs * steps_per_epoch

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_mf_mska_net(
        image_size  = args.img_size,
        patch_size  = args.patch_size,
        emb_dim     = args.emb_dim,
        num_layers  = args.num_layers,
        num_heads   = args.num_heads,
        mlp_dim     = args.mlp_dim,
        drop        = args.dropout,
        freq_dim    = args.freq_dim,
    )
    print(f"\nTotal parameters: {model.count_params():,}")

    # ── Optimiser + compile ───────────────────────────────────────────────
    lr_schedule = build_lr_schedule(args.lr, total_steps)
    model.compile(
        optimizer    = tf.keras.optimizers.Adam(lr_schedule),
        loss         = [combined_loss, combined_loss],
        loss_weights = [1.0, 0.4],
        metrics      = {"final_mask": [dice_coef, iou_metric]},
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_final_mask_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_final_mask_dice_coef",
            patience=args.patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.output_dir, "logs"),
            update_freq="epoch",
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(args.output_dir, "training_log.csv")
        ),
    ]

    # ── Generators ────────────────────────────────────────────────────────
    train_gen = dual_output_generator(
        train_df, args.batch_size,
        shuffle=True, augment_data=True, seed=args.seed)
    val_gen = dual_output_generator(
        val_df, args.batch_size,
        shuffle=False, augment_data=False)

    # ── Training ──────────────────────────────────────────────────────────
    print("\n=== Training ===")
    t0 = time.time()
    history = model.fit(
        train_gen,
        steps_per_epoch  = steps_per_epoch,
        validation_data  = val_gen,
        validation_steps = validation_steps,
        epochs           = args.epochs,
        callbacks        = callbacks,
    )
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed/60:.1f} min")

    # ── Save history ──────────────────────────────────────────────────────
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"History saved → {history_path}")

    # ── Final metrics dict ────────────────────────────────────────────────
    final = {
        "hardware":        hw_info,
        "hyperparameters": vars(args),
        "best_val_dice":   float(max(history.history.get(
                               "val_final_mask_dice_coef", [0]))),
        "best_val_iou":    float(max(history.history.get(
                               "val_final_mask_iou_metric", [0]))),
        "total_params":    model.count_params(),
        "training_time_s": round(elapsed, 1),
        "checkpoint":      ckpt_path,
    }
    with open(metrics_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Metrics saved  → {metrics_path}")

    return model, history, train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = get_args()
    train(args)
