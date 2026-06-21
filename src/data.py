"""
Data pipeline for MF-MSKA-Net: loading, preprocessing, and augmentation.

Dataset: BRISC 2025 Brain Tumor MRI Segmentation
Source : https://www.kaggle.com/datasets/briscdataset/brisc2025
Split  : 70 / 10 / 20  (train / val / test), seed = 42
"""

import os
import glob
import json
from typing import Generator, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE    = 256
GLOBAL_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Low-level I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str, size: int = IMG_SIZE) -> np.ndarray:
    """Load a grayscale MRI slice, resize, and normalise to [0, 1].

    Returns:
        np.ndarray of shape (size, size, 1), dtype float32.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, (size, size)).astype("float32") / 255.0
    return np.expand_dims(img, -1)


def load_mask(path: str, size: int = IMG_SIZE) -> np.ndarray:
    """Load a binary tumour mask, resize (nearest-neighbour), and binarise.

    Returns:
        np.ndarray of shape (size, size, 1), dtype float32 in {0, 1}.
    """
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    m = cv2.resize(m, (size, size),
                   interpolation=cv2.INTER_NEAREST).astype("float32") / 255.0
    m = (m > 0.5).astype("float32")
    return np.expand_dims(m, -1)


def has_tumor(mask_path: str) -> int:
    """Return 1 if the mask contains at least one foreground pixel, else 0."""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return int(m.max() > 0) if m is not None else 0


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment(img: np.ndarray, mask: np.ndarray,
            rng: Optional[np.random.RandomState] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random geometric and intensity augmentation to a training pair.

    Operations (each applied independently with p = 0.5):
        · Horizontal flip
        · Vertical flip
        · Rotation in [-20°, 20°]
        · Brightness scaling in [0.8, 1.2]

    Args:
        img : (H, W, 1) float32 image in [0, 1].
        mask: (H, W, 1) float32 binary mask.
        rng : Optional numpy RandomState for reproducibility.

    Returns:
        Augmented (img, mask) pair.
    """
    if rng is None:
        rng = np.random.RandomState()

    H, W = img.shape[:2]

    # Horizontal flip
    if rng.rand() > 0.5:
        img  = img[:, ::-1, :]
        mask = mask[:, ::-1, :]

    # Vertical flip
    if rng.rand() > 0.5:
        img  = img[::-1, :, :]
        mask = mask[::-1, :, :]

    # Rotation
    angle = rng.uniform(-20, 20)
    M     = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
    img   = cv2.warpAffine(img[..., 0],  M, (W, H))[..., np.newaxis]
    mask  = cv2.warpAffine(mask[..., 0], M, (W, H),
                            flags=cv2.INTER_NEAREST)[..., np.newaxis]

    # Brightness jitter
    img = np.clip(img * rng.uniform(0.8, 1.2), 0.0, 1.0)

    return img.astype("float32"), mask.astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset split
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(image_dir: str, mask_dir: str) -> pd.DataFrame:
    """Scan image/mask directories and return a paired DataFrame.

    Args:
        image_dir: Path to directory containing MRI images.
        mask_dir : Path to directory containing corresponding binary masks.

    Returns:
        pd.DataFrame with columns: ``image_path``, ``mask_path``, ``has_tumor``.
    """
    img_paths  = sorted(glob.glob(os.path.join(image_dir, "*.*")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir,  "*.*")))

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if len(img_paths) != len(mask_paths):
        raise ValueError(
            f"Image/mask count mismatch: {len(img_paths)} vs {len(mask_paths)}"
        )

    df = pd.DataFrame({"image_path": img_paths, "mask_path": mask_paths})
    df["has_tumor"] = df["mask_path"].map(has_tumor)
    return df


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.10,
    seed:       int   = GLOBAL_SEED,
    save_path:  Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 3-way split: train / val / test.

    The test fraction is ``1 - train_frac - val_frac``.

    Args:
        df         : Full dataset DataFrame from :func:`build_dataframe`.
        train_frac : Fraction of data for training (default 0.70).
        val_frac   : Fraction of data for validation (default 0.10).
        seed       : Random seed (default 42).
        save_path  : If given, save split indices to JSON for reproducibility.

    Returns:
        (train_df, val_df, test_df)
    """
    test_frac   = 1.0 - train_frac - val_frac
    n_classes   = df["has_tumor"].nunique()
    strat       = df["has_tumor"] if n_classes > 1 else None

    trainval_df, test_df = train_test_split(
        df, test_size=test_frac, random_state=seed, stratify=strat
    )

    # val_frac as proportion of trainval
    val_of_trainval = val_frac / (train_frac + val_frac)
    strat2 = trainval_df["has_tumor"] if n_classes > 1 else None
    train_df, val_df = train_test_split(
        trainval_df, test_size=val_of_trainval,
        random_state=seed, stratify=strat2
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # Persist indices for reproducibility
    if save_path:
        indices = {
            "seed":        seed,
            "train_frac":  train_frac,
            "val_frac":    val_frac,
            "train_index": train_df.index.tolist(),
            "val_index":   val_df.index.tolist(),
            "test_index":  test_df.index.tolist(),
        }
        with open(save_path, "w") as f:
            json.dump(indices, f, indent=2)
        print(f"Split indices saved → {save_path}")

    print(f"Split: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

def dual_output_generator(
    data_df:       pd.DataFrame,
    batch_size:    int,
    shuffle:       bool = True,
    augment_data:  bool = False,
    seed:          Optional[int] = None,
) -> Generator:
    """Infinite generator yielding (X, (Y_main, Y_aux)) batches.

    Both outputs ``Y_main`` and ``Y_aux`` are the same binary mask —
    ``Y_main`` supervises the final segmentation head and
    ``Y_aux`` supervises the SKA coarse mask head.

    Args:
        data_df     : DataFrame with ``image_path`` and ``mask_path`` columns.
        batch_size  : Number of samples per batch.
        shuffle     : Shuffle samples between epochs.
        augment_data: Apply random augmentation (training only).
        seed        : Random seed for the internal RNG.

    Yields:
        Tuple (X, (Y, Y)) where
            X: (B, 256, 256, 1) float32
            Y: (B, 256, 256, 1) float32
    """
    rng   = np.random.RandomState(seed)
    total = len(data_df)
    idx   = 0

    if shuffle:
        order = rng.permutation(total)
        data_df = data_df.iloc[order].reset_index(drop=True)

    while True:
        if idx >= total:
            idx = 0
            if shuffle:
                order   = rng.permutation(total)
                data_df = data_df.iloc[order].reset_index(drop=True)

        imgs, masks = [], []
        while len(imgs) < batch_size and idx < total:
            img  = load_image(data_df.iloc[idx]["image_path"])
            mask = load_mask( data_df.iloc[idx]["mask_path"])
            if augment_data:
                img, mask = augment(img, mask, rng=rng)
            imgs.append(img)
            masks.append(mask)
            idx += 1

        if imgs:
            X = np.array(imgs,  dtype="float32")
            Y = np.array(masks, dtype="float32")
            yield X, (Y, Y)


# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold Cross-Validation helper
# ─────────────────────────────────────────────────────────────────────────────

def get_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = GLOBAL_SEED,
):
    """Yield (train_df, val_df) pairs for stratified k-fold CV.

    Args:
        df      : Full dataset DataFrame.
        n_splits: Number of folds.
        seed    : Random seed.

    Yields:
        (fold_train_df, fold_val_df) DataFrames.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(
            skf.split(df["image_path"], df["has_tumor"])):
        yield (
            fold + 1,
            df.iloc[tr_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
        )
