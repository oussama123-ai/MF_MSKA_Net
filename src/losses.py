"""
Loss functions and evaluation metrics for MF-MSKA-Net.

Triple hybrid loss:
    L_total = 0.4 * L_Tversky + 0.3 * L_Focal + 0.3 * L_Dice

All functions accept tensors of arbitrary batch size and spatial shape.
"""

import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor,
              smooth: float = 1e-6) -> tf.Tensor:
    """Sørensen-Dice coefficient.

    Args:
        y_true: Ground-truth binary mask.
        y_pred: Predicted probability map (after sigmoid).
        smooth: Laplace smoothing term to avoid division by zero.

    Returns:
        Scalar Dice coefficient in [0, 1]; higher is better.
    """
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    inter  = tf.reduce_sum(y_true * y_pred)
    return (2.0 * inter + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor,
               smooth: float = 1e-6) -> tf.Tensor:
    """Intersection-over-Union (Jaccard index).

    Args:
        y_true: Ground-truth binary mask.
        y_pred: Predicted probability map.
        smooth: Smoothing term.

    Returns:
        Scalar IoU in [0, 1]; higher is better.
    """
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    inter  = tf.reduce_sum(y_true * y_pred)
    union  = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return (inter + smooth) / (union + smooth)


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss components
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
              smooth: float = 1e-6) -> tf.Tensor:
    """Dice loss = 1 - Dice coefficient."""
    return 1.0 - dice_coef(y_true, y_pred, smooth)


def tversky_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
                 alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1e-6) -> tf.Tensor:
    """Tversky loss.

    Parameters:
        alpha: FP penalty weight (default 0.3 — light penalty).
        beta : FN penalty weight (default 0.7 — heavy penalty, clinically motivated).

    Tversky Index:
        TI = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    """
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    return 1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
               gamma: float = 2.0, alpha_t: float = 0.25) -> tf.Tensor:
    """Focal loss for addressing class imbalance.

    Parameters:
        gamma  : Focusing parameter (default 2.0).
        alpha_t: Balancing factor (default 0.25).
    """
    y_true  = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred  = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    bce     = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_w = y_true * alpha_t + (1.0 - y_true) * (1.0 - alpha_t)
    return tf.reduce_mean(alpha_w * bce * tf.pow(1.0 - p_t, gamma))


# ─────────────────────────────────────────────────────────────────────────────
# Triple hybrid loss  (used in model.compile)
# ─────────────────────────────────────────────────────────────────────────────

def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Triple hybrid loss used to train MF-MSKA-Net.

    L_total = 0.4 * L_Tversky + 0.3 * L_Focal + 0.3 * L_Dice

    The Tversky weight (beta = 0.7) penalises false negatives more heavily
    than false positives, reflecting the clinical priority of not missing
    tumour tissue.
    """
    return (
        0.4 * tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7)
        + 0.3 * focal_loss(y_true, y_pred, gamma=2.0, alpha_t=0.25)
        + 0.3 * dice_loss(y_true, y_pred)
    )
