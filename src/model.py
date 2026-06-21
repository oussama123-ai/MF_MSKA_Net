"""
MF-MSKA-Net: Multi-Scale Morphological Skeleton Attention Network
with Hybrid Cross-Attention Fusion for Brain Tumor MRI Segmentation

Paper: "MF-MSKA-Net: A Multi-Scale Morphological Skeleton Attention Network
        with Hybrid Cross-Attention Fusion for Brain Tumor MRI Segmentation"
Author: Oussama El Othmani
GitHub: https://github.com/oussama123-ai/MF_MSKA_Net
"""

import tensorflow as tf
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MULTI-SCALE FREQUENCY ENCODER  (MSFE)
# ─────────────────────────────────────────────────────────────────────────────

class MSFE(tf.keras.layers.Layer):
    """Multi-Scale Frequency Encoder (MSFE).

    Dual-branch module:
      · Spatial branch  — three parallel dilated convolutions (d = 1, 2, 4)
      · Frequency branch — 2-D FFT log-magnitude projection
    Both branches are concatenated and recalibrated with a Squeeze-Excitation gate.

    Args:
        out_channels (int): Number of output feature channels.
    """

    def __init__(self, out_channels: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.out_ch = out_channels
        half = out_channels // 2
        third = out_channels // 3

        # Spatial branch — dilated convolutions
        self.sp_d1 = tf.keras.layers.Conv2D(third, 3, padding="same",
                                             dilation_rate=1, activation="relu")
        self.sp_d2 = tf.keras.layers.Conv2D(third, 3, padding="same",
                                             dilation_rate=2, activation="relu")
        self.sp_d4 = tf.keras.layers.Conv2D(third, 3, padding="same",
                                             dilation_rate=4, activation="relu")
        self.sp_bn = tf.keras.layers.BatchNormalization()

        # Frequency branch — FFT magnitude
        self.fq_c1 = tf.keras.layers.Conv2D(half, 3, padding="same", activation="relu")
        self.fq_c2 = tf.keras.layers.Conv2D(half, 3, padding="same", activation="relu")
        self.fq_bn = tf.keras.layers.BatchNormalization()

        # Fusion + SE gate
        self.fuse  = tf.keras.layers.Conv2D(out_channels, 1, activation="relu")
        self.se_gap = tf.keras.layers.GlobalAveragePooling2D()
        self.se_fc1 = tf.keras.layers.Dense(max(1, out_channels // 4), activation="relu")
        self.se_fc2 = tf.keras.layers.Dense(out_channels, activation="sigmoid")
        self.out_bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training: bool = False):
        # Spatial branch
        sp = tf.concat([self.sp_d1(x), self.sp_d2(x), self.sp_d4(x)], axis=-1)
        sp = self.sp_bn(sp, training=training)

        # Frequency branch
        x_gray = tf.cast(x[..., 0], tf.complex64)
        fft    = tf.signal.fftshift(tf.signal.fft2d(x_gray))
        mag    = tf.cast(tf.math.log1p(tf.abs(fft)), tf.float32)
        mag    = tf.expand_dims(mag, -1)
        # Normalise to [0, 1]
        mn  = tf.reduce_min(mag, axis=[1, 2, 3], keepdims=True)
        mx  = tf.reduce_max(mag, axis=[1, 2, 3], keepdims=True)
        mag = (mag - mn) / (mx - mn + 1e-6)
        fq  = self.fq_bn(self.fq_c2(self.fq_c1(mag)), training=training)

        # Fusion
        fused = self.fuse(tf.concat([sp, fq], axis=-1))

        # Squeeze-Excitation gate
        se = self.se_fc2(self.se_fc1(self.se_gap(fused)))
        se = tf.reshape(se, (-1, 1, 1, self.out_ch))
        return self.out_bn(fused * se, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"out_channels": self.out_ch})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MORPHOLOGICAL SKELETON ATTENTION  (SKA)
# ─────────────────────────────────────────────────────────────────────────────

class SKAModule(tf.keras.layers.Layer):
    """Morphological Skeleton Attention (SKA).

    Produces:
      · ``coarse_mask``  — sigmoid probability map (used as auxiliary output)
      · ``skel_bias``    — Gaussian proximity map around approximated skeleton
                           (injected into HCAF as structural bias)

    Note:
        The skeletonization step uses an iterative max-pool erosion approximation
        that is fully differentiable and GPU-compatible. This approximates binary
        thinning without the CPU-bound ``skimage.morphology.skeletonize`` call.
        A future release will integrate a learnable soft-skeleton head.
    """

    def __init__(self, gamma: float = 15.0, n_erode: int = 5,
                 n_dilate: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.gamma    = gamma
        self.n_erode  = n_erode
        self.n_dilate = n_dilate

        self.coarse_c1 = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")
        self.coarse_c2 = tf.keras.layers.Conv2D(1,  1, activation="sigmoid", dtype="float32")

    def _soft_skeleton_bias(self, coarse: tf.Tensor) -> tf.Tensor:
        """Differentiable skeleton-proximity map via morphological approximation."""
        mask = tf.cast(coarse > 0.5, tf.float32)

        # Approximate erosion (min-pool via negated max-pool)
        eroded = mask
        for _ in range(self.n_erode):
            eroded = -tf.nn.max_pool2d(-eroded, ksize=3, strides=1, padding="SAME")

        # Approximate distance-weighted dilation (proximity map)
        bias   = eroded
        dilated = eroded
        for i in range(1, self.n_dilate):
            dilated = tf.nn.max_pool2d(dilated, ksize=3, strides=1, padding="SAME")
            weight  = tf.exp(tf.cast(-i, tf.float32) / self.gamma)
            bias    = tf.maximum(bias, dilated * weight)

        # Fall back to uniform 0.5 bias when no foreground pixels are present
        has_fg  = tf.cast(tf.reduce_sum(mask, axis=[1, 2, 3], keepdims=True) > 10,
                          tf.float32)
        return bias * has_fg + 0.5 * (1.0 - has_fg)

    def call(self, feat_map: tf.Tensor, training: bool = False):
        coarse     = self.coarse_c2(self.coarse_c1(feat_map))
        skel_bias  = self._soft_skeleton_bias(coarse)
        return coarse, skel_bias

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "n_erode": self.n_erode,
                    "n_dilate": self.n_dilate})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HIERARCHICAL CROSS-ATTENTION FUSION  (HCAF)
# ─────────────────────────────────────────────────────────────────────────────

class HCAFBlock(tf.keras.layers.Layer):
    """Hierarchical Cross-Attention Fusion (HCAF) block.

    CNN skip features act as queries (Q); global ViT tokens act as keys (K)
    and values (V). The skeleton proximity map is projected and added to Q
    before multi-head attention, biasing the decoder toward structurally
    meaningful tumor regions.

    Args:
        channels  (int): Feature channel dimension.
        num_heads (int): Number of attention heads.
        dropout   (float): Dropout rate applied inside MHA.
    """

    def __init__(self, channels: int, num_heads: int = 4,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

        self.q_proj    = tf.keras.layers.Dense(channels)
        self.kv_proj   = tf.keras.layers.Dense(channels)
        self.mha       = tf.keras.layers.MultiHeadAttention(
                             num_heads=num_heads,
                             key_dim=max(1, channels // num_heads),
                             dropout=dropout)
        self.norm_q    = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_kv   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_out  = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.bias_proj = tf.keras.layers.Dense(channels)
        self.drop      = tf.keras.layers.Dropout(dropout)

    def call(self, skip_feat: tf.Tensor, trans_tokens: tf.Tensor,
             skeleton_bias: tf.Tensor, training: bool = False):
        H = tf.shape(skip_feat)[1]
        W = tf.shape(skip_feat)[2]
        HW = H * W

        # Flatten spatial dimensions → sequence
        q_seq = tf.reshape(skip_feat, (-1, HW, tf.shape(skip_feat)[-1]))
        q_seq = self.norm_q(q_seq)
        Q     = self.q_proj(q_seq)

        # Inject skeleton proximity bias into Q
        bias_seq = tf.reshape(tf.cast(skeleton_bias, Q.dtype), (-1, HW, 1))
        Q = Q + self.bias_proj(bias_seq)

        # Keys / Values from ViT tokens
        KV  = self.kv_proj(self.norm_kv(trans_tokens))
        out = self.mha(query=Q, key=KV, value=KV, training=training)
        out = self.norm_out(Q + self.drop(out, training=training))

        # Reshape back to spatial
        H_s = skip_feat.shape[1] or H
        W_s = skip_feat.shape[2] or W
        return tf.reshape(out, (-1, H_s, W_s, self.channels))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PATCH EXTRACTOR  (ViT tokeniser)
# ─────────────────────────────────────────────────────────────────────────────

class PatchExtractor(tf.keras.layers.Layer):
    """Extract non-overlapping patches and flatten each patch to a vector."""

    def __init__(self, patch_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.image.extract_patches(
            x,
            sizes  =[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates  =[1, 1, 1, 1],
            padding="VALID",
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FULL MF-MSKA-NET
# ─────────────────────────────────────────────────────────────────────────────

def build_mf_mska_net(
    image_size:  int   = 256,
    patch_size:  int   = 16,
    emb_dim:     int   = 192,
    num_layers:  int   = 4,
    num_heads:   int   = 6,
    mlp_dim:     int   = 384,
    drop:        float = 0.1,
    freq_dim:    int   = 64,
) -> tf.keras.Model:
    """Build the complete MF-MSKA-Net model.

    Architecture:
        Input (256×256×1)
        ├─ CNN Encoder (3 levels, 32/32/64 channels)
        ├─ MSFE branch (spatial + frequency)
        ├─ ViT encoder (patch embedding + L transformer layers)
        ├─ SKA module (coarse mask + skeleton proximity map)
        └─ HCAF decoder (2 cross-attention levels + 2 plain skip levels)
           └─ 1×1 Conv + Sigmoid → final tumour mask

    Returns:
        tf.keras.Model with two outputs:
            [0] ``final_mask``   — (B, H, W, 1) final tumour probability map
            [1] ``coarse_mask``  — (B, H, W, 1) auxiliary SKA coarse map
    """

    grid  = image_size // patch_size   # = 16 for 256×256, patch 16
    n_pat = grid * grid                # = 256 tokens
    p_dim = patch_size * patch_size    # = 256 values per patch (grayscale)

    # ── Input ────────────────────────────────────────────────────────────────
    img_in = tf.keras.Input((image_size, image_size, 1), name="image_input")

    # ── MSFE branch ──────────────────────────────────────────────────────────
    freq_feat = MSFE(out_channels=freq_dim, name="msfe")(img_in)

    # ── CNN Encoder ──────────────────────────────────────────────────────────
    def _enc_block(x, filters, name):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same",
                                   activation="relu", name=f"{name}_c1")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
        return x

    s1 = _enc_block(img_in,              32, "enc1")          # 256×256×32
    s2 = _enc_block(
             tf.keras.layers.MaxPool2D(2)(s1), 32, "enc2")   # 128×128×32
    s3 = _enc_block(
             tf.keras.layers.MaxPool2D(2)(s2), 64, "enc3")   # 64×64×64

    # Fuse MSFE features at encoder level 3
    freq_d3 = tf.keras.layers.MaxPool2D(4, name="freq_pool3")(freq_feat)  # 64×64×freq_dim
    s3 = tf.keras.layers.Conv2D(64, 1, activation="relu", name="s3_fuse")(
             tf.keras.layers.Concatenate()([s3, freq_d3]))

    s4 = _enc_block(
             tf.keras.layers.MaxPool2D(2)(s3), 64, "enc4")   # 32×32×64

    # ── ViT Encoder ──────────────────────────────────────────────────────────
    patches  = PatchExtractor(patch_size, name="patch_extractor")(img_in)
    flat     = tf.keras.layers.Reshape((n_pat, p_dim), name="flat_patches")(patches)
    proj     = tf.keras.layers.Dense(emb_dim, name="img_projection")(flat)

    pos_ids  = tf.keras.ops.arange(0, n_pat)
    pos_emb  = tf.keras.layers.Embedding(n_pat, emb_dim, name="pos_embedding")(pos_ids)
    vit_x    = proj + tf.expand_dims(pos_emb, 0)

    for i in range(num_layers):
        prefix = f"vit_layer{i}"
        n1     = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln1")(vit_x)
        attn   = tf.keras.layers.MultiHeadAttention(
                     num_heads=num_heads,
                     key_dim=emb_dim // num_heads,
                     dropout=drop,
                     name=f"{prefix}_mha")(n1, n1)
        vit_x  = tf.keras.layers.Add(name=f"{prefix}_add1")([vit_x, attn])
        n2     = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln2")(vit_x)
        ff     = tf.keras.layers.Dense(mlp_dim, activation="gelu",
                                       name=f"{prefix}_ff1")(n2)
        ff     = tf.keras.layers.Dropout(drop, name=f"{prefix}_drop")(ff)
        ff     = tf.keras.layers.Dense(emb_dim, name=f"{prefix}_ff2")(ff)
        vit_x  = tf.keras.layers.Add(name=f"{prefix}_add2")([vit_x, ff])

    # ── SKA module ───────────────────────────────────────────────────────────
    # Applied at full resolution after upsampling s4
    ska_input  = tf.keras.layers.UpSampling2D(8, name="ska_upsample")(s4)   # 256×256
    coarse_mask, skel_bias = SKAModule(name="ska")(ska_input)

    # ── HCAF Decoder ─────────────────────────────────────────────────────────
    # Reshape ViT tokens to spatial grid
    gm = tf.keras.layers.Reshape((grid, grid, emb_dim), name="vit_spatial")(vit_x)

    # Level 4 (32×32): cross-attend s4 to ViT tokens
    gm_up4   = tf.keras.layers.Conv2D(64, 1, activation="relu", name="gm_proj4")(
                    tf.keras.layers.UpSampling2D(2, name="gm_up4")(gm))        # 32×32×64
    tok4     = tf.keras.layers.Reshape((32 * 32, 64), name="tok4")(gm_up4)
    bias4    = tf.keras.layers.AveragePooling2D(8, name="bias_pool4")(skel_bias)
    fused4   = HCAFBlock(64, num_heads=4, name="hcaf4")(s4, tok4, bias4)
    d4       = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu",
                                      name="dec4_conv")(
                   tf.keras.layers.Add(name="dec4_add")([gm_up4, fused4]))
    d4       = tf.keras.layers.BatchNormalization(name="dec4_bn")(d4)
    d4_up    = tf.keras.layers.UpSampling2D(2, name="dec4_up")(d4)             # 64×64

    # Level 3 (64×64): cross-attend s3 to upsampled ViT tokens
    gm_up8   = tf.keras.layers.Conv2D(32, 1, activation="relu", name="gm_proj3")(
                    tf.keras.layers.UpSampling2D(4, name="gm_up3")(gm))        # 64×64×32
    tok3     = tf.keras.layers.Reshape((64 * 64, 32), name="tok3")(gm_up8)
    bias3    = tf.keras.layers.AveragePooling2D(4, name="bias_pool3")(skel_bias)
    fused3   = HCAFBlock(32, num_heads=4, name="hcaf3")(s3, tok3, bias3)
    d3_cat   = tf.keras.layers.Concatenate(name="dec3_cat")([d4_up, fused3])
    d3       = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu",
                                      name="dec3_conv")(d3_cat)
    d3       = tf.keras.layers.BatchNormalization(name="dec3_bn")(d3)
    d3_up    = tf.keras.layers.UpSampling2D(2, name="dec3_up")(d3)             # 128×128

    # Level 2 (128×128): plain skip from s2
    d2_cat   = tf.keras.layers.Concatenate(name="dec2_cat")([d3_up, s2])
    d2       = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu",
                                      name="dec2_conv")(d2_cat)
    d2       = tf.keras.layers.BatchNormalization(name="dec2_bn")(d2)
    d2_up    = tf.keras.layers.UpSampling2D(2, name="dec2_up")(d2)             # 256×256

    # Level 1 (256×256): fuse with s1 + MSFE features
    d1_cat   = tf.keras.layers.Concatenate(name="dec1_cat")([d2_up, s1, freq_feat])
    d1       = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu",
                                      name="dec1_conv1")(d1_cat)
    d1       = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu",
                                      name="dec1_conv2")(d1)

    # ── Segmentation head ─────────────────────────────────────────────────────
    d1_fp32     = tf.keras.layers.Lambda(
                      lambda t: tf.cast(t, tf.float32),
                      name="cast_fp32")(d1)
    final_mask  = tf.keras.layers.Conv2D(
                      1, 1, activation="sigmoid",
                      name="final_mask", dtype="float32")(d1_fp32)

    return tf.keras.Model(
        inputs  = img_in,
        outputs = [final_mask, coarse_mask],
        name    = "MF_MSKA_Net",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_mf_mska_net()
    model.summary(line_length=100)
    print(f"\nTotal parameters: {model.count_params():,}")
    dummy = np.random.rand(2, 256, 256, 1).astype("float32")
    out   = model.predict(dummy, verbose=0)
    print(f"final_mask  shape: {out[0].shape}")   # (2, 256, 256, 1)
    print(f"coarse_mask shape: {out[1].shape}")   # (2, 256, 256, 1)
