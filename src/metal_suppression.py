"""
metal_suppression.py — Instrument / metal suppression for intraoperative X-rays.

Intraoperative C-arm images contain bright surgical instruments (pedicle screws,
rods, retractors) whose gradients dominate the GO similarity metric and prevent
the optimizer from finding the correct bone-only registration pose.

Pipeline:
  1. Adaptive threshold  — instruments are always in the top ~3% of pixel
     intensities AND above a hard floor of 0.82.  We take whichever is lower
     so we don't over-mask low-contrast images.
  2. Morphological dilation — grow the mask by ~10px to cover partial-volume
     edge pixels that are only partially metal.
  3. Telea inpainting — fill the masked region with a smooth estimate of the
     surrounding bone/soft-tissue texture.  The inpainted image is NOT used for
     EPnP (we keep raw pixel coordinates for 2D landmark clicks); it IS used as
     the registration target for the GOS similarity metric.

Usage (in loaders or run scripts):
    from metal_suppression import suppress_metal
    clean, mask = suppress_metal(proj.image_raw)
    # use clean as registration target; proj.image_raw unchanged for EPnP
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def suppress_metal(
        xray: np.ndarray,
        hard_floor: float   = 0.82,
        percentile: float   = 97.0,
        dilate_px:  int     = 10,
        inpaint_r:  int     = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Suppress bright metal implants/instruments in an intraoperative X-ray.

    Parameters
    ----------
    xray        : float32 H×W image in [0, 1], bone-bright convention.
    hard_floor  : minimum threshold — never mask pixels below this value.
    percentile  : adaptive threshold level (97 → top 3% of image brightness).
    dilate_px   : dilation radius in pixels for the instrument mask.
    inpaint_r   : inpainting radius passed to cv2.inpaint (Telea method).

    Returns
    -------
    clean : float32 H×W — inpainted image with instrument regions filled.
    mask  : uint8  H×W — 255 where instruments were detected, 0 elsewhere.
    """
    assert xray.ndim == 2, "Expected 2-D grayscale image"
    assert xray.dtype == np.float32

    # ── Step 1: adaptive threshold ─────────────────────────────────────────
    # Instruments are the abnormally bright pixels — we take the HIGHER of the
    # image's p97 (top 3% brightest) and the hard_floor (0.82).  This ensures:
    #   • We never mask pixels below 0.82 (that would include bright bone)
    #   • On high-contrast images with many instruments (Swaroopa), the p97
    #     auto-adjusts upward to target only the very brightest metal.
    p_thresh = float(np.percentile(xray, percentile))
    thresh   = max(p_thresh, hard_floor)

    mask = (xray > thresh).astype(np.uint8) * 255

    # Early-exit: if < 0.1% of pixels are metal, nothing to suppress
    if mask.mean() < 0.1 * 255:
        return xray.copy(), mask

    # ── Step 2: morphological dilation ────────────────────────────────────
    ksize  = dilate_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask_d = cv2.dilate(mask, kernel)

    # ── Step 3: Telea inpainting ───────────────────────────────────────────
    img8  = (xray * 255.0).clip(0, 255).astype(np.uint8)
    clean8 = cv2.inpaint(img8, mask_d, inpaintRadius=inpaint_r,
                         flags=cv2.INPAINT_TELEA)
    clean  = clean8.astype(np.float32) / 255.0

    return clean, mask_d


# ---------------------------------------------------------------------------
# DRR-guided metal suppression
# ---------------------------------------------------------------------------

def suppress_metal_drr(
        xray:       np.ndarray,
        drr:        np.ndarray,
        ratio_thresh: float = 2.0,
        abs_thresh:   float = 0.60,
        dilate_px:    int   = 12,
        inpaint_r:    int   = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DRR-guided instrument suppression.

    Metal implants appear MUCH brighter in the X-ray than the CT-derived DRR
    predicts (no metal HU in the CT, or HU clamped).  We detect them as pixels
    where the X-ray intensity is both:
      (a) above abs_thresh (absolute floor — avoids over-masking dark regions)
      (b) at least ratio_thresh × the DRR value at the same pixel

    Both images must be float32 [0,1], bone-bright, same spatial size.
    The DRR is typically rendered at the EPnP pose (best available pose) then
    resized to match xray.

    Returns
    -------
    clean : float32 H×W — inpainted image with metal regions filled.
    mask  : uint8  H×W — 255 where metal was detected.
    """
    assert xray.shape == drr.shape, "xray and drr must be the same size"
    assert xray.ndim == 2
    assert xray.dtype == np.float32
    assert drr.dtype  == np.float32

    # Avoid division by zero in dark regions — add small epsilon
    drr_safe = drr + 1e-3

    # Pixels where xray is much brighter than DRR → metal
    ratio_mask = (xray / drr_safe) > ratio_thresh
    abs_mask   = xray > abs_thresh
    metal_mask = (ratio_mask & abs_mask).astype(np.uint8) * 255

    # If nothing detected, fall back to blind suppression
    if metal_mask.mean() < 0.02 * 255:
        return suppress_metal(xray)

    # Dilate to catch halos
    ksize  = dilate_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask_d = cv2.dilate(metal_mask, kernel)

    # Inpaint
    img8   = (xray * 255.0).clip(0, 255).astype(np.uint8)
    clean8 = cv2.inpaint(img8, mask_d, inpaintRadius=inpaint_r,
                         flags=cv2.INPAINT_TELEA)
    clean  = clean8.astype(np.float32) / 255.0

    return clean, mask_d


# ---------------------------------------------------------------------------
# Convenience: suppress + resize to target size
# ---------------------------------------------------------------------------

def suppress_and_resize(
        xray: np.ndarray,
        size: int,
        **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    suppress_metal() then resize to (size × size).

    Returns
    -------
    clean_small : float32 size×size
    mask_small  : uint8  size×size
    """
    clean, mask = suppress_metal(xray, **kwargs)
    clean_s = cv2.resize(clean, (size, size), interpolation=cv2.INTER_AREA)
    mask_s  = cv2.resize(mask,  (size, size), interpolation=cv2.INTER_NEAREST)
    return clean_s, mask_s


# ---------------------------------------------------------------------------
# Visualisation helper (for debugging)
# ---------------------------------------------------------------------------

def save_suppression_debug(xray: np.ndarray, out_path: str, **kwargs) -> None:
    """Save a side-by-side comparison: original | mask overlay | inpainted."""
    clean, mask = suppress_metal(xray, **kwargs)

    h, w = xray.shape
    canvas = np.zeros((h, w * 3 + 4), dtype=np.float32)
    canvas[:, :w]           = xray
    canvas[:, w+2:w*2+2]    = xray.copy()
    canvas[:, w*2+4:]       = clean

    # Red overlay on mask region in middle panel
    canvas_bgr = cv2.cvtColor((canvas * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mid_panel  = canvas_bgr[:, w+2:w*2+2]
    mid_panel[mask > 0] = [0, 0, 200]   # red = suppressed
    canvas_bgr[:, w+2:w*2+2] = mid_panel

    cv2.putText(canvas_bgr, "Original",  (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(canvas_bgr, "Mask",      (w+12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(canvas_bgr, "Inpainted", (w*2+14, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imwrite(out_path, canvas_bgr)
    print(f"  Saved debug image: {out_path}")
