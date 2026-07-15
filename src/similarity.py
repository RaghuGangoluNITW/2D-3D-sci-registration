"""
similarity.py — Multi-Modal Registration Similarity Metrics
============================================================

Implements several metrics designed for 2D/3D registration comparing
synthetic DRR (digitally reconstructed radiograph) to real X-ray:

1.  GO (Gradient Orientation) — Ketcha 2017
2.  NGF (Normalized Gradient Field) — Haber & Modersitzki 2006
3.  Edge-Bone similarity
4.  NCC — Normalized Cross Correlation
5.  MI — Mutual Information
6.  Local NCC (LNCC) — windowed NCC mean over local patches
7.  Multi-Scale NCC — NCC averaged over image pyramid levels
8.  Gradient NCC — NCC applied to gradient magnitude maps
9.  Normalised Mutual Information (NMI) — (H(X)+H(Y)) / H(X,Y)
10. Entropy of Difference — entropy of (DRR − Xray)
11. Correlation Ratio (CR) — Roche et al. 1998
12. Stochastic Rank Correlation (SRC) — approximate rank correlation
13. Gradient Difference (GD) — Penney et al. 1998
14. Pattern Intensity (PI) — Weese et al. 1997 / Birkfellner 2003
15. Gradient Multi-Scale NCC — NCC of gradient magnitudes at each pyramid level

GO:
    GO = 1/max(N, N_LB) * sum_{i: |∇DRR_i|>t AND |∇p_i|>t}  w_i
    w_i = (2/π) * ln(|θ_i| + 1)
    θ_i = angle difference in [0, π/2]
    N_LB = 0.30 * total_pixel_count

NGF (preferred for DRR↔fluoroscopy):
    n(u,ε) = ∇u / sqrt(|∇u|² + ε²)   — normalized gradient field
    NGF_cost = 1 - mean(〈n_DRR, n_Xray〉²)   — 1 - mean squared cosine
    Range: [0, 1], 0 = perfect alignment, 1 = orthogonal gradients.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from typing import Optional, List


# ---------------------------------------------------------------------------
# GO Metric (paper Eq. 2)
# ---------------------------------------------------------------------------

def gradient_orientation_similarity(
        drr: np.ndarray,
        rad: np.ndarray,
        sigma: float = 2.0,
        n_lb_frac: float = 0.30,
) -> float:
    """
    Compute Gradient Orientation (GO) similarity between DRR and radiograph.

    Args:
        drr   : (H, W) float32 DRR image (normalized 0–1)
        rad   : (H, W) float32 radiograph / topogram (normalized 0–1)
        sigma : Gaussian smoothing width (mm / pixel) for gradient computation
        n_lb_frac: lower bound fraction = 0.30 (paper)

    Returns:
        GO score in [0, 1], higher = better similarity.
        We invert to return a COST (lower = better) for the optimizer:
            cost = 1 - GO
    """
    # Resize rad to match drr if needed
    if drr.shape != rad.shape:
        from skimage.transform import resize
        rad = resize(rad.astype(np.float32), drr.shape,
                     anti_aliasing=True, preserve_range=True)

    # Smooth images before gradient computation
    drr_s = gaussian_filter(drr.astype(np.float32), sigma=sigma)
    rad_s = gaussian_filter(rad.astype(np.float32), sigma=sigma)

    # Compute image gradients (central differences)
    drr_gx = np.gradient(drr_s, axis=1)
    drr_gy = np.gradient(drr_s, axis=0)
    rad_gx = np.gradient(rad_s, axis=1)
    rad_gy = np.gradient(rad_s, axis=0)

    # Gradient magnitudes
    drr_mag = np.sqrt(drr_gx**2 + drr_gy**2)
    rad_mag = np.sqrt(rad_gx**2 + rad_gy**2)

    # Threshold = median gradient magnitude (paper: t = median of nonzero grads)
    t_drr = np.median(drr_mag)
    t_rad = np.median(rad_mag)

    # Mask: both images have gradient above threshold
    mask = (drr_mag > t_drr) & (rad_mag > t_rad)

    N = int(mask.sum())
    N_total = drr.size
    N_LB = int(n_lb_frac * N_total)

    if N == 0:
        return 0.0  # no overlap → GO = 0

    # Gradient directions at masked pixels
    drr_angle = np.arctan2(drr_gy[mask], drr_gx[mask])  # (-π, π)
    rad_angle = np.arctan2(rad_gy[mask], rad_gx[mask])

    # Angular difference (wrapped to [0, π] — direction has 180° ambiguity)
    theta = np.abs(drr_angle - rad_angle)
    # Wrap to [0, π]
    theta = np.where(theta > np.pi, 2*np.pi - theta, theta)
    # Directions are undirected — further wrap to [0, π/2]
    theta = np.where(theta > np.pi/2, np.pi - theta, theta)
    # theta in [0, π/2]: 0 = perfect alignment, π/2 = perpendicular

    # Weights (paper Eq. 2): w_i = (2/π) * ln(|θ_i| + 1)
    # Note: at θ=0 -> w=0 (perfect alignment gives max contribution when COST)
    # Wait — re-reading paper: w_i penalizes large θ.
    # GO = sum(w_i) / max(N, N_LB), w_i = (2/π)*ln(|θ|+1)
    # At θ=0: w=0 → contribution=0 (aligned pixels contribute nothing??)
    # That seems wrong. Let me re-read...
    #
    # Actually the paper defines GO as a SIMILARITY (higher = better).
    # w_i is the weight for EACH ALIGNED PIXEL. With w_i=(2/π)*ln(θ+1),
    # this INCREASES with angular error. So GO would be higher when gradients
    # are MORE misaligned? That can't be right.
    #
    # Most likely the paper defines it as a COST metric (lower = better registration),
    # OR the weight is (1 - (2/π)*ln(θ+1)) normalized.
    #
    # From context: "higher GO = better similarity" + their GO decreases with deformation.
    # The standard Gradient Orientation metric in registration literature is:
    #   GO = mean of cos(θ_i) for pixels where both gradients are strong
    # OR equivalently: count of pixels where |θ_i| < threshold.
    #
    # Given paper uses ln form, most likely interpretation:
    #   w_i = 1 - (2/π)*ln(|θ_i| + 1) if we want w_i∈[0,1] with w_i=1 at θ=0
    # But paper explicitly says w_i = (2/π)*ln(|θ_i|+1) which=0 at θ=0.
    #
    # RESOLUTION: The paper likely defines GO as:
    #   GO_cost = (1/max(N,N_LB)) * sum w_i    (this is what they MINIMIZE)
    # and the "similarity" increases as GO_cost DECREASES.
    # We'll return 1 - GO_cost/max_possible as a similarity in [0,1].

    # Paper formula as written (this is the COST, goes to 0 when perfectly aligned)
    w = (2.0/np.pi) * np.log(np.abs(theta) + 1.0)
    go_cost = w.sum() / max(N, N_LB)

    # The maximum possible cost (all pixels at θ=π/2):
    #   w_max = (2/π)*ln(π/2+1) ≈ 0.5936
    # So go_cost ∈ [0, w_max * N / max(N, N_LB)]
    # For a valid similarity:
    go_similarity = 1.0 - go_cost / ((2.0/np.pi) * np.log(np.pi/2 + 1.0) + 1e-8)
    return float(np.clip(go_similarity, 0, 1))


def go_cost(drr: np.ndarray, rad: np.ndarray, sigma: float = 2.0) -> float:
    """
    GO cost to MINIMIZE (1 - GO_similarity). Lower = better alignment.
    Used by the optimizer.

    Includes a coverage penalty: if the DRR has < 15% nonzero pixels the CT
    has been moved off-detector. Return worst cost (1.0) to prevent the
    optimizer settling on a blank-DRR local minimum.
    """
    # Coverage check — blank DRR means CT drifted off the detector
    coverage = np.count_nonzero(drr) / max(drr.size, 1)
    if coverage < 0.15:
        return 1.0  # worst possible cost
    return 1.0 - gradient_orientation_similarity(drr, rad, sigma=sigma)


# ---------------------------------------------------------------------------
# Normalized Cross Correlation (fallback / additional metric)
# ---------------------------------------------------------------------------

def normalized_cross_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """NCC between two images in [-1, 1]. Higher = more similar."""
    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)
    m1, m2 = img1.mean(), img2.mean()
    s1, s2 = img1.std(), img2.std()
    if s1 < 1e-8 or s2 < 1e-8:
        return 0.0
    ncc = ((img1 - m1) * (img2 - m2)).mean() / (s1 * s2)
    return float(ncc)


# ---------------------------------------------------------------------------
# Mutual Information (additional robust metric)
# ---------------------------------------------------------------------------

def mutual_information(img1: np.ndarray, img2: np.ndarray, bins: int = 32) -> float:
    """Normalized mutual information. Higher = more similar."""
    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)
    i1 = np.clip(img1, 0, 1)
    i2 = np.clip(img2, 0, 1)
    hist2d, _, _ = np.histogram2d(i1.ravel(), i2.ravel(), bins=bins)
    hist2d = hist2d + 1e-10  # avoid log(0)
    pxy = hist2d / hist2d.sum()
    px  = pxy.sum(axis=1)
    py  = pxy.sum(axis=0)
    Hx  = -np.sum(px * np.log(px + 1e-10))
    Hy  = -np.sum(py * np.log(py + 1e-10))
    Hxy = -np.sum(pxy * np.log(pxy + 1e-10))
    if (Hx + Hy) < 1e-10:
        return 0.0
    return float((Hx + Hy - Hxy))


# ---------------------------------------------------------------------------
# Normalized Gradient Field (NGF) — Haber & Modersitzki 2006
# ---------------------------------------------------------------------------

def _normalized_gradient_field(img: np.ndarray, sigma: float, eps: float):
    """
    Compute the Normalized Gradient Field of `img`.
    Returns (gx, gy) normalized so that ||(gx, gy)|| ≈ 1 wherever gradients exist.
    eps prevents division by zero in flat regions.
    """
    s = gaussian_filter(img.astype(np.float32), sigma=sigma)
    gx = np.gradient(s, axis=1)
    gy = np.gradient(s, axis=0)
    mag = np.sqrt(gx**2 + gy**2 + eps**2)
    return gx / mag, gy / mag


def precompute_xray_ngf(
    xray: np.ndarray,
    sigma: float = 1.5,
    eps: float = 0.01,
) -> tuple:
    """Precompute the NGF of the X-ray image once per projection.

    Returns (nx, ny) normalized gradient fields as float32 arrays.
    Pass these to fast_combined_cost to avoid recomputing them every call.
    """
    nx, ny = _normalized_gradient_field(xray, sigma, eps)
    return nx.astype(np.float32), ny.astype(np.float32)


def ngf_cost(
    drr: np.ndarray,
    rad: np.ndarray,
    sigma: float = 1.5,
    eps: float = 0.01,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Normalized Gradient Field (NGF) cost.

    NGF similarity = mean(〈n_drr, n_rad〉²)   ∈ [0, 1]
    where n_u = ∇u / sqrt(|∇u|² + ε²)

    We return the *cost* = 1 − NGF_similarity, so lower = better.

    Args:
        drr  : (H, W) float32 synthetic DRR in [0, 1]
        rad  : (H, W) float32 real X-ray in [0, 1]
        sigma: Gaussian pre-smoothing width (pixels)
        eps  : NGF regularization (controls sensitivity in flat regions)
        mask : optional binary mask (H, W), restrict comparison to bone region

    Returns:
        cost ∈ [0, 1],  0 = perfect gradient alignment
    """
    if drr.shape != rad.shape:
        from skimage.transform import resize
        rad = resize(rad.astype(np.float32), drr.shape,
                     anti_aliasing=True, preserve_range=True)

    # Coverage guard — blank DRR → optimizer false minimum
    coverage = np.count_nonzero(drr) / max(drr.size, 1)
    if coverage < 0.10:
        return 1.0

    nx_drr, ny_drr = _normalized_gradient_field(drr, sigma, eps)
    nx_rad, ny_rad = _normalized_gradient_field(rad, sigma, eps)

    # Squared cosine similarity: 〈n_drr, n_rad〉²
    dot_sq = (nx_drr * nx_rad + ny_drr * ny_rad) ** 2

    if mask is not None:
        m = (mask > 0)
        if m.sum() < 100:
            return 1.0
        score = dot_sq[m].mean()
    else:
        score = dot_sq.mean()

    return float(1.0 - np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Edge / Bone-boundary based metric
# ---------------------------------------------------------------------------

def _compute_edges(img: np.ndarray, sigma: float = 1.5, low: float = 0.05,
                   high: float = 0.20) -> np.ndarray:
    """
    Compute soft edge map using Gaussian derivatives. Returns float in [0, 1].
    Using soft edges (gradient magnitude) rather than hard Canny thresholding
    provides a smooth, differentiable landscape for the optimizer.
    """
    import cv2
    smoothed = gaussian_filter(img.astype(np.float32), sigma=sigma)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    if mag.max() > 1e-8:
        mag /= mag.max()
    return mag.astype(np.float32)


def bone_edge_ncc(
    drr: np.ndarray,
    xray_edge: np.ndarray,
    sigma: float = 1.5,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Edge-based NCC cost.

    Compare the DRR edge map to a *precomputed* X-ray bone boundary edge map.
    The X-ray edge map (`xray_edge`) is fixed per projection — only the DRR
    edge map changes as the pose changes, making this efficient.

    Args:
        drr       : (H, W) float32 synthetic DRR in [0, 1]
        xray_edge : (H, W) float32 precomputed X-ray edge map (fixed, in [0, 1])
        sigma     : edge smoothing width (pixels)
        mask      : optional binary bone-region mask

    Returns:
        cost ∈ [0, 1],  0 = perfect edge alignment (NCC=1)
    """
    coverage = np.count_nonzero(drr) / max(drr.size, 1)
    if coverage < 0.10:
        return 1.0

    drr_edge = _compute_edges(drr, sigma=sigma)

    if mask is not None:
        m = (mask > 0)
        if m.sum() < 100:
            return 1.0
        e1 = drr_edge[m]
        e2 = xray_edge[m]
    else:
        e1 = drr_edge.ravel()
        e2 = xray_edge.ravel()

    s1, s2 = e1.std(), e2.std()
    if s1 < 1e-8 or s2 < 1e-8:
        return 1.0

    ncc = ((e1 - e1.mean()) * (e2 - e2.mean())).mean() / (s1 * s2)
    cost = float(1.0 - np.clip(ncc, -1.0, 1.0)) / 2.0   # map [-1,1] → [0,1]
    return cost


def precompute_xray_edge(xray: np.ndarray, sigma: float = 1.5,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Precompute the X-ray edge map once per projection (before optimization loop).
    Optionally restrict to bone mask region.

    Returns:
        edge_map: (H, W) float32 in [0, 1]
    """
    edge = _compute_edges(xray, sigma=sigma)
    if mask is not None:
        edge = edge * (mask > 0).astype(np.float32)
    return edge


# ---------------------------------------------------------------------------
# Combined multi-modal cost (NGF + edge blend)
# ---------------------------------------------------------------------------

def combined_cost(
    drr: np.ndarray,
    rad: np.ndarray,
    xray_edge: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    ngf_weight: float = 0.7,
    edge_weight: float = 0.3,
    ngf_sigma: float = 1.5,
    ngf_eps: float = 0.01,
    edge_sigma: float = 1.5,
) -> float:
    """
    Blended NGF + edge-NCC cost for multi-modal 2D/3D registration.

    Args:
        drr        : (H, W) synthetic DRR
        rad        : (H, W) real X-ray (used for NGF)
        xray_edge  : (H, W) precomputed X-ray edge map; if None, computed on the fly
        mask       : optional binary bone mask
        ngf_weight : weight for NGF cost (default 0.7)
        edge_weight: weight for edge NCC cost (default 0.3)

    Returns:
        blended cost ∈ [0, 1]
    """
    c_ngf = ngf_cost(drr, rad, sigma=ngf_sigma, eps=ngf_eps, mask=mask)

    if xray_edge is None:
        xray_edge = precompute_xray_edge(rad, sigma=edge_sigma, mask=mask)
    c_edge = bone_edge_ncc(drr, xray_edge, sigma=edge_sigma, mask=mask)

    return float(ngf_weight * c_ngf + edge_weight * c_edge)


def fast_combined_cost(
    drr: np.ndarray,
    xray_ngf: tuple,
    xray_edge: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ngf_weight: float = 0.6,
    edge_weight: float = 0.4,
    ngf_sigma: float = 1.5,
    ngf_eps: float = 0.01,
    edge_sigma: float = 1.5,
) -> float:
    """Fast combined NGF + edge cost using precomputed X-ray structures.

    This is the speed-optimised version for the optimizer hot loop.
    The X-ray NGF (``xray_ngf``) and edge map (``xray_edge``) are precomputed
    once per projection outside the optimizer loop.

    Args:
        drr       : (H, W) float32 DRR synthesized at candidate pose
        xray_ngf  : (nx, ny) tuple of precomputed X-ray NGF arrays (from precompute_xray_ngf)
        xray_edge : (H, W) float32 precomputed X-ray edge map (from precompute_xray_edge)
        mask      : optional binary bone mask (H, W)
        ngf_weight: blending weight for NGF term (default 0.6)
        edge_weight: blending weight for edge NCC term (default 0.4)

    Returns:
        blended cost ∈ [0, 1], lower = better alignment
    """
    # Coverage guard — blank DRR means CT drifted off detector
    coverage = np.count_nonzero(drr) / max(drr.size, 1)
    if coverage < 0.10:
        return 1.0

    # ── NGF component: compare DRR NGF to precomputed X-ray NGF ────────────
    nx_drr, ny_drr = _normalized_gradient_field(drr, ngf_sigma, ngf_eps)
    nx_rad, ny_rad = xray_ngf  # precomputed — no gaussian_filter for X-ray
    dot_sq = (nx_drr * nx_rad + ny_drr * ny_rad) ** 2

    if mask is not None:
        m = (mask > 0)
        if m.sum() < 100:
            return 1.0
        ngf_score = dot_sq[m].mean()
    else:
        ngf_score = dot_sq.mean()
    c_ngf = float(1.0 - np.clip(ngf_score, 0.0, 1.0))

    # ── Edge NCC component ─────────────────────────────────────────────────
    c_edge = bone_edge_ncc(drr, xray_edge, sigma=edge_sigma, mask=mask)

    return float(ngf_weight * c_ngf + edge_weight * c_edge)


# ---------------------------------------------------------------------------
# Local NCC (LNCC) — windowed mean NCC over local patches
# ---------------------------------------------------------------------------

def local_ncc(
        img1: np.ndarray,
        img2: np.ndarray,
        window: int = 9,
) -> float:
    """
    Local Normalized Cross-Correlation.

    Computes NCC inside every (window × window) neighbourhood using fast
    box-filter arithmetic, then returns the mean over all valid pixels.
    This is more robust than global NCC for local intensity variations.

    Returns cost ∈ [0, 1], 0 = perfect alignment.
    """
    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)

    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    w = int(window)

    mu_a  = uniform_filter(a,     size=w)
    mu_b  = uniform_filter(b,     size=w)
    mu_aa = uniform_filter(a * a, size=w)
    mu_bb = uniform_filter(b * b, size=w)
    mu_ab = uniform_filter(a * b, size=w)

    sigma_a  = np.sqrt(np.maximum(mu_aa - mu_a**2, 0.0))
    sigma_b  = np.sqrt(np.maximum(mu_bb - mu_b**2, 0.0))
    cov_ab   = mu_ab - mu_a * mu_b

    denom = sigma_a * sigma_b
    mask  = denom > 1e-6
    lncc  = np.zeros_like(a)
    lncc[mask] = cov_ab[mask] / denom[mask]

    mean_lncc = float(lncc[mask].mean()) if mask.sum() > 0 else 0.0
    return float(1.0 - np.clip(mean_lncc, -1.0, 1.0)) / 2.0   # → [0, 1]


# ---------------------------------------------------------------------------
# Multi-Scale NCC
# ---------------------------------------------------------------------------

def multiscale_ncc(
        img1: np.ndarray,
        img2: np.ndarray,
        scales: List[int] = None,
        weights: List[float] = None,
) -> float:
    """
    Multi-Scale NCC: global NCC averaged over a Gaussian image pyramid.

    Each scale halves the image resolution. Global NCC is computed at each
    level and the results are averaged (optionally weighted).

    Returns cost ∈ [0, 1], 0 = perfect alignment.
    """
    if scales is None:
        scales = [1, 2, 4]          # original, ½, ¼
    if weights is None:
        weights = [1.0] * len(scales)

    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)

    import cv2
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)

    total, w_sum = 0.0, 0.0
    for scale, wt in zip(scales, weights):
        if scale == 1:
            la, lb = a, b
        else:
            h, w_img = max(1, a.shape[0] // scale), max(1, a.shape[1] // scale)
            la = cv2.resize(a, (w_img, h), interpolation=cv2.INTER_AREA)
            lb = cv2.resize(b, (w_img, h), interpolation=cv2.INTER_AREA)
        total += wt * normalized_cross_correlation(la, lb)
        w_sum += wt

    mean_ncc = total / (w_sum + 1e-12)
    return float(1.0 - np.clip(mean_ncc, -1.0, 1.0)) / 2.0


# ---------------------------------------------------------------------------
# Gradient NCC
# ---------------------------------------------------------------------------

def gradient_ncc(
        img1: np.ndarray,
        img2: np.ndarray,
        sigma: float = 1.5,
) -> float:
    """
    Gradient NCC: NCC applied to the gradient *magnitude* maps.

    More sensitive to structural alignment than intensity NCC, especially
    across modalities (DRR vs X-ray).

    Returns cost ∈ [0, 1], 0 = perfect alignment.
    """
    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)

    a = gaussian_filter(img1.astype(np.float32), sigma=sigma)
    b = gaussian_filter(img2.astype(np.float32), sigma=sigma)

    gxa, gya = np.gradient(a, axis=1), np.gradient(a, axis=0)
    gxb, gyb = np.gradient(b, axis=1), np.gradient(b, axis=0)

    mag_a = np.sqrt(gxa**2 + gya**2)
    mag_b = np.sqrt(gxb**2 + gyb**2)

    ncc_val = normalized_cross_correlation(mag_a, mag_b)
    return float(1.0 - np.clip(ncc_val, -1.0, 1.0)) / 2.0


# ---------------------------------------------------------------------------
# Normalised Mutual Information (NMI)
# ---------------------------------------------------------------------------

def normalised_mutual_information(
        img1: np.ndarray,
        img2: np.ndarray,
        bins: int = 32,
) -> float:
    """
    Normalised Mutual Information: NMI = (H(X) + H(Y)) / H(X, Y).

    Range > 1, with 1 meaning independence and higher values indicating
    more mutual information. We return a cost = 1 / NMI so that lower is
    better and perfect alignment → cost near 0 (NMI → ∞ is impossible; in
    practice NMI ∈ [1, 2] for most image pairs).

    Returns cost ∈ [0, 1] (approximately).
    """
    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)

    a = np.clip(img1, 0, 1).ravel()
    b = np.clip(img2, 0, 1).ravel()

    hist2d, _, _ = np.histogram2d(a, b, bins=bins)
    hist2d = hist2d + 1e-10
    pxy = hist2d / hist2d.sum()
    px  = pxy.sum(axis=1)
    py  = pxy.sum(axis=0)

    Hx  = -float(np.sum(px  * np.log(px  + 1e-10)))
    Hy  = -float(np.sum(py  * np.log(py  + 1e-10)))
    Hxy = -float(np.sum(pxy * np.log(pxy + 1e-10)))

    if Hxy < 1e-10:
        return 0.0

    nmi = (Hx + Hy) / Hxy   # ≥ 1 for any non-degenerate distribution
    # Map to cost: perfect alignment → nmi high → cost low
    # Use 2 - nmi clamped to [0, 1] (NMI ≤ 2 for binary-entropy marginals)
    return float(np.clip(2.0 - nmi, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Entropy of Difference
# ---------------------------------------------------------------------------

def entropy_of_difference(
        img1: np.ndarray,
        img2: np.ndarray,
        bins: int = 64,
        sigma: float = 0.0,
) -> float:
    """
    Entropy of the Difference Image.

    H(img1 − img2): at perfect alignment the difference is near-zero and its
    entropy is low. Returns the normalised entropy (divided by log(bins)) so
    the cost is in [0, 1].

    Args:
        sigma : optional Gaussian pre-smoothing (0 = none)
    """
    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)

    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    if sigma > 0:
        a = gaussian_filter(a, sigma=sigma)
        b = gaussian_filter(b, sigma=sigma)

    diff = (a - b).ravel()
    hist, _ = np.histogram(diff, bins=bins, range=(-1.0, 1.0), density=False)
    hist = hist.astype(np.float64) + 1e-10
    p    = hist / hist.sum()
    H    = -float(np.sum(p * np.log(p)))
    H_max = np.log(bins)
    return float(np.clip(H / H_max, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Correlation Ratio (CR) — Roche et al. 1998
# ---------------------------------------------------------------------------

def correlation_ratio(
        img1: np.ndarray,
        img2: np.ndarray,
        bins: int = 32,
) -> float:
    """
    Correlation Ratio η²(img2 | img1).

    Measures how well img2 is a function of img1 (functional dependency).
    CR = 1 − Var(E[img2|img1]) / Var(img2)

    A value near 1 indicates high dependency (good alignment for multi-modal
    registration). Returns cost = 1 − CR ∈ [0, 1].

    Reference: Roche et al., MICCAI 1998.
    """
    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)

    a = np.clip(img1, 0, 1).ravel()
    b = np.clip(img2, 0, 1).ravel()

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_idx = np.digitize(a, edges) - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)

    var_b = float(np.var(b))
    if var_b < 1e-10:
        return 0.0

    # Weighted variance of bin means
    cond_var = 0.0
    for k in range(bins):
        mask = bin_idx == k
        n_k  = int(mask.sum())
        if n_k < 2:
            continue
        mu_k    = float(b[mask].mean())
        cond_var += n_k * (mu_k - b.mean()) ** 2

    cond_var /= len(b)
    cr = float(np.clip(cond_var / var_b, 0.0, 1.0))
    return float(1.0 - cr)


# ---------------------------------------------------------------------------
# Stochastic Rank Correlation (SRC)
# ---------------------------------------------------------------------------

def stochastic_rank_correlation(
        img1: np.ndarray,
        img2: np.ndarray,
        n_samples: int = 2000,
        seed: int = 0,
) -> float:
    """
    Stochastic Rank Correlation (SRC).

    Approximates the Spearman rank correlation between DRR and X-ray using
    a random subsample of pixels.  Cheaper than computing exact ranks over
    full-resolution images.

    Returns cost = (1 − SRC) / 2 ∈ [0, 1], lower = better.

    Reference: Brehler et al.; also used as a similarity in 2D/3D reg.
    """
    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)

    rng  = np.random.default_rng(seed)
    flat_a = img1.ravel().astype(np.float32)
    flat_b = img2.ravel().astype(np.float32)

    n = min(n_samples, len(flat_a))
    idx = rng.choice(len(flat_a), n, replace=False)

    from scipy.stats import spearmanr
    r, _ = spearmanr(flat_a[idx], flat_b[idx])
    if not np.isfinite(r):
        return 0.5
    return float((1.0 - float(r)) / 2.0)


# ---------------------------------------------------------------------------
# Gradient Difference (GD) — Penney et al. 1998
# ---------------------------------------------------------------------------

def gradient_difference(
        img1: np.ndarray,
        img2: np.ndarray,
        sigma: float = 1.0,
        A: float = None,
) -> float:
    """
    Gradient Difference similarity metric (Penney et al. 1998).

    GD = Σ_i  A² / (A² + (∂img1/∂x_i − ∂img2/∂x_i)²)

    A is a variance parameter controlling sensitivity. When A is large the
    metric is robust to intensity offsets; small A amplifies small gradient
    differences. Summed over both x and y gradient directions.

    Returns cost = 1 − GD_normalised ∈ [0, 1].
    """
    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)

    a = gaussian_filter(img1.astype(np.float32), sigma=sigma)
    b = gaussian_filter(img2.astype(np.float32), sigma=sigma)

    dax, day = np.gradient(a, axis=1), np.gradient(a, axis=0)
    dbx, dby = np.gradient(b, axis=1), np.gradient(b, axis=0)

    if A is None:
        # Set A to std of gradient differences (robust default)
        diff_x = dax - dbx
        diff_y = day - dby
        A = float(np.std(np.concatenate([diff_x.ravel(), diff_y.ravel()])))
        A = max(A, 1e-4)

    A2 = A * A
    gd_x = A2 / (A2 + (dax - dbx) ** 2)
    gd_y = A2 / (A2 + (day - dby) ** 2)

    gd = float((gd_x + gd_y).mean()) / 2.0   # normalise to [0, 1]
    return float(1.0 - np.clip(gd, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Pattern Intensity (PI) — Weese et al. 1997 / Birkfellner 2003
# ---------------------------------------------------------------------------

def pattern_intensity(
        img1: np.ndarray,
        img2: np.ndarray,
        sigma: float = 3.0,
        r: float = None,
) -> float:
    """
    Pattern Intensity (PI) similarity metric.

    PI = Σ_{i} Σ_{j ∈ neighbourhood(i)}  σ² / (σ² + (D_i − D_j)²)

    where D = img1 − img2 (difference image) and σ controls the width.
    High PI → difference image has little spatial structure → good alignment.

    Approximated efficiently using a uniform local variance:
        PI ≈ mean( σ² / (σ² + local_var(D)) )

    Returns cost = 1 − PI_normalised ∈ [0, 1].
    """
    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)

    D = img1.astype(np.float32) - img2.astype(np.float32)

    # Local variance of D using box filter
    win = max(3, int(sigma * 2) | 1)   # odd window size ≥ 3
    mu_D  = uniform_filter(D,       size=win)
    mu_D2 = uniform_filter(D * D,   size=win)
    local_var = np.maximum(mu_D2 - mu_D**2, 0.0)

    if r is None:
        r = float(np.std(D))
        r = max(r, 1e-4)

    r2 = r * r
    pi_map = r2 / (r2 + local_var)
    pi     = float(pi_map.mean())   # ∈ [0, 1], 1 = flat difference = perfect alignment
    return float(1.0 - np.clip(pi, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Gradient Multi-Scale NCC
# ---------------------------------------------------------------------------

def gradient_multiscale_ncc(
        img1: np.ndarray,
        img2: np.ndarray,
        scales: List[int] = None,
        sigma: float = 1.5,
        weights: List[float] = None,
) -> float:
    """
    Gradient Multi-Scale NCC.

    Computes the gradient magnitude map at each scale level of a Gaussian
    pyramid, then evaluates global NCC on those gradient maps and averages
    across scales.  Combines the benefits of multi-scale NCC (capture range)
    and gradient NCC (structural sensitivity).

    Returns cost ∈ [0, 1], 0 = perfect alignment.
    """
    if scales is None:
        scales = [1, 2, 4]
    if weights is None:
        weights = [1.0] * len(scales)

    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)

    import cv2
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)

    total, w_sum = 0.0, 0.0
    for scale, wt in zip(scales, weights):
        if scale == 1:
            la, lb = a, b
        else:
            h_s = max(1, a.shape[0] // scale)
            w_s = max(1, a.shape[1] // scale)
            la  = cv2.resize(a, (w_s, h_s), interpolation=cv2.INTER_AREA)
            lb  = cv2.resize(b, (w_s, h_s), interpolation=cv2.INTER_AREA)

        # Gradient magnitudes at this scale
        la_s = gaussian_filter(la, sigma=sigma)
        lb_s = gaussian_filter(lb, sigma=sigma)
        ga   = np.sqrt(np.gradient(la_s, axis=1)**2 + np.gradient(la_s, axis=0)**2)
        gb   = np.sqrt(np.gradient(lb_s, axis=1)**2 + np.gradient(lb_s, axis=0)**2)

        total += wt * normalized_cross_correlation(ga, gb)
        w_sum += wt

    mean_ncc = total / (w_sum + 1e-12)
    return float(1.0 - np.clip(mean_ncc, -1.0, 1.0)) / 2.0


# ---------------------------------------------------------------------------
# Normalised Gradient Information (NGI)  — Pluim et al. 2000
# ---------------------------------------------------------------------------

def normalised_gradient_information(
        fixed:  np.ndarray,
        moving: np.ndarray,
        sigma:  float = 1.5,
) -> float:
    """
    Normalised Gradient Information (NGI), Pluim et al. 2000.

    NGI(I0, I1) =
        sum_{i,j} [ (alpha(i,j)+1)/2 * min(|∇I0|, |∇I1|) ]
      / sum_{i,j} [ (beta(i,j) +1)/2 * min(|∇I1|, |∇I1|) ]

    where:
        alpha(i,j) = (∇I0 · ∇I1) / (|∇I0| |∇I1|)   cosine similarity
        beta(i,j)  = (∇I1 · ∇I1) / (|∇I1| |∇I1|) = 1  always

    So the denominator simplifies to sum_{i,j} |∇I1(i,j)|  (over non-zero pixels).

    NGI ∈ [0, 1]; NGI = 1 when moving = fixed.
    Returns cost = 1 - NGI ∈ [0, 1], 0 = perfect alignment.
    """
    if fixed.shape != moving.shape:
        from skimage.transform import resize
        moving = resize(moving.astype(np.float32), fixed.shape,
                        anti_aliasing=True, preserve_range=True)

    p1 = gaussian_filter(fixed.astype(np.float32),  sigma=sigma)   # I0
    p2 = gaussian_filter(moving.astype(np.float32), sigma=sigma)   # I1

    gx1, gy1 = np.gradient(p1, axis=1), np.gradient(p1, axis=0)
    gx2, gy2 = np.gradient(p2, axis=1), np.gradient(p2, axis=0)

    mag1 = np.sqrt(gx1**2 + gy1**2)   # |∇I0|
    mag2 = np.sqrt(gx2**2 + gy2**2)   # |∇I1|

    eps  = 1e-8
    mask = (mag1 > eps) & (mag2 > eps)   # both gradients non-zero

    # alpha = cosine similarity of ∇I0 and ∇I1
    dot   = gx1 * gx2 + gy1 * gy2
    alpha = np.where(mask, dot / (mag1 * mag2 + eps), 0.0)
    alpha = np.clip(alpha, -1.0, 1.0)

    w_alpha  = (alpha + 1.0) / 2.0              # (alpha+1)/2 ∈ [0,1]
    min_mag  = np.minimum(mag1, mag2)           # min(|∇I0|, |∇I1|)
    m        = mask.astype(np.float32)

    # numerator: sum[ (alpha+1)/2 * min(|∇I0|,|∇I1|) ]  over joint-nonzero pixels
    numerator = float(np.sum(m * w_alpha * min_mag))

    # denominator: sum[ (beta+1)/2 * |∇I1| ] = sum[ |∇I1| ]  (beta=1, I1=fixed=p1)
    denom_mask = (mag1 > eps).astype(np.float32)
    denominator = float(np.sum(denom_mask * mag1))

    if denominator < eps:
        return 1.0                               # degenerate: fixed has no gradients

    ngi = numerator / denominator
    return float(np.clip(1.0 - ngi, 0.0, 1.0))  # cost: lower = better


# ---------------------------------------------------------------------------
# Local Gradient NCC (LGNCC)
# ---------------------------------------------------------------------------

def local_gradient_ncc(
        img1:   np.ndarray,
        img2:   np.ndarray,
        window: int   = 9,
        sigma:  float = 1.5,
) -> float:
    """
    Local Gradient NCC (LGNCC).

    Applies LNCC to the gradient magnitude maps rather than the raw
    intensity images, combining the local robustness of LNCC with the
    structural sensitivity of gradient NCC.

    Returns cost ∈ [0, 1], 0 = perfect alignment.
    """
    if img1.shape != img2.shape:
        from skimage.transform import resize
        img2 = resize(img2.astype(np.float32), img1.shape,
                      anti_aliasing=True, preserve_range=True)

    # Gradient magnitude maps
    p1 = gaussian_filter(img1.astype(np.float32), sigma=sigma)
    p2 = gaussian_filter(img2.astype(np.float32), sigma=sigma)
    a  = np.sqrt(np.gradient(p1, axis=1)**2 + np.gradient(p1, axis=0)**2)
    b  = np.sqrt(np.gradient(p2, axis=1)**2 + np.gradient(p2, axis=0)**2)

    # Local NCC on gradient magnitudes
    w = int(window)
    mu_a  = uniform_filter(a,     size=w)
    mu_b  = uniform_filter(b,     size=w)
    mu_aa = uniform_filter(a * a, size=w)
    mu_bb = uniform_filter(b * b, size=w)
    mu_ab = uniform_filter(a * b, size=w)

    sigma_a = np.sqrt(np.maximum(mu_aa - mu_a**2, 0.0))
    sigma_b = np.sqrt(np.maximum(mu_bb - mu_b**2, 0.0))
    cov_ab  = mu_ab - mu_a * mu_b

    denom = sigma_a * sigma_b
    mask  = denom > 1e-6
    lncc_map = np.zeros_like(a)
    lncc_map[mask] = cov_ab[mask] / denom[mask]

    mean_lncc = float(lncc_map[mask].mean()) if mask.sum() > 0 else 0.0
    return float(1.0 - np.clip(mean_lncc, -1.0, 1.0)) / 2.0


if __name__ == '__main__':
    # Quick test: shifted image should have lower GO than aligned
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    np.random.seed(42)
    from skimage.data import shepp_logan_phantom
    phantom = shepp_logan_phantom().astype(np.float32)

    go_same   = go_cost(phantom, phantom)
    shifted   = np.roll(phantom, 30, axis=0)
    go_shift  = go_cost(phantom, shifted)
    ncc_same  = normalized_cross_correlation(phantom, phantom)
    ncc_shift = normalized_cross_correlation(phantom, shifted)

    print(f"GO cost (same img):    {go_same:.4f}  (should be near 0)")
    print(f"GO cost (shifted 30px): {go_shift:.4f} (should be > 0)")
    print(f"NCC (same img):        {ncc_same:.4f}  (should be near 1)")
    print(f"NCC (shifted 30px):    {ncc_shift:.4f} (should be < 1)")
