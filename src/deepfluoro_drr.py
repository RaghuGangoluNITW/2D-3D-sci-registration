"""
deepfluoro_drr.py — Calibrated DRR Generator for DeepFluoro Dataset
====================================================================
Generates Digitally Reconstructed Radiographs (DRRs) that match the
calibrated C-arm geometry stored in the DeepFluoro dataset.

Projection geometry (validated in deepfluoro_loader.py):
  - Source position in world XYZ mm: src = -R_proj^T @ t_proj  [then XZY->XYZ]
  - Camera principal axis: R_proj^T @ [0,0,1] mapped XZY->XYZ
  - SDD = 1020 mm  (= FX_POS * PIXEL_SPACING_MM = 5257.73 * 0.194)
  - Full-res image: 1536x1536, pixel spacing = 0.194 mm
  - ds4x image:     359x359,   pixel spacing = 0.776 mm (=0.194*4)

Key class: DeepFluoroDRR
  drr_img = gen.generate(proj, n_steps=400)
  drr_img_perturbed = gen.generate_from_extrinsic(R_new, t_new, ...)

The DRR is generated at downsampled resolution (default: matching ds4x 359x359)
using GPU-accelerated ray casting via torch.nn.functional.grid_sample.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path

# Import from the loader (always in same directory)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from deepfluoro_loader import (
    DeepFluoroProjection, DeepFluoroSpecimen,
    FX_POS, FY_POS, CX, CY,
    PIXEL_SPACING_MM, SDD_MM,
    FULL_RES_SIZE,
    xzy, xzy_inv,
    source_position_world,
    principal_axis_world,
    detector_axes_world,
    project_world_to_image,
)

# Default optimizer DRR output size (pixels, square)
_OPT_SIZE: int = 180


# ---------------------------------------------------------------------------
# DRR generator
# ---------------------------------------------------------------------------

class DeepFluoroDRR:
    """
    GPU-accelerated DRR generator calibrated to the DeepFluoro C-arm geometry.

    The generator takes a ``DeepFluoroSpecimen`` (CT volume + metadata) and
    generates DRRs matching any ``DeepFluoroProjection``'s calibrated geometry.
    It can also generate DRRs for perturbed or arbitrary poses for use in
    the registration optimizer.

    Usage::

        gen = DeepFluoroDRR(specimen)
        # Match GT projection geometry exactly:
        drr = gen.generate(proj)
        # Match GT geometry at ds4x resolution:
        drr = gen.generate(proj, output_size=359, pixel_spacing_mm=0.776)
        # Generate with a custom extrinsic (for optimizer):
        drr = gen.generate_from_extrinsic(R_new, t_new)
    """

    def __init__(self,
                 specimen: DeepFluoroSpecimen,
                 hu_threshold: float = 0.0,
                 device: Optional[str] = None):
        """
        Initialise the DRR generator from a loaded specimen.

        Args:
            specimen     : loaded DeepFluoroSpecimen (CT + metadata)
            hu_threshold : HU values below this are treated as air (zero attenuation).
                           Default -200 clips only extreme air pockets.
                           Use 0 for water+bone only, 300+ for bone-only DRR.
            device       : 'cuda', 'cpu', or None (auto-detect)
        """
        self.specimen = specimen
        self.hu_threshold = hu_threshold

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # CT metadata
        self.ct_origin  = specimen.ct_origin.astype(np.float64)   # (3,) XYZ mm
        self.ct_spacing = specimen.ct_spacing.astype(np.float64)   # (3,) mm
        ct_vol = specimen.ct_volume  # stored as (n_Z, n_Y, n_X) — (Z, Y, X) ordering

        # Derive volume bounds in world XYZ mm
        # ct_vol.shape = (n_Z, n_Y, n_X), ct_origin = [X0, Y0, Z0], ct_spacing = [sx, sy, sz]
        n_Z, n_Y, n_X = ct_vol.shape
        sx, sy, sz = self.ct_spacing
        ox, oy, oz = self.ct_origin
        self.world_min = self.ct_origin.copy()
        self.world_max = np.array([ox + (n_X-1)*sx, oy + (n_Y-1)*sy, oz + (n_Z-1)*sz])

        # ── Beer-Lambert attenuation volume ──────────────────────────────────
        # Convert HU to linear attenuation coefficient (mu), normalized:
        #   mu(HU) = mu_water * (1 + HU/1000)   [standard HU definition]
        #   mu_water ~ 0.02 mm^-1 at diagnostic X-ray energies
        # We omit the mu_water constant since DRR is normalized anyway.
        # The line integral I = exp(-sum(mu(x) * step_mm)) gives proper X-ray contrast.
        #
        # Importantly: we include soft tissue (HU > -200) for realistic contrast,
        # NOT just bone. Zeroing soft tissue was causing flat appearance.
        vol_hu = ct_vol.astype(np.float32)
        # mu ∝ (1 + HU/1000), clip air (HU < threshold) to 0
        vol_mu = np.clip(vol_hu / 1000.0 + 1.0, 0.0, None)  # ≥0, bone~1.7, water~1.0, air~0
        vol_mu[vol_hu < hu_threshold] = 0.0

        # Keep raw mu values — step_mm scaling happens at render time
        # (so we can change n_steps without reloading the volume)
        # Normalise so max mu = 1.0 for GPU numerical stability
        mu_max = float(vol_mu.max())
        self._mu_max = mu_max if mu_max > 0 else 1.0
        vol_mu /= self._mu_max

        # Upload to GPU as 5D tensor: (1, 1, D, H, W)
        # The CT volume is stored as (Z, Y, X) in the HDF5 file:
        #   dim0 = Z (size = n_Z), dim1 = Y (size = n_Y), dim2 = X (size = n_X)
        # grid_sample on (1, 1, D, H, W) uses grid [x_norm, y_norm, z_norm] where:
        #   grid[...,0] = x_norm -> W (dim2 = X axis)
        #   grid[...,1] = y_norm -> H (dim1 = Y axis)
        #   grid[...,2] = z_norm -> D (dim0 = Z axis)
        # So the volume is stored as (Z, Y, X) and grid_sample naturally handles
        # this when we pass grid coords in (x, y, z) order.
        self._vol = (
            torch.from_numpy(vol_mu)
            .unsqueeze(0).unsqueeze(0)
            .to(self.device)
        )
        self._vol_shape = ct_vol.shape  # (n_Z, n_Y, n_X) — stored as (Z, Y, X)

        print(f"  DeepFluoroDRR: CT={ct_vol.shape}, "
              f"world=[{self.world_min.round(1)} -> {self.world_max.round(1)}], "
              f"device={self.device}")

    # ── Coordinate conversion ─────────────────────────────────────────────

    def _world_to_norm(self, pts: torch.Tensor) -> torch.Tensor:
        """Convert world XYZ mm to normalised grid_sample coordinates.

        The CT volume is stored as (Z, Y, X) — i.e. _vol_shape = (n_Z, n_Y, n_X).
        PyTorch grid_sample on a 5D tensor (1, 1, D, H, W) uses a grid with
        last dimension [x_norm, y_norm, z_norm] mapped as:
          grid[..., 0] = x_norm  →  W axis  (dim2 = X in world, size n_X)
          grid[..., 1] = y_norm  →  H axis  (dim1 = Y in world, size n_Y)
          grid[..., 2] = z_norm  →  D axis  (dim0 = Z in world, size n_Z)

        Normalisation: norm = (voxel_index / (size - 1)) * 2 - 1

        Args:
            pts : (..., 3) world XYZ mm  (pts[...,0]=X, pts[...,1]=Y, pts[...,2]=Z)

        Returns:
            (..., 3) grid_sample coordinates [x_norm, y_norm, z_norm]
        """
        # ct_origin = [X0, Y0, Z0], ct_spacing = [sx, sy, sz]
        # _vol_shape = (n_Z, n_Y, n_X)
        n_Z, n_Y, n_X = self._vol_shape

        X0 = float(self.ct_origin[0]);  sx = float(self.ct_spacing[0])
        Y0 = float(self.ct_origin[1]);  sy = float(self.ct_spacing[1])
        Z0 = float(self.ct_origin[2]);  sz = float(self.ct_spacing[2])

        # Voxel indices (float)
        vx = (pts[..., 0] - X0) / sx   # 0 .. n_X-1
        vy = (pts[..., 1] - Y0) / sy   # 0 .. n_Y-1
        vz = (pts[..., 2] - Z0) / sz   # 0 .. n_Z-1

        # Normalise to [-1, 1]
        nx = vx / (n_X - 1) * 2 - 1   # grid x -> W dim (n_X)
        ny = vy / (n_Y - 1) * 2 - 1   # grid y -> H dim (n_Y)
        nz = vz / (n_Z - 1) * 2 - 1   # grid z -> D dim (n_Z)

        return torch.stack([nx, ny, nz], dim=-1)

    # ── Main generation methods ───────────────────────────────────────────

    def generate(self,
                 proj: DeepFluoroProjection,
                 output_size: int = _OPT_SIZE,
                 pixel_spacing_mm: float = PIXEL_SPACING_MM * (FULL_RES_SIZE / _OPT_SIZE),
                 n_steps: int = 400,
                 chunk_size: int = 512,
                 ) -> np.ndarray:
        """
        Generate a DRR matching the calibrated geometry of a projection.

        Args:
            proj             : DeepFluoroProjection with validated R_proj, t_proj
            output_size      : output image size in pixels (square)
            pixel_spacing_mm : detector pixel spacing in mm
                               (default: 0.776mm = 4 * 0.194mm for ds4x)
            n_steps          : number of ray-casting steps
            chunk_size       : pixels per GPU batch (reduce if OOM)

        Returns:
            (output_size, output_size) float32 DRR image, range [0, 1]
        """
        if proj.R_proj is None:
            raise RuntimeError("Projection has no extrinsic. Load via DeepFluoroLoader.")
        return self.generate_from_extrinsic(
            proj.R_proj, proj.t_proj,
            output_size=output_size,
            pixel_spacing_mm=pixel_spacing_mm,
            n_steps=n_steps,
            chunk_size=chunk_size,
        )

    def generate_from_extrinsic(self,
                                 R_proj: np.ndarray,
                                 t_proj: np.ndarray,
                                 output_size: int = _OPT_SIZE,
                                 pixel_spacing_mm: float = PIXEL_SPACING_MM * (FULL_RES_SIZE / _OPT_SIZE),
                                 n_steps: int = 400,
                                 chunk_size: int = 512,
                                 sdd_mm: Optional[float] = None,
                                 ) -> np.ndarray:
        """
        Generate a DRR for an arbitrary projection extrinsic.

        Fully vectorized GPU ray casting — no Python loop over pixels.
        All N×N rays sampled simultaneously via a single grid_sample call
        on tensor shape (1, 1, n_steps, N, N).

        Args:
            R_proj           : (3, 3) rotation
            t_proj           : (3,)  translation
            output_size      : output image size (pixels, square)
            pixel_spacing_mm : detector pixel spacing (mm per pixel at output_size)
            n_steps          : ray-casting steps (80-150 sufficient for optimizer)
            chunk_size       : unused (kept for API compatibility)
            sdd_mm           : source-to-detector distance in mm.
                               If None, computed from the CT bounding box so rays
                               always span the full volume (works for any dataset).

        Returns:
            (output_size, output_size) float32 DRR in [0, 1]
        """
        dev = self.device

        # ── Derive geometry from extrinsic ─────────────────────────────────
        src_xzy = -R_proj.T @ t_proj
        src_xyz = np.array([src_xzy[0], src_xzy[2], src_xzy[1]])

        pax_xzy = R_proj.T @ np.array([0., 0., 1.])
        pax_xyz = np.array([pax_xzy[0], pax_xzy[2], pax_xzy[1]])

        u_xzy = R_proj.T @ np.array([1., 0., 0.])
        u_xyz = np.array([u_xzy[0], u_xzy[2], u_xzy[1]])

        v_xzy = R_proj.T @ np.array([0., 1., 0.])
        v_xyz = np.array([v_xzy[0], v_xzy[2], v_xzy[1]])

        # ── Compute SDD so rays always span the full CT volume ─────────────
        # If sdd_mm is provided, use it directly.
        # Otherwise, project the 8 CT bbox corners onto the principal axis and
        # set SDD = max projected distance + margin, so the detector is always
        # BEHIND the CT volume from the source's perspective.
        if sdd_mm is not None:
            _sdd = float(sdd_mm)
        else:
            corners = np.array([
                [self.world_min[0], self.world_min[1], self.world_min[2]],
                [self.world_max[0], self.world_min[1], self.world_min[2]],
                [self.world_min[0], self.world_max[1], self.world_min[2]],
                [self.world_max[0], self.world_max[1], self.world_min[2]],
                [self.world_min[0], self.world_min[1], self.world_max[2]],
                [self.world_max[0], self.world_min[1], self.world_max[2]],
                [self.world_min[0], self.world_max[1], self.world_max[2]],
                [self.world_max[0], self.world_max[1], self.world_max[2]],
            ])
            # Project each corner onto principal axis (distance from source)
            t_corners = (corners - src_xyz) @ pax_xyz
            _sdd = float(t_corners.max()) + 50.0   # 50mm margin past far edge

        det_center = src_xyz + _sdd * pax_xyz

        src_t   = torch.tensor(src_xyz,    dtype=torch.float32, device=dev)
        u_ax_t  = torch.tensor(u_xyz,      dtype=torch.float32, device=dev)
        v_ax_t  = torch.tensor(v_xyz,      dtype=torch.float32, device=dev)
        det_t   = torch.tensor(det_center, dtype=torch.float32, device=dev)

        # ── Build detector pixel grid ──────────────────────────────────────
        N = output_size
        half = (N - 1) / 2.0
        u_coords = (torch.arange(N, device=dev, dtype=torch.float32) - half) * pixel_spacing_mm
        v_coords = (torch.arange(N, device=dev, dtype=torch.float32) - half) * pixel_spacing_mm
        vv, uu = torch.meshgrid(v_coords, u_coords, indexing='ij')  # (N, N)

        # Detector points: (N, N, 3)
        det_pts = (det_t.view(1, 1, 3)
                   + uu.unsqueeze(-1) * u_ax_t.view(1, 1, 3)
                   + vv.unsqueeze(-1) * v_ax_t.view(1, 1, 3))

        # ── Ray directions ─────────────────────────────────────────────────
        ray_dir  = det_pts - src_t.view(1, 1, 3)
        ray_len  = ray_dir.norm(dim=-1, keepdim=True)        # (N, N, 1)
        ray_norm = ray_dir / (ray_len + 1e-8)

        mean_len = float(ray_len.mean().item())
        step_mm  = mean_len / n_steps
        t_vals   = torch.linspace(0.0, mean_len, n_steps, device=dev)

        # ── Chunked ray casting (avoids OOM for large N×n_steps) ──────────
        # Peak tensor: (n_steps, chunk_rows, N, 3) — tune chunk_rows to fit VRAM.
        # At N=1024, n_steps=400: full alloc = 400×1024×1024×3×4B = 4.7 GiB.
        # With chunk_rows=64: 400×64×1024×3×4B = 300 MB per chunk — fits comfortably.
        EFF_MU_MM   = 0.018
        line_integral_np = np.zeros((N, N), dtype=np.float32)

        # Decide chunk size: aim for ≤ 512 MB per chunk
        bytes_per_row = n_steps * N * 3 * 4  # float32
        max_bytes     = 512 * 1024 * 1024
        chunk_rows    = max(1, int(max_bytes // bytes_per_row))
        chunk_rows    = min(chunk_rows, N)

        for row_start in range(0, N, chunk_rows):
            row_end  = min(row_start + chunk_rows, N)
            rn_chunk = ray_norm[row_start:row_end]   # (R, N, 3)
            R_chunk  = row_end - row_start

            # (n_steps, R, N, 3)
            pts = (src_t.view(1, 1, 1, 3)
                   + t_vals.view(n_steps, 1, 1, 1) * rn_chunk.unsqueeze(0))

            grid  = self._world_to_norm(pts)          # (n_steps, R, N, 3)
            g5    = grid.unsqueeze(0)                 # (1, n_steps, R, N, 3)
            vals  = F.grid_sample(
                self._vol, g5,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            ).squeeze(0).squeeze(0)                   # (n_steps, R, N)

            li_chunk = (vals * (EFF_MU_MM * step_mm)).sum(dim=0)  # (R, N)
            line_integral_np[row_start:row_end] = li_chunk.cpu().numpy()
            del pts, grid, g5, vals, li_chunk

        line_integral = torch.from_numpy(line_integral_np).to(dev)
        transmitted   = torch.exp(-line_integral)
        drr_t         = 1.0 - transmitted

        drr = drr_t.cpu().numpy().astype(np.float32)

        # Normalise to [0, 1]
        drr_max = drr.max()
        if drr_max > 1e-6:
            drr /= drr_max
        return drr

    # ── Convenience helpers ───────────────────────────────────────────────

    def generate_gt_pair(self,
                          proj: DeepFluoroProjection,
                          output_size: int = _OPT_SIZE,
                          pixel_spacing_mm: float = PIXEL_SPACING_MM * (FULL_RES_SIZE / _OPT_SIZE),
                          n_steps: int = 400,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate DRR + return the matching fluoroscopy image (image_raw) as a pair.

        Both images are in DRR-matching orientation — ready for NCC comparison.

        Returns:
            (drr, xray) where both are (output_size, output_size) float32 [0,1]
        """
        import cv2
        drr  = self.generate(proj, output_size, pixel_spacing_mm, n_steps)
        xray = cv2.resize(proj.image_raw, (output_size, output_size),
                          interpolation=cv2.INTER_AREA)
        return drr, xray


# ---------------------------------------------------------------------------
# Utility: reproj error at a given pose (for optimizer callbacks)
# ---------------------------------------------------------------------------

def reproj_error_for_extrinsic(
        proj: DeepFluoroProjection,
        R_cand: np.ndarray,
        t_cand: np.ndarray,
        pts3d_world: np.ndarray,
        lm_names,
        pixel_spacing_mm: float = PIXEL_SPACING_MM,
) -> float:
    """Mean reprojection error (mm) for a candidate extrinsic.

    Projects the 3D landmarks with (R_cand, t_cand) and computes mean
    Euclidean distance to GT 2D landmark positions.

    Args:
        proj            : DeepFluoroProjection with gt_landmarks_2d
        R_cand, t_cand  : candidate projection extrinsic
        pts3d_world     : (N, 3) world XYZ mm
        lm_names        : names for each row of pts3d_world
        pixel_spacing_mm: mm per pixel

    Returns:
        mean PDE in mm
    """
    uv_pred = project_world_to_image(pts3d_world, R_cand, t_cand)
    errs = []
    for i, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            gt = proj.gt_landmarks_2d[name]
            if 0 < gt[0] < FULL_RES_SIZE and 0 < gt[1] < FULL_RES_SIZE:
                errs.append(np.linalg.norm(uv_pred[i] - gt) * pixel_spacing_mm)
    return float(np.mean(errs)) if errs else float('inf')


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from deepfluoro_loader import DeepFluoroLoader

    print("Loading specimen 17-1882 (2 projections) ...")
    loader = DeepFluoroLoader()
    spec   = loader.load_specimen('17-1882', max_projections=2, verbose=True)
    proj   = spec.projections[0]

    print(f"\nGenerating DRR for projection {proj.proj_key} ...")
    gen = DeepFluoroDRR(spec, hu_threshold=0.0)
    drr, xray = gen.generate_gt_pair(proj, output_size=_OPT_SIZE, n_steps=400)

    print(f"DRR:  shape={drr.shape}, max={drr.max():.3f}, nonzero={np.count_nonzero(drr)}")
    print(f"Xray: shape={xray.shape}, max={xray.max():.3f}")

    # Save side-by-side comparison
    out_path = Path('/home/supermicro/Documents/2D_3D_Raghu/results/drr_vs_xray.png')
    out_path.parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(drr, cmap='gray', aspect='equal')
    axes[0].set_title(f'DRR — {proj.proj_key} ({_OPT_SIZE}x{_OPT_SIZE})')
    axes[0].axis('off')
    axes[1].imshow(xray, cmap='gray', aspect='equal')
    axes[1].set_title(f'Fluoroscopy — {proj.proj_key} ({_OPT_SIZE}x{_OPT_SIZE})')
    axes[1].axis('off')
    plt.suptitle(f'Specimen {spec.specimen_id} — GT calibrated geometry',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
