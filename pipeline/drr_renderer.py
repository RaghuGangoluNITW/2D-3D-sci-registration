"""
drr_renderer.py
================
Self-contained, GPU-accelerated DRR (Digitally Reconstructed Radiograph)
renderer for the POC. Deliberately independent of the main mslevel_reg
codebase so this proof-of-concept can be reviewed and run standalone.

Takes a CT volume (loaded via nibabel from a .nii.gz, e.g. VerSe) and a
camera pose (rotation + translation + simple pinhole intrinsics), and
renders a synthetic X-ray via GPU ray-marching (Beer-Lambert attenuation
model, same approach as the main project's DeepFluoroDRR).

This is the "generate many DRRs from the CT, each with known camera
angle and known 3D->2D landmark projection" step -- the half of the
pipeline that is NOT in question. Every DRR this renders comes with
its own landmark pixel positions computed for free (no manual marking),
because we choose the pose and can project any known 3D point through it.
"""
import numpy as np
import torch
import torch.nn.functional as F


class CTVolume:
    """Thin wrapper around a loaded CT volume + world-coordinate metadata."""

    def __init__(self, volume_zyx: np.ndarray, spacing_xyz: np.ndarray,
                 origin_xyz: np.ndarray, hu_threshold: float = 150.0):
        """
        volume_zyx   : (Z, Y, X) HU array
        spacing_xyz  : (3,) mm/voxel, order (sx, sy, sz)
        origin_xyz   : (3,) mm, world position of voxel (0,0,0)
        hu_threshold : voxels below this are treated as air (0 attenuation)
        """
        self.spacing = np.asarray(spacing_xyz, dtype=np.float64)
        self.origin = np.asarray(origin_xyz, dtype=np.float64)
        self.shape_zyx = volume_zyx.shape  # (n_Z, n_Y, n_X)

        n_Z, n_Y, n_X = self.shape_zyx
        sx, sy, sz = self.spacing
        ox, oy, oz = self.origin
        self.world_min = self.origin.copy()
        self.world_max = np.array([ox + (n_X - 1) * sx,
                                     oy + (n_Y - 1) * sy,
                                     oz + (n_Z - 1) * sz])

        vol_hu = volume_zyx.astype(np.float32)
        # mu proportional to (1 + HU/1000); clip air to 0
        vol_mu = np.clip(vol_hu / 1000.0 + 1.0, 0.0, None)
        vol_mu[vol_hu < hu_threshold] = 0.0
        mu_max = float(vol_mu.max())
        self._mu_max = mu_max if mu_max > 0 else 1.0
        vol_mu /= self._mu_max
        self.vol_mu = vol_mu  # (Z, Y, X), normalized [0,1]


class DRRRenderer:
    """GPU ray-marching DRR renderer. See deepfluoro_drr.py for the
    original version this is adapted from (same Beer-Lambert approach)."""

    def __init__(self, ct: CTVolume, device: str = None):
        self.ct = ct
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._vol = torch.from_numpy(ct.vol_mu).unsqueeze(0).unsqueeze(0).to(self.device)
        self._vol_shape = ct.shape_zyx

    def _world_to_norm(self, pts: torch.Tensor) -> torch.Tensor:
        n_Z, n_Y, n_X = self._vol_shape
        X0, Y0, Z0 = self.ct.origin
        sx, sy, sz = self.ct.spacing
        vx = (pts[..., 0] - X0) / sx
        vy = (pts[..., 1] - Y0) / sy
        vz = (pts[..., 2] - Z0) / sz
        nx = vx / (n_X - 1) * 2 - 1
        ny = vy / (n_Y - 1) * 2 - 1
        nz = vz / (n_Z - 1) * 2 - 1
        return torch.stack([nx, ny, nz], dim=-1)

    def render(self, src_xyz: np.ndarray, pax_xyz: np.ndarray,
               u_xyz: np.ndarray, v_xyz: np.ndarray,
               output_size: int = 256, pixel_spacing_mm: float = 1.5,
               n_steps: int = 200, sdd_mm: float = None) -> np.ndarray:
        """
        Render a DRR from an explicit camera basis:
          src_xyz  : (3,) source (X-ray emitter) position, world mm
          pax_xyz  : (3,) unit principal axis (source -> detector center)
          u_xyz,v_xyz : (3,) unit detector in-plane axes (orthogonal to pax)
        Returns (output_size, output_size) float32 in [0,1].
        """
        dev = self.device
        src_t = torch.tensor(src_xyz, dtype=torch.float32, device=dev)
        pax_t = torch.tensor(pax_xyz, dtype=torch.float32, device=dev)
        u_t = torch.tensor(u_xyz, dtype=torch.float32, device=dev)
        v_t = torch.tensor(v_xyz, dtype=torch.float32, device=dev)

        if sdd_mm is None:
            corners = np.array([[x, y, z] for x in (self.ct.world_min[0], self.ct.world_max[0])
                                 for y in (self.ct.world_min[1], self.ct.world_max[1])
                                 for z in (self.ct.world_min[2], self.ct.world_max[2])])
            t_corners = (corners - src_xyz) @ pax_xyz
            sdd_mm = float(t_corners.max()) + 50.0
        det_center = src_xyz + sdd_mm * pax_xyz
        det_t = torch.tensor(det_center, dtype=torch.float32, device=dev)

        N = output_size
        half = (N - 1) / 2.0
        coords = (torch.arange(N, device=dev, dtype=torch.float32) - half) * pixel_spacing_mm
        vv, uu = torch.meshgrid(coords, coords, indexing='ij')
        det_pts = (det_t.view(1, 1, 3) + uu.unsqueeze(-1) * u_t.view(1, 1, 3)
                   + vv.unsqueeze(-1) * v_t.view(1, 1, 3))

        ray_dir = det_pts - src_t.view(1, 1, 3)
        ray_len = ray_dir.norm(dim=-1, keepdim=True)
        ray_norm = ray_dir / (ray_len + 1e-8)
        mean_len = float(ray_len.mean().item())
        step_mm = mean_len / n_steps
        t_vals = torch.linspace(0.0, mean_len, n_steps, device=dev)

        EFF_MU_MM = 0.018
        li = np.zeros((N, N), dtype=np.float32)
        bytes_per_row = n_steps * N * 3 * 4
        chunk_rows = max(1, min(N, int(512 * 1024 * 1024 // bytes_per_row)))

        for r0 in range(0, N, chunk_rows):
            r1 = min(r0 + chunk_rows, N)
            rn = ray_norm[r0:r1]
            pts = (src_t.view(1, 1, 1, 3)
                   + t_vals.view(n_steps, 1, 1, 1) * rn.unsqueeze(0))
            grid = self._world_to_norm(pts).unsqueeze(0)
            vals = F.grid_sample(self._vol, grid, mode='bilinear',
                                  padding_mode='zeros', align_corners=True
                                  ).squeeze(0).squeeze(0)
            li[r0:r1] = (vals * (EFF_MU_MM * step_mm)).sum(dim=0).cpu().numpy()

        transmitted = np.exp(-li)
        drr = 1.0 - transmitted
        m = drr.max()
        if m > 1e-6:
            drr = drr / m
        return drr.astype(np.float32)

    @staticmethod
    def camera_basis_from_azimuth_elevation(center_xyz: np.ndarray,
                                              sid_mm: float,
                                              azimuth_deg: float,
                                              elevation_deg: float = 0.0):
        """
        Build a simple orbit camera around `center_xyz`: source sits on a
        sphere of radius sid_mm around the center, looking back at it.
        azimuth=0 -> source on +Y axis (approx AP), azimuth=90 -> +X axis
        (approx lateral). elevation tilts the source up/down.
        Returns (src_xyz, pax_xyz, u_xyz, v_xyz).

        u_xyz convention fixed to match DeepFluoro's real calibrated
        camera (deepfluoro_loader.detector_axes_world): verified by
        comparing against 6 real projections at different azimuths --
        pax/v always matched (dot~+0.99) but u was consistently FLIPPED
        (dot~-0.99) under the original `u = cross(up, pax)` formula. This
        is a left-right detector mirroring, not a coincidence (same sign
        flip at every azimuth tested). Fixed by negating u so this
        renderer's detector-u axis matches real device convention --
        otherwise every DRR in a bank built from this function is a
        horizontal mirror of what the corresponding real X-ray at that
        pose actually shows, which silently prevented retrieval from ever
        finding a genuinely close match (see DeepFluoro POC session: best
        bank entry still 126mm off despite covering the right az/el/dist
        range).
        """
        az = np.deg2rad(azimuth_deg)
        el = np.deg2rad(elevation_deg)
        direction = np.array([
            np.sin(az) * np.cos(el),
            np.cos(az) * np.cos(el),
            np.sin(el),
        ])
        src = center_xyz + sid_mm * direction
        pax = (center_xyz - src)
        pax = pax / np.linalg.norm(pax)
        up = np.array([0., 0., -1.])
        if abs(np.dot(pax, up)) > 0.95:
            up = np.array([1., 0., 0.])
        u = np.cross(pax, up)
        u = u / np.linalg.norm(u)
        # Real device convention is LEFT-handed: cross(u,v) = -pax (verified
        # against deepfluoro_loader's detector_axes_world across 6 real
        # projections: cross(u,v).pax = -1.0 exactly). v = cross(u, pax)
        # gives that handedness; the original v = cross(pax, u) gave the
        # opposite (right-handed) one, which is why v_dot flipped sign
        # once u was corrected.
        v = np.cross(u, pax)
        return src, pax, u, v

    def project_points(self, pts_xyz: np.ndarray, src_xyz: np.ndarray,
                        pax_xyz: np.ndarray, u_xyz: np.ndarray, v_xyz: np.ndarray,
                        output_size: int, pixel_spacing_mm: float,
                        sdd_mm: float) -> np.ndarray:
        """
        Project 3D world points onto the detector plane for this camera,
        returning (N, 2) pixel coordinates (u_px, v_px) in the rendered
        image's pixel frame. This is how we get "free" ground-truth 2D
        landmark positions for every DRR -- no manual marking, just math,
        since we chose the camera pose ourselves.
        """
        pts = np.atleast_2d(pts_xyz)
        rel = pts - src_xyz
        depth = rel @ pax_xyz
        u_comp = rel @ u_xyz
        v_comp = rel @ v_xyz
        # scale to the detector plane (at distance sdd_mm from source)
        scale = sdd_mm / depth
        u_mm = u_comp * scale
        v_mm = v_comp * scale
        half = (output_size - 1) / 2.0
        u_px = u_mm / pixel_spacing_mm + half
        v_px = v_mm / pixel_spacing_mm + half
        return np.stack([u_px, v_px], axis=1)
