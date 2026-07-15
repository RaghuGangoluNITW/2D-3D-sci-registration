"""
drr.py — Digitally Reconstructed Radiograph Generator
======================================================
GPU-accelerated ray-casting DRR using PyTorch, faithful to Ketcha 2017.

Geometry:
  - Lateral (PA) view: source along Y (Posterior axis) from CT center
  - Source-to-detector distance = SDD mm
  - 6DOF pose: (xr, yr, zr) translation in LPS mm + Euler rotations
  - Ray casting via torch.nn.functional.grid_sample on GPU

CT volume convention after data_loader:
  nrrd.read -> (X, Y, Z) = (512, 512, Nslices)
  after transpose(2,1,0) -> (Z, Y, X) stored in memory
  LPS: X=Left, Y=Posterior, Z=Superior
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

SDD = 1000.0  # mm


def rotation_matrix_np(eta: float, theta: float, phi: float) -> np.ndarray:
    """Rotation matrix from Euler angles (deg): Rz(phi) @ Ry(theta) @ Rx(eta)."""
    er, tr, pr = np.deg2rad([eta, theta, phi])
    Rx = np.array([[1,0,0],[0,np.cos(er),-np.sin(er)],[0,np.sin(er),np.cos(er)]])
    Ry = np.array([[np.cos(tr),0,np.sin(tr)],[0,1,0],[-np.sin(tr),0,np.cos(tr)]])
    Rz = np.array([[np.cos(pr),-np.sin(pr),0],[np.sin(pr),np.cos(pr),0],[0,0,1]])
    return Rz @ Ry @ Rx


class DRRGenerator:
    """
    GPU DRR generator. Lateral cone-beam: source along -Y (Anterior side),
    detector along +Y side. Matches a lateral spine radiograph geometry.
    """

    def __init__(self, ct_volume, ct_spacing, ct_origin,
                 hu_threshold=150.0, view='lateral', sdd=SDD):
        self.spacing   = np.array(ct_spacing, dtype=np.float64)   # (sx,sy,sz) -> (L,P,S)
        self.origin    = np.array(ct_origin,  dtype=np.float64)   # LPS mm of voxel [0,0,0]
        self.hu_thr    = hu_threshold
        self.sdd       = sdd
        self.view      = view
        self.dev       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vol_zyx   = ct_volume.shape                          # (Z,Y,X)

        Z, Y, X = ct_volume.shape
        sx, sy, sz = self.spacing
        ox, oy, oz = self.origin
        self.min_lps  = self.origin.copy()
        self.max_lps  = np.array([ox+(X-1)*sx, oy+(Y-1)*sy, oz+(Z-1)*sz])
        self.ctr_lps  = (self.min_lps + self.max_lps) / 2.0

        vol = ct_volume.astype(np.float32)
        vol[vol < hu_threshold] = 0.0
        vol /= 1000.0   # scale to avoid huge integrals
        self._vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(self.dev)
        print(f"  DRRGen: shape={ct_volume.shape} center={self.ctr_lps.round(1)} dev={self.dev}")

    def _to_norm(self, pts: torch.Tensor) -> torch.Tensor:
        """(...,3) LPS world -> normalised grid_sample coords (x=W=L, y=H=P, z=D=S)."""
        o = torch.tensor(self.origin,  dtype=torch.float32, device=self.dev)
        s = torch.tensor(self.spacing, dtype=torch.float32, device=self.dev)
        Z,Y,X = self.vol_zyx
        vL = (pts[...,0]-o[0])/s[0];  nL = vL/(X-1)*2-1
        vP = (pts[...,1]-o[1])/s[1];  nP = vP/(Y-1)*2-1
        vS = (pts[...,2]-o[2])/s[2];  nS = vS/(Z-1)*2-1
        return torch.stack([nL,nP,nS], dim=-1)

    def generate(self, xr=0., yr=0., zr=0., eta=0., theta=0., phi=0.,
                 det_h=256, det_w=256, pix_mm=2.0, n_steps=300):
        """Generate DRR. Returns (H,W) float32 array."""
        dv = self.dev
        R  = torch.tensor(rotation_matrix_np(eta,theta,phi), dtype=torch.float32, device=dv)

        # CT center after translation
        ctr = torch.tensor(self.ctr_lps+[xr,yr,zr], dtype=torch.float32, device=dv)

        # Source / detector positions
        # Lateral: source along -Y (anterior side), sees spine from the side
        # AP:      source along +Y (posterior side), sees spine front-on
        if self.view == 'lateral':
            src_off = torch.tensor([0., -self.sdd, 0.], dtype=torch.float32, device=dv)
            u_base  = torch.tensor([1., 0., 0.], dtype=torch.float32, device=dv)  # L-R
            v_base  = torch.tensor([0., 0., 1.], dtype=torch.float32, device=dv)  # S-I
        else:  # ap
            src_off = torch.tensor([0.,  self.sdd, 0.], dtype=torch.float32, device=dv)
            u_base  = torch.tensor([1., 0., 0.], dtype=torch.float32, device=dv)  # L-R
            v_base  = torch.tensor([0., 0., 1.], dtype=torch.float32, device=dv)  # S-I

        source   = ctr + R @ src_off      # (3,)
        det_ctr  = ctr - R @ src_off      # (3,)

        # Detector axes (same L-R / S-I basis for both views; source direction differs)
        u_ax = R @ u_base
        v_ax = R @ v_base

        # Pixel grid
        u = (torch.arange(det_w,device=dv)-det_w/2.+0.5)*pix_mm
        v = (torch.arange(det_h,device=dv)-det_h/2.+0.5)*pix_mm
        vv,uu = torch.meshgrid(v,u,indexing='ij')
        det_pts = det_ctr.view(1,1,3) + uu.unsqueeze(-1)*u_ax.view(1,1,3) \
                                      + vv.unsqueeze(-1)*v_ax.view(1,1,3)

        ray_dir = det_pts - source.view(1,1,3)
        ray_len = ray_dir.norm(dim=-1, keepdim=True)
        ray_norm= ray_dir/(ray_len+1e-8)

        mean_len = float(ray_len.mean().item())
        n_s = max(n_steps, int(mean_len/pix_mm))
        t   = torch.linspace(0, mean_len, n_s, device=dv)

        N = det_h*det_w
        rd_flat = ray_norm.reshape(N,3)
        intg    = torch.zeros(N, device=dv)
        chunk   = 256

        for i in range(0, N, chunk):
            c  = rd_flat[i:i+chunk]            # (c,3)
            s_ = source.unsqueeze(0).expand(c.shape[0],-1)  # (c,3)
            pts = s_.unsqueeze(1) + c.unsqueeze(1)*t.view(1,-1,1)  # (c,ns,3)
            g   = self._to_norm(pts)           # (c,ns,3)
            cs  = c.shape[0]
            g5  = g.reshape(1,1,1,cs*n_s,3)
            vals= F.grid_sample(self._vol,g5,mode='bilinear',padding_mode='zeros',align_corners=True)
            vals= vals.reshape(cs,n_s)
            intg[i:i+cs] = vals.sum(1)*pix_mm

        return intg.reshape(det_h,det_w).cpu().numpy().astype(np.float32)

    def project_labels(self, label_lps: np.ndarray,
                       xr=0.,yr=0.,zr=0.,eta=0.,theta=0.,phi=0.) -> np.ndarray:
        """Project (N,3) LPS label points -> (N,2) detector (u,v) mm."""
        R   = rotation_matrix_np(eta,theta,phi)
        ctr = self.ctr_lps + np.array([xr,yr,zr])

        if self.view == 'lateral':
            src_off = np.array([0., -self.sdd, 0.])
        else:  # ap
            src_off = np.array([0.,  self.sdd, 0.])

        source  = ctr + R @ src_off
        det_ctr = ctr - R @ src_off
        view_dir= det_ctr - source; view_dir/=(np.linalg.norm(view_dir)+1e-10)

        u_ax = R @ np.array([1., 0., 0.])   # L-R
        v_ax = R @ np.array([0., 0., 1.])   # S-I

        out = []
        for pt in label_lps:
            ray = pt - source
            denom = np.dot(ray, view_dir)
            if abs(denom)<1e-10: out.append(np.zeros(2)); continue
            t_p = np.dot(det_ctr-source, view_dir)/denom
            hit = source + t_p*ray
            u   = np.dot(hit-det_ctr, u_ax)
            v   = np.dot(hit-det_ctr, v_ax)
            out.append(np.array([u,v]))
        return np.array(out)


def normalize_image(img: np.ndarray) -> np.ndarray:
    nz = img[img>0]
    if len(nz)==0: return img.astype(np.float32)
    p1,p99 = np.percentile(nz,[1,99])
    if p99>p1: return np.clip((img-p1)/(p99-p1),0,1).astype(np.float32)
    return img.astype(np.float32)


def preprocess_topogram(topo: np.ndarray,
                         target_pix_mm=2.0, orig_pix_mm=0.6) -> np.ndarray:
    from skimage.transform import resize
    scale = orig_pix_mm/target_pix_mm
    nh = max(1,int(topo.shape[0]*scale)); nw = max(1,int(topo.shape[1]*scale))
    t2 = resize(topo.astype(np.float32),(nh,nw),anti_aliasing=True,preserve_range=True)
    return normalize_image(t2)


def remove_metal_artifacts(img: np.ndarray,
                            metal_threshold: float = 0.88,
                            max_metal_fraction: float = 0.08,
                            dilate_px: int = 6) -> np.ndarray:
    """
    Suppress surgical metal hardware (screws, rods) in a C-arm image.

    In X-ray images (before inversion), metal appears as the BRIGHTEST
    pixels (near 1.0). After inversion, metal becomes the DARKEST.
    We detect metal on the ORIGINAL (pre-inversion) image and replace
    those pixels with the local median so the GO metric only sees bone.

    Args:
        img                 : (H, W) float32 in [0,1], BEFORE inversion
        metal_threshold     : pixels above this value are considered metal
        max_metal_fraction  : if detected metal > this fraction of image,
                              raise threshold adaptively (avoids masking
                              bright backgrounds/borders as metal)
        dilate_px           : dilation radius to cover metal bloom/halo

    Returns:
        (H, W) float32 with metal pixels replaced by local median
    """
    from scipy.ndimage import binary_dilation, median_filter

    # Adaptive threshold: if naive mask is too large it's probably background
    threshold = metal_threshold
    metal_mask = img > threshold
    if metal_mask.mean() > max_metal_fraction:
        # Step up threshold until metal fraction is reasonable
        for t in [0.91, 0.93, 0.95, 0.97, 0.99, 1.01]:
            metal_mask = img > t
            if metal_mask.mean() <= max_metal_fraction:
                threshold = t
                break
        else:
            return img  # can't isolate metal safely, skip

    if metal_mask.sum() == 0:
        return img  # no metal detected, return unchanged

    # Dilate mask to cover X-ray bloom around metal edges
    struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=bool)
    metal_mask_dilated = binary_dilation(metal_mask, structure=struct)

    # Replace metal pixels with local median (neutral, no strong gradient)
    img_clean = img.copy()
    median_bg = median_filter(img, size=15)
    img_clean[metal_mask_dilated] = median_bg[metal_mask_dilated]

    return img_clean


def preprocess_carm(carm_img: np.ndarray,
                    target_size: int = 256,
                    invert: bool = True,
                    remove_metal: bool = True) -> np.ndarray:
    """
    Preprocess an intraoperative C-arm image for 3D-2D registration.

    C-arm images are X-ray projections where dense structures (bone) appear
    DARK (low intensity) and soft tissue is bright.
    DRRs generated from CT have bone appearing BRIGHT (high attenuation).
    -> We invert the C-arm image so that bone = bright, matching DRR polarity.

    Processing steps:
      1. Convert to float32 [0,1] (assumed already done)
      2. Remove metal artifacts (screws/rods) — replaces with local median
      3. Invert if requested (bone -> bright)
      4. Resize to target_size x target_size
      5. Apply CLAHE for local contrast enhancement
      6. Normalize to [0,1]

    Args:
        carm_img    : (H, W) float32 grayscale image in [0,1]
        target_size : output size in pixels (square); DRR is also this size
        invert      : True = invert intensities (X-ray bone dark -> bright)
        remove_metal: True = suppress surgical hardware before registration

    Returns:
        (target_size, target_size) float32 in [0,1]
    """
    from skimage.transform import resize
    from skimage.exposure import equalize_adapthist

    img = carm_img.astype(np.float32)

    # Step 1: Remove metal artifacts BEFORE inversion (metal = bright in raw X-ray)
    if remove_metal:
        img = remove_metal_artifacts(img)

    # Step 2: Invert (X-ray: bone dark -> bright to match DRR polarity)
    if invert:
        img = 1.0 - img

    # Step 3: Resize to square target_size
    img = resize(img, (target_size, target_size),
                 anti_aliasing=True, preserve_range=True)

    # Step 4: CLAHE for local contrast (paper §2.4 mentions adaptive equalization)
    img = equalize_adapthist(np.clip(img, 0, 1), clip_limit=0.03)

    return img.astype(np.float32)


if __name__ == '__main__':
    import sys; sys.path.insert(0,'src')
    from pathlib import Path
    from data_loader import discover_sessions
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sessions = discover_sessions(Path('/home/supermicro/Documents/2D_3D_Raghu/data'))
    preop    = sessions['ARJUN']['PREOP']
    gen      = DRRGenerator(preop.ct_volume, preop.ct_spacing, preop.ct_origin)

    print("Generating DRR...")
    drr = gen.generate(det_h=256,det_w=256,pix_mm=2.0,n_steps=300)
    print(f"DRR: {drr.shape}, nonzero={np.count_nonzero(drr)}, max={drr.max():.3f}")

    label_pts = np.array([c.position for c in preop.centroids])
    projs     = gen.project_labels(label_pts)
    print("Projections (u,v mm):", projs.round(1))

    fig,axes = plt.subplots(1,2,figsize=(12,6))
    dn = normalize_image(drr)
    axes[0].imshow(dn,cmap='gray',aspect='auto')
    pix = 2.0
    for i,c in enumerate(preop.centroids):
        up = projs[i,0]/pix+128; vp = -projs[i,1]/pix+128
        axes[0].plot(up,vp,'r+',ms=12)
        axes[0].annotate(c.label,(up+2,vp),color='r',fontsize=8)
    axes[0].set_title('DRR (ARJUN PREOP)')
    axes[1].imshow(normalize_image(preop.topogram),cmap='gray',aspect='auto')
    axes[1].set_title('CT Topogram (ARJUN PREOP)')
    plt.tight_layout()
    plt.savefig('/home/supermicro/Documents/2D_3D_Raghu/drr_test.png',dpi=150)
    print("Saved drr_test.png")
