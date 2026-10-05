"""
sanity_check_deepfluoro.py
=============================
Verifies the DeepFluoro loader + DRRRenderer camera convention by
rendering a DRR directly from the known ground-truth pose of a real
projection, and comparing it visually to the real fluoroscopy image.
If this looks right, the renderer/camera math checks out before we
build the full retrieval bank.

DeepFluoro's own pose convention (from the main project's
deepfluoro_loader.py, cross-checked here independently):
  pt_xzy = pt_world[[0, 2, 1]]      (swap Y<->Z)
  P_cam  = R_proj @ pt_xzy + t_proj
  u = FX_POS * P_cam[0]/P_cam[2] + CX
  src_world = (R_proj.T @ (-t_proj))[[0,2,1]]   (undo the swap)
  principal_axis_world = (R_proj.T @ [0,0,1])[[0,2,1]]
  detector u-axis = (R_proj.T @ [1,0,0])[[0,2,1]]
  detector v-axis = (R_proj.T @ [0,1,0])[[0,2,1]]
"""
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'mslevel_reg' / 'src'))
from deepfluoro_loader import DeepFluoroLoader, FX_POS, CX, CY, PIXEL_SPACING_MM
from load_deepfluoro import H5_PATH
from drr_renderer import CTVolume, DRRRenderer

OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')
OUT_DIR.mkdir(exist_ok=True)


def main():
    loader = DeepFluoroLoader(full_res_path=H5_PATH)
    spec = loader.load_specimen('17-1882', max_projections=1, verbose=True)
    proj = spec.projections[0]
    print(proj)

    # DeepFluoro volume is already (Z,Y,X), matching CTVolume's expectation
    # (confirmed: deepfluoro_drr.py's explicit comment + this sanity check
    # itself -- a WRONG transpose here previously produced a scrambled,
    # visually-incorrect DRR despite a geometrically correct camera pose).
    ct = CTVolume(spec.ct_volume, spec.ct_spacing, spec.ct_origin, hu_threshold=150.0)
    renderer = DRRRenderer(ct)

    R_proj, t_proj = proj.R_proj, proj.t_proj
    src = proj.source_position()
    pax = proj.principal_axis()
    u_axis, v_axis = proj.detector_axes()

    print('src', src.round(1), 'pax', pax.round(3))

    drr = renderer.render(src, pax, u_axis, v_axis, output_size=384,
                           pixel_spacing_mm=PIXEL_SPACING_MM * (1536 / 384),
                           n_steps=250, sdd_mm=FX_POS * PIXEL_SPACING_MM)

    xray_small = cv2.resize(proj.image_raw, (384, 384), interpolation=cv2.INTER_AREA)
    drr_u8 = (np.clip(drr, 0, 1) * 255).astype('uint8')
    xray_u8 = (np.clip(xray_small, 0, 1) * 255).astype('uint8')
    side = np.hstack([xray_u8, drr_u8])
    cv2.imwrite(str(OUT_DIR / 'deepfluoro_sanity_check.png'), side)
    print('saved comparison. DRR coverage:', np.count_nonzero(drr) / drr.size)


if __name__ == '__main__':
    main()
