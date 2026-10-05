import sys, json, cv2
sys.path.insert(0, '.')
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\mslevel_reg\src')
import numpy as np
from deepfluoro_loader import DeepFluoroLoader, PIXEL_SPACING_MM
from drr_renderer import CTVolume, DRRRenderer
from pathlib import Path

BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\deepfluoro_data\17-1882\drr_bank')
manifest = json.loads((BANK_DIR/'manifest.json').read_text())
names = manifest['landmark_names']
out_size = manifest['output_size']
pix_mm_bank = manifest['pixel_spacing_mm']
sdd_mm = manifest['sdd_mm']
FULL_RES_SIZE = 1536
real_pix_mm = PIXEL_SPACING_MM

loader = DeepFluoroLoader(
    full_res_path=r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\deepfluoro\ipcai_2020_full_res_data.h5')
spec = loader.load_specimen('17-1882', verbose=False)
ct = CTVolume(spec.ct_volume, spec.ct_spacing, spec.ct_origin, hu_threshold=150.0)
renderer = DRRRenderer(ct)
pts3d_all = np.array([spec.landmarks_3d[n] for n in names])

p1 = [p for p in spec.valid_projections(max_reproj_px=2.0) if p.proj_key == '036'][0]
p2 = [p for p in spec.valid_projections(max_reproj_px=2.0) if p.proj_key == '045'][0]
src1, pax1 = p1.source_position(), p1.principal_axis()
u1, v1 = p1.detector_axes()
src2, pax2 = p2.source_position(), p2.principal_axis()
u2, v2 = p2.detector_axes()

entries = manifest['entries']

# Find bank entry closest to proj 036 by source position
dists = [np.linalg.norm(np.array(e['src_xyz']) - src1) for e in entries]
best_idx = int(np.argmin(dists))
best = entries[best_idx]
best_src = np.array(best['src_xyz'])
best_center = np.array(best['center_xyz'])
center_real = pts3d_all.mean(axis=0)

print('Real src1 (036):', src1.round(1))
print('Bank src closest:', best_src.round(1), 'dist:', round(dists[best_idx], 1))
print('Bank center:', best_center.round(1), 'real center:', center_real.round(1))

# Test 1: delta = identity (zero displacement) -- should work exactly for real camera
def project_at_delta(pts3d, real_src, real_pax, real_u, real_v, delta_R, delta_t):
    src_eff = delta_R.T @ (real_src - delta_t)
    pax_eff = delta_R.T @ real_pax
    u_eff = delta_R.T @ real_u
    v_eff = delta_R.T @ real_v
    return renderer.project_points(pts3d, src_eff, pax_eff, u_eff, v_eff,
                                    FULL_RES_SIZE, real_pix_mm, sdd_mm)

uv = project_at_delta(pts3d_all, src1, pax1, u1, v1, np.eye(3), np.zeros(3))
errs = [np.linalg.norm(uv[i]-p1.gt_landmarks_2d[n])*real_pix_mm
        for i,n in enumerate(names) if n in p1.gt_landmarks_2d]
print('delta=I,0 PDE (036):', round(np.mean(errs), 4), 'mm')

uv2 = project_at_delta(pts3d_all, src2, pax2, u2, v2, np.eye(3), np.zeros(3))
errs2 = [np.linalg.norm(uv2[i]-p2.gt_landmarks_2d[n])*real_pix_mm
         for i,n in enumerate(names) if n in p2.gt_landmarks_2d]
print('delta=I,0 PDE (045):', round(np.mean(errs2), 4), 'mm')

# Test 2: use the bank entry's own R_init/t_init (from camera basis) as delta
R_init = np.stack([np.array(best['u_xyz']), np.array(best['v_xyz']), np.array(best['pax_xyz'])], axis=0)
t_init = -R_init @ best_src
uv_b = project_at_delta(pts3d_all, src1, pax1, u1, v1, R_init, t_init)
errs_b = [np.linalg.norm(uv_b[i]-p1.gt_landmarks_2d[n])*real_pix_mm
          for i,n in enumerate(names) if n in p1.gt_landmarks_2d]
print('delta=R_init,t_init PDE (036):', round(np.mean(errs_b), 1), 'mm  <-- BAD if huge')

# Test 3: delta_t as the translation offset between bank jittered center and real center
delta_t_center = best_center - center_real
uv_c = project_at_delta(pts3d_all, src1, pax1, u1, v1, np.eye(3), delta_t_center)
errs_c = [np.linalg.norm(uv_c[i]-p1.gt_landmarks_2d[n])*real_pix_mm
          for i,n in enumerate(names) if n in p1.gt_landmarks_2d]
print('delta=I,center_offset PDE (036):', round(np.mean(errs_c), 1), 'mm  <-- SHOULD be small')

# The correct interpretation: delta describes the CT PATIENT POSITION relative
# to an assumed 'default' reference. Since this renderer's CT always sits at
# spec.ct_origin with spec.ct_spacing, the only translation that matters is
# the difference in where the C-arm was centered.
# delta = (0, center_jitter) in the multi-view test should give us the right init.
