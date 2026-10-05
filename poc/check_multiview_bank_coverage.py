import sys, json
sys.path.insert(0, '.')
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\mslevel_reg\src')
import numpy as np
from deepfluoro_loader import DeepFluoroLoader, PIXEL_SPACING_MM
from drr_renderer import CTVolume, DRRRenderer
from pathlib import Path

BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\deepfluoro_data\17-1882\drr_bank')
manifest = json.loads((BANK_DIR/'manifest.json').read_text())
entries = manifest['entries']
names = manifest['landmark_names']
sdd_mm = manifest['sdd_mm']
FULL_RES_SIZE = 1536

loader = DeepFluoroLoader(
    full_res_path=r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\deepfluoro\ipcai_2020_full_res_data.h5')
spec = loader.load_specimen('17-1882', verbose=False)
ct = CTVolume(spec.ct_volume, spec.ct_spacing, spec.ct_origin, hu_threshold=150.0)
renderer = DRRRenderer(ct)
pts3d_all = np.array([spec.landmarks_3d[n] for n in names])
center_world = pts3d_all.mean(axis=0)

p1 = [p for p in spec.valid_projections(max_reproj_px=2.0) if p.proj_key == '036'][0]
p2 = [p for p in spec.valid_projections(max_reproj_px=2.0) if p.proj_key == '045'][0]
src1, pax1 = p1.source_position(), p1.principal_axis()
u1, v1 = p1.detector_axes()
src2, pax2 = p2.source_position(), p2.principal_axis()
u2, v2 = p2.detector_axes()


def proj_at_delta(pts3d, real_src, real_pax, real_u, real_v, dR, dt):
    se = dR.T @ (real_src - dt)
    pe = dR.T @ real_pax / np.linalg.norm(dR.T @ real_pax)
    ue = dR.T @ real_u / np.linalg.norm(dR.T @ real_u)
    ve = dR.T @ real_v / np.linalg.norm(dR.T @ real_v)
    return renderer.project_points(pts3d, se, pe, ue, ve, FULL_RES_SIZE, PIXEL_SPACING_MM, sdd_mm)


pos_errs = []
for e in entries:
    ctr = np.array(e['center_xyz'])
    dt = ctr - center_world
    uv1 = proj_at_delta(pts3d_all, src1, pax1, u1, v1, np.eye(3), dt)
    uv2 = proj_at_delta(pts3d_all, src2, pax2, u2, v2, np.eye(3), dt)
    errs = []
    for real_p, uv in [(p1, uv1), (p2, uv2)]:
        for i, n in enumerate(names):
            if n in real_p.gt_landmarks_2d:
                errs.append(np.linalg.norm(uv[i] - real_p.gt_landmarks_2d[n]) * PIXEL_SPACING_MM)
    pos_errs.append(np.mean(errs))

pos_errs = np.array(pos_errs)
order = np.argsort(pos_errs)
print(f'Min achievable PDE across 500 bank entries (both views): {pos_errs.min():.1f}mm')
print(f'Mean: {pos_errs.mean():.1f}mm  Median: {np.median(pos_errs):.1f}mm')
print('Best 5 bank entries:')
for i in order[:5]:
    ctr = np.array(entries[i]['center_xyz'])
    print(f'  entry {i}: PDE={pos_errs[i]:.1f}mm  center_jitter={np.linalg.norm(ctr-center_world):.1f}mm')
print()
print('Note: if min bank PDE > ~5mm, the bank is too coarse')
print('The real centroid is at:', center_world.round(1))
print('True delta_t to reach true pose is (0,0,0) -- meaning bank needs a')
print('center entry at exactly center_world with az/el matching the real pose')
print()
print('True az(036):', round(float(np.degrees(np.arctan2(
    (src1-center_world)[0], (src1-center_world)[1]))), 1))
print('True az(045):', round(float(np.degrees(np.arctan2(
    (src2-center_world)[0], (src2-center_world)[1]))), 1))
