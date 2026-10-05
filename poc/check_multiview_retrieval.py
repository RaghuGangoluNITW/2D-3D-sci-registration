import sys, json, cv2
sys.path.insert(0, '.')
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\mslevel_reg\src')
import numpy as np
from similarity import gradient_orientation_similarity
from deepfluoro_loader import DeepFluoroLoader, PIXEL_SPACING_MM
from drr_renderer import CTVolume, DRRRenderer
from pathlib import Path

BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\deepfluoro_data\17-1882\drr_bank')
manifest = json.loads((BANK_DIR/'manifest.json').read_text())
entries = manifest['entries']
names = manifest['landmark_names']
out_size = manifest['output_size']
pix_mm_bank = manifest['pixel_spacing_mm']
sdd_mm = manifest['sdd_mm']
FULL_RES_SIZE = 1536

loader = DeepFluoroLoader(
    full_res_path=r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\deepfluoro\ipcai_2020_full_res_data.h5')
spec = loader.load_specimen('17-1882', verbose=False)
ct = CTVolume(spec.ct_volume, spec.ct_spacing, spec.ct_origin, hu_threshold=150.0)
renderer = DRRRenderer(ct)
pts3d_all = np.array([spec.landmarks_3d[n] for n in names])
center_world = pts3d_all.mean(axis=0)


def preprocess_xray(xray_raw, sz):
    u8 = (np.clip(xray_raw, 0, 1) * 255).astype('uint8')
    resized = cv2.resize(u8, (sz, sz), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(resized).astype(np.float32) / 255.0


p1 = [p for p in spec.valid_projections(max_reproj_px=2.0) if p.proj_key == '036'][0]
p2 = [p for p in spec.valid_projections(max_reproj_px=2.0) if p.proj_key == '045'][0]

t1 = preprocess_xray(p1.image_raw, out_size)
t2 = preprocess_xray(p2.image_raw, out_size)

go_joint = []
for e in entries:
    drr_img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    g1 = gradient_orientation_similarity(drr_img, t1, sigma=2.0)
    g2 = gradient_orientation_similarity(drr_img, t2, sigma=2.0)
    go_joint.append(g1 + g2)

go_joint = np.array(go_joint)
order_go = np.argsort(-go_joint)

# Best GO entry
bi = order_go[0]
print(f'Best GO entry: idx={bi}  joint_GO={go_joint[bi]:.4f}')
# rank of the known-good entries (1.9mm = entry 412, 2.6mm = entry 145)
for good_idx, pos_mm in [(412, 1.9), (145, 2.6), (219, 4.1)]:
    rank = int(np.where(order_go == good_idx)[0][0])
    print(f'  Entry {good_idx} (pos={pos_mm}mm) ranked #{rank}/500 by GO, GO={go_joint[good_idx]:.4f}')

print()
print('Top 5 by joint GO:')
for i in order_go[:5]:
    print(f'  idx={i}  GO={go_joint[i]:.4f}')

# Compute true pos error for top 5 GO entries using correct delta
def proj_at_delta(pts3d, rp, dR, dt):
    src, pax = rp.source_position(), rp.principal_axis()
    u, v = rp.detector_axes()
    se = dR.T @ (src - dt); pe = dR.T @ pax / np.linalg.norm(dR.T @ pax)
    ue = dR.T @ u / np.linalg.norm(dR.T @ u); ve = dR.T @ v / np.linalg.norm(dR.T @ v)
    return renderer.project_points(pts3d, se, pe, ue, ve, FULL_RES_SIZE, PIXEL_SPACING_MM, sdd_mm)

for i in order_go[:10]:
    ctr = np.array(entries[i]['center_xyz'])
    dt = ctr - center_world
    uv1 = proj_at_delta(pts3d_all, p1, np.eye(3), dt)
    uv2 = proj_at_delta(pts3d_all, p2, np.eye(3), dt)
    errs = []
    for rp, uv in [(p1, uv1), (p2, uv2)]:
        errs += [np.linalg.norm(uv[j] - rp.gt_landmarks_2d[n]) * PIXEL_SPACING_MM
                 for j, n in enumerate(names) if n in rp.gt_landmarks_2d]
    print(f'  GO rank {int(np.where(order_go==i)[0][0])}: entry {i}  GO={go_joint[i]:.4f}  pos_err={np.mean(errs):.1f}mm')
