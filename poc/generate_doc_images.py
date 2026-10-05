"""
generate_doc_images.py
========================
Generates all supporting images needed for the pipeline regression
document: CT slices, DRR bank samples, real C-arm samples, and the
final multi-view registration result composite.
"""
import sys, json, cv2
sys.path.insert(0, '.')
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\mslevel_reg\src')
import numpy as np
from pathlib import Path
from deepfluoro_loader import DeepFluoroLoader, PIXEL_SPACING_MM
from drr_renderer import CTVolume, DRRRenderer

H5 = r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\deepfluoro\ipcai_2020_full_res_data.h5'
BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\deepfluoro_data\17-1882\drr_bank')
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\doc_images')
OUT_DIR.mkdir(exist_ok=True, parents=True)

loader = DeepFluoroLoader(full_res_path=H5)
spec = loader.load_specimen('17-1882', verbose=False)
ct = CTVolume(spec.ct_volume, spec.ct_spacing, spec.ct_origin, hu_threshold=150.0)
renderer = DRRRenderer(ct)
names, pts = spec.get_landmark_array()
center = pts.mean(axis=0)


# ── 1. CT SLICES ─────────────────────────────────────────────────────────────
# The CT volume is (Z,Y,X) -- pick the axial, coronal and sagittal slices
# nearest the pelvis centroid.
vol = spec.ct_volume  # (Z, Y, X)
# Convert centroid to voxel indices
ox, oy, oz = spec.ct_origin
sx, sy, sz = spec.ct_spacing
cx_vox = int((center[0] - ox) / sx)
cy_vox = int((center[1] - oy) / sy)
cz_vox = int((center[2] - oz) / sz)
cx_vox = max(0, min(cx_vox, vol.shape[2]-1))
cy_vox = max(0, min(cy_vox, vol.shape[1]-1))
cz_vox = max(0, min(cz_vox, vol.shape[0]-1))

def ct_slice_to_u8(arr2d):
    lo, hi = np.percentile(arr2d, [0.5, 99.5])
    return np.clip((arr2d.astype(np.float32) - lo) / max(hi - lo, 1e-5), 0, 1)

# Axial (z-plane through centroid)
axial = ct_slice_to_u8(vol[cz_vox, :, :])
# Coronal (y-plane through centroid)
coronal = ct_slice_to_u8(vol[:, cy_vox, :])
# Sagittal (x-plane through centroid)
sagittal = ct_slice_to_u8(vol[:, :, cx_vox])

for name, slc in [('ct_axial', axial), ('ct_coronal', coronal), ('ct_sagittal', sagittal)]:
    h, w = slc.shape
    # Pad or crop to square
    side = max(h, w)
    pad = np.zeros((side, side), dtype=np.float32)
    y0 = (side - h) // 2; x0 = (side - w) // 2
    pad[y0:y0+h, x0:x0+w] = slc
    # Resize to 400x400 for doc
    img = cv2.resize((pad * 255).astype('uint8'), (400, 400), interpolation=cv2.INTER_LINEAR)
    # Add label
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lbl = name.replace('ct_', '').upper() + ' SLICE'
    cv2.putText(img_bgr, lbl, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imwrite(str(OUT_DIR / f'{name}.png'), img_bgr)
    print(f'Saved {name}.png')

# ── 2. DRR BANK SAMPLES ───────────────────────────────────────────────────────
manifest = json.loads((BANK_DIR / 'manifest.json').read_text())
all_entries = manifest['entries']
sdd_mm = manifest['sdd_mm']
pix_mm_bank = manifest['pixel_spacing_mm']
out_size = manifest['output_size']

# Pick 4 representative entries spanning the azimuth range
# and 1 LAT-like (high elevation)
azimuths_target = [-20, -5, 10, 25]
sample_indices = []
for az_t in azimuths_target:
    best_i = int(np.argmin([abs(e['azimuth_deg'] - az_t) for e in all_entries]))
    sample_indices.append(best_i)
# also one high-elevation
elev_best = int(np.argmax([e['elevation_deg'] for e in all_entries]))
sample_indices.append(elev_best)

drr_imgs = []
for idx in sample_indices:
    e = all_entries[idx]
    drr_path = BANK_DIR / e['file']
    drr_img = cv2.imread(str(drr_path), cv2.IMREAD_GRAYSCALE)
    az = e['azimuth_deg']; el = e['elevation_deg']
    drr_bgr = cv2.cvtColor(drr_img, cv2.COLOR_GRAY2BGR)
    lbl = f'Az={az:.0f}  El={el:.0f}'
    cv2.putText(drr_bgr, lbl, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    drr_imgs.append(drr_bgr)

# Make a strip
strip = np.hstack(drr_imgs)
cv2.imwrite(str(OUT_DIR / 'drr_bank_samples.png'), strip)
print('Saved drr_bank_samples.png')

# One large single DRR (AP-like, az~0)
ap_idx = int(np.argmin([abs(e['azimuth_deg'] - 0) + abs(e['elevation_deg'] - 5) for e in all_entries]))
drr_ap = cv2.imread(str(BANK_DIR / all_entries[ap_idx]['file']), cv2.IMREAD_GRAYSCALE)
drr_ap_large = cv2.resize(drr_ap, (500, 500))
drr_ap_bgr = cv2.cvtColor(drr_ap_large, cv2.COLOR_GRAY2BGR)
cv2.putText(drr_ap_bgr, f'SYNTHETIC DRR  (Az={all_entries[ap_idx]["azimuth_deg"]:.0f})  From CT',
            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
cv2.imwrite(str(OUT_DIR / 'drr_ap_sample.png'), drr_ap_bgr)
print('Saved drr_ap_sample.png')

# ── 3. REAL C-ARM SAMPLES ─────────────────────────────────────────────────────
# Use some of the real whole-pelvis projections
import h5py
with h5py.File(H5, 'r') as f:
    pelvis_keys = [pk for pk, pg in f['17-1882']['projections'].items()
                   if int(pg['gt-poses']['left-femur-good-fov'][()]) == 0
                   and int(pg['gt-poses']['right-femur-good-fov'][()]) == 0]

pelvis_projs = {p.proj_key: p for p in spec.valid_projections(max_reproj_px=2.0)
                if p.proj_key in pelvis_keys}

# Pick 4 spanning az range
carm_keys = ['045', '013', '032', '036']
carm_imgs = []
for pk in carm_keys:
    if pk in pelvis_projs:
        p = pelvis_projs[pk]
        img = (np.clip(p.image_raw, 0, 1) * 255).astype('uint8')
        img_rs = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        img_bgr = cv2.cvtColor(img_rs, cv2.COLOR_GRAY2BGR)
        src = p.source_position(); rel = src - center; d = np.linalg.norm(rel)
        az = np.degrees(np.arctan2(rel[0], rel[1]))
        cv2.putText(img_bgr, f'{pk}  Az={az:.0f}', (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        carm_imgs.append(img_bgr)

carm_strip = np.hstack(carm_imgs)
cv2.imwrite(str(OUT_DIR / 'carm_samples.png'), carm_strip)
print('Saved carm_samples.png')

# One large individual C-arm (036, the +29 deg oblique used in the result)
p036 = pelvis_projs.get('036')
if p036:
    img = (np.clip(p036.image_raw, 0, 1) * 255).astype('uint8')
    img_rs = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
    img_bgr = cv2.cvtColor(img_rs, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_bgr, 'REAL C-ARM (Proj 036, Az=+29)',
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.imwrite(str(OUT_DIR / 'carm_036_large.png'), img_bgr)

p045 = pelvis_projs.get('045')
if p045:
    img = (np.clip(p045.image_raw, 0, 1) * 255).astype('uint8')
    img_rs = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
    img_bgr = cv2.cvtColor(img_rs, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_bgr, 'REAL C-ARM (Proj 045, Az=-18)',
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.imwrite(str(OUT_DIR / 'carm_045_large.png'), img_bgr)
    print('Saved carm large images')

# ── 4. FINAL REGISTRATION RESULT ─────────────────────────────────────────────
# Already saved in results folder — copy/enhance them
from shutil import copy2
for fname in ['deepfluoro_multiview_036_045_view1.png',
              'deepfluoro_multiview_036_045_view2.png']:
    src_path = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results') / fname
    if src_path.exists():
        img = cv2.imread(str(src_path))
        if img is not None:
            # Make larger for doc
            h, w = img.shape[:2]
            img_big = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(OUT_DIR / fname), img_big)
            print(f'Saved {fname} (enlarged)')

# ── 5. PIPELINE DIAGRAM COMPOSITE ────────────────────────────────────────────
# Build a 5-step pipeline flow image showing the stages
flow_h, flow_w = 200, 900
flow = np.ones((flow_h, flow_w, 3), dtype='uint8') * 30

steps = [
    (70, 'STEP 1', 'Pre-op CT', '3D Volume'),
    (220, 'STEP 2', 'DRR Bank', '500 Synthetic X-rays'),
    (370, 'STEP 3', 'Intra-op C-arm', 'Real Fluoroscopy'),
    (520, 'STEP 4', 'GO Retrieval', 'Top-10 Matches'),
    (700, 'STEP 5', 'Refinement', 'Joint 2-View Optim'),
]

flow_tall = np.ones((320, 1100, 3), dtype='uint8') * 25
# Draw boxes
colors = [(60,120,220),(80,180,100),(200,80,60),(180,140,50),(50,180,180)]
for i, (x, s1, s2, s3) in enumerate(steps):
    x -= 60; y_top = 30; bw = 130; bh = 100
    cv2.rectangle(flow_tall, (x, y_top), (x+bw, y_top+bh), colors[i], -1)
    cv2.rectangle(flow_tall, (x, y_top), (x+bw, y_top+bh), (255,255,255), 1)
    cv2.putText(flow_tall, s1, (x+6, y_top+22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
    cv2.putText(flow_tall, s2, (x+6, y_top+48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.putText(flow_tall, s3, (x+6, y_top+70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
    if i < 4:
        cv2.arrowedLine(flow_tall, (x+bw+4, y_top+50), (x+bw+66, y_top+50), (180,180,180), 2, tipLength=0.3)

# PDE result box
cv2.rectangle(flow_tall, (20, 160), (1080, 300), (40,40,60), -1)
cv2.rectangle(flow_tall, (20, 160), (1080, 300), (50,180,180), 2)
cv2.putText(flow_tall, 'RESULT: 4.3 mm PDE (under 5 mm target)', (40, 210),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,230,50), 2)
cv2.putText(flow_tall, 'Pair 036+045 (az=+29/-18 deg, 47 deg separation)  |  Specimen 17-1882', (40, 248),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
cv2.putText(flow_tall, 'Zero manual clicks during intra-operative phase  |  CT labels pre-operative only', (40, 278),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,220,180), 1)

cv2.imwrite(str(OUT_DIR / 'pipeline_flow.png'), flow_tall)
print('Saved pipeline_flow.png')
print('\nAll doc images generated in:', OUT_DIR)
