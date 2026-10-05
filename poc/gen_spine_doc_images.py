"""Generate all supporting images for the VerSe spine regression document."""
import sys, json, cv2
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\src')
import numpy as np
from pathlib import Path
from load_verse_ct import load_verse_case
from drr_renderer import CTVolume, DRRRenderer

CASE_DIR = r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\dataset-verse19validation'
BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\drr_bank')
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\doc_images_spine')
RES_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')
OUT_DIR.mkdir(exist_ok=True)

case = load_verse_case(CASE_DIR)
vol = case['volume_zyx']  # (Z,Y,X)
centroids = case['centroids_world']
lm_names = sorted(centroids.keys())
pts = np.array([centroids[n] for n in lm_names])
center = pts.mean(axis=0)
ox, oy, oz = case['origin_xyz']
sx, sy, sz = case['spacing_xyz']
ct = CTVolume(vol, case['spacing_xyz'], case['origin_xyz'], hu_threshold=180.0)
renderer = DRRRenderer(ct)

# Centroid voxel index
cx = int((center[0]-ox)/sx); cy = int((center[1]-oy)/sy); cz = int((center[2]-oz)/sz)
cx = max(0, min(cx, vol.shape[2]-1))
cy = max(0, min(cy, vol.shape[1]-1))
cz = max(0, min(cz, vol.shape[0]-1))


def to_u8(s):
    lo, hi = np.percentile(s, [0.5, 99.5])
    return ((np.clip(s.astype(float)-lo, 0, hi-lo) / max(hi-lo, 1)) * 255).astype('uint8')


def pad_square_resize(slc, size=400):
    h, w = slc.shape
    side = max(h, w)
    pad = np.zeros((side, side), dtype='uint8')
    pad[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = slc
    return cv2.resize(pad, (size, size))


# ── CT slices ────────────────────────────────────────────────────────────────
# Use slice where vertebral bodies are most visible
axial_z = cz
best_content = 0
for dz in range(-15, 16, 3):
    z = max(0, min(cz+dz, vol.shape[0]-1))
    s = to_u8(vol[z, :, :])
    c = np.count_nonzero(s > 60) / s.size
    if c > best_content:
        best_content = c; axial_z = z
axial = to_u8(vol[axial_z, :, :])
coronal = to_u8(vol[:, cy, :])
sagittal = to_u8(vol[:, :, cx])

for name, slc, lbl in [
    ('ct_axial',   axial,   'AXIAL SLICE   (VerSe sub-verse010)'),
    ('ct_coronal', coronal, 'CORONAL SLICE (VerSe sub-verse010)'),
    ('ct_sagittal',sagittal,'SAGITTAL SLICE(VerSe sub-verse010)'),
]:
    img = pad_square_resize(slc, 400)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, lbl, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
    cv2.imwrite(str(OUT_DIR / (name+'.png')), bgr)
    print('Saved', name)

# ── DRR bank samples ──────────────────────────────────────────────────────────
manifest = json.loads((BANK_DIR / 'manifest.json').read_text())
entries = manifest['entries']
ap_entries = [e for e in entries if e['base_view'] == 'AP']
lat_entries = [e for e in entries if e['base_view'] == 'LAT']

ap_azimuths = [-18, -8, 0, 8, 18]
lat_azimuths = [72, 82, 90, 98, 108]

ap_imgs = []
for az_t in ap_azimuths:
    idx = int(np.argmin([abs(e['azimuth_deg']-az_t) for e in ap_entries]))
    e = ap_entries[idx]
    img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    az_val = round(e['azimuth_deg'], 0)
    cv2.putText(bgr, f'AP Az={az_val:.0f}', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    ap_imgs.append(bgr)
cv2.imwrite(str(OUT_DIR / 'drr_ap_samples.png'), np.hstack(ap_imgs))
print('Saved drr_ap_samples')

lat_imgs = []
for az_t in lat_azimuths:
    idx = int(np.argmin([abs(e['azimuth_deg']-az_t) for e in lat_entries]))
    e = lat_entries[idx]
    img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    az_val = round(e['azimuth_deg'], 0)
    cv2.putText(bgr, f'LAT Az={az_val:.0f}', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    lat_imgs.append(bgr)
cv2.imwrite(str(OUT_DIR / 'drr_lat_samples.png'), np.hstack(lat_imgs))
print('Saved drr_lat_samples')

# ── AP DRR with landmark overlay ─────────────────────────────────────────────
e0 = ap_entries[int(np.argmin([abs(e['azimuth_deg']) for e in ap_entries]))]
img = cv2.imread(str(BANK_DIR / e0['file']), cv2.IMREAD_GRAYSCALE)
lg = cv2.resize(img, (400, 400))
bgr = cv2.cvtColor(lg, cv2.COLOR_GRAY2BGR)
scale = 400 / 256
for n in lm_names:
    x, y = e0['landmarks_px'][n]
    xi, yi = int(x*scale), int(y*scale)
    cv2.circle(bgr, (xi, yi), 7, (0, 255, 0), -1)
    cv2.putText(bgr, n, (xi+8, yi+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(bgr, 'AP DRR with Auto-Computed Landmarks', (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
cv2.imwrite(str(OUT_DIR / 'drr_ap_with_landmarks.png'), bgr)
print('Saved drr_ap_with_landmarks')

# ── LAT DRR with landmark overlay ────────────────────────────────────────────
e0l = lat_entries[int(np.argmin([abs(e['azimuth_deg']-90) for e in lat_entries]))]
img = cv2.imread(str(BANK_DIR / e0l['file']), cv2.IMREAD_GRAYSCALE)
lg = cv2.resize(img, (400, 400))
bgr = cv2.cvtColor(lg, cv2.COLOR_GRAY2BGR)
for n in lm_names:
    x, y = e0l['landmarks_px'][n]
    xi, yi = int(x*scale), int(y*scale)
    cv2.circle(bgr, (xi, yi), 7, (0, 200, 255), -1)
    cv2.putText(bgr, n, (xi+8, yi+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
cv2.putText(bgr, 'LAT DRR with Auto-Computed Landmarks', (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
cv2.imwrite(str(OUT_DIR / 'drr_lat_with_landmarks.png'), bgr)
print('Saved drr_lat_with_landmarks')

# ── Pipeline flow diagram ────────────────────────────────────────────────────
flow = np.ones((280, 950, 3), dtype='uint8') * 25
steps = [
    ('STEP 1', 'Pre-op CT', 'Landmark Annotation', (60, 120, 220)),
    ('STEP 2', 'DRR Bank', '300 AP+LAT Synthetic X-rays', (80, 180, 100)),
    ('STEP 3', 'Two Target Views', 'AP + LAT DRRs (Synthetic)', (200, 80, 60)),
    ('STEP 4', 'Joint GO Retrieval', 'Top-10 Multi-Start Matches', (180, 140, 50)),
    ('STEP 5', 'Joint Refinement', 'AP+LAT Nelder-Mead Result', (50, 180, 180)),
]
for i, (s1, s2, s3, col) in enumerate(steps):
    x = 50 + i*185; y = 30; bw = 155; bh = 90
    cv2.rectangle(flow, (x, y), (x+bw, y+bh), col, -1)
    cv2.rectangle(flow, (x, y), (x+bw, y+bh), (255, 255, 255), 1)
    cv2.putText(flow, s1, (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(flow, s2, (x+5, y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)
    cv2.putText(flow, s3, (x+5, y+68), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)
    if i < 4:
        cv2.arrowedLine(flow, (x+bw+3, y+45), (x+bw+55, y+45), (180, 180, 180), 2, tipLength=0.35)

cv2.rectangle(flow, (20, 150), (930, 270), (40, 40, 60), -1)
cv2.rectangle(flow, (20, 150), (930, 270), (50, 180, 180), 2)
cv2.putText(flow, 'RESULT:  Mean PDE = 0.32mm  (AP=0.52mm, LAT=0.11mm)  --- TARGET <2mm ACHIEVED',
            (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 230, 50), 2)
cv2.putText(flow, 'Sub-2mm on 7/10 starts  |  AP+LAT (89 deg sep)  |  VerSe sub-verse010  |  L1-L5 landmarks',
            (30, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
cv2.putText(flow, 'Zero manual intervention on X-ray images  |  Pre-op CT annotation only  |  Synthetic DRR targets',
            (30, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 180), 1)
cv2.imwrite(str(OUT_DIR / 'pipeline_flow_spine.png'), flow)
print('Saved pipeline_flow_spine')

# ── Copy+enlarge result images ─────────────────────────────────────────────────
for fname in ['verse_multiview_result_AP.png',
              'verse_multiview_result_LAT.png',
              'verse_multiview_combined_result.png',
              'verse_multiview_target_ap.png',
              'verse_multiview_target_lat.png']:
    src = RES_DIR / fname
    if src.exists():
        img = cv2.imread(str(src))
        if img is not None:
            h, w = img.shape[:2]
            factor = 3 if h <= 256 else 2
            large = cv2.resize(img, (w*factor, h*factor), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(OUT_DIR / fname), large)
            print(f'Copied+enlarged {fname}')

print('\nAll spine doc images ready in', OUT_DIR)
