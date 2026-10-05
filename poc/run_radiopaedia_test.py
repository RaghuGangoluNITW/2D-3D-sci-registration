"""
run_radiopaedia_test.py
========================
Runs the GO retrieval + refinement pipeline on the real Radiopaedia AP
lumbar spine X-ray against the VerSe DRR bank.

This is a QUALITATIVE test only -- no PDE is possible because the
Radiopaedia X-ray is from a different anonymous patient than the VerSe CT.
What this demonstrates:
  1. GO retrieval correctly identifies AP-oriented lumbar spine poses
     from the real X-ray (same qualitative anatomy, correct orientation)
  2. The cross-domain score spread (narrow: ~6% range) explains why
     numeric convergence is harder than the synthetic test
  3. The refinement step improves GO score, showing the optimizer is
     working but without a matching CT it cannot be validated numerically
"""
import sys, json, cv2
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\src')
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\mslevel_reg\src')
import numpy as np
from pathlib import Path
from similarity import gradient_orientation_similarity
from scipy.optimize import minimize
from load_verse_ct import load_verse_case
from drr_renderer import CTVolume, DRRRenderer

BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\drr_bank')
XRAY_PATH = r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\real_xray\radiopaedia_ap_lumbar_46571.jpg'
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\doc_images_spine')
OUT_DIR.mkdir(exist_ok=True)

manifest = json.loads((BANK_DIR / 'manifest.json').read_text())
ap_entries = [e for e in manifest['entries'] if e['base_view'] == 'AP']
out_size = manifest['output_size']
pix_mm = manifest['pixel_spacing_mm']
sid_mm = manifest['sid_mm']

# Load and preprocess real X-ray
xray_raw = cv2.imread(XRAY_PATH, cv2.IMREAD_GRAYSCALE)
xray_rs = cv2.resize(xray_raw, (out_size, out_size), interpolation=cv2.INTER_AREA)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
xray = clahe.apply(xray_rs).astype(np.float32) / 255.0

# GO scores against all AP bank entries
go_scores = []
for e in ap_entries:
    drr = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    go_scores.append(gradient_orientation_similarity(drr, xray, sigma=2.0))
go_scores = np.array(go_scores)
order = np.argsort(-go_scores)

best_e = ap_entries[order[0]]
best_az = round(best_e['azimuth_deg'], 1)
best_el = round(best_e['elevation_deg'], 1)
print(f'Retrieval: best GO={go_scores.max():.4f}  Az={best_az}  El={best_el}')
print(f'Score range: {go_scores.min():.3f} - {go_scores.max():.3f}  '
      f'Mean={go_scores.mean():.3f}  SNR={go_scores.max()/go_scores.mean():.3f}')

# Load CT and refine
case = load_verse_case(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\dataset-verse19validation')
ct = CTVolume(case['volume_zyx'], case['spacing_xyz'], case['origin_xyz'], hu_threshold=180.0)
renderer = DRRRenderer(ct)
center0 = np.array(best_e['center_xyz'])

BOUNDS = np.array([20., 20., 20., 8., 8.])

def cost(x):
    if np.any(np.abs(x) > BOUNDS):
        return 1.0
    dcx, dcy, dcz, daz, del_ = x
    c = center0 + np.array([dcx, dcy, dcz])
    s, p, u, v = DRRRenderer.camera_basis_from_azimuth_elevation(
        c, sid_mm, best_e['azimuth_deg']+daz, best_e['elevation_deg']+del_)
    drr = renderer.render(s, p, u, v, output_size=out_size,
                           pixel_spacing_mm=pix_mm, n_steps=140, sdd_mm=sid_mm)
    if np.count_nonzero(drr) / drr.size < 0.1:
        return 1.0
    return 1.0 - gradient_orientation_similarity(drr, xray, sigma=2.0)

x0 = np.zeros(5)
step = np.array([3., 3., 3., 3., 3.])
simplex = np.vstack([x0] + [x0 + step[i] * np.eye(5)[i] for i in range(5)])
res = minimize(cost, x0=x0, method='Nelder-Mead',
               options=dict(xatol=0.02, fatol=1e-4, maxiter=300, maxfev=300,
                            initial_simplex=simplex, adaptive=True))

dcx, dcy, dcz, daz, del_ = res.x
c_final = center0 + np.array([dcx, dcy, dcz])
s_f, p_f, u_f, v_f = DRRRenderer.camera_basis_from_azimuth_elevation(
    c_final, sid_mm, best_e['azimuth_deg'] + daz, best_e['elevation_deg'] + del_)
drr_final = renderer.render(s_f, p_f, u_f, v_f, output_size=out_size,
                             pixel_spacing_mm=pix_mm, n_steps=200, sdd_mm=sid_mm)
final_go = gradient_orientation_similarity(drr_final, xray, sigma=2.0)

print(f'Refinement: init_GO={go_scores.max():.4f} -> final_GO={final_go:.4f}  '
      f'nit={res.nit}  cost={res.fun:.4f}')
print(f'Final Az={round(best_e["azimuth_deg"]+daz, 1)} El={round(best_e["elevation_deg"]+del_, 1)}')
print('Note: No PDE measurable -- Radiopaedia patient != VerSe CT patient')

# Save three-panel comparison
xray_big = cv2.resize((xray * 255).astype('uint8'), (400, 400))
drr_init = cv2.imread(str(BANK_DIR / best_e['file']), cv2.IMREAD_GRAYSCALE)
drr_init_big = cv2.resize(drr_init, (400, 400))
drr_fin_big = cv2.resize((np.clip(drr_final, 0, 1) * 255).astype('uint8'), (400, 400))

xray_bgr = cv2.cvtColor(xray_big, cv2.COLOR_GRAY2BGR)
init_bgr = cv2.cvtColor(drr_init_big, cv2.COLOR_GRAY2BGR)
fin_bgr = cv2.cvtColor(drr_fin_big, cv2.COLOR_GRAY2BGR)

cv2.putText(xray_bgr, 'REAL AP LUMBAR X-RAY', (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
cv2.putText(xray_bgr, '(Radiopaedia rID 46571)', (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 0), 1)

go_init_str = f'RETRIEVED DRR  GO={go_scores.max():.3f}'
az_str = f'Az={best_az} deg'
cv2.putText(init_bgr, go_init_str, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 0), 1)
cv2.putText(init_bgr, az_str, (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

go_fin_str = f'REFINED DRR  GO={final_go:.3f}'
cv2.putText(fin_bgr, go_fin_str, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1)
cv2.putText(fin_bgr, 'No PDE (diff. patient)', (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 150, 100), 1)

side = np.hstack([xray_bgr, init_bgr, fin_bgr])
cv2.imwrite(str(OUT_DIR / 'radiopaedia_full_result.png'), side)
print('Saved radiopaedia_full_result.png')

# Also save a GO score distribution bar chart as image
n_bins = 12
hist, edges = np.histogram(go_scores, bins=n_bins)
bar_w, bar_h = 60, 200
chart = np.ones((bar_h + 60, n_bins * bar_w + 40, 3), dtype='uint8') * 30
max_count = max(hist)
colors = [(80, 80, 200)] * n_bins
best_bin = int(np.digitize(go_scores.max(), edges) - 1)
colors[min(best_bin, n_bins-1)] = (0, 200, 60)

for i, (count, color) in enumerate(zip(hist, colors)):
    bar_height = int(count / max_count * (bar_h - 20))
    x1 = 20 + i * bar_w + 2
    x2 = x1 + bar_w - 4
    y1 = bar_h - bar_height + 10
    y2 = bar_h + 10
    cv2.rectangle(chart, (x1, y1), (x2, y2), color, -1)
    mid_score = (edges[i] + edges[i+1]) / 2
    cv2.putText(chart, f'{mid_score:.2f}', (x1, bar_h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)

cv2.putText(chart, 'GO Score Distribution (Real X-ray vs Synthetic DRRs)',
            (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
cv2.putText(chart, 'Green bar = best match (rank 1)',
            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 60), 1)
cv2.imwrite(str(OUT_DIR / 'radiopaedia_go_histogram.png'), chart)
print('Saved radiopaedia_go_histogram.png')
