"""
test_cyclegan_detector_comparison.py
=====================================
Compare the pedicle heatmap detector on:
  (A) Raw JYOTHI C-arm frames  (as tested before: AP PCK=0.444, LAT PCK=0.083)
  (B) CycleGAN-converted versions of the same frames (DRR-style)

The detector was trained on synthetic DRRs, so converting the real X-ray
toward DRR style at inference time should reduce the domain gap.

Outputs
-------
  POC/results/cyclegan_compare_AP.png   -- 2-panel: raw | converted  (AP, frame 9)
  POC/results/cyclegan_compare_LAT.png  -- 2-panel: raw | converted  (LAT, frame 14)
  POC/results/cyclegan_compare_combined.png -- 4-panel combined
"""

import sys, cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pydicom
import torch

from landmark_dataset import CLASS_NAMES, NUM_CLASSES, IMG_SIZE, HMAP_SIZE
from landmark_model import LandmarkHeatmapNet

ROOT      = Path(__file__).parent.parent
CKPT_BASE = ROOT / 'checkpoints'
CARM_BASE = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\jyothi\carm_extracted\jyothi carm\20260707\0')
CONV_DIR  = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\jyothi_data\converted')
OUT_DIR   = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')
OUT_DIR.mkdir(exist_ok=True)

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIXEL_MM  = 0.288   # corrected pixel spacing (from DICOM, not the old 0.2mm default)


# ── Image loading ─────────────────────────────────────────────────────────────

def load_raw_dicom(frame_idx: int) -> np.ndarray:
    ds  = pydicom.dcmread(str(CARM_BASE / str(frame_idx)), force=True)
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / max(hi - lo, 1.0)

def load_converted_png(frame_idx: int) -> np.ndarray:
    path = CONV_DIR / f'frame_{frame_idx:03d}_converted.png'
    img  = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


# ── Detector ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path) -> tuple:
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    model = LandmarkHeatmapNet(pretrained=False, num_classes=NUM_CLASSES,
                               dropout=0.0).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    val_pck    = ckpt.get('val_pck',    float('nan'))
    val_err_mm = ckpt.get('val_err_mm', float('nan'))
    return model, val_pck, val_err_mm


@torch.no_grad()
def detect(model, img: np.ndarray):
    """
    img  : float32 [0,1], any (H,W) — grayscale, already normalised.
           May be raw (DICOM) or CycleGAN-converted PNG.
    Returns: detections {name: [u,v]}, confidences {name: float},
             mean_conf_centroids, mean_conf_pedicles
    """
    orig_h, orig_w = img.shape[:2]

    # CLAHE  (same pre-processing as training)
    u8   = (np.clip(img, 0, 1) * 255).astype('uint8')
    cl   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
    rs   = cv2.resize(cl.astype(np.float32) / 255.0,
                      (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    t    = torch.from_numpy(rs).unsqueeze(0).unsqueeze(0).to(DEVICE)
    hmap = model(t).sigmoid()[0].cpu().numpy()  # (NUM_CLASSES, HMAP_SIZE, HMAP_SIZE)

    sx = orig_w / HMAP_SIZE
    sy = orig_h / HMAP_SIZE
    dets  = {}
    confs = {}
    for ci, name in enumerate(CLASS_NAMES):
        h    = hmap[ci]
        peak = float(h.max())
        if peak > 0.05:
            py, px = np.unravel_index(h.argmax(), h.shape)
            dets[name]  = np.array([px * sx, py * sy])
            confs[name] = peak

    c_means = [v for k, v in confs.items() if '_PED' not in k]
    p_means = [v for k, v in confs.items() if '_PED' in k]
    return (dets, confs,
            float(np.mean(c_means)) if c_means else 0.0,
            float(np.mean(p_means)) if p_means else 0.0)


# ── Visualise ─────────────────────────────────────────────────────────────────

def draw_detections(img: np.ndarray, dets: dict, confs: dict,
                    label: str, frame_idx: int) -> np.ndarray:
    h, w   = img.shape[:2]
    u8     = (np.clip(img, 0, 1) * 255).astype('uint8')
    bgr    = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    for name, uv in dets.items():
        x, y = int(uv[0]), int(uv[1])
        conf = confs.get(name, 0)
        if '_LEFT_PED' in name:
            col, r = (255, 200, 0), 5
        elif '_RIGHT_PED' in name:
            col, r = (0, 165, 255), 5
        else:
            col, r = (0, 220, 0), 7
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(bgr, (x, y), r, col, -1)
            short = name.replace('_LEFT_PED', 'LP').replace('_RIGHT_PED', 'RP')
            cv2.putText(bgr, short, (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)
    n_c  = sum(1 for n in dets if '_PED' not in n)
    n_p  = sum(1 for n in dets if '_PED' in n)
    mean_c = np.mean([v for k, v in confs.items() if '_PED' not in k]) if n_c else 0
    mean_p = np.mean([v for k, v in confs.items() if '_PED' in k])     if n_p else 0

    cv2.putText(bgr, f'Frame {frame_idx}  [{label}]',
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.putText(bgr, f'Centroid: {n_c}  conf={mean_c:.2f}',
                (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1)
    cv2.putText(bgr, f'Pedicle:  {n_p}  conf={mean_p:.2f}',
                (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1)
    return bgr


# ── Per-landmark comparison table ─────────────────────────────────────────────

def print_comparison(view: str, frame_idx: int,
                     dets_raw, confs_raw,
                     dets_conv, confs_conv):
    print(f'\n  {view} frame {frame_idx} — landmark comparison')
    print(f'  {"Landmark":<22}  {"Raw u,v":>20}  {"Raw conf":>9}  '
          f'{"Conv u,v":>20}  {"Conv conf":>9}  {"Delta conf":>10}')
    print('  ' + '-' * 96)
    all_names = sorted(set(list(dets_raw.keys()) + list(dets_conv.keys())))
    for name in all_names:
        r_uv   = dets_raw.get(name)
        c_uv   = dets_conv.get(name)
        r_conf = confs_raw.get(name, 0.0)
        c_conf = confs_conv.get(name, 0.0)
        r_str  = f'[{r_uv[0]:6.1f},{r_uv[1]:6.1f}]' if r_uv is not None else '       ---      '
        c_str  = f'[{c_uv[0]:6.1f},{c_uv[1]:6.1f}]' if c_uv is not None else '       ---      '
        delta  = c_conf - r_conf
        arrow  = '↑' if delta > 0.02 else ('↓' if delta < -0.02 else '~')
        print(f'  {name:<22}  {r_str:>20}  {r_conf:>9.3f}  '
              f'{c_str:>20}  {c_conf:>9.3f}  {delta:>+8.3f} {arrow}')

    # Summary
    improved = [n for n in all_names
                if confs_conv.get(n, 0) - confs_raw.get(n, 0) > 0.02]
    degraded = [n for n in all_names
                if confs_raw.get(n, 0)  - confs_conv.get(n, 0) > 0.02]
    print(f'\n  Improved ({len(improved)}): {", ".join(improved) if improved else "none"}')
    print(f'  Degraded ({len(degraded)}): {", ".join(degraded) if degraded else "none"}')

    # Mean confidence
    c_r = [confs_raw.get(n, 0)  for n in all_names if '_PED' not in n]
    c_c = [confs_conv.get(n, 0) for n in all_names if '_PED' not in n]
    p_r = [confs_raw.get(n, 0)  for n in all_names if '_PED' in n]
    p_c = [confs_conv.get(n, 0) for n in all_names if '_PED' in n]
    print(f'\n  Mean centroid conf:  raw={np.mean(c_r):.3f}  converted={np.mean(c_c):.3f}  '
          f'delta={np.mean(c_c)-np.mean(c_r):+.3f}')
    print(f'  Mean pedicle  conf:  raw={np.mean(p_r):.3f}  converted={np.mean(p_c):.3f}  '
          f'delta={np.mean(p_c)-np.mean(p_r):+.3f}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Device:', DEVICE)
    print('=' * 70)
    print('  JYOTHI DETECTOR: RAW vs CycleGAN-CONVERTED COMPARISON')
    print('=' * 70)

    ckpt_ap  = CKPT_BASE / 'landmark_detector_ap_pedicles'  / 'fold_JYOTHI.pth'
    ckpt_lat = CKPT_BASE / 'landmark_detector_lat_pedicles' / 'fold_JYOTHI.pth'

    model_ap,  pck_ap,  err_ap  = load_model(ckpt_ap)
    model_lat, pck_lat, err_lat = load_model(ckpt_lat)
    print(f'\nCheckpoint AP : val_PCK@10mm={pck_ap:.3f}  val_err={err_ap:.1f}mm')
    print(f'Checkpoint LAT: val_PCK@10mm={pck_lat:.3f}  val_err={err_lat:.1f}mm')
    print('(These are TRAINING val metrics; we now measure inference on converted frames)')

    # ── AP ─────────────────────────────────────────────────────────────────
    print('\n' + '-' * 70)
    print('  AP  (frame 9)')
    print('-' * 70)
    raw_ap   = load_raw_dicom(9)
    conv_ap  = load_converted_png(9)
    dets_raw_ap,  confs_raw_ap,  rc_ap, rp_ap  = detect(model_ap, raw_ap)
    dets_conv_ap, confs_conv_ap, cc_ap, cp_ap  = detect(model_ap, conv_ap)
    print(f'  Raw      detections: {len(dets_raw_ap):2d}  '
          f'(centroid conf={rc_ap:.3f}, pedicle conf={rp_ap:.3f})')
    print(f'  Converted detections: {len(dets_conv_ap):2d}  '
          f'(centroid conf={cc_ap:.3f}, pedicle conf={cp_ap:.3f})')
    print_comparison('AP', 9, dets_raw_ap, confs_raw_ap, dets_conv_ap, confs_conv_ap)

    vis_raw_ap  = draw_detections(raw_ap,  dets_raw_ap,  confs_raw_ap,  'RAW',       9)
    vis_conv_ap = draw_detections(conv_ap, dets_conv_ap, confs_conv_ap, 'CONVERTED', 9)

    # ── LAT ────────────────────────────────────────────────────────────────
    print('\n' + '-' * 70)
    print('  LAT (frame 14)')
    print('-' * 70)
    raw_lat   = load_raw_dicom(14)
    conv_lat  = load_converted_png(14)
    dets_raw_lat,  confs_raw_lat,  rc_lat, rp_lat  = detect(model_lat, raw_lat)
    dets_conv_lat, confs_conv_lat, cc_lat, cp_lat  = detect(model_lat, conv_lat)
    print(f'  Raw      detections: {len(dets_raw_lat):2d}  '
          f'(centroid conf={rc_lat:.3f}, pedicle conf={rp_lat:.3f})')
    print(f'  Converted detections: {len(dets_conv_lat):2d}  '
          f'(centroid conf={cc_lat:.3f}, pedicle conf={cp_lat:.3f})')
    print_comparison('LAT', 14, dets_raw_lat, confs_raw_lat, dets_conv_lat, confs_conv_lat)

    vis_raw_lat  = draw_detections(raw_lat,  dets_raw_lat,  confs_raw_lat,  'RAW',       14)
    vis_conv_lat = draw_detections(conv_lat, dets_conv_lat, confs_conv_lat, 'CONVERTED', 14)

    # ── Save images ────────────────────────────────────────────────────────
    def hstack_pair(a, b):
        h = max(a.shape[0], b.shape[0])
        def pad(x):
            if x.shape[0] < h:
                x = np.vstack([x, np.zeros((h-x.shape[0], x.shape[1], 3), dtype='uint8')])
            return x
        return np.hstack([pad(a), pad(b)])

    panel_ap  = hstack_pair(vis_raw_ap,  vis_conv_ap)
    panel_lat = hstack_pair(vis_raw_lat, vis_conv_lat)

    cv2.imwrite(str(OUT_DIR / 'cyclegan_compare_AP.png'),  panel_ap)
    cv2.imwrite(str(OUT_DIR / 'cyclegan_compare_LAT.png'), panel_lat)

    h = max(panel_ap.shape[0], panel_lat.shape[0])
    def pad_h(x):
        if x.shape[0] < h:
            x = np.vstack([x, np.zeros((h-x.shape[0], x.shape[1], 3), dtype='uint8')])
        return x
    combined = np.hstack([pad_h(panel_ap), pad_h(panel_lat)])
    cv2.imwrite(str(OUT_DIR / 'cyclegan_compare_combined.png'), combined)

    print('\n' + '=' * 70)
    print('  SAVED:')
    print(f'    {OUT_DIR}/cyclegan_compare_AP.png')
    print(f'    {OUT_DIR}/cyclegan_compare_LAT.png')
    print(f'    {OUT_DIR}/cyclegan_compare_combined.png')
    print('=' * 70)


if __name__ == '__main__':
    main()
