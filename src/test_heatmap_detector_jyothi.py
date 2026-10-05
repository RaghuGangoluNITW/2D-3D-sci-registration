"""
test_heatmap_detector_jyothi.py
=================================
Tests the pre-trained heatmap landmark detector (pedicle-aware) on
JYOTHI's raw C-arm frames. This is the key missing piece: detecting
2D pedicle projections in real X-rays to match the 3D pedicle tips
already auto-extracted from the CT.

Checkpoints used:
  AP:  checkpoints/landmark_detector_ap_pedicles/fold_JYOTHI.pth
  LAT: checkpoints/landmark_detector_lat_pedicles/fold_JYOTHI.pth
"""
import sys, cv2, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pydicom
import torch
from landmark_dataset import CLASS_NAMES, NUM_CLASSES, IMG_SIZE, HMAP_SIZE
from landmark_model import LandmarkHeatmapNet

CKPT_BASE = Path(__file__).parent.parent / 'checkpoints'
CARM_BASE = r'C:\Users\rgangolu\OneDrive - Infor\analysis\jyothi\carm_extracted\jyothi carm\20260707\0'
OUT_DIR   = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)


def load_raw_frame(frame_idx, out_size=None):
    ds = pydicom.dcmread(f'{CARM_BASE}\\{frame_idx}', force=True)
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1.0)
    if out_size:
        arr = cv2.resize(arr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return arr


def run_detector(ckpt_path, xray_float32):
    """Run the heatmap detector and return {name: [u,v]} for all detected landmarks."""
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    model = LandmarkHeatmapNet(pretrained=False, num_classes=NUM_CLASSES, dropout=0.0).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    val_pck = ckpt.get('val_pck', float('nan'))
    print(f'  Checkpoint: {ckpt_path.name}  val_PCK@10mm={val_pck:.3f}')

    orig_h, orig_w = xray_float32.shape[:2]
    img = xray_float32.copy()
    img_u8 = (np.clip(img, 0, 1)*255).astype('uint8')
    img_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img_u8).astype(np.float32)/255.
    img_rs = cv2.resize(img_clahe, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        hmap = logits.sigmoid()[0].cpu().numpy()  # (NUM_CLASSES, HMAP_SIZE, HMAP_SIZE)

    sx = orig_w / HMAP_SIZE
    sy = orig_h / HMAP_SIZE

    detections = {}
    confidences = {}
    for ci, name in enumerate(CLASS_NAMES):
        h = hmap[ci]
        peak_val = float(h.max())
        if peak_val > 0.05:
            py, px = np.unravel_index(h.argmax(), h.shape)
            u = float(px) * sx
            v = float(py) * sy
            detections[name] = np.array([u, v])
            confidences[name] = peak_val

    return detections, confidences, hmap


def visualize(frame_idx, xray_float32, detections, confidences, tag):
    h, w = xray_float32.shape[:2]
    img_u8 = (np.clip(xray_float32, 0, 1)*255).astype('uint8')
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    # Color code: centroids=green, left pedicles=cyan, right pedicles=orange
    for name, uv in detections.items():
        x, y = int(uv[0]), int(uv[1])
        conf = confidences.get(name, 0)
        if '_LEFT_PED' in name:
            col = (255, 200, 0)  # cyan
            r = 6
        elif '_RIGHT_PED' in name:
            col = (0, 165, 255)  # orange
            r = 6
        else:
            col = (0, 255, 0)    # green
            r = 8
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(bgr, (x, y), r, col, -1)
            cv2.putText(bgr, f'{name.split("_")[0]}', (x+4, y-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)

    # Legend
    cv2.putText(bgr, f'Frame {frame_idx}  {tag}', (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.putText(bgr, f'Detected: {len(detections)} landmarks', (5, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(bgr, 'GREEN=centroid  CYAN=L_ped  ORANGE=R_ped', (5, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    out_path = OUT_DIR / f'jyothi_heatmap_detect_{tag}.png'
    cv2.imwrite(str(out_path), bgr)
    print(f'  Saved: {out_path.name}')
    return bgr


def main():
    print('='*60)
    print('  JYOTHI HEATMAP DETECTOR TEST — AP + LAT Pedicles')
    print('='*60)

    # ── AP detection on frame 9 ──────────────────────────────────────
    print('\nAP detection (frame 9, positioner=-19 deg)...')
    ap_frame = load_raw_frame(9)
    ckpt_ap = CKPT_BASE / 'landmark_detector_ap_pedicles' / 'fold_JYOTHI.pth'
    dets_ap, confs_ap, _ = run_detector(ckpt_ap, ap_frame)
    print(f'  Detected {len(dets_ap)} landmarks:')
    for n, uv in sorted(dets_ap.items()):
        print(f'    {n}: [{uv[0]:.1f}, {uv[1]:.1f}]  conf={confs_ap[n]:.3f}')
    vis_ap = visualize(9, ap_frame, dets_ap, confs_ap, 'AP')

    # ── LAT detection on frame 14 ─────────────────────────────────────
    print('\nLAT detection (frame 14)...')
    lat_frame = load_raw_frame(14)
    ckpt_lat = CKPT_BASE / 'landmark_detector_lat_pedicles' / 'fold_JYOTHI.pth'
    dets_lat, confs_lat, _ = run_detector(ckpt_lat, lat_frame)
    print(f'  Detected {len(dets_lat)} landmarks:')
    for n, uv in sorted(dets_lat.items()):
        print(f'    {n}: [{uv[0]:.1f}, {uv[1]:.1f}]  conf={confs_lat[n]:.3f}')
    vis_lat = visualize(14, lat_frame, dets_lat, confs_lat, 'LAT')

    # Combined view
    h = max(vis_ap.shape[0], vis_lat.shape[0])
    def pad(img):
        if img.shape[0] < h:
            pad = np.zeros((h-img.shape[0], img.shape[1], 3), dtype='uint8')
            return np.vstack([img, pad])
        return img
    combined = np.hstack([pad(vis_ap), pad(vis_lat)])
    cv2.imwrite(str(OUT_DIR / 'jyothi_heatmap_detect_combined.png'), combined)
    print('\nSaved combined detection image.')

    print('\n' + '='*60)
    print('  SUMMARY')
    print('='*60)
    ap_peds  = [n for n in dets_ap  if '_PED' in n]
    lat_peds = [n for n in dets_lat if '_PED' in n]
    ap_cents = [n for n in dets_ap  if '_PED' not in n]
    lat_cents= [n for n in dets_lat if '_PED' not in n]
    print(f'  AP  frame: {len(ap_cents)} centroids + {len(ap_peds)} pedicles = {len(dets_ap)} total')
    print(f'  LAT frame: {len(lat_cents)} centroids + {len(lat_peds)} pedicles = {len(dets_lat)} total')
    print(f'  {"PEDICLES DETECTED IN AP VIEW" if ap_peds else "No pedicles detected in AP"}')
    print(f'  {"PEDICLES DETECTED IN LAT VIEW" if lat_peds else "No pedicles detected in LAT"}')
    if ap_peds or lat_peds:
        print('\n  Pipeline is now FULLY unblocked:')
        print('  3D pedicle tips (CT auto) + 2D pedicle positions (X-ray CNN)')
        print('  -> Non-collinear EPnP/multi-view registration is now possible.')

    return dets_ap, dets_lat


if __name__ == '__main__':
    main()
