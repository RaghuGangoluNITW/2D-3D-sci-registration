"""
check_jyothi_bank_coverage.py
Check best-matching azimuths for real JYOTHI frames against the bank
to confirm coordinate mapping before multi-view test.
"""
import sys, json, cv2, pydicom
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\src')
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\mslevel_reg\src')
import numpy as np
from pathlib import Path
from similarity import gradient_orientation_similarity

BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\jyothi_data\drr_bank')
CARM_BASE = r'C:\Users\rgangolu\OneDrive - Infor\analysis\jyothi\carm_extracted\jyothi carm\20260707\0'

manifest = json.loads((BANK_DIR/'manifest.json').read_text())
entries = manifest['entries']
ap_entries  = [e for e in entries if e['base_view']=='AP']
lat_entries = [e for e in entries if e['base_view']=='LAT']
out_size = manifest['output_size']


def load_and_preprocess(frame_idx):
    """Load raw DICOM frame, apply polarity inversion + CLAHE + metal suppression."""
    ds = pydicom.dcmread(f'{CARM_BASE}\\{frame_idx}', force=True)
    arr = ds.pixel_array.astype(np.float32)
    arr = arr / arr.max()
    u8 = (np.clip(arr, 0, 1)*255).astype('uint8')
    resized = cv2.resize(u8, (out_size, out_size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(resized)
    # Invert to bone-bright convention (same as JYOTHI single-view test)
    img = 1.0 - (eq.astype(np.float32)/255.0)
    # Metal suppression
    p_thresh = float(np.percentile(img, 95.0))
    thresh = max(p_thresh, 0.75)
    mask = (img > thresh).astype(np.uint8) * 255
    if mask.mean() > 1e-6:
        ksize = 13
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask_d = cv2.dilate(mask, kernel)
        img8 = (img * 255.0).clip(0,255).astype(np.uint8)
        img8 = cv2.inpaint(img8, mask_d, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        img = img8.astype(np.float32) / 255.0
    return img


print('Loading and preprocessing real frames...')
frame_ap  = load_and_preprocess(9)   # AP frame
frame_lat = load_and_preprocess(14)  # LAT frame
print(f'AP  frame: shape={frame_ap.shape}  mean={frame_ap.mean():.3f}')
print(f'LAT frame: shape={frame_lat.shape} mean={frame_lat.mean():.3f}')

# Score AP bank entries vs real AP frame
print('\nScoring AP bank vs real AP frame...')
go_ap = []
for e in ap_entries:
    drr = cv2.imread(str(BANK_DIR/e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
    go_ap.append(gradient_orientation_similarity(drr, frame_ap, sigma=2.0))
go_ap = np.array(go_ap)
order_ap = np.argsort(-go_ap)
print(f'Best AP: GO={go_ap.max():.4f}  az={ap_entries[order_ap[0]]["azimuth_deg"]:.1f}  el={ap_entries[order_ap[0]]["elevation_deg"]:.1f}')
print(f'Top-5 AP azimuths: {[round(ap_entries[order_ap[i]]["azimuth_deg"],1) for i in range(5)]}')
print(f'Score range: {go_ap.min():.3f} - {go_ap.max():.3f}  mean={go_ap.mean():.3f}  SNR={go_ap.max()/go_ap.mean():.3f}')

# Score LAT bank entries vs real LAT frame
print('\nScoring LAT bank vs real LAT frame...')
go_lat = []
for e in lat_entries:
    drr = cv2.imread(str(BANK_DIR/e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
    go_lat.append(gradient_orientation_similarity(drr, frame_lat, sigma=2.0))
go_lat = np.array(go_lat)
order_lat = np.argsort(-go_lat)
print(f'Best LAT: GO={go_lat.max():.4f}  az={lat_entries[order_lat[0]]["azimuth_deg"]:.1f}  el={lat_entries[order_lat[0]]["elevation_deg"]:.1f}')
print(f'Top-5 LAT azimuths: {[round(lat_entries[order_lat[i]]["azimuth_deg"],1) for i in range(5)]}')
print(f'Score range: {go_lat.min():.3f} - {go_lat.max():.3f}  mean={go_lat.mean():.3f}  SNR={go_lat.max()/go_lat.mean():.3f}')

print('\nConclusion:')
print(f'AP real angle: DICOM primary=-19.16° -> best bank match az={ap_entries[order_ap[0]]["azimuth_deg"]:.1f}°')
print(f'LAT real angle: DICOM secondary=-90.3° -> best bank match az={lat_entries[order_lat[0]]["azimuth_deg"]:.1f}°')
