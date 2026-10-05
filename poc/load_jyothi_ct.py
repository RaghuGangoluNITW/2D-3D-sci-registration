"""
load_jyothi_ct.py
====================
Loads JYOTHI's real CT (converted from the raw DICOM series) and her
real, doctor-verified centroid annotations (CENTROID.mrk.json, Slicer
markup format) into the format drr_renderer.CTVolume expects.

This gives the POC pipeline a REAL CT + REAL doctor-verified 3D ground
truth + REAL labeled X-rays (with real 2D ground-truth positions) for
the first time -- unlike the VerSe test (synthetic target only) or the
Radiopaedia test (no ground truth at all). This is the one case where a
true, defensible numeric PDE can be computed end-to-end.
"""
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

JYOTHI_ROOT = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\jyothi\jyothi_extracted\jyothi')
CT_SERIES_UID = '1.3.12.2.1107.5.1.4.96940.30000026070403245011500121822'  # L_Spine 1.0 I26s 3, 371 slices
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\jyothi_data')


def convert_ct():
    """One-time conversion of the raw DICOM series to NRRD (cached)."""
    out_path = OUT_DIR / 'jyothi_ct.nrrd'
    if out_path.exists():
        return out_path
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(JYOTHI_ROOT), CT_SERIES_UID)
    reader.SetFileNames(files)
    img = reader.Execute()
    OUT_DIR.mkdir(exist_ok=True)
    sitk.WriteImage(img, str(out_path))
    print(f'Converted CT: {img.GetSize()} spacing={img.GetSpacing()} -> {out_path}')
    return out_path


def load_mrk_3d(path: Path):
    d = json.loads(path.read_text())
    return {cp['label'].strip(): np.array(cp['position'], dtype=np.float64)
            for cp in d['markups'][0]['controlPoints']
            if cp.get('positionStatus', 'defined') == 'defined'}


def load_mrk_2d(path: Path):
    d = json.loads(path.read_text())
    out = {}
    for cp in d['markups'][0]['controlPoints']:
        if cp.get('positionStatus', 'defined') == 'defined':
            pos = cp['position']
            out[cp['label'].strip()] = np.array([pos[0], pos[1]], dtype=np.float64)
    return out


def load_jyothi():
    ct_path = convert_ct()
    img = sitk.ReadImage(str(ct_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)
    origin = np.array(img.GetOrigin(), dtype=np.float64)

    centroids = load_mrk_3d(JYOTHI_ROOT / 'CT label' / 'CENTROID.mrk.json')

    # Discover labeled real X-ray sets (AP-SET-*, LAT-SET-*)
    sets = []
    for view_tag, subfolder in [('AP', 'XRAY-AP'), ('LAT', 'XRAY-LAT')]:
        vdir = JYOTHI_ROOT / 'XRAY Label' / subfolder
        if not vdir.exists():
            continue
        for sdir in sorted(vdir.iterdir()):
            nrrds = list(sdir.glob('*.nrrd'))
            jsons = list(sdir.glob('*.mrk.json'))
            if not nrrds or not jsons:
                continue
            lm2d = load_mrk_2d(jsons[0])
            if len(lm2d) < 3:
                continue
            xray_img = sitk.ReadImage(str(nrrds[0]))
            xray_arr = sitk.GetArrayFromImage(xray_img).squeeze().astype(np.float32)
            lo, hi = xray_arr.min(), xray_arr.max()
            if hi > lo:
                xray_arr = (xray_arr - lo) / (hi - lo)
            sets.append(dict(name=sdir.name, view=view_tag, image=xray_arr,
                              landmarks_2d=lm2d))

    return dict(
        volume_zyx=arr, spacing_xyz=spacing, origin_xyz=origin,
        centroids_world=centroids, xray_sets=sets,
    )


if __name__ == '__main__':
    data = load_jyothi()
    print('CT shape (Z,Y,X):', data['volume_zyx'].shape)
    print('spacing:', data['spacing_xyz'])
    print('centroids:', {k: v.round(1).tolist() for k, v in data['centroids_world'].items()})
    print(f'\n{len(data["xray_sets"])} labeled X-ray sets found:')
    for s in data['xray_sets']:
        print(f'  {s["name"]} ({s["view"]}): image {s["image"].shape}, '
              f'labels={sorted(s["landmarks_2d"].keys())}')
