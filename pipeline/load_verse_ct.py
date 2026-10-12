"""
load_verse_ct.py
==================
Loads a VerSe dataset CT case (public, CC BY-SA 4.0, Sekuboyina et al.
2021) and its vertebra centroid annotations into the format expected by
drr_renderer.CTVolume.

VerSe stores:
  - sub-verseNNN_ct.nii.gz          : CT volume (NIfTI)
  - sub-verseNNN_seg-subreg_ctd.json: centroid list, voxel coords, with a
                                       'direction' header describing axis
                                       orientation (e.g. ["P","I","R"])

VerSe label convention (from https://github.com/anjany/verse):
  1-7   : C1-C7
  8-19  : T1-T12
  20-25 : L1-L6
  26    : sacrum (unused)
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

LABEL_NAMES = {
    20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6',
    19: 'T12', 18: 'T11',
}


def load_verse_case(case_dir: Path):
    """
    case_dir: path to .../dataset-verseXX.../ (contains rawdata/ and
              derivatives/ subfolders for one patient, as extracted from
              the official VerSe zip).

    Returns: dict(volume_zyx, spacing_xyz, origin_xyz, centroids_world,
                  nii_affine, raw_centroids_voxel)
    """
    case_dir = Path(case_dir)
    raw_dirs = list(case_dir.glob('rawdata/sub-*'))
    der_dirs = list(case_dir.glob('derivatives/sub-*'))
    if not raw_dirs or not der_dirs:
        raise FileNotFoundError(f'Could not find rawdata/derivatives under {case_dir}')
    raw_dir, der_dir = raw_dirs[0], der_dirs[0]

    ct_path = next(raw_dir.glob('*_ct.nii.gz'))
    ctd_path = next(der_dir.glob('*_seg-subreg_ctd.json'))

    img = nib.load(str(ct_path))
    # nibabel returns data in the NIfTI's own (i,j,k) voxel order; the
    # affine maps voxel (i,j,k,1) -> world (x,y,z,1) in mm (RAS+ convention
    # by nibabel's canonicalization below).
    img = nib.as_closest_canonical(img)  # standardize to RAS+ orientation
    data = img.get_fdata().astype(np.float32)
    affine = img.affine

    # nibabel voxel order here is (X, Y, Z) i.e. (i,j,k) = (X-index, Y-index, Z-index)
    # our renderer wants volume as (Z, Y, X), spacing (sx,sy,sz), origin (ox,oy,oz)
    # with a simple axis-aligned affine (RAS+ canonical gives us that).
    volume_xyz = data  # shape (nx, ny, nz)
    volume_zyx = np.transpose(volume_xyz, (2, 1, 0)).copy()  # (nz, ny, nx)

    spacing = np.array([affine[0, 0], affine[1, 1], affine[2, 2]], dtype=np.float64)
    origin = np.array([affine[0, 3], affine[1, 3], affine[2, 3]], dtype=np.float64)
    # RAS+ canonical should give a diagonal affine (axis-aligned); spacing
    # signs may be negative if an axis is flipped relative to voxel index
    # increasing direction -- take abs for spacing magnitude, and handle
    # sign by checking if volume needs flipping to match origin convention.
    flip = spacing < 0
    spacing = np.abs(spacing)
    if flip[0]:
        volume_zyx = volume_zyx[:, :, ::-1]
        origin[0] = affine[0, 3] + (volume_zyx.shape[2] - 1) * -affine[0, 0]
    if flip[1]:
        volume_zyx = volume_zyx[:, ::-1, :]
        origin[1] = affine[1, 3] + (volume_zyx.shape[1] - 1) * -affine[1, 1]
    if flip[2]:
        volume_zyx = volume_zyx[::-1, :, :]
        origin[2] = affine[2, 3] + (volume_zyx.shape[0] - 1) * -affine[2, 2]
    volume_zyx = np.ascontiguousarray(volume_zyx)

    # ── Centroids: voxel coords in the ORIGINAL (pre-canonicalization) NIfTI
    # orientation + the 'direction' header. Simplest robust approach: use
    # the ORIGINAL img's affine (before as_closest_canonical) to map these
    # voxel coords straight to world mm, since the json's X/Y/Z coordinates
    # are voxel indices in that original array -- this sidesteps needing to
    # track the canonicalization permutation ourselves.
    img_orig = nib.load(str(ct_path))
    affine_orig = img_orig.affine

    ctd_raw = json.loads(ctd_path.read_text())
    direction_hdr = ctd_raw[0] if 'direction' in ctd_raw[0] else None
    centroids_world = {}
    for entry in ctd_raw:
        if 'label' not in entry:
            continue
        lbl = entry['label']
        name = LABEL_NAMES.get(lbl)
        if name is None:
            continue
        vox = np.array([entry['X'], entry['Y'], entry['Z'], 1.0])
        world = affine_orig @ vox
        centroids_world[name] = world[:3]

    return dict(
        volume_zyx=volume_zyx,
        spacing_xyz=spacing,
        origin_xyz=origin,
        centroids_world=centroids_world,
        direction_header=direction_hdr,
    )


if __name__ == '__main__':
    import sys
    case = load_verse_case(sys.argv[1] if len(sys.argv) > 1 else
                            r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\dataset-verse19validation')
    print('volume shape (Z,Y,X):', case['volume_zyx'].shape)
    print('spacing:', case['spacing_xyz'])
    print('origin:', case['origin_xyz'])
    print('centroids (world mm):')
    for k, v in case['centroids_world'].items():
        print(f'  {k}: {v.round(1)}')
