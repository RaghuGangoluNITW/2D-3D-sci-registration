"""
load_deepfluoro.py
=====================
Loads one DeepFluoro (Grupp et al., IPCAI 2020) cadaveric pelvis specimen
directly from the public ipcai_2020_full_res_data.h5 file, independent of
the main project's mslevel_reg/src/deepfluoro_loader.py (reused for
reference/validation only, not imported, to keep this POC standalone).

Real CT + real fluoroscopy + real, independent 3D/2D pelvis landmarks
(ASIS, FH, GSN, IOF, IPS, MOF, SPS, left/right) -- NOT collinear like
spine vertebrae (PCA singular-value ratio 0.72/0.52 vs ~0.11-0.13 for
a spine dataset's vertebral centroids), so this dataset should not suffer the
bas-relief pose ambiguity that broke a collinear-landmark spine test.

IMPORTANT: the 3D/2D landmark positions used here come from the dataset's
own 2020 annotation (done once, by the original researchers, for their
own published paper -- not something we are asking any doctor to do, and
not something done live during any surgery). They are used ONLY to score
PDE after the fact. The actual registration pipeline (DRR bank -> GO
retrieval -> pose init -> GO refinement) never reads these 2D landmarks;
it only reads the real fluoroscopy pixel data.

CT volume convention: stored as (Z, Y, X) -- confirmed against the main
project's deepfluoro_drr.py (explicit comment + validated render matching
the real X-ray almost exactly), NOT (X, Y, Z) as an earlier version of
this file incorrectly assumed. The earlier assumption passed a bogus
"sanity check" (voxel indices landed in-bounds under BOTH orderings,
since the three dimension sizes are similar enough that an index fitting
one axis's bound also happens to fit another's) -- an inconclusive check
mistaken for a real one. The actual proof was rendering a DRR from the
known ground-truth pose and comparing it visually to the real X-ray;
with the wrong transpose applied, the DRR was a scrambled, wrong-looking
image despite using a geometrically-correct camera. Fixed here: no
transpose needed, this loader's volume_zyx is already (Z,Y,X)-ordered.
"""
import h5py
import numpy as np

H5_PATH = r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\deepfluoro\ipcai_2020_full_res_data.h5'

SPECIMEN_IDS = ['17-1882', '17-1905', '18-0725', '18-1109', '18-2799', '18-2800']

FULL_RES_SIZE = 1536
FX_POS = 5257.731934          # positive-convention focal length (px)
PIXEL_SPACING_MM = 0.194
SDD_MM = FX_POS * PIXEL_SPACING_MM   # ~1020mm


def load_specimen(specimen_id: str, max_projections=None):
    """Returns dict with:
      volume_zyx   : (Z,Y,X) float32 HU-like array (matches drr_renderer.CTVolume)
      spacing_xyz  : (3,) mm/voxel
      origin_xyz   : (3,) mm, world position of voxel (0,0,0)
      landmarks_3d : {name: (3,) world mm}
      projections  : list of dicts with image (1536,1536) float32 [0,1],
                     gt_landmarks_2d {name:(2,)}, rot_180_for_up (bool)
    """
    with h5py.File(H5_PATH, 'r') as f:
        sg = f[specimen_id]
        vg = sg['vol']
        # HDF5 stores the volume as (Z, Y, X) -- confirmed against
        # deepfluoro_drr.py's explicit comment and a validated render.
        volume_zyx = vg['pixels'][()].astype(np.float32)
        origin_xyz = vg['origin'][()].flatten().astype(np.float64)
        spacing_xyz = vg['spacing'][()].flatten().astype(np.float64)

        landmarks_3d = {}
        for name in sg['vol-landmarks'].keys():
            landmarks_3d[name] = sg['vol-landmarks'][name][()].flatten().astype(np.float64)

        proj_keys = sorted(sg['projections'].keys())
        if max_projections is not None:
            proj_keys = proj_keys[:max_projections]

        projections = []
        for pk in proj_keys:
            pg = sg['projections'][pk]
            rot_flag = bool(pg['rot-180-for-up'][()])
            img = pg['image']['pixels'][()].astype(np.float32)
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn)
            # img is already in "DRR-matching" (raw) orientation when
            # rot_flag is False; if True, stored image is upright and we
            # must undo the 180 deg rotation to get back to raw/DRR frame.
            if rot_flag:
                img_raw = np.rot90(img, 2).copy()
            else:
                img_raw = img

            gt_2d = {}
            for lm_name in pg['gt-landmarks'].keys():
                uv = pg['gt-landmarks'][lm_name][()].flatten().astype(np.float64)
                if rot_flag:
                    uv = np.array([FULL_RES_SIZE - 1.0 - uv[0],
                                   FULL_RES_SIZE - 1.0 - uv[1]])
                gt_2d[lm_name] = uv

            projections.append(dict(
                proj_key=pk, image=img_raw, gt_landmarks_2d=gt_2d,
                rot_180_for_up=rot_flag,
            ))

    return dict(
        volume_zyx=volume_zyx, spacing_xyz=spacing_xyz, origin_xyz=origin_xyz,
        landmarks_3d=landmarks_3d, projections=projections,
    )


if __name__ == '__main__':
    data = load_specimen('17-1882', max_projections=3)
    print('volume shape (Z,Y,X):', data['volume_zyx'].shape)
    print('spacing:', data['spacing_xyz'], 'origin:', data['origin_xyz'])
    print('landmarks:', sorted(data['landmarks_3d'].keys()))
    for p in data['projections']:
        print(p['proj_key'], 'image', p['image'].shape, 'rot180=', p['rot_180_for_up'],
              'n_gt_2d=', len(p['gt_landmarks_2d']))
