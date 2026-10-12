# Data Directory

Place your datasets in this folder as described below.

## DeepFluoro Dataset (required for 4.3 mm pelvis test)

Download from: https://doi.org/10.7281/T1/IFSXNV (free, CC BY-NC 4.0)

```
data/
└── deepfluoro/
    └── ipcai_2020_full_res_data.h5   ← ~2.5 GB HDF5 file
```

## VerSe19 Dataset (required for 0.32 mm spine test)

Download sub-verse010 from: https://osf.io/nqjyw/

```
data/
└── verse/
    ├── sub-verse010_ct.nii.gz        ← CT volume
    └── sub-verse010_seg-vert_msk.nii.gz  ← vertebra mask
```

## NIMS Real Patient Dataset

Not publicly available. Contact NIMS team.

```
data/
└── nims/
    ├── patient_ct.nrrd               ← Pre-operative CT (NRRD)
    ├── centroids.mrk.json            ← 3D vertebra centroid labels (3D Slicer format)
    └── carm_frames/                  ← DICOM C-arm frames
        ├── 001, 002, ...             ← Raw DICOM (no extension, Ziehm format)
```
