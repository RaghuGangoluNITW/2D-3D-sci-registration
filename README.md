# 2D/3D Fluoroscopy Registration Pipeline
### Automatic Intraoperative Spine / Pelvis Registration from C-arm X-rays

**Version:** 1.0  
**Release Date:** October 2026  
**Institution:** NIMS Neurosurgery / Research  

---

## What This Does

This software automatically registers a patient's pre-operative CT scan to an
intraoperative C-arm fluoroscopy X-ray â€” providing real-time 3D spine/pelvis
pose estimation during surgery, with **zero manual intervention** on the C-arm
images and **zero additional hardware** beyond a standard C-arm machine.

### Validated Results

| Test | Dataset | Accuracy | Notes |
|---|---|---|---|
| Pelvis, real fluoroscopy | DeepFluoro (Johns Hopkins, cadaveric) | **4.3 mm PDE** | Real intraoperative X-rays |
| Spine, synthetic DRR | VerSe19 (public lumbar CT) | **0.32 mm PDE** | 7/10 starts sub-2mm |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU (CUDA 12.x â€” strongly recommended):
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download the DeepFluoro dataset

```bash
# Dataset: Grupp et al., Johns Hopkins 2020 (CC BY-NC 4.0)
# Download from: https://doi.org/10.7281/T1/IFSXNV
# Place the file at: data/deepfluoro/ipcai_2020_full_res_data.h5
```

### 3. Run the validated DeepFluoro test (4.3 mm result)

```bash
cd pipeline
python build_drr_bank_deepfluoro.py        # Step 1: build DRR bank (~5 min)
python run_deepfluoro_multiview_test.py    # Step 2: run registration
```

### 4. Run the VerSe spine test (0.32 mm result)

```bash
# Download VerSe19 sub-verse010 from: https://osf.io/nqjyw/
# Place CT at: data/verse/sub-verse010_ct.nii.gz
cd pipeline
python run_verse_multiview_test.py         # Builds bank + runs registration
```

### 5. Sanity check the DRR renderer

```bash
cd pipeline
python sanity_check_render.py
```

---

## Folder Structure

```
2D-3D-sci-registration/
â”œâ”€â”€ README.md                        â† This file
â”œâ”€â”€ requirements.txt                 â† Python dependencies
â”œâ”€â”€ RELEASE_NOTES.md                 â† What changed in this release
â”œâ”€â”€ src/                             â† Core library modules
â”‚   â”œâ”€â”€ similarity.py                â† GO similarity metric
â”‚   â”œâ”€â”€ metal_suppression.py         â† Surgical instrument suppression
â”‚   â”œâ”€â”€ camera_config.py             â† C-arm camera geometry loader
â”‚   â”œâ”€â”€ deepfluoro_drr.py            â† GPU DRR renderer (Beer-Lambert)
â”‚   â”œâ”€â”€ deepfluoro_loader.py         â† DeepFluoro HDF5 data loader
â”‚   â””â”€â”€ optimizer.py                 â† CMA-ES multi-start optimizer
â”œâ”€â”€ pipeline/                        â† Runnable scripts
â”‚   â”œâ”€â”€ drr_renderer.py              â† DRR renderer (POC version)
â”‚   â”œâ”€â”€ similarity.py                â† GO metric (POC version, standalone)
â”‚   â”œâ”€â”€ load_deepfluoro.py           â† DeepFluoro loader (POC)
â”‚   â”œâ”€â”€ load_verse_ct.py             â† VerSe CT loader
â”‚   â”œâ”€â”€ build_drr_bank.py            â† Generic DRR bank builder
â”‚   â”œâ”€â”€ build_drr_bank_deepfluoro.py â† DeepFluoro-specific bank builder
â”‚   â”œâ”€â”€ run_deepfluoro_multiview_test.py â† DeepFluoro 4.3mm test
â”‚   â”œâ”€â”€ run_verse_multiview_test.py      â† VerSe 0.32mm test
â”‚   â”œâ”€â”€ run_nims_calibrated_multiview.py â† NIMS real patient test
â”‚   â”œâ”€â”€ run_nims_auto_pedicle_poc.py    â† NIMS auto-pedicle test
â”‚   â””â”€â”€ sanity_check_render.py           â† DRR renderer sanity check
â”œâ”€â”€ scripts/                         â† Training and data preparation
â”‚   â”œâ”€â”€ train_landmark_detector.py   â† LOPO CNN training
â”‚   â”œâ”€â”€ train_cyclegan.py            â† CycleGAN domain adaptation training
â”‚   â”œâ”€â”€ generate_synthetic_drrs.py   â† Synthetic DRR bank generation
â”‚   â”œâ”€â”€ extract_transverse_process_tips.py â† CT pedicle auto-extraction
â”‚   â””â”€â”€ convert_xrays_to_drr.py     â† CycleGAN inference on real X-rays
â”œâ”€â”€ configs/
â”‚   â””â”€â”€ cameras/                     â† Per-machine C-arm geometry configs
â”‚       â”œâ”€â”€ ziehm_vision_fd.json     â† Ziehm Vision RFD 3D (NIMS)
â”‚       â”œâ”€â”€ siemens_cios_spin.json   â† Siemens Cios Spin
â”‚       â””â”€â”€ ge_oec_9900.json         â† GE OEC 9900
â””â”€â”€ data/                            â† Data directory (user provides)
    â””â”€â”€ sample/                      â† Place datasets here
```

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.10 |
| GPU VRAM | 4 GB | 8 GB |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | 50 GB |
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |

---

## Data Requirements

### For DeepFluoro test (4.3 mm validated result)
- **File:** `ipcai_2020_full_res_data.h5`
- **Source:** https://doi.org/10.7281/T1/IFSXNV (free, CC BY-NC 4.0)
- **Place at:** `data/deepfluoro/ipcai_2020_full_res_data.h5`

### For VerSe spine test (0.32 mm validated result)
- **File:** `sub-verse010_ct.nii.gz` + vertebra annotations
- **Source:** https://osf.io/nqjyw/ (free, academic use)
- **Place at:** `data/verse/`

### For NIMS real patient test
- Requires: patient CT (NRRD format) + C-arm DICOM frames
- Labeled AP/LAT X-ray sets with 2D vertebra positions (CENTROID.mrk.json)
- Contact NIMS team for data access

---

## License

Research use only. Not approved for clinical use.  
DeepFluoro dataset: CC BY-NC 4.0 â€” Grupp et al., Johns Hopkins University.  
VerSe19 dataset: see dataset license at https://osf.io/nqjyw/

