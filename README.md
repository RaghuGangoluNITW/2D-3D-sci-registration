# 2D/3D Fluoroscopy Registration Pipeline
### Automatic Intraoperative Spine / Pelvis Registration from C-arm X-rays

**Version:** 1.0
**Release Date:** October 2026
**Institution:** NIMS Neurosurgery / Research

---

## What This Does

This software automatically registers a patient's pre-operative CT scan to an
intraoperative C-arm fluoroscopy X-ray, providing real-time 3D spine/pelvis
pose estimation during surgery with **zero manual intervention** on the C-arm
images and **zero additional hardware** beyond a standard C-arm machine.

### Validated Results

| Test | Dataset | Accuracy | Notes |
|---|---|---|---|
| Pelvis, real fluoroscopy | DeepFluoro (Johns Hopkins, cadaveric) | **4.3 mm PDE** | Real intraoperative X-rays |
| Spine, synthetic DRR | VerSe19 (public lumbar CT) | **0.32 mm PDE** | 7/10 starts sub-2mm |

---

## Quick Start

### Option A: Streamlit GUI (recommended)

```bash
pip install -r requirements.txt
pip install -r gui/requirements_gui.txt
cd gui
streamlit run app.py --server.port 9500
# Open browser: http://127.0.0.1:9500
```

### Option B: Command line (DeepFluoro 4.3 mm result)

```bash
pip install -r requirements.txt
# Download DeepFluoro: https://doi.org/10.7281/T1/IFSXNV
# Place at: data/deepfluoro/ipcai_2020_full_res_data.h5
cd pipeline
python build_drr_bank_deepfluoro.py
python run_deepfluoro_multiview_test.py
```

### Option C: Command line (VerSe spine 0.32 mm result)

```bash
# Download VerSe19 sub-verse010: https://osf.io/nqjyw/
# Place at: data/verse/sub-verse010_ct.nii.gz
cd pipeline
python run_verse_multiview_test.py
```

---

## Folder Structure

```
2D-3D-sci-registration/
|
+-- README.md                          <- This file
+-- requirements.txt                   <- Python package dependencies
+-- RELEASE_NOTES.md                   <- What changed in this release
|
+-- gui/                               <- Streamlit web application
|   +-- app.py                         <- Main 5-step registration wizard
|   +-- requirements_gui.txt           <- GUI-specific dependencies
|   \-- README_GUI.md                  <- GUI quick-start guide
|
+-- src/                               <- Core library modules
|   +-- similarity.py                  <- Gradient Orientation (GO) metric
|   +-- metal_suppression.py           <- Surgical instrument suppression
|   +-- camera_config.py               <- C-arm camera geometry loader
|   +-- deepfluoro_drr.py              <- GPU DRR renderer (Beer-Lambert)
|   +-- deepfluoro_loader.py           <- DeepFluoro HDF5 data loader
|   +-- landmark_model.py              <- LandmarkHeatmapNet CNN (18-class)
|   +-- landmark_detector.py           <- Inference wrapper for CNN
|   +-- landmark_dataset.py            <- Dataset constants and loading
|   +-- cyclegan.py                    <- CycleGAN X-ray <-> DRR transfer
|   +-- optimizer.py                   <- CMA-ES multi-start optimizer
|   \-- generic_patient_loader.py      <- Patient data folder loader
|
+-- pipeline/                          <- Runnable end-to-end scripts
|   +-- drr_renderer.py                <- DRR renderer (standalone POC version)
|   +-- load_deepfluoro.py             <- DeepFluoro dataset loader (POC)
|   +-- load_verse_ct.py               <- VerSe19 CT loader
|   +-- build_drr_bank.py              <- Generic DRR bank builder
|   +-- build_drr_bank_deepfluoro.py   <- DeepFluoro-specific bank builder
|   +-- run_deepfluoro_multiview_test.py  <- DeepFluoro 4.3mm test
|   +-- run_verse_multiview_test.py       <- VerSe 0.32mm test
|   +-- run_nims_multiview_test.py        <- NIMS real patient test
|   +-- run_nims_auto_pedicle_test.py     <- NIMS auto-pedicle test
|   \-- sanity_check_render.py            <- DRR renderer GPU sanity check
|
+-- scripts/                           <- Training and data preparation
|   +-- train_landmark_detector.py     <- LOPO CNN training (18-class heatmap)
|   +-- train_cyclegan.py              <- CycleGAN domain adaptation training
|   +-- generate_synthetic_drrs.py     <- Synthetic DRR bank generation from CT
|   +-- extract_transverse_process_tips.py  <- CT pedicle auto-extraction
|   \-- convert_xrays_to_drr.py        <- CycleGAN inference on real X-rays
|
+-- configs/
|   \-- cameras/                       <- Per-machine C-arm geometry configs
|       +-- ziehm_vision_fd.json       <- Ziehm Vision RFD 3D (NIMS machine)
|       +-- siemens_cios_spin.json     <- Siemens Cios Spin
|       \-- ge_oec_9900.json           <- GE OEC 9900 Elite
|
\-- data/                              <- Data directory (user provides)
    \-- sample/
        \-- README_DATA.md             <- Instructions for placing datasets
```

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.10 |
| GPU VRAM | 4 GB (CUDA) | 8 GB |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | 50 GB |
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |

---

## Data Requirements

### DeepFluoro (4.3 mm pelvis test)
- Source: https://doi.org/10.7281/T1/IFSXNV (free, CC BY-NC 4.0)
- Place file at: `data/deepfluoro/ipcai_2020_full_res_data.h5`

### VerSe19 (0.32 mm spine test)
- Source: https://osf.io/nqjyw/ (free, academic use)
- Place files at: `data/verse/sub-verse010_ct.nii.gz`

### NIMS real patient
- Requires: patient CT (NRRD) + C-arm DICOM frames + 3D vertebra labels
- Contact NIMS team for data access

---

## License

Research use only. Not approved for clinical use.
DeepFluoro dataset: CC BY-NC 4.0 -- Grupp et al., Johns Hopkins University.
VerSe19 dataset: see https://osf.io/nqjyw/
