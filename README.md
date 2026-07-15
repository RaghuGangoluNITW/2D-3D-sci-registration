# 2D-3D Intraoperative Spine Registration

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Intraoperative Spine Level Identification via 2D/3D Registration**  
> CT ↔ C-arm X-ray registration using EPnP initialisation + CMA-ES optimisation  
> Evaluated on 9 patients, 51 C-arm sets — mean final PDE: **1.63 mm**

---

## 📋 Overview

This repository provides a complete pipeline for **2D/3D registration** of preoperative CT volumes to intraoperative C-arm fluoroscopy images for spine surgery. It identifies vertebral levels (L1–L5) in the operating room without requiring additional DICOM metadata.

### Pipeline Summary

```
Preop CT (NRRD)         Intraop C-arm (NRRD)
     +                         +
CT Centroids              X-ray Centroids
 (.mrk.json)               (.mrk.json)
     |                         |
     └────────┬────────────────┘
              ▼
       EPnP Initialisation
       (cv2.solvePnP)
              ▼
    CMA-ES Optimisation
    (DRR ↔ X-ray similarity)
              ▼
    Registered 6-DOF Pose
    + Per-vertebra Deformation
              ▼
    PDE metric (mm) + Report PDF
```

### Key Results

| Metric | Value |
|---|---|
| Patients | 9 |
| C-arm sets | 51 (AP + LAT) |
| Mean final PDE | **1.63 mm** |
| Sets ≤ 2 mm | **84%** |
| Success rate | **100%** |
| Mean runtime / set | ~8 s (GPU) |

---

## 🗂️ Repository Structure

```
2D-3D-sci-registration/
├── src/                          # Core library modules
│   ├── generic_patient_loader.py   # Unified loader for all patients
│   ├── deepfluoro_drr.py           # GPU DRR renderer (PyTorch)
│   ├── deepfluoro_loader.py        # DeepFluoro dataset support
│   ├── optimizer.py                # CMA-ES optimisation
│   ├── similarity.py               # GO cost + NCC metrics
│   ├── deformable_registration.py  # Per-vertebra deformation
│   ├── drr.py                      # DRR utilities
│   ├── registration.py             # Registration helpers
│   └── ...                         # Patient-specific loaders
│
├── run_new_patients.py           # ★ Main pipeline runner
├── generate_report.py            # PDF report generator
├── analyse_loss_pde.py           # Loss landscape analysis
│
├── scripts/                      # Visualisation & diagnostic scripts
│
├── data/                         # Dataset folder
│   ├── BIKNA/
│   │   ├── CT label/
│   │   │   ├── <CT_scan>.nrrd        # Preop CT volume (not in repo — see below)
│   │   │   └── CENTROID.mrk.json     # 3D vertebra centroids (LPS mm)
│   │   └── XRAY Label/
│   │       ├── XRAY-AP/
│   │       │   ├── AP-SET-1/
│   │       │   │   ├── <xray>.nrrd       # C-arm image (not in repo — see below)
│   │       │   │   └── CENTROID.mrk.json # 2D pixel centroids
│   │       │   └── AP-SET-2/ ...
│   │       └── XRAY-LAT/ ...
│   └── <PATIENT>/ ...            # Same structure for each patient
│
├── results/
│   └── newdata/
│       ├── all_results.json          # Combined results for all patients
│       ├── registration_report.pdf   # ★ Full evaluation report (PDF)
│       └── <PATIENT>/
│           ├── results.json              # Per-patient metrics
│           └── <SET>_overlay.png         # 3-panel visualisation
│
├── manuscript/                   # LaTeX manuscript files
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/RaghuGangoluNITW/2D-3D-sci-registration.git
cd 2D-3D-sci-registration
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your data

Each patient must follow this folder layout under `data/`:

```
data/<PATIENT_NAME>/
    CT label/
        <any_name>.nrrd         ← preop CT volume (exported from 3D Slicer)
        CENTROID.mrk.json       ← 3D vertebra centroids (L1–L5) in LPS mm
    XRAY Label/
        XRAY-AP/
            AP-SET-1/
                <any_name>.nrrd         ← 1024×1024 C-arm AP image
                CENTROID.mrk.json       ← 2D pixel centroids (u,v,0)
            AP-SET-2/ ...
        XRAY-LAT/
            LAT-SET-1/ ...
```

**Landmark files** (`.mrk.json`) are standard [3D Slicer](https://www.slicer.org/) Markups JSON files.  
Label names must be from: `L1, L2, L3, L4, L5` (and optionally `D11, D12`).

> ⚠️ **CT NRRDs and X-ray NRRDs are not included** in this repository due to file size.  
> The `data/` folder contains only the `.mrk.json` centroid annotation files.  
> Please contact the authors or obtain the clinical data through your institution.

### 4. Run the pipeline

```bash
# Run all patients found in data/
python run_new_patients.py

# Run specific patients
python run_new_patients.py --patients BIKNA JYOTHI

# Check geometry only (DRR render test, no optimisation)
python run_new_patients.py --geometry_only --verbose

# Adjust CMA-ES settings
python run_new_patients.py --starts 32 --rot 15 --trans 30
```

### 5. Generate the PDF report

```bash
python generate_report.py
# Output: results/newdata/registration_report.pdf
```

---

## ⚙️ Camera Model

The pipeline assumes a **Ziehm Vision FD** C-arm (fixed intrinsics):

| Parameter | Value |
|---|---|
| SID | 1110 mm |
| Pixel spacing | 0.2 mm/px |
| Image size | 1024 × 1024 px |
| Fx = Fy | 5550 px |

To use a different C-arm, update the constants in `src/generic_patient_loader.py`:
```python
CAM_SID_MM   = 1110.0
CAM_PIX_MM   = 0.2
CAM_IMG_SIZE = 1024
```

---

## 📐 Algorithm Details

### Phase 1 — EPnP Initialisation
- Matches 3D CT centroids to 2D X-ray centroids via `cv2.solvePnP` (EPnP for ≥4 landmarks, SQPNP for 3)
- Provides initial 6-DOF extrinsic pose (R, t)

### Phase 2 — CMA-ES Optimisation (3 resolution stages)
| Stage | DRR size | Pixel spacing | Similarity |
|---|---|---|---|
| Coarse | 64 px | 3.2 mm/px | NCC |
| Medium | 180 px | 1.14 mm/px | 0.5×GO + 0.5×NCC |
| Fine | 256 px | 0.8 mm/px | NCC |

A **safety net** reverts to the EPnP pose if the optimised PDE exceeds 1.5× the EPnP PDE.

### Phase 3 — Per-vertebra Deformation
Closed-form backprojection of 2D centroids onto 3D viewing rays gives intraoperative vertebra positions, capturing prone-positioning spinal deformation (~1–3 mm).

---

## 📊 Results

Full results are in `results/newdata/`. See the [PDF report](results/newdata/registration_report.pdf).

| Patient | Sets | EPnP PDE | Final PDE | Success |
|---|---|---|---|---|
| BIKNA | 6 | 1.77 mm | 1.77 mm | 6/6 |
| JYOTHI | 6 | 0.48 mm | 0.57 mm | 6/6 |
| NAYEEMA BEGUM | 6 | 0.52 mm | 0.52 mm | 6/6 |
| RASVANTI | 6 | 2.00 mm | 2.11 mm | 6/6 |
| SARKHI | 6 | 1.29 mm | 1.43 mm | 6/6 |
| SHRILATHA | 6 | 3.06 mm | 3.06 mm | 6/6 |
| SHRINIVAS | 6 | 2.08 mm | 2.20 mm | 6/6 |
| Swaroop | 3 | 0.97 mm | 0.97 mm | 3/3 |
| VIJAYALAXMI | 6 | 1.68 mm | 1.49 mm | 6/6 |
| **COMBINED** | **51** | — | **1.63 mm** | **51/51** |

---

## 🛠️ Dependencies

- Python 3.9+
- PyTorch (CUDA recommended)
- SimpleITK
- OpenCV (`opencv-python`)
- NumPy, SciPy
- Matplotlib, Pillow
- cmaes

See `requirements.txt` for pinned versions.

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{gangolu2026spine,
  title   = {Intraoperative Spine Level Identification via 2D/3D Registration},
  author  = {Gangolu, Raghu and others},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Under review}
}
```

---

## 👤 Author

**Raghu Gangolu**  
National Institute of Technology Warangal (NITW)  
GitHub: [@RaghuGangoluNITW](https://github.com/RaghuGangoluNITW)

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
