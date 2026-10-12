# 2D/3D Registration GUI — Quick Start

## Run

```bash
cd gui
streamlit run app.py
```

Opens at http://localhost:8501

## Workflow

| Step | What you do |
|---|---|
| 1 — Load CT | Upload NRRD file from pre-operative CT |
| 2 — Mark Centroids | Click on CT slices to mark L1-L5 vertebra centres |
| 3 — Load C-arm | Upload intraoperative DICOM frames |
| 4 — Select Frames | Pick AP and LAT frames, select your C-arm machine |
| 5 — Run & View | Click Run, see side-by-side result overlay |

## Tips

- Mark at least L3, L4, L5 (3 minimum; all 6 for best results)
- AP frame = C-arm roughly in front of patient (primary angle ~0°)
- LAT frame = C-arm roughly to the side (secondary angle ~-90°)
- GPU strongly recommended — CPU will take ~10x longer

## Data formats

- CT: `.nrrd` (exported from 3D Slicer)
- C-arm: raw DICOM files (any C-arm manufacturer)

## Requirements

```bash
pip install -r requirements.txt
pip install -r gui/requirements_gui.txt
```
