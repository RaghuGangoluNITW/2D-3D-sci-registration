# Release Notes â€” v1.0
**2D/3D Fluoroscopy Registration Pipeline**  
**Date:** October 2026  
**Branch:** release/v1.0

---

## What Is This Release?

This is the **first public release** of the NIMS 2D/3D fluoroscopy registration
pipeline. It delivers a fully-functional, validated registration system for
research and evaluation use.

---

## Validated Results Included in This Release

| Test | Result | Status |
|---|---|---|
| DeepFluoro cadaveric pelvis (real fluoroscopy, 2 views) | **4.3 mm PDE** | âœ“ Validated |
| VerSe19 lumbar spine (synthetic DRR target, 2 views) | **0.32 mm PDE** | âœ“ Validated |
| NIMS real spine patient (real C-arm, NIMS hospital) | Registration tested | âš  Gaps identified (see below) |

Both public-dataset results were achieved with **zero manual annotation** on
any C-arm or fluoroscopy image.

---

## What Is Included

### Core pipeline (pipeline/)
- `drr_renderer.py` â€” GPU-accelerated DRR renderer using Beer-Lambert
  line integration. Renders synthetic X-rays from CT volumes.
- `similarity.py` â€” Gradient Orientation (GO) similarity metric.
  Measures how well a synthetic DRR matches a real X-ray.
- `load_deepfluoro.py` â€” Loads the DeepFluoro HDF5 dataset (CT +
  fluoroscopy + ground-truth landmarks).
- `load_verse_ct.py` â€” Loads VerSe19 NIfTI CT volumes with vertebra
  centroid annotations.
- `build_drr_bank.py` â€” Generic DRR bank builder. Renders N DRRs from
  a CT across a range of camera angles.
- `build_drr_bank_deepfluoro.py` â€” DeepFluoro-specific bank builder.
  Uses the dataset's own camera geometry for bank generation.
- `run_deepfluoro_multiview_test.py` â€” Full end-to-end 2-view
  registration test on DeepFluoro. Produces the 4.3 mm result.
- `run_verse_multiview_test.py` â€” Full end-to-end 2-view spine
  registration test on VerSe19. Produces the 0.32 mm result.
- `run_nims_calibrated_multiview.py` â€” NIMS real patient test using
  doctor-labeled camera poses as calibration seed.
- `run_nims_auto_pedicle_poc.py` â€” NIMS test with auto-extracted
  pedicle landmarks from CT HU thresholding.
- `sanity_check_render.py` â€” Quick check that the DRR renderer works
  on your hardware.

### Core library (src/)
- `deepfluoro_drr.py` â€” Low-level GPU DRR renderer (CTVolume + DRRRenderer).
- `deepfluoro_loader.py` â€” DeepFluoro dataset loader (HDF5).
- `similarity.py` â€” GO similarity + NGF + edge-NCC + combined metrics.
- `metal_suppression.py` â€” Blind and DRR-guided surgical instrument
  suppression via adaptive thresholding + Telea inpainting.
- `camera_config.py` â€” Loads and validates C-arm camera geometry JSON files.
- `optimizer.py` â€” CMA-ES multi-start optimizer (Ketcha 2017 stages 1-4).

### Training scripts (scripts/)
- `train_landmark_detector.py` â€” LOPO cross-validation training of
  LandmarkHeatmapNet (MobileNetV3-Small + UNet decoder, 18-class).
- `train_cyclegan.py` â€” CycleGAN (UNet-32 generator + PatchGAN-32)
  training for real X-ray â†’ synthetic DRR style transfer.
- `generate_synthetic_drrs.py` â€” Generates synthetic DRR training data
  from patient CT volumes.
- `extract_transverse_process_tips.py` â€” Auto-extracts pedicle tip
  positions from CT using HU thresholding (no model needed).
- `convert_xrays_to_drr.py` â€” Applies trained CycleGAN to convert real
  X-ray images to synthetic DRR style for training augmentation.

### Camera configurations (configs/cameras/)
- `ziehm_vision_fd.json` â€” Ziehm Vision RFD 3D (NIMS machine).
  SID=1050mm, pixel_spacing=0.288mm/px. Corrected from DICOM.
- `siemens_cios_spin.json` â€” Siemens Cios Spin. SID=1050mm,
  pixel_spacing=0.154mm/px.
- `ge_oec_9900.json` â€” GE OEC 9900 Elite. SID=1020mm,
  pixel_spacing=0.184mm/px.

---

## Known Gaps (NIMS Real Patient)

The pipeline achieves 4.3mm on the public pelvis dataset but the current
NIMS real spine patient result requires three improvements:

| Gap | Impact | Fix Required |
|---|---|---|
| Landmark collinearity | AP vertebra centroids (L3-L5) are near-collinear â†’ PnP unstable | Add skin fiducial markers or pedicle tips for non-collinear references |
| C-arm isocenter calibration | Isocenter position not independently measured | One-time phantom calibration from C-arm technician |
| CNN training data | ~30 real X-ray images too few for reliable landmark detection | Collect 13+ additional patient datasets and retrain |

None of these are algorithmic gaps â€” they are data and calibration tasks.

---

## What Was Fixed in This Release vs. Previous Work

| Fix | Description |
|---|---|
| Camera config corrected | `ziehm_vision_fd.json` SID corrected 1110â†’1050mm, pixel spacing 0.2â†’0.288mm/px (44% error) |
| Metal suppression improved | Threshold p95/0.75 â†’ p97/0.82, dilation 13px â†’ 21px, no early-exit for thin K-wires |
| Training crash fixed | `train_landmark_detector.py` â€” added `redirect_stderr` to suppress timm backbone warning that was killing PowerShell training at 0 epochs |
| CycleGAN inference | 100 NIMS C-arm frames converted to DRR style (`pipeline/nims_cyclegan_convert.py`) |

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- `torch==2.4.1` + `torchvision==0.19.1` (GPU strongly recommended)
- `timm==1.0.26` (MobileNetV3-Small backbone)
- `SimpleITK` / `pynrrd` (medical imaging I/O)
- `scipy` (Nelder-Mead optimiser)
- `opencv-python-headless` (image processing + CLAHE)

---

## Not Included in This Release

- Trained model checkpoints (`.pth` files) â€” too large for Git. Contact
  NIMS team for checkpoint files.
- Patient data â€” not shareable. Use public datasets for testing.
- DICOM integration layer â€” planned for v1.1.
- Real-time surgeon display overlay â€” planned for v1.2.

---

## Citation

If you use this pipeline in research, please cite:

```
NIMS Neurosurgery 2D/3D Fluoroscopy Registration Pipeline, v1.0 (2026).
GitHub: https://github.com/RaghuGangoluNITW/2D-3D-sci-registration

DeepFluoro dataset:
Grupp RB et al. "Automatic annotation of hip anatomy in fluoroscopy for
robust and efficient 2D/3D registration." IJCARS 15 (2020): 759-769.
https://doi.org/10.1007/s11548-020-02162-7
```

