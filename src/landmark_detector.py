"""
landmark_detector.py
=====================
CNN + SSM inference pipeline for automatic 2D vertebra landmark detection.

Pipeline
--------
  1. CNN (MobileNetV3-Small + UNet decoder + per-class heads)
     → predicts heatmaps → picks confident peaks (L3/L4/L5 most reliable)
  2. SpineSSM.fit_and_complete()
     → fits a 5-mode Statistical Shape Model to the CNN detections
     → fills in missing vertebrae (D12/L1/L2) from anatomical prior
     → snaps all coords onto the learned valid-spine manifold

Result: full 6-vertebra prediction even when CNN only fires on 2-3 vertebrae.

Usage::
    from landmark_detector import LandmarkDetector

    det = LandmarkDetector()          # loads best checkpoint + SSM
    preds = det.predict(xray_np)      # (H,W) float32
    # Returns: {'D12': array([u,v]), 'L1': ..., ..., 'L5': ...}
    # Always returns all 6 vertebrae (SSM fills any CNN misses)

    preds, confs = det.predict(xray_np, return_conf=True)
    # confs: {'D12': 0.12, ...}  — CNN peak confidence (0 if SSM-filled)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from landmark_dataset import (
    CLASS_NAMES, NUM_CLASSES, CLS2IDX,
    IMG_SIZE, HMAP_SIZE, _load_xray,
)
from landmark_model import LandmarkHeatmapNet
from spine_ssm import SpineSSM

CKPT_DIR     = Path(__file__).parent.parent / 'checkpoints' / 'landmark_detector'
_LN_REG_DIR  = Path(__file__).parent.parent / 'checkpoints' / 'ssm_regressor_ap'
_BN_REG_DIR  = Path(__file__).parent.parent / 'checkpoints' / 'ssm_regressor_ap'
SSM_REG_DIR  = _LN_REG_DIR if _LN_REG_DIR.exists() and any(_LN_REG_DIR.glob('fold_*.pth')) else _BN_REG_DIR
SSM_PATH     = Path(__file__).parent.parent / 'checkpoints' / 'spine_ssm.pkl'
DEFAULT_CONF = 0.05   # min heatmap peak for CNN detection to be trusted


# ── SSM Regressor Detector ────────────────────────────────────────────────────

class SSMRegressorDetector:
    """
    Direct SSM parameter regression for spine landmark localisation.

    Runs MobileNetV3-Small → GAP → MLP → 8 SSM params → decode to pixel coords.
    Much simpler task than heatmap detection: regress 8 scalars describing
    spine centroid position, scale, and shape variation.

    Usage::
        det = SSMRegressorDetector(patient='SARKHI')
        coords = det.predict(xray_np)   # {name: np.array([u,v])}
    """

    def __init__(self,
                 checkpoint: Optional[str | Path] = None,
                 ssm_path:   Optional[str | Path] = None,
                 patient:    Optional[str] = None,
                 view:       Optional[str] = None,   # 'AP' or 'LAT' for view-specific ckpts
                 device:     Optional[str] = None):
        from ssm_regressor import SSMRegressor, decode_params

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load SSM — prefer BUU SSM if it exists (trained with BUU data)
        if ssm_path:
            sp = Path(ssm_path)
        else:
            buu_ssm = Path(__file__).parent.parent / 'checkpoints' / 'spine_ssm_buu.pkl'
            sp = buu_ssm if buu_ssm.exists() else SSM_PATH
        self.ssm = SpineSSM.load(str(sp))

        # Locate checkpoint: patient+view specific, patient-only, or best available
        if checkpoint is not None:
            ckpt_path = Path(checkpoint)
        elif patient is not None:
            # Try view-specific first (fold_SARKHI_AP.pth), then fall back to combined
            view_ckpt = SSM_REG_DIR / f'fold_{patient}_{view}.pth' if view else None
            combined_ckpt = SSM_REG_DIR / f'fold_{patient}.pth'
            if view_ckpt and view_ckpt.exists():
                ckpt_path = view_ckpt
            elif combined_ckpt.exists():
                ckpt_path = combined_ckpt
            else:
                ckpts = sorted(SSM_REG_DIR.glob('fold_*.pth'))
                ckpt_path = ckpts[0] if ckpts else None
        else:
            ckpts = sorted(SSM_REG_DIR.glob('fold_*.pth'))
            ckpt_path = ckpts[0] if ckpts else None

        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(
                f'No SSM regressor checkpoint found. '
                'Run scripts/train_ssm_regressor.py first.')

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        # Support old checkpoints (mobilenet) and new checkpoints (resnet50)
        backbone_name = ckpt.get('backbone_name', 'mobilenetv3_small_100')
        self.model = SSMRegressor(
            n_modes=self.ssm.n_modes, dropout=0.0, pretrained=False,
            backbone_name=backbone_name,
        ).to(self.device)
        missing, unexpected = self.model.load_state_dict(ckpt['model_state'], strict=False)
        head_bn_keys = [k for k in unexpected if k.startswith('head') and 'running' in k]
        if head_bn_keys:
            raise RuntimeError(
                f'Checkpoint {ckpt_path.name} has BatchNorm head but model uses LayerNorm. '
                f'Retrain with scripts/train_ssm_regressor.py --ckpt_dir checkpoints/ssm_regressor_r50')
        self.model.eval()
        self._decode = decode_params

        val_err = ckpt.get('val_err_mm', float('nan'))
        print(f'[SSMRegressorDetector] {ckpt_path.name}  '
              f'(val_err={val_err:.1f}mm)')

    @torch.no_grad()
    def predict(self,
                xray:        np.ndarray,
                return_conf: bool = False,
                ) -> 'Dict[str, np.ndarray]':
        """
        Parameters
        ----------
        xray : (H, W) float32 in [0,1].

        Returns
        -------
        coords : {vertebra_name: np.array([u_px, v_px])} — all 6 vertebrae.
        confs  : (only if return_conf=True) all 1.0 (regression, no threshold).
        """
        orig_h, orig_w = xray.shape[:2]

        # CLAHE + resize (same as training)
        img    = xray.astype(np.float32)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img_cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_u8).astype(np.float32) / 255.0
        img_rs = cv2.resize(img_cl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(self.device)

        params = self.model(tensor)[0].cpu().numpy()
        coords = self._decode(params, (orig_h, orig_w), self.ssm)

        if return_conf:
            # Regression has no per-landmark confidence — return 1.0 for all
            confs = {name: 1.0 for name in coords}
            return coords, confs
        return coords

    def predict_from_nrrd(self, nrrd_path: str | Path, **kwargs):
        img = _load_xray(str(nrrd_path))
        return self.predict(img, **kwargs)


def _find_best_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Pick the checkpoint with the highest val_pck from lopo_summary.json,
    or fall back to the lexicographically last fold_*.pth.
    """
    summary_path = ckpt_dir / 'lopo_summary.json'
    if summary_path.exists():
        import json
        summary = json.load(open(summary_path))
        lopo = summary.get('lopo_pck', {})
        if lopo:
            best_patient = max(lopo, key=lambda k: lopo[k] if lopo[k] == lopo[k] else -1)
            candidate = ckpt_dir / f'fold_{best_patient}.pth'
            if candidate.exists():
                return candidate
    # Fallback: any fold checkpoint
    ckpts = sorted(ckpt_dir.glob('fold_*.pth'))
    return ckpts[-1] if ckpts else None


class LandmarkDetector:
    """
    CNN + SSM pipeline for full 6-vertebra localisation.

    Parameters
    ----------
    checkpoint     : path to a .pth CNN checkpoint, or None to auto-select.
    ssm_path       : path to spine_ssm.pkl, or None to auto-find.
    device         : 'cuda' / 'cpu' / None (auto).
    conf_threshold : CNN heatmap peak threshold for trusting a detection.
    use_ssm        : if False, return only raw CNN detections (no completion).
    """

    def __init__(self,
                 checkpoint:      Optional[str | Path] = None,
                 ssm_path:        Optional[str | Path] = None,
                 device:          Optional[str] = None,
                 conf_threshold:  float = DEFAULT_CONF,
                 use_ssm:         bool  = True,
                 img_size:        int   = IMG_SIZE):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.conf_threshold = conf_threshold
        self.img_size       = img_size
        self.use_ssm        = use_ssm

        # ── Load CNN checkpoint ───────────────────────────────────────────────
        if checkpoint is None:
            ckpt_path = _find_best_checkpoint(CKPT_DIR)
            if ckpt_path is None:
                raise FileNotFoundError(
                    f'No landmark detector checkpoint found in {CKPT_DIR}. '
                    'Run scripts/train_landmark_detector.py first.')
        else:
            ckpt_path = Path(checkpoint)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model = LandmarkHeatmapNet(pretrained=False, dropout=0.0).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        val_pck = ckpt.get('val_pck', float('nan'))
        print(f'[LandmarkDetector] CNN: {ckpt_path.name}  (val PCK@10mm={val_pck:.3f})')

        # ── Load SSM ──────────────────────────────────────────────────────────
        self.ssm: Optional[SpineSSM] = None
        if use_ssm:
            sp = Path(ssm_path) if ssm_path else SSM_PATH
            if sp.exists():
                self.ssm = SpineSSM.load(str(sp))
                print(f'[LandmarkDetector] SSM: {sp.name}  '
                      f'({self.ssm.n_modes} modes)')
            else:
                print(f'[LandmarkDetector] WARNING: SSM not found at {sp}. '
                      f'Run: python src/spine_ssm.py')
                self.use_ssm = False

    # ── Core prediction ───────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self,
                xray:        np.ndarray,
                return_conf: bool = False,
                ) -> 'Dict[str, np.ndarray] | Tuple[Dict[str, np.ndarray], Dict[str, float]]':
        """
        Parameters
        ----------
        xray : (H, W) float32 array in [0,1] or raw uint16.
        return_conf : if True, also return CNN confidence dict.

        Returns
        -------
        coords : {vertebra_name: np.array([u_px, v_px])} in original image space.
                 With SSM enabled, always contains all 6 vertebrae.
                 CNN-detected coords have conf > threshold; SSM-completed ones
                 have conf = 0.0 in the confs dict.
        confs  : (only if return_conf=True) {name: float} — CNN peak confidence.
        """
        orig_h, orig_w = xray.shape[:2]

        # ── Preprocess ────────────────────────────────────────────────────────
        img = xray.astype(np.float32)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)
        # Apply same CLAHE used during training
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_u8).astype(np.float32) / 255.0

        img_rs = cv2.resize(img, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(self.device)

        # ── CNN forward ───────────────────────────────────────────────────────
        logits = self.model(tensor)                    # (1, C, HMAP_SIZE, HMAP_SIZE)
        hmap   = logits.sigmoid()[0].cpu().numpy()     # (C, HMAP_SIZE, HMAP_SIZE)

        sx = orig_w / HMAP_SIZE
        sy = orig_h / HMAP_SIZE

        cnn_coords: Dict[str, np.ndarray] = {}
        confs:      Dict[str, float]      = {}

        for i, name in enumerate(CLASS_NAMES):
            ch   = hmap[i]
            peak = float(ch.max())
            confs[name] = peak
            if peak < self.conf_threshold:
                continue
            flat_idx = int(ch.argmax())
            hx = (flat_idx % HMAP_SIZE + 0.5) * sx
            hy = (flat_idx // HMAP_SIZE + 0.5) * sy
            cnn_coords[name] = np.array([hx, hy], dtype=np.float64)

        # ── SSM completion ────────────────────────────────────────────────────
        if self.use_ssm and self.ssm is not None:
            # Pass CNN detections to SSM — it fits the shape model and fills
            # any missing vertebrae from the anatomical prior.
            # Even if CNN detected nothing, SSM returns the mean spine shape.
            coords = self.ssm.fit_and_complete(
                {k: v.tolist() for k, v in cnn_coords.items()},
                img_hw=(orig_h, orig_w),
            )
            # SSM-completed coords that CNN didn't detect get conf=0.0
            for name in CLASS_NAMES:
                if name not in confs:
                    confs[name] = 0.0
        else:
            coords = cnn_coords

        if return_conf:
            return coords, confs
        return coords

    # ── Batch predict from file ───────────────────────────────────────────────
    def predict_from_nrrd(self, nrrd_path: str | Path,
                          **kwargs) -> Dict[str, np.ndarray]:
        """Convenience: load NRRD and predict."""
        img = _load_xray(str(nrrd_path))
        return self.predict(img, **kwargs)

    # ── Visualisation helper ──────────────────────────────────────────────────
    def draw_predictions(self,
                         xray:    np.ndarray,
                         coords:  Dict[str, np.ndarray],
                         gt:      Optional[Dict[str, np.ndarray]] = None,
                         radius:  int = 18) -> np.ndarray:
        """
        Return an RGB uint8 overlay image with predicted (green) and
        optional GT (red) landmarks marked.
        """
        img_u8 = (np.clip(xray, 0, 1) * 255).astype(np.uint8)
        out    = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

        COLORS = {
            'D12': (255, 165,   0),   # orange
            'L1':  (  0, 255,   0),   # green
            'L2':  (  0, 200, 100),
            'L3':  (  0, 160, 200),
            'L4':  (  0, 100, 255),
            'L5':  (100,   0, 255),
        }
        for name, uv in coords.items():
            u, v = int(round(uv[0])), int(round(uv[1]))
            color = COLORS.get(name, (0, 255, 0))
            cv2.circle(out, (u, v), radius, color, 3)
            cv2.putText(out, name, (u + radius + 4, v + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if gt:
            for name, uv in gt.items():
                u, v = int(round(uv[0])), int(round(uv[1]))
                cv2.circle(out, (u, v), radius // 2, (0, 0, 255), -1)

        return out


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, json as _json
    if len(sys.argv) < 2:
        print('Usage: python landmark_detector.py <xray.nrrd> [checkpoint.pth]')
        sys.exit(1)
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    det  = LandmarkDetector(checkpoint=ckpt)
    coords, confs = det.predict_from_nrrd(sys.argv[1], return_conf=True)
    print('Predictions:')
    for name in CLASS_NAMES:
        uv = coords.get(name)
        c  = confs.get(name, 0.0)
        if uv is not None:
            print(f'  {name}: ({uv[0]:.1f}, {uv[1]:.1f})  conf={c:.3f}')
        else:
            print(f'  {name}: --- (conf={c:.3f} < threshold)')
