"""
camera_config.py — C-arm camera model loader
=============================================
Replaces hardcoded camera constants with per-machine JSON config files
stored in configs/cameras/.

Usage
-----
    from camera_config import load_camera_config, list_cameras

    cam = load_camera_config("ziehm_vision_fd")   # default
    cam = load_camera_config("siemens_cios_spin")
    cam = load_camera_config()                     # uses default

    print(cam.fx, cam.fy, cam.cx, cam.cy)
    print(cam.K)            # 3x3 intrinsic matrix

    # Pass as --camera argument from command line
    python run_new_patients.py --camera siemens_cios_spin

To add a new camera:
    1. Create configs/cameras/<your_model>.json (see existing files for format)
    2. Run: python -c "from src.camera_config import list_cameras; print(list_cameras())"
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_SRC_DIR     = Path(__file__).parent
_CONFIGS_DIR = _SRC_DIR.parent / "configs" / "cameras"
DEFAULT_CAMERA = "ziehm_vision_fd"


# ── dataclass ─────────────────────────────────────────────────────────────────
@dataclass
class CameraConfig:
    """
    Intrinsic parameters for a C-arm X-ray machine.

    Attributes
    ----------
    name              : human-readable model name
    sid_mm            : Source-to-Image Distance in mm
    pixel_spacing_mm  : pixel size at the detector plane (mm/px)
    image_size_px     : square image dimension in pixels
    """
    name:             str
    sid_mm:           float
    pixel_spacing_mm: float
    image_size_px:    int

    # ── derived intrinsics ────────────────────────────────────────────────
    @property
    def fx(self) -> float:
        """Focal length in pixels (x-axis)."""
        return self.sid_mm / self.pixel_spacing_mm

    @property
    def fy(self) -> float:
        """Focal length in pixels (y-axis)."""
        return self.sid_mm / self.pixel_spacing_mm

    @property
    def cx(self) -> float:
        """Principal point x (pixels) — image centre."""
        return (self.image_size_px - 1) / 2.0

    @property
    def cy(self) -> float:
        """Principal point y (pixels) — image centre."""
        return (self.image_size_px - 1) / 2.0

    @property
    def K(self) -> np.ndarray:
        """3×3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0.,      self.cx],
            [0.,      self.fy, self.cy],
            [0.,      0.,      1.     ],
        ], dtype=np.float64)

    def __str__(self) -> str:
        return (
            f"CameraConfig({self.name!r}: "
            f"SID={self.sid_mm}mm, "
            f"pix={self.pixel_spacing_mm}mm/px, "
            f"size={self.image_size_px}px, "
            f"Fx={self.fx:.1f}px)"
        )


# ── loader ────────────────────────────────────────────────────────────────────
def load_camera_config(name: str = DEFAULT_CAMERA) -> CameraConfig:
    """
    Load a camera config by model name.

    Parameters
    ----------
    name : filename stem of the JSON config (without .json extension),
           e.g. "ziehm_vision_fd", "siemens_cios_spin", "ge_oec_9900".
           Defaults to "ziehm_vision_fd".

    Returns
    -------
    CameraConfig instance

    Raises
    ------
    FileNotFoundError if the named config does not exist.
    """
    path = _CONFIGS_DIR / f"{name}.json"
    if not path.exists():
        available = list_cameras()
        raise FileNotFoundError(
            f"Camera config '{name}' not found.\n"
            f"Config directory: {_CONFIGS_DIR}\n"
            f"Available configs: {available}\n"
            f"To add a new camera, create: {_CONFIGS_DIR}/{name}.json"
        )
    d = json.loads(path.read_text())
    return CameraConfig(
        name=d["name"],
        sid_mm=float(d["sid_mm"]),
        pixel_spacing_mm=float(d["pixel_spacing_mm"]),
        image_size_px=int(d["image_size_px"]),
    )


def list_cameras() -> list:
    """Return list of available camera config names (without .json extension)."""
    if not _CONFIGS_DIR.exists():
        return []
    return sorted(p.stem for p in _CONFIGS_DIR.glob("*.json"))


# ── module-level default (backward compat) ────────────────────────────────────
# Modules that previously imported CAM_SID_MM etc. from generic_patient_loader
# can instead do:   from camera_config import DEFAULT_CAM as cam
# and use cam.sid_mm, cam.fx, cam.pixel_spacing_mm etc.
DEFAULT_CAM: CameraConfig = load_camera_config(DEFAULT_CAMERA)


if __name__ == "__main__":
    print(f"Available camera configs: {list_cameras()}")
    for cname in list_cameras():
        cam = load_camera_config(cname)
        print(f"  {cam}")
