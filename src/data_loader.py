"""
data_loader.py
==============
Loads NRRD CT volumes, centroid labels, and intraoperative C-arm images.

Dataset layout (per patient CT session):
  - "3 L_Spine 1.0 B60s*.nrrd"    -> high-res 3D CT (1.0 mm slices)  ← PRIMARY for DRR
  - "2 L_Spine 3.0 B70s*.nrrd"    -> thick-slice 3D CT (3.0 mm slices)
  - "4 L_Spine 1.0 I26s*.nrrd"    -> iodine/contrast CT
  - "1 CT Topogram*.nrrd"          -> 2D CT Topogram (scout view, fallback only)
  - "502 Patient Protocol*.nrrd"   -> 1-slice protocol image (ignore)
  - "centroids.mrk.json"           -> 3D vertebral centroid labels (LPS mm)
  - "CENTROID.mrk.json"            -> same, alternate naming

Intraoperative C-arm images (separate folder per patient):
  ARJUN:
    - arjun c arm lateral/  -> 4 x JPEG lateral C-arm images (~999×1010 RGB)
    - Arjun carm ap/        -> 5 x JPEG AP C-arm images (~1024×1024 RGB)
  RAMULAMMA:
    - 00000001.TIF .. 00000085.TIF  -> 85 frames of uint16 1024×1024 fluoroscopy
    - CarmDummyData_19Jan2026/LAT/lt 2.bmp.dcm  -> DICOM lateral (946×1659 RGB 8-bit)
    - CarmDummyData_19Jan2026/AP/lt.bmp.dcm     -> DICOM AP (946×1659 RGB 8-bit)

The Ketcha 2017 paper:
  - PREOP CT  = 3D volume (source, has 3D labels)
  - 2D target = intraoperative C-arm radiograph  ← now correctly loaded
  - Registration: find 6DOF pose such that DRR(PREOP CT, pose) ≈ C-arm image

Coordinate system: LPS (Left-Posterior-Superior), mm
"""

import os
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import nrrd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Centroid:
    """One vertebral label point in 3D (LPS mm)."""
    label: str           # e.g. "L5", "L4", "L1", "D12", "D11"
    position: np.ndarray  # shape (3,) in LPS mm


@dataclass
class CarmImage:
    """One intraoperative C-arm image with metadata."""
    view: str                       # 'lateral' or 'ap'
    image: np.ndarray               # (H, W) float32 in [0,1], grayscale
    source_path: Path               # original file path
    pixel_spacing_mm: float = 0.0   # mm/pixel (0 = unknown)
    sdd_mm: float = 0.0             # source-to-detector distance (0 = unknown)


@dataclass
class PatientSession:
    """All data for one patient/session (preop or postop)."""
    patient_name: str         # e.g. "ARJUN"
    session_type: str         # "PREOP" or "POSTOP"
    folder: Path

    # CT volumes
    ct_volume: Optional[np.ndarray] = None      # 3D array (Z, Y, X) in HU
    ct_header: Optional[dict] = None            # NRRD header
    ct_spacing: Optional[np.ndarray] = None     # (sx, sy, sz) mm
    ct_origin: Optional[np.ndarray] = None      # LPS origin of [0,0,0] voxel (mm)
    ct_path: Optional[Path] = None

    # Topogram (2D CT scout, fallback when no C-arm available)
    topogram: Optional[np.ndarray] = None       # 2D array (H, W)
    topo_header: Optional[dict] = None
    topo_path: Optional[Path] = None

    # Intraoperative C-arm images (PRIMARY 2D target for registration)
    carm_lateral: Optional[CarmImage] = None    # best lateral C-arm frame
    carm_ap: Optional[CarmImage] = None         # best AP C-arm frame
    carm_all_lateral: List[CarmImage] = field(default_factory=list)   # all lateral frames
    carm_all_ap: List[CarmImage] = field(default_factory=list)        # all AP frames

    # Labels
    centroids: List[Centroid] = field(default_factory=list)

    def get_target_image(self, view: str = 'lateral') -> Optional[np.ndarray]:
        """
        Return the best available 2D target image for registration.
        Priority: C-arm > topogram.
        Args:
            view: 'lateral' or 'ap'
        """
        if view == 'lateral' and self.carm_lateral is not None:
            return self.carm_lateral.image
        if view == 'ap' and self.carm_ap is not None:
            return self.carm_ap.image
        # Fallback to topogram (scout view)
        return self.topogram

    def __repr__(self):
        n_labels = len(self.centroids)
        ct_shape = self.ct_volume.shape if self.ct_volume is not None else None
        carm_lat = f"CarmLat({self.carm_lateral.source_path.name[:20]}...)" if self.carm_lateral else "None"
        carm_ap  = f"CarmAP({self.carm_ap.source_path.name[:20]}...)"  if self.carm_ap  else "None"
        return (f"PatientSession({self.patient_name} {self.session_type} | "
                f"CT:{ct_shape} | C-armLat:{carm_lat} | C-armAP:{carm_ap} | Labels:{n_labels})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nrrd_get_spacing_and_origin(header: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract voxel spacing (dx, dy, dz) and world origin from NRRD header.
    Returns (spacing_xyz, origin_xyz) in mm, LPS convention.
    """
    # Space directions: each row is a direction vector (with spacing encoded)
    sd = header.get('space directions', None)
    if sd is not None:
        # sd shape: (3,3) for 3D volumes
        spacing = np.array([np.linalg.norm(v) for v in sd[:3]])
    else:
        spacing = np.array([1.0, 1.0, 1.0])

    origin_raw = header.get('space origin', None)
    if origin_raw is not None:
        origin = np.array(origin_raw, dtype=float)
    else:
        origin = np.zeros(3)

    return spacing, origin


def _load_centroids(json_path: Path) -> List[Centroid]:
    """Parse a 3D Slicer .mrk.json fiducial file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    centroids = []
    for markup in data.get('markups', []):
        for cp in markup.get('controlPoints', []):
            pos = np.array(cp['position'], dtype=float)  # LPS mm
            lbl = cp.get('label', cp.get('id', '?'))
            centroids.append(Centroid(label=lbl, position=pos))
    return centroids


def _find_file(folder: Path, pattern: str) -> Optional[Path]:
    """Find first file matching a glob pattern (case-insensitive)."""
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def _find_best_centroid_file(folder: Path) -> Optional[Path]:
    """
    Find the most useful centroid file. Prefer the one with more labels,
    and prefer lowercase 'centroids.mrk.json' over 'CENTROID.mrk.json'
    if both exist (the all-caps ones seem to be copies).
    """
    candidates = list(folder.glob('*.mrk.json'))
    if not candidates:
        return None
    # Score by number of control points
    best = None
    best_n = -1
    for c in candidates:
        try:
            with open(c) as f:
                d = json.load(f)
            n = sum(len(m.get('controlPoints', [])) for m in d.get('markups', []))
            if n > best_n:
                best_n = n
                best = c
        except Exception:
            continue
    return best


def _load_topogram(folder: Path) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Path]]:
    """
    Load the CT Topogram (2D projection image, 512×512×1).
    Prefers the simple single-frame '* 0.nrrd' variant.
    Returns (image_2d, header, path) or (None, None, None).
    """
    # Try the single-frame variant first (ends with '0.nrrd')
    candidates = sorted(folder.glob('*Topogram* 0.nrrd')) + \
                 sorted(folder.glob('*Topogram*.nrrd'))
    for c in candidates:
        if '.seq.nrrd' in c.name:
            continue
        try:
            data, hdr = nrrd.read(str(c))
            # Shape should be (512, 512, 1) or similar
            # Squeeze the last singleton dimension
            if data.ndim == 3 and data.shape[2] == 1:
                img = data[:, :, 0].astype(np.float32)
            elif data.ndim == 2:
                img = data.astype(np.float32)
            else:
                img = data[..., 0].astype(np.float32) if data.shape[-1] == 1 else data.astype(np.float32)
            return img, hdr, c
        except Exception as e:
            print(f"  [warn] Could not load topogram {c.name}: {e}")
            continue
    return None, None, None


def _jpeg_to_gray_float(path: Path) -> np.ndarray:
    """Load JPEG/PNG C-arm image, convert RGB -> grayscale float32 [0,1]."""
    from PIL import Image as PILImage
    img = PILImage.open(str(path)).convert('L')   # grayscale
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def _tif_to_gray_float(path: Path) -> np.ndarray:
    """Load 16-bit TIF C-arm fluoroscopy frame -> float32 [0,1]."""
    from PIL import Image as PILImage
    img = PILImage.open(str(path))
    arr = np.array(img, dtype=np.float32)
    # 16-bit TIF: values in [0, 65535]
    if arr.max() > 1.0:
        arr = arr / arr.max()
    return arr


def _dcm_to_gray_float(path: Path) -> np.ndarray:
    """Load DICOM C-arm image -> float32 [0,1]."""
    import pydicom
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    # If RGB, convert to grayscale via luminance
    if arr.ndim == 3:
        arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    if arr.max() > 1.0:
        arr = arr / arr.max()
    return arr


def _bmp_to_gray_float(path: Path) -> np.ndarray:
    """Load BMP image -> float32 [0,1]."""
    from PIL import Image as PILImage
    img = PILImage.open(str(path)).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def _select_sharpest(images: List[np.ndarray]) -> int:
    """
    Select the sharpest image from a list using Laplacian variance.
    Returns index of sharpest image.
    """
    from scipy.ndimage import laplace
    best_idx, best_score = 0, -1.0
    for i, img in enumerate(images):
        lap = laplace(img)
        score = float(np.var(lap))
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx


def _load_carm_images(carm_folder: Path,
                      patient_name: str
                      ) -> Tuple[List['CarmImage'], List['CarmImage']]:
    """
    Load C-arm intraoperative images from the dedicated C-arm folder.
    Returns (lateral_list, ap_list) where each is a list of CarmImage.

    Supported formats:
      - JPEG: ARJUN C-arm lateral and AP images
      - TIFF 16-bit: RAMULAMMA fluoroscopy sequence (treat all as lateral)
      - DICOM BMP-wrapped: RAMULAMMA AP/LAT DICOM files

    For TIFF sequences, loads all frames and picks the sharpest as the primary.
    """
    lateral_list: List[CarmImage] = []
    ap_list: List[CarmImage]      = []

    if not carm_folder.exists():
        return lateral_list, ap_list

    # Walk the folder
    for root, dirs, files in os.walk(str(carm_folder)):
        root_path = Path(root)
        folder_lower = root_path.name.lower()

        # ---- JPEG C-arm images (ARJUN) ----
        jpegs = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg'))])
        if jpegs:
            view = 'lateral' if 'lateral' in folder_lower else ('ap' if 'ap' in folder_lower else 'lateral')
            for fname in jpegs:
                p = root_path / fname
                try:
                    arr = _jpeg_to_gray_float(p)
                    ci = CarmImage(view=view, image=arr, source_path=p)
                    if view == 'lateral':
                        lateral_list.append(ci)
                    else:
                        ap_list.append(ci)
                except Exception as e:
                    print(f"  [warn] Could not load C-arm JPEG {p.name}: {e}")

        # ---- TIFF fluoroscopy sequence (RAMULAMMA) ----
        tifs = sorted([f for f in files if f.upper().endswith('.TIF') and
                       f[:-4].isdigit()])  # only numbered frames
        if tifs:
            # Load ALL frames, then pick sharpest as primary
            frames = []
            frame_paths = []
            for fname in tifs:
                p = root_path / fname
                try:
                    arr = _tif_to_gray_float(p)
                    frames.append(arr)
                    frame_paths.append(p)
                except Exception:
                    pass

            if frames:
                # All TIF frames treated as lateral (C-arm fluoroscopy, lateral view)
                best_i = _select_sharpest(frames)
                print(f"  [carm] TIFF seq: {len(frames)} frames, sharpest={tifs[best_i]}")
                for i, (arr, p) in enumerate(zip(frames, frame_paths)):
                    ci = CarmImage(view='lateral', image=arr, source_path=p)
                    if i == best_i:
                        lateral_list.insert(0, ci)   # sharpest goes first
                    else:
                        lateral_list.append(ci)

        # ---- DICOM C-arm (RAMULAMMA CarmDummyData) ----
        dcms = [f for f in files if f.lower().endswith('.dcm')]
        if dcms:
            for fname in dcms:
                p = root_path / fname
                # Infer view from parent folder name
                parent_lower = root_path.name.lower()
                if parent_lower == 'lat':
                    view = 'lateral'
                elif parent_lower == 'ap':
                    view = 'ap'
                else:
                    view = 'lateral'
                try:
                    arr = _dcm_to_gray_float(p)
                    ci = CarmImage(view=view, image=arr, source_path=p)
                    if view == 'lateral':
                        lateral_list.append(ci)
                    else:
                        ap_list.append(ci)
                    print(f"  [carm] DICOM {view}: {p.name}, shape={arr.shape}")
                except Exception as e:
                    print(f"  [warn] Could not load DICOM {p.name}: {e}")

    return lateral_list, ap_list


def _find_carm_folder(data_root: Path, patient_name: str) -> Optional[Path]:
    """
    Find the C-arm folder for a patient in the data root.
    The C-arm folders are siblings of the CT session folders under data_root.
    """
    pname_upper = patient_name.upper()
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        dname = d.name.upper()
        if 'C ARM' not in dname and 'CARM' not in dname:
            continue
        if pname_upper not in dname:
            continue
        # The actual images are in the inner subdirectory
        inner_dirs = [x for x in d.iterdir() if x.is_dir()]
        if inner_dirs:
            return inner_dirs[0]
        return d
    return None


def _load_ct_volume(folder: Path) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[Path]]:
    """
    Load the best available 3D CT volume.
    Priority: high-res B60s (1.0mm) > B70s (3.0mm) > I26s (iodine, fallback)
    Returns (volume_zyx, header, path).
    The volume is returned as (Z, Y, X) with HU values.
    """
    priority_patterns = [
        '*B60s*.nrrd',   # 1 mm bone reconstruction — best for DRR
        '*B70s*.nrrd',   # 3 mm — coarser but okay
        '*I26s*.nrrd',   # iodine series
    ]
    for pattern in priority_patterns:
        candidates = sorted(folder.glob(pattern))
        for c in candidates:
            if '.seq.nrrd' in c.name:
                continue
            try:
                data, hdr = nrrd.read(str(c))
                # NRRD pynrrd loads as (X, Y, Z) by default (fastest varying = last)
                # but nrrd.read actually returns data in file order.
                # The NRRD header says dimension=3 with sizes = Nx Ny Nz
                # In practice for these files: sizes = 512 512 Nz
                # After nrrd.read: shape is (512, 512, Nz) = (X, Y, Z)
                # We want (Z, Y, X) for typical medical image convention
                if data.ndim == 3:
                    vol = data.transpose(2, 1, 0).astype(np.float32)  # -> (Z, Y, X)
                else:
                    vol = data.astype(np.float32)
                return vol, hdr, c
            except Exception as e:
                print(f"  [warn] Could not load CT {c.name}: {e}")
                continue
    return None, None, None


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_patient_session(folder: Path,
                          carm_folder: Optional[Path] = None) -> 'PatientSession':
    """
    Load a single patient session from a folder.
    Auto-detects patient name and session type from folder path.

    Args:
        folder     : CT session folder (contains NRRD + .mrk.json)
        carm_folder: Optional intraoperative C-arm image folder.
                     If provided, loads C-arm images and attaches to session.
    """
    folder_str = folder.name.upper()

    # Parse patient name and session type
    if 'ARJUN' in folder_str:
        patient_name = 'ARJUN'
    elif 'RAMULAMMA' in folder_str:
        patient_name = 'RAMULAMMA'
    elif 'SAMRAJYAM' in folder_str:
        patient_name = 'SAMRAJYAM'
    else:
        patient_name = folder.name

    if 'PREOP' in folder_str or 'PRE OP' in folder_str:
        session_type = 'PREOP'
    elif 'POST' in folder_str:
        session_type = 'POSTOP'
    else:
        session_type = 'UNKNOWN'

    session = PatientSession(
        patient_name=patient_name,
        session_type=session_type,
        folder=folder
    )

    # Load CT volume
    ct_vol, ct_hdr, ct_path = _load_ct_volume(folder)
    if ct_vol is not None:
        spacing, origin = _nrrd_get_spacing_and_origin(ct_hdr)
        session.ct_volume = ct_vol
        session.ct_header = ct_hdr
        session.ct_spacing = spacing
        session.ct_origin = origin
        session.ct_path = ct_path

    # Load topogram (fallback 2D target when no C-arm available)
    topo, topo_hdr, topo_path = _load_topogram(folder)
    if topo is not None:
        session.topogram = topo
        session.topo_header = topo_hdr
        session.topo_path = topo_path

    # Load centroids
    centroid_file = _find_best_centroid_file(folder)
    if centroid_file is not None:
        session.centroids = _load_centroids(centroid_file)

    # Load C-arm images (PRIMARY 2D target)
    if carm_folder is not None and carm_folder.exists():
        print(f"  Loading C-arm images from: {carm_folder.name} ...")
        lat_list, ap_list = _load_carm_images(carm_folder, patient_name)
        session.carm_all_lateral = lat_list
        session.carm_all_ap      = ap_list
        # Primary = first (sharpest already sorted to front)
        if lat_list:
            session.carm_lateral = lat_list[0]
            print(f"    -> {len(lat_list)} lateral C-arm frame(s), primary: {lat_list[0].source_path.name}")
        if ap_list:
            session.carm_ap = ap_list[0]
            print(f"    -> {len(ap_list)} AP C-arm frame(s), primary: {ap_list[0].source_path.name}")

    return session


def discover_sessions(data_root: Path) -> Dict[str, Dict[str, PatientSession]]:
    """
    Walk data_root, discover all patient sessions and their C-arm images.

    The data root contains:
      - CT session folders (named PATIENT_NAME PREOP/POSTOP-...)
      - C-arm image folders (named "Patient's C arm intra op images-...")

    C-arm folders are attached to PREOP sessions (since C-arm is taken
    during the same surgical procedure, providing the 2D registration target).
    Some implementations attach to POSTOP; here we follow the paper's logic
    where C-arm is the intraoperative 2D target acquired alongside the surgery.

    Returns dict: {patient_name: {'PREOP': session, 'POSTOP': session}}
    """
    sessions: Dict[str, Dict[str, PatientSession]] = {}

    # First pass: discover and load all CT sessions
    for outer in sorted(data_root.iterdir()):
        if not outer.is_dir():
            continue
        outer_upper = outer.name.upper()

        # Skip C-arm folders (no CT data inside)
        if 'C ARM' in outer_upper or 'CARM' in outer_upper:
            continue

        # The actual data is in the inner subdirectory
        inner_dirs = [d for d in outer.iterdir() if d.is_dir()]
        if not inner_dirs:
            continue
        inner = inner_dirs[0]  # always one inner folder

        print(f"Loading CT: {inner.name} ...")
        sess = load_patient_session(inner)
        print(f"  -> {sess}")

        pname = sess.patient_name
        if pname not in sessions:
            sessions[pname] = {}

        if sess.session_type in sessions[pname]:
            existing = sessions[pname][sess.session_type]
            # Keep the one with more centroids
            if len(sess.centroids) > len(existing.centroids):
                sessions[pname][sess.session_type] = sess
        else:
            sessions[pname][sess.session_type] = sess

    # Second pass: attach C-arm images to the appropriate session
    # C-arm is taken intraoperatively -> attach to PREOP session
    # (The PREOP CT + C-arm = the registration pair; POSTOP CT is the GT)
    for pname, patient_sessions in sessions.items():
        carm_folder = _find_carm_folder(data_root, pname)
        if carm_folder is None:
            print(f"  [warn] No C-arm folder found for {pname}")
            continue

        print(f"  Loading C-arm images for {pname} from: {carm_folder.name} ...")
        lat_list, ap_list = _load_carm_images(carm_folder, pname)

        # Attach to PREOP session (primary registration source)
        if 'PREOP' in patient_sessions:
            sess = patient_sessions['PREOP']
            sess.carm_all_lateral = lat_list
            sess.carm_all_ap      = ap_list
            if lat_list:
                sess.carm_lateral = lat_list[0]
                print(f"    PREOP lateral: {len(lat_list)} frame(s), primary={lat_list[0].source_path.name}")
            if ap_list:
                sess.carm_ap = ap_list[0]
                print(f"    PREOP AP: {len(ap_list)} frame(s), primary={ap_list[0].source_path.name}")

    return sessions


def get_registration_pairs(sessions: Dict[str, Dict[str, PatientSession]]):
    """
    Return list of (preop_session, postop_session, patient_name) pairs
    where both preop and postop exist.
    """
    pairs = []
    for pname, s in sessions.items():
        if 'PREOP' in s and 'POSTOP' in s:
            pairs.append((s['PREOP'], s['POSTOP'], pname))
        else:
            print(f"  [warn] {pname} missing {'PREOP' if 'PREOP' not in s else 'POSTOP'}")
    return pairs


if __name__ == '__main__':
    data_root = Path('/home/supermicro/Documents/2D_3D_Raghu/data')
    sessions = discover_sessions(data_root)
    print("\n=== Sessions discovered ===")
    for pname, s in sessions.items():
        print(f"\nPatient: {pname}")
        for stype, sess in s.items():
            print(f"  {stype}: {sess}")
            if sess.carm_lateral:
                print(f"    C-arm lateral: {sess.carm_lateral.image.shape}")
            if sess.carm_ap:
                print(f"    C-arm AP:      {sess.carm_ap.image.shape}")
            for c in sess.centroids:
                print(f"    Label: {c.label}  pos: {c.position.round(1)}")

    print("\n=== Registration pairs ===")
    pairs = get_registration_pairs(sessions)
    for pre, post, pname in pairs:
        has_carm = "YES" if pre.carm_lateral else "NO"
        print(f"  {pname}: PREOP labels={len(pre.centroids)}, C-arm={has_carm}, POSTOP labels={len(post.centroids)}")

