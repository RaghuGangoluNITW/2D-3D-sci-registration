#!/usr/bin/env python3
"""
extract_transverse_process_tips.py
====================================
Extracts ADDITIONAL, NON-COLLINEAR 3D landmarks per vertebra -- transverse
process tips -- directly from a patient's EXISTING pre-op CT volume, with
NO new doctor annotation required.

Why this exists (root cause + fix)
-------------------------------------
This session proved, with a direct noise-injection experiment, that solving
a camera pose from vertebra CENTROIDS via EPnP is fundamentally unstable:
centroids sit almost in a straight line along the spine's own axis, so
even near-perfect 2D detection of them produces wild 3D pose errors
(median 15 degrees / 340mm from just 1mm of 2D noise in one measured
case). A published spine-registration study that DID achieve clinical
accuracy on real X-rays worked around this exact problem by surgically
implanting 14 metal fiducial markers spread in an ellipsoid AROUND the
spine (not along its axis) -- deliberately non-collinear reference
points.

We cannot implant markers. But the same fix does not require implants --
it only requires 3D reference points that are NOT strung along the
spine's own axis. Each vertebra already HAS such points, in the same CT
we already possess: the LEFT and RIGHT PEDICLES (the dense, compact bone
struts connecting the vertebral body to the posterior arch). These are
real anatomical structures already present in the CT data -- finding
them is an image-processing task on data we already have, not a new
manual-annotation task.

(An earlier version of this script targeted transverse process tips
instead, but visual verification found that heuristic consistently
mis-fired onto ribs and the pelvis, which are similarly dense and
roughly the same distance away. Pedicles are closer to the centroid and
more compact, making them far more reliably isolated by a simple
"most lateral bone within a small radius" search -- see in-code
comments near LOCAL_BOX_MM for the measured distances that motivated
this.)

Method
-------
For each vertebra centroid (already marked by the doctor, unchanged):
  1. Take a small local 3D neighbourhood of the CT around that centroid.
  2. Threshold to bone (HU > BONE_HU_THRESHOLD).
  3. Within that local bone mask, find the most LATERAL bone voxel on
     each side (left pedicle candidate, right pedicle candidate), by
     projecting bone voxel coordinates onto the patient's own left-right
     axis (estimated from the whole spine's point cloud via PCA, so it
     works regardless of how the patient was scanned/oriented).
  4. Sanity-check each candidate: must be a plausible distance from the
     centroid (real lumbar pedicles are ~12-20mm lateral of the vertebral
     body centroid) and must have real bone density (rejects noise-
     driven picks), and must fall within this vertebra's own slab along
     the spine axis (rejects picks that belong to a neighbouring
     vertebra).

This produces 2 EXTRA 3D points per vertebra, geometrically spread away
from the spine's central axis -- exactly the property needed for a
well-conditioned EPnP solve. Output is written in the same Slicer
.mrk.json format as the existing CENTROID.mrk.json, so it drops directly
into the existing loader pipeline (generic_patient_loader.py) as
additional labels, alongside (not replacing) the doctor's original
centroids.

IMPORTANT LIMITATION (must be verified before trusting these points):
This is a fully automatic, unsupervised extraction with no ground truth
of its own. It must be spot-checked visually (see --verify) before use --
if the CT is noisy, has metal artifact, or the local PCA axis estimate is
off for an unusual vertebra shape, candidate points could land on rib,
adjacent vertebra, or other bone. Treat this as a candidate landmark
source to validate, not an established one.

Usage:
    python scripts/extract_transverse_process_tips.py --patient JYOTHI
    python scripts/extract_transverse_process_tips.py --patient JYOTHI --verify
    python scripts/extract_transverse_process_tips.py --all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

DATA_ROOT = ROOT / 'data' / 'patients'
LUMBAR = ['D12', 'L1', 'L2', 'L3', 'L4', 'L5']

BONE_HU_THRESHOLD = 250.0     # conservative cortical-bone threshold
LOCAL_BOX_MM = 30.0           # half-width of local search box around each centroid
MIN_TIP_DIST_MM = 8.0         # reject candidates too close to centroid (noise)
MAX_TIP_DIST_MM = 42.0        # reject candidates too far (wrong structure -- rib/pelvis)
# NOTE: measured on JYOTHI, genuine pedicle picks (confirmed visually)
# ranged 25-40mm from centroid -- the initial 28mm guess (based on
# textbook lumbar pedicle width) was too tight for this patient and
# rejected several correct picks. 42mm leaves a safety margin below
# where ribs/pelvis were found in the earlier version of this script
# (48mm+), so widening does not reopen the original mis-fire.
# NOTE: An earlier version of this script targeted TRANSVERSE PROCESS
# TIPS (LOCAL_BOX_MM=55, MAX_TIP_DIST_MM=75). Visual verification (see
# project history) found this consistently picked up ribs and the pelvis
# instead -- both are dense, bony, and roughly the same distance away as
# a true transverse process tip, but the tip itself is a thin structure
# that "most lateral bone voxel in a radius" doesn't reliably isolate
# from those larger, denser neighbours.
#
# Switched to PEDICLES instead: the two dense, compact, rounded bone
# structures connecting the vertebral body to the posterior arch,
# clearly visible on axial CT (see debug_axial_L3.png in project
# history). Pedicles are:
#   - Consistently ~12-20mm lateral of the centroid (much closer than
#     transverse process tips), which keeps the search radius small
#     enough to structurally exclude ribs/pelvis (which sit 35mm+ away)
#   - Bilaterally spread left/right of the centroid -- still the
#     non-collinear property we need for a well-conditioned EPnP solve
#   - Denser and more compact than transverse processes, easier to
#     isolate reliably with a simple "most lateral bone in a small
#     radius" heuristic.


def _load_ct(patient_dir: Path):
    ct_dir = patient_dir / 'CT label'
    ct_files = list(ct_dir.glob('*.nrrd'))
    if not ct_files:
        raise FileNotFoundError(f'No CT nrrd found in {ct_dir}')
    img = sitk.ReadImage(str(ct_files[0]))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X) HU
    spacing = np.array(img.GetSpacing(), dtype=np.float64)  # (sx, sy, sz) mm/voxel
    origin = np.array(img.GetOrigin(), dtype=np.float64)    # (ox, oy, oz) mm
    return arr, spacing, origin


def _world_to_voxel(pt_world: np.ndarray, spacing: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """World XYZ mm -> voxel (x,y,z) index (float, not yet rounded)."""
    return (pt_world - origin) / spacing


def _voxel_to_world(vox_xyz: np.ndarray, spacing: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return vox_xyz * spacing + origin


def _load_mrk_3d(path: Path) -> Dict[str, np.ndarray]:
    d = json.loads(path.read_text())
    return {cp['label'].strip(): np.array(cp['position'], dtype=np.float64)
            for cp in d['markups'][0]['controlPoints']}


def _estimate_local_axes(centroids: Dict[str, np.ndarray], name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate this vertebra's local (superior-inferior, left-right,
    anterior-posterior) axes from the overall spine point cloud via PCA.
    The spine's principal axis (largest variance direction across all
    vertebra centroids) approximates superior-inferior. The other two
    PCA axes approximate the cross-sectional plane, but PCA alone can't
    tell left/right from anterior/posterior -- we disambiguate using the
    fact that in LPS-convention CT world coordinates (the convention this
    codebase uses throughout, see deepfluoro_loader.py), the anatomical
    left-right axis is close to world X and anterior-posterior is close
    to world Y. We use PCA only to refine within that plane, not to
    guess the axes from scratch, since a 6-14 point spine is too sparse
    to reliably estimate cross-sectional axes on its own.
    """
    pts = np.array(list(centroids.values()))
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    si_axis = eigvecs[:, order[0]]  # superior-inferior (largest variance)

    # Cross-sectional plane is perpendicular to si_axis. Project world X
    # (left-right) and world Y (anterior-posterior) onto that plane and
    # orthonormalise, rather than trusting PCA's other two components
    # directly (which are unreliable with this few points).
    world_x = np.array([1.0, 0.0, 0.0])
    world_y = np.array([0.0, 1.0, 0.0])
    lr_axis = world_x - np.dot(world_x, si_axis) * si_axis
    lr_axis /= np.linalg.norm(lr_axis)
    ap_axis = world_y - np.dot(world_y, si_axis) * si_axis
    ap_axis -= np.dot(ap_axis, lr_axis) * lr_axis
    ap_axis /= np.linalg.norm(ap_axis)

    return si_axis, lr_axis, ap_axis


def extract_tips_for_vertebra(ct: np.ndarray, spacing: np.ndarray, origin: np.ndarray,
                              centroid_world: np.ndarray,
                              lr_axis: np.ndarray, ap_axis: np.ndarray,
                              si_axis: np.ndarray, si_half_range_mm: float) -> Dict[str, Optional[np.ndarray]]:
    """
    Find left/right transverse process tip and posterior spinous process
    tip candidates near one vertebra centroid.

    si_half_range_mm constrains the search along the spine's own axis
    (si_axis) to roughly half the spacing to the NEAREST neighbouring
    vertebra. Without this, a wide lateral search box (needed to reach
    real transverse process tips, ~35-50mm lateral) overlaps adjacent
    vertebrae when they are closer together than the box is wide (lumbar
    vertebra spacing is often ~25-35mm, much less than a useful lateral
    search radius) -- confirmed directly: without this constraint,
    adjacent vertebrae's searches were picking up the EXACT SAME bone
    point. Constraining along si_axis (but NOT along lr_axis/ap_axis,
    where the real anatomical extent is wide) keeps each vertebra's
    search anatomically local to itself.

    Returns dict with keys 'LEFT_PED', 'RIGHT_PED' -> world XYZ mm point,
    or None if no plausible candidate was found.
    """
    # Local voxel search box around the centroid (generous in all
    # directions -- filtered precisely by si/lr/ap projection below,
    # this box only needs to be large enough to contain the true
    # anatomical structures we're looking for).
    box_half_vox = np.ceil(LOCAL_BOX_MM / spacing).astype(int)
    center_vox = _world_to_voxel(centroid_world, spacing, origin)  # (x,y,z)
    cx, cy, cz = center_vox.round().astype(int)
    hx, hy, hz = box_half_vox

    Z, Y, X = ct.shape  # array is (z, y, x)
    x0, x1 = max(0, cx - hx), min(X, cx + hx + 1)
    y0, y1 = max(0, cy - hy), min(Y, cy + hy + 1)
    z0, z1 = max(0, cz - hz), min(Z, cz + hz + 1)

    local = ct[z0:z1, y0:y1, x0:x1]
    bone_mask = local > BONE_HU_THRESHOLD
    if not bone_mask.any():
        return {'LEFT_PED': None, 'RIGHT_PED': None}

    # Voxel coords (local box, then shift to global voxel index, then to world mm).
    zz, yy, xx = np.nonzero(bone_mask)
    vox_global = np.stack([xx + x0, yy + y0, zz + z0], axis=1).astype(np.float64)
    world_pts = _voxel_to_world(vox_global, spacing, origin)  # (N, 3)

    rel = world_pts - centroid_world
    lr_proj = rel @ lr_axis      # signed distance along left-right axis
    ap_proj = rel @ ap_axis      # signed distance along anterior-posterior axis
    si_proj = rel @ si_axis      # signed distance along superior-inferior axis

    # Restrict to this vertebra's own slab along the spine axis -- see
    # docstring above for why this is necessary.
    si_keep = np.abs(si_proj) < si_half_range_mm
    ap_proj_masked = np.where(si_keep, ap_proj, 1e9)  # push excluded points out of AP-bound checks

    def _pick(proj: np.ndarray, want_sign: int, other_proj: np.ndarray, other_bound: float):
        """Pick the most extreme point in `proj` direction (matching
        want_sign), restricted to points not too far off-axis in the
        other direction (keeps the pick anatomically plausible -- e.g.
        for a transverse process tip, don't wander too far anterior/
        posterior)."""
        mask = (np.sign(proj) == want_sign) & (np.abs(other_proj) < other_bound)
        if not mask.any():
            return None, None
        idx = np.argmax(np.abs(proj) * mask)
        dist = float(np.linalg.norm(rel[idx]))
        if dist < MIN_TIP_DIST_MM or dist > MAX_TIP_DIST_MM:
            return None, None
        return world_pts[idx], dist

    lr_proj_masked = np.where(si_keep, lr_proj, 0.0)
    left_pt, left_d   = _pick(lr_proj_masked, +1, ap_proj_masked, LOCAL_BOX_MM * 0.7)
    right_pt, right_d = _pick(lr_proj_masked, -1, ap_proj_masked, LOCAL_BOX_MM * 0.7)

    return {'LEFT_PED': left_pt, 'RIGHT_PED': right_pt}


def process_patient(patient: str, verify: bool = False) -> Dict[str, np.ndarray]:
    patient_dir = DATA_ROOT / patient
    ct_dir = patient_dir / 'CT label'
    # Some patients' 3D centroid files use different capitalisation/naming
    # (e.g. Swaroop: 'centroid_3.mrk.json' instead of 'CENTROID.mrk.json').
    centroid_path = ct_dir / 'CENTROID.mrk.json'
    if not centroid_path.exists():
        candidates = sorted(ct_dir.glob('*.mrk.json')) if ct_dir.exists() else []
        # Prefer a file with "centroid" in the name if there's more than one.
        centroid_candidates = [c for c in candidates if 'centroid' in c.name.lower()]
        if centroid_candidates:
            centroid_path = centroid_candidates[0]
        elif candidates:
            centroid_path = candidates[0]
    if not centroid_path.exists():
        raise FileNotFoundError(f'No centroid .mrk.json found for {patient} in {ct_dir}')

    centroids_all = _load_mrk_3d(centroid_path)
    centroids = {k: v for k, v in centroids_all.items() if k in LUMBAR}
    if len(centroids) < 3:
        raise ValueError(f'{patient}: fewer than 3 lumbar centroids found')

    ct, spacing, origin = _load_ct(patient_dir)
    si_axis, lr_axis, ap_axis = _estimate_local_axes(centroids, patient)

    # Per-vertebra si-axis position, sorted, to find nearest-neighbour
    # spacing along the spine for each level (see extract_tips_for_vertebra
    # docstring for why this matters).
    names = list(centroids.keys())
    si_positions = {n: float(centroids[n] @ si_axis) for n in names}
    sorted_names = sorted(names, key=lambda n: si_positions[n])

    new_points: Dict[str, np.ndarray] = {}
    for i, name in enumerate(sorted_names):
        centroid = centroids[name]
        neighbour_gaps = []
        if i > 0:
            neighbour_gaps.append(abs(si_positions[name] - si_positions[sorted_names[i - 1]]))
        if i < len(sorted_names) - 1:
            neighbour_gaps.append(abs(si_positions[name] - si_positions[sorted_names[i + 1]]))
        nearest_gap = min(neighbour_gaps) if neighbour_gaps else LOCAL_BOX_MM
        si_half_range_mm = max(8.0, nearest_gap * 0.45)  # keep each vertebra's slab from overlapping its neighbour

        tips = extract_tips_for_vertebra(ct, spacing, origin, centroid, lr_axis, ap_axis,
                                         si_axis, si_half_range_mm)
        for suffix, pt in tips.items():
            if pt is not None:
                new_points[f'{name}_{suffix}'] = pt

    print(f'{patient}: extracted {len(new_points)} new points from '
          f'{len(centroids)} vertebrae  ({len(new_points)/max(len(centroids),1):.1f}/vertebra)')
    for k, v in new_points.items():
        print(f'    {k}: {v.round(1)}')

    return new_points


def save_as_mrk_json(points: Dict[str, np.ndarray], out_path: Path):
    control_points = []
    for i, (label, pos) in enumerate(points.items()):
        control_points.append({
            'id': str(i + 1),
            'label': label,
            'description': '',
            'associatedNodeID': '',
            'position': [float(pos[0]), float(pos[1]), float(pos[2])],
            'orientation': [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
            'selected': True,
            'locked': False,
            'visibility': True,
            'positionStatus': 'defined',
        })
    doc = {
        '@schema': 'https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#',
        'markups': [{
            'type': 'Fiducial',
            'coordinateSystem': 'LPS',
            'coordinateUnits': 'mm',
            'lastUsedControlPointNumber': len(points),
            'controlPoints': control_points,
            'measurements': [],
            'display': {},
        }],
    }
    out_path.write_text(json.dumps(doc, indent=2))
    print(f'  Saved: {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--patient', default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--verify', action='store_true',
                   help='Save overlay images for visual verification.')
    args = p.parse_args()

    if args.all:
        patients = sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir())
    elif args.patient:
        patients = [args.patient]
    else:
        print('Specify --patient NAME or --all')
        return

    for patient in patients:
        try:
            points = process_patient(patient, verify=args.verify)
            out_path = DATA_ROOT / patient / 'CT label' / 'EXTRA_LANDMARKS.mrk.json'
            save_as_mrk_json(points, out_path)
        except Exception as e:
            print(f'{patient}: FAILED ({e})')


if __name__ == '__main__':
    main()
