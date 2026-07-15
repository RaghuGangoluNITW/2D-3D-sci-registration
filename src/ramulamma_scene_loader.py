"""
ramulamma_scene_loader.py
=========================
Load Ramulamma intra-op data strictly from the Slicer scene
`data/testing/RAMULAMMA INTRAOP/2026-02-25-Scene.mrml`.

This loader uses only what the scene declares as active:
- Active intra-op volume: `vtkMRMLScalarVolumeNode9` -> `Data/1.nrrd`
- Active markup node: `vtkMRMLMarkupsFiducialNode3` -> `Data/CENTROIDS-2.mrk.json`

The active markup contains 3 labelled 2D vertebral centroids: L3, L4, L5
on slice/frame z=6 of the active 107-frame stack.

We keep the pre-op CT and 3D centroids from the Ramulamma PREOP folder.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import nrrd
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))

from deepfluoro_loader import DeepFluoroProjection, DeepFluoroSpecimen, xzy

SCENE_PATH = Path('/home/supermicro/Documents/2D_3D_Raghu/data/testing/RAMULAMMA INTRAOP/2026-02-25-Scene.mrml')
PREOP_CT = Path('/home/supermicro/Documents/2D_3D_Raghu/data/testing/RAMULAMMA PREOP/RAMULAMMA PREOP/4 L_Spine  1.0  B60s.nrrd')
PREOP_3D = Path('/home/supermicro/Documents/2D_3D_Raghu/data/testing/RAMULAMMA PREOP/RAMULAMMA PREOP/centroids.mrk.json')

RAMU_SCENE_SID_MM: float = 1110.0
RAMU_SCENE_PIX_MM: float = 0.297656
RAMU_SCENE_IMG_SIZE: int = 1024
RAMU_SCENE_FX: float = RAMU_SCENE_SID_MM / RAMU_SCENE_PIX_MM
RAMU_SCENE_FY: float = RAMU_SCENE_FX
RAMU_SCENE_CX: float = (RAMU_SCENE_IMG_SIZE - 1) / 2.0
RAMU_SCENE_CY: float = RAMU_SCENE_CX
RAMU_SCENE_K: np.ndarray = np.array([
    [RAMU_SCENE_FX, 0., RAMU_SCENE_CX],
    [0., RAMU_SCENE_FY, RAMU_SCENE_CY],
    [0., 0., 1.],
], dtype=np.float64)


class RamuSceneProjection(DeepFluoroProjection):
    def project(self, pts3d_world: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts3d_world)
        p_cam = (self.R_proj @ xzy(pts).T).T + self.t_proj
        u = RAMU_SCENE_FX * p_cam[:, 0] / p_cam[:, 2] + RAMU_SCENE_CX
        v = RAMU_SCENE_FY * p_cam[:, 1] / p_cam[:, 2] + RAMU_SCENE_CY
        return np.stack([u, v], axis=1)


def compute_pde_scene(proj: RamuSceneProjection,
                      R: np.ndarray,
                      t: np.ndarray,
                      pts3d: np.ndarray,
                      lm_names: List[str]) -> Dict[str, float]:
    tmp = RamuSceneProjection.__new__(RamuSceneProjection)
    tmp.R_proj = R
    tmp.t_proj = t
    uv_pred = tmp.project(pts3d)
    pde = {}
    for i, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            gt = proj.gt_landmarks_2d[name]
            err_px = float(np.linalg.norm(uv_pred[i] - gt))
            pde[name] = err_px * RAMU_SCENE_PIX_MM
    return pde


def _load_3d_landmarks(path: Path) -> Dict[str, np.ndarray]:
    with open(path) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    return {cp['label'].strip(): np.array(cp['position'], dtype=np.float64) for cp in cps}


def _parse_active_scene(scene_path: Path) -> Tuple[Path, Path, int]:
    root = ET.parse(scene_path).getroot()
    selection = root.find('Selection')
    refs = selection.attrib['references']
    active_volume_id = refs.split('ActiveVolume:')[1].split(';')[0]
    active_place_id = refs.split('ActivePlaceNode:')[1].split(';')[0]

    volumes = {e.attrib['id']: e.attrib for e in root.findall('Volume')}
    storages = {e.attrib['id']: e.attrib for e in root.findall('VolumeArchetypeStorage')}
    markups = {e.attrib['id']: e.attrib for e in root.findall('MarkupsFiducial')}
    mkstor = {e.attrib['id']: e.attrib for e in root.findall('MarkupsJsonStorage')}

    vol = volumes[active_volume_id]
    vol_store_id = vol['references'].split('storage:')[1].split(';')[0]
    vol_file = scene_path.parent / storages[vol_store_id]['fileName']

    mk = markups[active_place_id]
    mk_store_id = mk['references'].split('storage:')[1].split(';')[0]
    mk_file = scene_path.parent / mkstor[mk_store_id]['fileName']

    with open(mk_file) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    frame_idx = int(round(cps[0]['position'][2]))
    return vol_file, mk_file, frame_idx


def _load_scene_2d_markups(markup_path: Path) -> Dict[str, np.ndarray]:
    with open(markup_path) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    return {cp['label'].strip(): np.array(cp['position'][:2], dtype=np.float64) for cp in cps}


def _load_scene_frame(volume_path: Path, frame_idx: int) -> np.ndarray:
    data, _ = nrrd.read(str(volume_path))
    img = data[:, :, frame_idx].T.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def _solve_scene_pnp(pts3d_world: np.ndarray,
                     pts2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    pts3d_xzy = xzy(pts3d_world).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        pts3d_xzy,
        pts2d.astype(np.float64),
        RAMU_SCENE_K,
        np.zeros(4),
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        raise RuntimeError('Scene SQPnP failed')
    R, _ = cv2.Rodrigues(rvec)
    p_cam = (R @ pts3d_xzy.T).T + tvec.reshape(1, 3)
    uv = np.stack([
        RAMU_SCENE_FX * p_cam[:, 0] / p_cam[:, 2] + RAMU_SCENE_CX,
        RAMU_SCENE_FY * p_cam[:, 1] / p_cam[:, 2] + RAMU_SCENE_CY,
    ], axis=1)
    err_px = float(np.mean(np.linalg.norm(uv - pts2d, axis=1)))
    return R.astype(np.float64), tvec.flatten().astype(np.float64), err_px


class RamuSceneLoader:
    def __init__(self,
                 scene_path: Path = SCENE_PATH,
                 preop_ct: Path = PREOP_CT,
                 preop_3d: Path = PREOP_3D):
        self.scene_path = Path(scene_path)
        self.preop_ct = Path(preop_ct)
        self.preop_3d = Path(preop_3d)

    def load(self, verbose: bool = True) -> DeepFluoroSpecimen:
        if verbose:
            print('[RamuScene] Loading from active MRML scene ...')

        volume_path, markup_path, frame_idx = _parse_active_scene(self.scene_path)
        if verbose:
            print(f'  Active volume : {volume_path.name}')
            print(f'  Active markups: {markup_path.name}')
            print(f'  Active frame  : z={frame_idx}')

        img = sitk.ReadImage(str(self.preop_ct))
        ct = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X) — keep for DRR generator
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin = np.array(img.GetOrigin(), dtype=np.float64)

        lm_3d = _load_3d_landmarks(self.preop_3d)
        lm_2d = _load_scene_2d_markups(markup_path)
        common = [name for name in sorted(lm_2d.keys()) if name in lm_3d]
        pts3d = np.array([lm_3d[name] for name in common], dtype=np.float64)
        pts2d = np.array([lm_2d[name] for name in common], dtype=np.float64)

        xray = _load_scene_frame(volume_path, frame_idx)
        R_init, t_init, reproj_px = _solve_scene_pnp(pts3d, pts2d)
        if verbose:
            print(f'  2D labels     : {common}')
            print(f'  SQPnP reproj  : {reproj_px:.4f} px ({reproj_px * RAMU_SCENE_PIX_MM:.4f} mm)')

        proj = RamuSceneProjection(
            specimen_id='ramulamma_scene',
            proj_index=0,
            proj_key=f'scene_frame_{frame_idx}',
            image_raw=xray,
            image_display=xray,
            R_proj=R_init,
            t_proj=t_init,
            gt_landmarks_2d=lm_2d,
            rot_180_for_up=False,
            reproj_error_px=reproj_px,
        )

        spec = DeepFluoroSpecimen(
            specimen_id='ramulamma_scene',
            ct_volume=ct,
            ct_spacing=spacing,
            ct_origin=origin,
            landmarks_3d=lm_3d,
            projections=[proj],
        )
        return spec
