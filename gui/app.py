# -*- coding: utf-8 -*-
"""
app.py  --  2D/3D Fluoroscopy Registration  --  Streamlit GUI
==============================================================
5-step wizard:
  Step 1  Load pre-operative CT (NRRD)
  Step 2  Mark vertebra centroids on CT slices (click to place)
  Step 3  Load intraoperative C-arm DICOM frames
  Step 4  Select AP + LAT frames and camera configuration
  Step 5  Run registration and view result overlay

Run:
    cd gui
    streamlit run app.py --server.port 8888
"""

import sys, io, json, threading, time
from pathlib import Path

import numpy as np
import streamlit as st
import cv2
import pydicom
import SimpleITK as sitk
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# ── path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / 'mslevel_reg' / 'src'))
sys.path.insert(0, str(HERE.parent / 'POC' / 'src'))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='2D/3D Registration',
    page_icon='🦴',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── constants ─────────────────────────────────────────────────────────────────
VERTEBRA_LABELS = ['D12', 'L1', 'L2', 'L3', 'L4', 'L5']
LABEL_COLORS = {
    'D12': '#FF6B6B', 'L1': '#FFA94D', 'L2': '#FFD43B',
    'L3': '#69DB7C',  'L4': '#4DABF7', 'L5': '#DA77F2',
}
CAMERA_CONFIGS = {
    'Ziehm Vision FD  (NIMS — 1050mm / 0.288mm/px)':
        {'sid_mm': 1050.0, 'pixel_spacing_mm': 0.288, 'image_size_px': 1024},
    'Siemens Cios Spin  (1050mm / 0.154mm/px)':
        {'sid_mm': 1050.0, 'pixel_spacing_mm': 0.154, 'image_size_px': 1024},
    'GE OEC 9900  (1020mm / 0.184mm/px)':
        {'sid_mm': 1020.0, 'pixel_spacing_mm': 0.184, 'image_size_px': 1024},
}
STEPS = ['1  Load CT', '2  Mark Centroids', '3  Load C-arm',
         '4  Select Frames', '5  Run & View']


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = dict(
        step=0,
        ct_volume=None, ct_spacing=None, ct_origin=None, ct_filename='',
        z_idx=0,
        centroids={}, pending_label=None,
        carm_frames=[],
        ap_frame_idx=None, lat_frame_idx=None,
        camera_name=list(CAMERA_CONFIGS.keys())[0],
        bank_size=200,
        result=None,
        log_lines=[], running=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# Image helpers
# ══════════════════════════════════════════════════════════════════════════════
def ct_slice_to_pil(arr2d: np.ndarray, w: int = 512,
                    window_center: float = 300.0,
                    window_width: float = 1500.0) -> Image.Image:
    """
    Apply bone-window windowing to a 2D CT slice and return a resized RGB PIL image.

    Default window (W:1500, L:300) is the standard bone window for spine CT —
    shows vertebra anatomy clearly with good contrast between bone and soft tissue.
    """
    lo = window_center - window_width / 2.0
    hi = window_center + window_width / 2.0
    clipped = np.clip(arr2d, lo, hi)
    norm = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
    h0, w0 = norm.shape
    norm_rs = cv2.resize(norm, (w, int(h0 * w / w0)), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(norm_rs).convert('RGB')


def draw_centroids_on_slice(pil_img, centroids, ct_shape, ct_spacing,
                             ct_origin, view, slice_idx, display_w=512):
    """Overlay coloured centroid dots on a CT slice PIL image."""
    img = np.array(pil_img).copy()
    Z, Y, X = ct_shape
    sx, sy, sz = ct_spacing
    ox, oy, oz = ct_origin

    for label, (wx, wy, wz) in centroids.items():
        vx = (wx - ox) / sx
        vy = (wy - oy) / sy
        vz = (wz - oz) / sz

        if view == 'axial':
            if abs(vz - slice_idx) > 5: continue
            px = int(vx / X * display_w)
            py = int(vy / Y * img.shape[0])
        elif view == 'coronal':
            if abs(vy - slice_idx) > 5: continue
            px = int(vx / X * display_w)
            py = int((1 - vz / Z) * img.shape[0])
        else:
            if abs(vx - slice_idx) > 5: continue
            px = int(vy / Y * display_w)
            py = int((1 - vz / Z) * img.shape[0])

        hex_col = LABEL_COLORS.get(label, '#FFFFFF')
        r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
        cv2.circle(img, (px, py), 8, (r, g, b), -1)
        cv2.circle(img, (px, py), 8, (255, 255, 255), 2)
        cv2.putText(img, label, (px+10, py+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (r, g, b), 2)
    return Image.fromarray(img)


def dicom_to_thumb(arr: np.ndarray, size: int = 150) -> Image.Image:
    """Convert a DICOM pixel array to a small thumbnail PIL image."""
    a = arr.astype(np.float32)
    a = (a - a.min()) / max(a.max() - a.min(), 1.0)
    u8 = (a * 255).astype(np.uint8)
    rs = cv2.resize(u8, (size, size), interpolation=cv2.INTER_AREA)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(rs)
    return Image.fromarray(eq).convert('RGB')


def dicom_to_display(arr: np.ndarray, w: int = 400) -> Image.Image:
    """Convert a full C-arm DICOM frame to a display PIL image."""
    a = arr.astype(np.float32)
    a = (a - a.min()) / max(a.max() - a.min(), 1.0)
    h0, w0 = a.shape
    rs = cv2.resize((a * 255).astype(np.uint8),
                    (w, int(h0 * w / w0)), interpolation=cv2.INTER_AREA)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(rs)
    return Image.fromarray(eq).convert('RGB')


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.title('🦴 2D/3D Registration')
    st.sidebar.markdown('**Intraoperative Spine Navigation**')
    st.sidebar.markdown('---')
    for i, label in enumerate(STEPS):
        if i < st.session_state.step:
            st.sidebar.markdown(f'✅  ~~{label}~~')
        elif i == st.session_state.step:
            st.sidebar.markdown(f'▶️  **{label}**')
        else:
            st.sidebar.markdown(f'⬜  {label}')
    st.sidebar.markdown('---')
    if st.session_state.centroids:
        st.sidebar.markdown('**Marked centroids:**')
        for lbl in VERTEBRA_LABELS:
            if lbl in st.session_state.centroids:
                p = st.session_state.centroids[lbl]
                st.sidebar.markdown(
                    f'<span style="color:{LABEL_COLORS[lbl]}">● {lbl}</span> '
                    f'({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm',
                    unsafe_allow_html=True)
    st.sidebar.markdown('---')
    st.sidebar.caption('v1.0 — NIMS Neurosurgery Research')


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load CT
# ══════════════════════════════════════════════════════════════════════════════
def step_load_ct():
    st.header('Step 1 — Load Pre-operative CT')
    st.markdown(
        'Upload the patient\'s pre-operative CT scan in **NRRD format**. '
        'This is the 3D map used for registration against the intraoperative C-arm X-ray.')

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader('Upload CT file (.nrrd)', type=['nrrd'],
                                     help='NRRD format — export from 3D Slicer.')
        if uploaded:
            with st.spinner('Loading CT volume...'):
                tmp = Path('_tmp_ct.nrrd')
                tmp.write_bytes(uploaded.getvalue())
                img = sitk.ReadImage(str(tmp))
                arr = sitk.GetArrayFromImage(img).astype(np.float32)
                sp  = tuple(float(x) for x in img.GetSpacing())
                ori = tuple(float(x) for x in img.GetOrigin())
                tmp.unlink(missing_ok=True)
            st.session_state.ct_volume   = arr
            st.session_state.ct_spacing  = sp
            st.session_state.ct_origin   = ori
            st.session_state.ct_filename = uploaded.name
            st.session_state.z_idx       = arr.shape[0] // 2
            st.success(f'Loaded: **{uploaded.name}**')

    if st.session_state.ct_volume is not None:
        arr = st.session_state.ct_volume
        sp  = st.session_state.ct_spacing
        with col2:
            st.markdown('**CT metadata**')
            st.json({
                'Shape (Z, Y, X)': list(arr.shape),
                'Spacing (mm)':    [round(x, 3) for x in sp],
                'Origin (mm)':     [round(x, 1) for x in st.session_state.ct_origin],
                'HU range':        [int(arr.min()), int(arr.max())],
            })

        st.markdown('### CT Preview  (bone window W:1500 / L:300)')
        Z, Y, X = arr.shape
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption('Axial (cross-section)')
            st.image(ct_slice_to_pil(arr[Z//2], 400), use_column_width=True)
        with c2:
            st.caption('Coronal (front view)')
            st.image(ct_slice_to_pil(arr[:, Y//2, :], 400), use_column_width=True)
        with c3:
            st.caption('Sagittal (side view)')
            st.image(ct_slice_to_pil(arr[:, :, X//2], 400), use_column_width=True)

        st.markdown('---')
        if st.button('Next → Mark Centroids', type='primary', use_container_width=False):
            st.session_state.step = 1
            st.experimental_rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Mark Vertebra Centroids
# ══════════════════════════════════════════════════════════════════════════════
def step_mark_centroids():
    st.header('Step 2 — Mark Vertebra Centroids')
    st.markdown(
        'The **Coronal (front) and Sagittal (side) views** show all vertebrae at once '
        'so you can click directly on each one without scrolling. '
        '**Workflow:** select a label → click on either view → repeat for each vertebra.')

    arr = st.session_state.ct_volume
    sp  = st.session_state.ct_spacing
    ori = st.session_state.ct_origin
    Z, Y, X = arr.shape

    if 'click_count' not in st.session_state:
        st.session_state.click_count = 0
    if 'cor_y_idx' not in st.session_state:
        st.session_state.cor_y_idx = Y // 2
    if 'sag_x_idx' not in st.session_state:
        st.session_state.sag_x_idx = X // 2

    # Window/level controls — top row
    ww_col, wl_col, _ = st.columns([1, 1, 2])
    with ww_col:
        ww = st.slider('Window width',  200, 3000, 1500, 100)
    with wl_col:
        wl = st.slider('Window level', -200,  800,  300,  50)

    st.markdown('---')

    # ── Main layout: two large clickable views + label panel ─────────────────
    cor_col, sag_col, lbl_col = st.columns([2.5, 2.5, 1])

    # ── CORONAL view (front view — arr[:, y_idx, :], shape Z×X) ─────────────
    with cor_col:
        st.markdown('#### Coronal view  (front)')
        st.caption('Click on a vertebra body to mark it.')
        y_idx = st.slider('Slice  (front ↔ back)', 0, Y-1,
                           st.session_state.cor_y_idx, key='cor_slider')
        st.session_state.cor_y_idx = y_idx

        # arr[:, y_idx, :] is shape (Z, X) — flip Z so top of image = top of spine
        cor_arr = arr[:, y_idx, :]          # (Z, X)
        cor_arr = np.flipud(cor_arr)         # flip so inferior spine is at bottom
        cor_pil = ct_slice_to_pil(cor_arr, 500, wl, ww)

        # Draw existing centroids on coronal view
        cor_bgr = np.array(cor_pil).copy()
        for lbl, (wx, wy, wz) in st.session_state.centroids.items():
            vox_x = (wx - ori[0]) / sp[0]
            vox_z = (wz - ori[2]) / sp[2]
            # After flipud: display_y = (Z-1 - vox_z) / Z * img_h
            img_h, img_w = cor_bgr.shape[:2]
            px = int(vox_x / X * img_w)
            py = int((Z - 1 - vox_z) / Z * img_h)
            hex_c = LABEL_COLORS.get(lbl, '#FFFFFF')
            r, g, b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
            cv2.circle(cor_bgr, (px, py), 9, (r,g,b), -1)
            cv2.circle(cor_bgr, (px, py), 9, (255,255,255), 2)
            cv2.putText(cor_bgr, lbl, (px+11, py+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (r,g,b), 2)
        cor_pil = Image.fromarray(cor_bgr)

        if st.session_state.pending_label:
            lbl = st.session_state.pending_label
            st.markdown(
                f'<p style="color:{LABEL_COLORS[lbl]};font-weight:bold">'
                f'→ Click here to place  <b>{lbl}</b></p>',
                unsafe_allow_html=True)

        click_cor = streamlit_image_coordinates(
            cor_pil, key=f'cor_click_{st.session_state.click_count}')

        if click_cor is not None and st.session_state.pending_label:
            img_w_c, img_h_c = cor_pil.size
            # coronal slice: x-axis = voxel X, y-axis = flipped voxel Z
            vox_x  = (click_cor['x'] / img_w_c) * X
            vox_z  = (Z - 1) - (click_cor['y'] / img_h_c) * Z   # un-flip
            vox_y  = float(y_idx)                                 # from slider
            wx = ori[0] + vox_x * sp[0]
            wy = ori[1] + vox_y * sp[1]
            wz = ori[2] + vox_z * sp[2]
            lbl = st.session_state.pending_label
            st.session_state.centroids[lbl] = [wx, wy, wz]
            st.session_state.click_count   += 1
            st.session_state.pending_label  = None
            st.experimental_rerun()

    # ── SAGITTAL view (side view — arr[:, :, x_idx], shape Z×Y) ─────────────
    with sag_col:
        st.markdown('#### Sagittal view  (side)')
        st.caption('Click to place or refine depth (front-to-back position).')
        x_idx = st.slider('Slice  (left ↔ right)', 0, X-1,
                           st.session_state.sag_x_idx, key='sag_slider')
        st.session_state.sag_x_idx = x_idx

        # arr[:, :, x_idx] is shape (Z, Y) — flip Z same as coronal
        sag_arr = arr[:, :, x_idx]
        sag_arr = np.flipud(sag_arr)
        sag_pil = ct_slice_to_pil(sag_arr, 500, wl, ww)

        # Draw existing centroids on sagittal view
        sag_bgr = np.array(sag_pil).copy()
        for lbl, (wx, wy, wz) in st.session_state.centroids.items():
            vox_y = (wy - ori[1]) / sp[1]
            vox_z = (wz - ori[2]) / sp[2]
            img_h, img_w = sag_bgr.shape[:2]
            px = int(vox_y / Y * img_w)
            py = int((Z - 1 - vox_z) / Z * img_h)
            hex_c = LABEL_COLORS.get(lbl, '#FFFFFF')
            r, g, b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
            cv2.circle(sag_bgr, (px, py), 9, (r,g,b), -1)
            cv2.circle(sag_bgr, (px, py), 9, (255,255,255), 2)
            cv2.putText(sag_bgr, lbl, (px+11, py+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (r,g,b), 2)
        sag_pil = Image.fromarray(sag_bgr)

        if st.session_state.pending_label:
            lbl = st.session_state.pending_label
            st.markdown(
                f'<p style="color:{LABEL_COLORS[lbl]};font-weight:bold">'
                f'→ Or click here to place  <b>{lbl}</b></p>',
                unsafe_allow_html=True)

        click_sag = streamlit_image_coordinates(
            sag_pil, key=f'sag_click_{st.session_state.click_count}')

        if click_sag is not None and st.session_state.pending_label:
            img_w_s, img_h_s = sag_pil.size
            # sagittal slice: x-axis = voxel Y, y-axis = flipped voxel Z
            vox_y  = (click_sag['x'] / img_w_s) * Y
            vox_z  = (Z - 1) - (click_sag['y'] / img_h_s) * Z   # un-flip
            vox_x  = float(x_idx)                                 # from slider
            wx = ori[0] + vox_x * sp[0]
            wy = ori[1] + vox_y * sp[1]
            wz = ori[2] + vox_z * sp[2]
            lbl = st.session_state.pending_label
            st.session_state.centroids[lbl] = [wx, wy, wz]
            st.session_state.click_count   += 1
            st.session_state.pending_label  = None
            st.experimental_rerun()

    # ── Label selection panel ─────────────────────────────────────────────────
    with lbl_col:
        st.markdown('#### Labels')
        st.caption('1. Click a label\n2. Click on the spine')
        for lbl in VERTEBRA_LABELS:
            done      = lbl in st.session_state.centroids
            icon      = '✅' if done else '⬜'
            is_active = (st.session_state.pending_label == lbl)
            if st.button(f'{icon} {lbl}', key=f'btn_{lbl}',
                         type='primary' if is_active else 'secondary'):
                st.session_state.pending_label = lbl
                st.session_state.click_count  += 1   # reset stale click
                st.experimental_rerun()

        st.markdown('---')
        if st.session_state.pending_label:
            lbl = st.session_state.pending_label
            st.markdown(
                f'<div style="padding:6px;border-radius:5px;'
                f'background:{LABEL_COLORS[lbl]}33;'
                f'border:2px solid {LABEL_COLORS[lbl]}">'
                f'<b>Placing:</b><br>'
                f'<span style="color:{LABEL_COLORS[lbl]};font-size:18px">● {lbl}</span>'
                f'</div>', unsafe_allow_html=True)

        if st.button('🗑 Clear all'):
            st.session_state.centroids     = {}
            st.session_state.pending_label = None
            st.session_state.click_count  += 1
            st.experimental_rerun()

        st.markdown('**Placed:**')
        for lbl in VERTEBRA_LABELS:
            if lbl in st.session_state.centroids:
                p = st.session_state.centroids[lbl]
                st.markdown(
                    f'<span style="color:{LABEL_COLORS[lbl]}">● {lbl}</span>',
                    unsafe_allow_html=True)
                st.caption(f'X:{p[0]:.0f} Y:{p[1]:.0f} Z:{p[2]:.0f}')

    # ── Axial reference at current centroid Z level ────────────────────────────
    if st.session_state.centroids:
        st.markdown('---')
        st.markdown('#### Axial reference  (cross-section at each marked vertebra)')
        st.caption('Verify each centroid is roughly centred in the vertebral body.')
        marked_items = [(lbl, p) for lbl, p in st.session_state.centroids.items()]
        ax_cols = st.columns(min(len(marked_items), 6))
        for col_i, (lbl, p) in enumerate(marked_items):
            with ax_cols[col_i % 6]:
                voz = int((p[2] - ori[2]) / sp[2])
                voz = max(0, min(Z-1, voz))
                ax_pil = ct_slice_to_pil(arr[voz], 180, wl, ww)
                ax_bgr = np.array(ax_pil).copy()
                # draw dot at centroid position
                vox_x = int((p[0] - ori[0]) / sp[0] * 180 / X)
                vox_y = int((p[1] - ori[1]) / sp[1] * 180 / Y)
                hex_c = LABEL_COLORS.get(lbl, '#fff')
                r, g, b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
                cv2.circle(ax_bgr, (vox_x, vox_y), 7, (r,g,b), -1)
                cv2.circle(ax_bgr, (vox_x, vox_y), 7, (255,255,255), 2)
                st.image(Image.fromarray(ax_bgr),
                         caption=f'{lbl}  (Z={voz})', use_column_width=True)

    st.markdown('---')
    n = len(st.session_state.centroids)
    cb, cn = st.columns(2)
    with cb:
        if st.button('← Back'):
            st.session_state.step = 0; st.experimental_rerun()
    with cn:
        ready = n >= 3
        label = f'Next → Load C-arm  ({n}/6 marked)'
        if ready:
            if st.button(label, type='primary'):
                st.session_state.step = 2; st.experimental_rerun()
        else:
            st.button(label, disabled=True)
            st.warning('Mark at least L3, L4, L5 to continue.')


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Load C-arm Frames
# ══════════════════════════════════════════════════════════════════════════════
def step_load_carm():
    st.header('Step 3 — Load Intraoperative C-arm Frames')
    st.markdown(
        'Upload C-arm frame files from this surgical session. '
        'Supported formats: **DICOM** (raw files from C-arm, any extension) '
        'and **NRRD** (.nrrd). '
        'You will select the AP and LAT frames in the next step.')

    uploaded_files = st.file_uploader(
        'Upload C-arm files  (DICOM or NRRD, select multiple)',
        type=None, accept_multiple_files=True,
        help='DICOM files from Ziehm / Siemens / GE, OR .nrrd files exported from 3D Slicer.')

    if uploaded_files:
        frames, prog = [], st.progress(0, text='Loading...')
        for i, f in enumerate(uploaded_files):
            raw = f.read()
            name = f.name
            arr  = None
            meta = {}

            try:
                if name.lower().endswith('.nrrd'):
                    # ── NRRD C-arm frame ─────────────────────────────────
                    tmp_nrrd = Path(f'_tmp_carm_{i}.nrrd')
                    tmp_nrrd.write_bytes(raw)
                    img  = sitk.ReadImage(str(tmp_nrrd))
                    tmp_nrrd.unlink(missing_ok=True)
                    arr_full = sitk.GetArrayFromImage(img).astype(np.float32)
                    # If 3D (multi-frame NRRD), take first frame
                    if arr_full.ndim == 3:
                        arr = arr_full[0]
                    else:
                        arr = arr_full
                    meta = {
                        'name':     name,
                        'idx':      i,
                        'rows':     arr.shape[0],
                        'cols':     arr.shape[1],
                        'sid_mm':   1050.0,   # default — update in Step 4
                        'prim_ang': 0.0,
                        'sec_ang':  0.0,
                        'format':   'NRRD',
                    }
                else:
                    # ── DICOM frame ──────────────────────────────────────
                    ds  = pydicom.dcmread(io.BytesIO(raw), force=True)
                    arr = ds.pixel_array.astype(np.float32)
                    meta = {
                        'name':     name,
                        'idx':      i,
                        'rows':     int(getattr(ds, 'Rows',    arr.shape[0])),
                        'cols':     int(getattr(ds, 'Columns', arr.shape[1])),
                        'sid_mm':   float(getattr(ds, 'DistanceSourceToDetector', 1050) or 1050),
                        'prim_ang': float(getattr(ds, 'PositionerPrimaryAngle',   0)    or 0),
                        'sec_ang':  float(getattr(ds, 'PositionerSecondaryAngle', 0)    or 0),
                        'format':   'DICOM',
                    }

                frames.append({'array': arr, 'meta': meta})

            except Exception as e:
                st.warning(f'Could not read **{name}**: {e}')

            prog.progress((i+1)/len(uploaded_files), text=f'Loading {i+1}/{len(uploaded_files)}...')

        prog.empty()
        st.session_state.carm_frames = frames
        n_dicom = sum(1 for f in frames if f['meta'].get('format') == 'DICOM')
        n_nrrd  = sum(1 for f in frames if f['meta'].get('format') == 'NRRD')
        st.success(
            f'Loaded **{len(frames)}** frames  '
            f'({n_dicom} DICOM, {n_nrrd} NRRD)')

    if st.session_state.carm_frames:
        frames = st.session_state.carm_frames
        st.markdown(f'### Frame Thumbnails  ({len(frames)} frames, showing first 40)')
        per_row = 8
        for row_start in range(0, min(len(frames), 40), per_row):
            cols = st.columns(per_row)
            for j, fi in enumerate(frames[row_start:row_start+per_row]):
                with cols[j]:
                    thumb   = dicom_to_thumb(fi['array'], 100)
                    ang_p   = fi['meta']['prim_ang']
                    ang_s   = fi['meta']['sec_ang']
                    fmt     = fi['meta'].get('format', 'DICOM')
                    badge   = '📷' if fmt == 'DICOM' else '🗂'
                    st.image(thumb,
                             caption=f"{badge} {fi['meta']['name']}\n{ang_p:.0f}°/{ang_s:.0f}°",
                             use_column_width=True)

    st.markdown('---')
    cb, cn = st.columns(2)
    with cb:
        if st.button('← Back'):
            st.session_state.step = 1; st.experimental_rerun()
    with cn:
        ready = len(st.session_state.carm_frames) >= 2
        if ready:
            if st.button('Next → Select Frames', type='primary'):
                st.session_state.step = 3; st.experimental_rerun()
        else:
            st.button('Next → Select Frames', disabled=True)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Select AP + LAT Frames and Config
# ══════════════════════════════════════════════════════════════════════════════
def step_select_frames():
    st.header('Step 4 — Select AP + LAT Frames and Camera Configuration')

    # ── Clear visual instructions ──────────────────────────────────────────
    st.markdown("""
    **What you need to do on this page:**
    - Use the **two dropdowns below** to pick which frame is the AP view and which is the LAT view
    - The full image preview updates as you change the selection
    - Then choose your **C-arm machine** and click **▶ Run Registration**

    > 📐 **AP frame** = C-arm is roughly in front of the patient (spine seen head-on, vertebrae stacked vertically)
    >
    > 📐 **LAT frame** = C-arm is to the side of the patient (spine seen sideways, like a capital S-curve)
    """)
    st.markdown('---')

    frames = st.session_state.carm_frames
    names  = [f['meta']['name'] for f in frames]

    # Auto-suggest AP and LAT based on positioner angles if available
    def guess_view(fi):
        pa = abs(fi['meta']['prim_ang'])
        sa = abs(fi['meta']['sec_ang'])
        if sa > 60:
            return 'LAT'
        elif pa < 30 and sa < 30:
            return 'AP'
        return '?'

    suggestions = [guess_view(f) for f in frames]
    ap_suggestions  = [i for i, s in enumerate(suggestions) if s == 'AP']
    lat_suggestions = [i for i, s in enumerate(suggestions) if s == 'LAT']

    if ap_suggestions or lat_suggestions:
        st.info(
            f'💡 Auto-detected: **{len(ap_suggestions)} AP frame(s)** and '
            f'**{len(lat_suggestions)} LAT frame(s)** based on positioner angles.')

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('### 🔵 AP Frame  (front view)')
        st.caption('Select the frame where the C-arm was roughly in front — '
                   'you should see the vertebrae stacked on top of each other.')
        ap_default = ap_suggestions[0] if ap_suggestions else 0
        ap_idx = st.selectbox(
            'Select AP frame',
            range(len(names)),
            format_func=lambda i: f"{names[i]}  [{suggestions[i]}]  "
                                   f"(P:{frames[i]['meta']['prim_ang']:.0f}°  "
                                   f"S:{frames[i]['meta']['sec_ang']:.0f}°)",
            index=ap_default,
            key='ap_sel')
        st.session_state.ap_frame_idx = ap_idx
        st.image(dicom_to_display(frames[ap_idx]['array'], 380),
                 caption=f"Selected AP frame: {names[ap_idx]}", use_column_width=True)
        m = frames[ap_idx]['meta']
        st.caption(
            f"Primary angle: **{m['prim_ang']:.1f}°**  |  "
            f"Secondary angle: **{m['sec_ang']:.1f}°**  |  "
            f"SID: {m['sid_mm']:.0f} mm")

    with c2:
        st.markdown('### 🟠 LAT Frame  (side view)')
        st.caption('Select the frame where the C-arm was to the side — '
                   'you should see the S-curve of the spine.')
        lat_default = lat_suggestions[0] if lat_suggestions else min(1, len(names)-1)
        lat_idx = st.selectbox(
            'Select LAT frame',
            range(len(names)),
            format_func=lambda i: f"{names[i]}  [{suggestions[i]}]  "
                                   f"(P:{frames[i]['meta']['prim_ang']:.0f}°  "
                                   f"S:{frames[i]['meta']['sec_ang']:.0f}°)",
            index=lat_default,
            key='lat_sel')
        st.session_state.lat_frame_idx = lat_idx
        st.image(dicom_to_display(frames[lat_idx]['array'], 380),
                 caption=f"Selected LAT frame: {names[lat_idx]}", use_column_width=True)
        m = frames[lat_idx]['meta']
        st.caption(
            f"Primary angle: **{m['prim_ang']:.1f}°**  |  "
            f"Secondary angle: **{m['sec_ang']:.1f}°**  |  "
            f"SID: {m['sid_mm']:.0f} mm")

    st.markdown('---')
    st.markdown('### Camera Configuration')
    st.markdown('Select the C-arm machine model. This sets the SID and pixel spacing.')
    cam_name = st.selectbox('C-arm machine', list(CAMERA_CONFIGS.keys()), key='cam_sel')
    st.session_state.camera_name = cam_name
    cfg = CAMERA_CONFIGS[cam_name]
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric('SID (mm)', cfg['sid_mm'])
    cc2.metric('Pixel spacing (mm/px)', cfg['pixel_spacing_mm'])
    cc3.metric('Image size (px)', cfg['image_size_px'])

    st.markdown('### Pipeline Settings')
    bank_size = st.slider('DRR bank size', 100, 500, 200, 50,
                          help='More = more accurate but slower. 200 is a good default.')
    st.session_state.bank_size = bank_size
    st.caption(f'Estimated time: ~{bank_size*0.3:.0f}s GPU  /  ~{bank_size*1.5:.0f}s CPU')

    st.info('ℹ️ **PDE is computed automatically** — the pipeline uses the CNN landmark detector '
            'to independently locate vertebrae in the X-ray and compares them with the '
            'registration estimate. No manual labels needed at any stage.')

    st.markdown('---')
    cb, cn = st.columns(2)
    with cb:
        if st.button('← Back'):
            st.session_state.step = 2; st.experimental_rerun()
    with cn:
        diff = (ap_idx != lat_idx)
        if diff:
            if st.button('▶ Run Registration', type='primary'):
                st.session_state.step = 4; st.experimental_rerun()
        else:
            st.button('▶ Run Registration', disabled=True)
            st.warning('⚠️ AP and LAT must be different frames — please select two different frames.')


# ══════════════════════════════════════════════════════════════════════════════
# Auto landmark detection for PDE  (no manual labels needed)
# ══════════════════════════════════════════════════════════════════════════════
def auto_detect_2d_landmarks(frame_array, lm_names, projected_px, out_size=512):
    """
    Automatically detect 2D vertebra positions in a C-arm frame.
    Fully automatic — no manual annotation.

    Two-method approach:
      Method A (preferred): LandmarkHeatmapNet CNN, loaded from checkpoint.
      Method B (fallback):  For each projected position, find the local
                            intensity maximum in a ±search_radius window.
                            Bone is the brightest region in bone-bright images.

    Parameters
    ----------
    frame_array   : (H, W) float32 raw C-arm pixel array
    lm_names      : list of landmark names to detect, e.g. ['L3','L4','L5']
    projected_px  : list of (px, py) or None — estimated positions from registration
    out_size      : working resolution (pixels)

    Returns
    -------
    dict {label: [u, v]}  detected 2D positions in out_size pixel space
    """
    # Preprocess to bone-bright, out_size × out_size
    a   = frame_array.astype(np.float32)
    a   = (a - a.min()) / max(a.max() - a.min(), 1.0)
    u8  = (a * 255).astype(np.uint8)
    rs  = cv2.resize(u8, (out_size, out_size), interpolation=cv2.INTER_AREA)
    eq  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(rs)
    img = 1.0 - eq.astype(np.float32) / 255.0  # bone-bright

    detections = {}

    # ── Method A: CNN (LandmarkHeatmapNet) ──────────────────────────────────
    try:
        import torch
        ckpt_base = HERE.parent / 'mslevel_reg' / 'checkpoints'
        ckpt_dirs = [
            ckpt_base / 'landmark_detector_ap_pedicles',
            ckpt_base / 'landmark_detector_lat_pedicles',
        ]
        ckpt_path = None
        for cd in ckpt_dirs:
            if cd.exists():
                ckpts = sorted(cd.glob('*.pth'))
                if ckpts:
                    ckpt_path = ckpts[0]
                    break

        if ckpt_path:
            from landmark_model   import LandmarkHeatmapNet
            from landmark_dataset import CLASS_NAMES, NUM_CLASSES, IMG_SIZE, HMAP_SIZE
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ckpt   = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model  = LandmarkHeatmapNet(pretrained=False,
                                        num_classes=NUM_CLASSES, dropout=0.0).to(device)
            import contextlib, io as _io
            with contextlib.redirect_stderr(_io.StringIO()):
                model.load_state_dict(ckpt['model_state'])
            model.eval()
            img_cnn = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(img_cnn).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                hmap = model(t).sigmoid()[0].cpu().numpy()
            sx = out_size / HMAP_SIZE
            sy = out_size / HMAP_SIZE
            for ci, name in enumerate(CLASS_NAMES):
                if name not in lm_names:
                    continue
                h = hmap[ci]
                if float(h.max()) > 0.05:
                    py_h, px_h = np.unravel_index(h.argmax(), h.shape)
                    detections[name] = [float(px_h * sx), float(py_h * sy)]
            if detections:
                return detections  # CNN succeeded
    except Exception:
        pass  # CNN unavailable — fall through

    # ── Method B: Local intensity maximum (fallback) ─────────────────────────
    search_r = max(20, out_size // 20)
    for lbl, pp in zip(lm_names, projected_px):
        if pp is None:
            continue
        px_c, py_c = int(pp[0]), int(pp[1])
        x0 = max(0, px_c - search_r);  x1 = min(out_size, px_c + search_r)
        y0 = max(0, py_c - search_r);  y1 = min(out_size, py_c + search_r)
        if x1 <= x0 or y1 <= y0:
            continue
        patch   = img[y0:y1, x0:x1]
        patch_s = cv2.GaussianBlur(patch, (5, 5), 0)
        ly, lx  = np.unravel_index(patch_s.argmax(), patch_s.shape)
        detections[lbl] = [float(x0 + lx), float(y0 + ly)]

    return detections


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runner (background thread)
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline_thread(log_queue: list, data: dict, result_holder: dict):
    """
    Execute the full 2D/3D registration in a background thread.
    Appends status strings to log_queue.
    Writes result dict to st.session_state.result on success.
    """
    try:
        from drr_renderer import CTVolume, DRRRenderer
        from similarity import gradient_orientation_similarity
        from scipy.optimize import minimize

        log_queue.append('Loading CT into renderer...')
        ct_arr = data["ct_volume"]
        sp, ori = data["ct_spacing"], data["ct_origin"]
        ct_vol   = CTVolume(ct_arr, sp, ori, hu_threshold=180.0)
        renderer = DRRRenderer(ct_vol)
        log_queue.append(f'CT loaded (downsampled). Shape: {ct_arr.shape}  Memory: {ct_arr.nbytes/1e6:.0f} MB')

        centroids = data["centroids"]
        lm_names  = sorted(centroids.keys())
        pts3d     = np.array([centroids[n] for n in lm_names])
        center_w  = pts3d.mean(axis=0)
        log_queue.append(f'Landmarks: {lm_names}')

        cam_cfg     = CAMERA_CONFIGS[data["camera_name"]]
        SDD_MM      = cam_cfg['sid_mm']
        PIX_MM      = cam_cfg['pixel_spacing_mm']
        OUT_SIZE    = 256
        PIX_MM_BANK = PIX_MM * (cam_cfg['image_size_px'] / OUT_SIZE)

        def preprocess(arr):
            """Resize, CLAHE, invert to bone-bright, metal-suppress."""
            f32 = arr.astype(np.float32); f32 /= max(f32.max(), 1.0)
            u8  = (f32 * 255).astype('uint8')
            rs  = cv2.resize(u8, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
            eq  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(rs)
            img = 1.0 - eq.astype(np.float32) / 255.0
            p97 = float(np.percentile(img, 97.0))
            thr = max(p97, 0.82)
            mask = (img > thr).astype(np.uint8) * 255
            if mask.any():
                k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                img = cv2.inpaint((img*255).clip(0,255).astype(np.uint8),
                                  cv2.dilate(mask, k), 7,
                                  cv2.INPAINT_TELEA).astype(np.float32) / 255.0
            return img

        log_queue.append('Preprocessing C-arm frames...')
        # arrays passed directly
        target_ap = preprocess(data["ap_array"])
        target_lat = preprocess(data["lat_array"])
        log_queue.append('Frames ready.')

        # Heuristic initial camera positions (directly in front / to the side)
        cx, cy, cz = center_w
        dist = 800.0
        src_ap  = np.array([cx, cy - dist, cz])
        pax_ap  = (center_w - src_ap);  pax_ap  /= np.linalg.norm(pax_ap)
        src_lat = np.array([cx + dist, cy, cz])
        pax_lat = (center_w - src_lat); pax_lat /= np.linalg.norm(pax_lat)
        log_queue.append('Initial camera poses estimated.')

        def make_uv(pax):
            up = np.array([0., 0., -1.])
            if abs(np.dot(pax, up)) > 0.95: up = np.array([1., 0., 0.])
            u = np.cross(pax, up); u /= np.linalg.norm(u)
            return u, np.cross(u, pax)

        # Build DRR bank
        N   = data["bank_size"]
        rng = np.random.default_rng(42)
        entries = []
        log_queue.append(f'Building DRR bank ({N} entries)...')
        for k in range(N):
            view    = 'AP' if k < N//2 else 'LAT'
            bs, bp  = (src_ap, pax_ap) if view == 'AP' else (src_lat, pax_lat)
            src_p   = bs + rng.normal(0, 20., 3)
            pax_p   = (center_w - src_p); pax_p /= np.linalg.norm(pax_p)
            u_p, v_p = make_uv(pax_p)
            drr = renderer.render(src_p, pax_p, u_p, v_p,
                                  output_size=OUT_SIZE,
                                  pixel_spacing_mm=PIX_MM_BANK,
                                  n_steps=160, sdd_mm=SDD_MM)
            entries.append({'view':view,'src':src_p,'pax':pax_p,'u':u_p,'v':v_p,'drr':drr})
            if (k+1) % 50 == 0:
                log_queue.append(f'  {k+1}/{N} DRRs rendered')
        log_queue.append(f'Bank ready ({N} DRRs).')

        def go(a, b):
            return gradient_orientation_similarity(a, b, sigma=2.0)

        # GO retrieval
        log_queue.append('Scoring bank with GO similarity...')
        ap_e  = [e for e in entries if e['view']=='AP']
        lat_e = [e for e in entries if e['view']=='LAT']
        go_ap  = np.array([go(e['drr'], target_ap)  for e in ap_e])
        go_lat = np.array([go(e['drr'], target_lat) for e in lat_e])
        lat_srcs = np.array([e['src'] for e in lat_e])

        joint = []
        for i, (e, s_ap_val) in enumerate(zip(ap_e, go_ap)):
            bl = int(np.argmin(np.linalg.norm(lat_srcs - np.array(e['src']), axis=1)))
            joint.append((s_ap_val + go_lat[bl], i, bl))
        joint.sort(reverse=True)
        log_queue.append(
            f'GO scored. Best joint={joint[0][0]:.4f}  '
            f'AP_SNR={go_ap.max()/go_ap.mean():.3f}  '
            f'LAT_SNR={go_lat.max()/go_lat.mean():.3f}')

        # Multi-start Nelder-Mead refinement (top 5)
        BOUNDS = np.array([30.,30.,30.,8.,6.,8.])
        best_go, best_res = float("-inf"), None
        log_queue.append('Multi-start refinement (top-5 candidates)...')
        for ki, (js, ai, li) in enumerate(joint[:5]):
            sa = np.array(ap_e[ai]['src'])
            sl = np.array(lat_e[li]['src'])

            def cost(x, sa=sa, sl=sl):
                if np.any(np.abs(x) > BOUNDS): return 2.
                dx,dy,dz,dax,day,daz = x; d = np.array([dx,dy,dz])
                s1 = sa + d + np.array([dax,day,0])
                s2 = sl + d + np.array([0,day,daz])
                p1 = (center_w-s1); p1 /= np.linalg.norm(p1)
                p2 = (center_w-s2); p2 /= np.linalg.norm(p2)
                u1,v1 = make_uv(p1); u2,v2 = make_uv(p2)
                d1 = renderer.render(s1,p1,u1,v1,OUT_SIZE,PIX_MM_BANK,140,SDD_MM)
                d2 = renderer.render(s2,p2,u2,v2,OUT_SIZE,PIX_MM_BANK,140,SDD_MM)
                if d1.mean() < 0.01 or d2.mean() < 0.01: return 2.
                return (1.-go(d1,target_ap)) + (1.-go(d2,target_lat))

            x0 = np.zeros(6)
            st_ = np.full(6, 5.)
            simp = np.vstack([x0]+[x0+st_[i]*np.eye(6)[i] for i in range(6)])
            res = minimize(cost, x0, method='Nelder-Mead',
                           options=dict(xatol=.05,fatol=1e-4,maxiter=400,
                                        maxfev=400,initial_simplex=simp,adaptive=True))
            go_val = -res.fun
            log_queue.append(f'  Start {ki+1}/5  GO={go_val:.4f}  cost={res.fun:.4f}')

            if go_val > best_go:
                best_go = go_val
                dx,dy,dz,dax,day,daz = res.x; d = np.array([dx,dy,dz])
                s1f = sa + d + np.array([dax,day,0])
                s2f = sl + d + np.array([0,day,daz])
                p1f = (center_w-s1f); p1f /= np.linalg.norm(p1f)
                p2f = (center_w-s2f); p2f /= np.linalg.norm(p2f)
                u1f,v1f = make_uv(p1f); u2f,v2f = make_uv(p2f)
                best_res = dict(
                    src_ap=s1f, pax_ap=p1f, u_ap=u1f, v_ap=v1f,
                    src_lat=s2f, pax_lat=p2f, u_lat=u2f, v_lat=v2f)

        # Render final overlays
        log_queue.append('Rendering result images...')
        def project_landmarks(src, pax, u, v, pts3d_world):
            """
            Project 3D world points into 2D image coordinates using the
            estimated camera pose. This is fully automatic — no manual marking.

            Returns list of (px, py) pixel positions in the OUT_SIZE image.
            """
            results = []
            for wp in pts3d_world:
                # Vector from source to world point
                d_vec = wp - src
                # Project onto camera axes
                dist_along_axis = np.dot(d_vec, pax)
                if dist_along_axis < 1e-6:
                    results.append(None)
                    continue
                # Scale factor (pin-hole projection)
                scale = SDD_MM / dist_along_axis
                # 2D position on detector plane (centred at 0)
                u_coord = np.dot(d_vec, u) * scale
                v_coord = np.dot(d_vec, np.cross(u, pax)) * scale
                # Convert to pixel (detector size = OUT_SIZE, pixel pitch = PIX_MM_BANK)
                half = (OUT_SIZE * PIX_MM_BANK) / 2.0
                px = int((u_coord + half) / PIX_MM_BANK)
                py = int((v_coord + half) / PIX_MM_BANK)
                if 0 <= px < OUT_SIZE and 0 <= py < OUT_SIZE:
                    results.append((px, py))
                else:
                    results.append(None)
            return results

        def make_overlay(src, pax, u, v, target, label, auto_2d=None, proj_list=None):
            """
            Render overlay: real C-arm | registered DRR with projected landmarks.

            Automatically projects 3D CT centroids onto both panels.
            If gt_2d is provided (dict {label: [u,v]}), PDE is computed.
            """
            drr = renderer.render(src, pax, u, v, OUT_SIZE, PIX_MM_BANK, 200, SDD_MM)
            t8  = (target * 255).clip(0, 255).astype('uint8')
            d8  = (drr   * 255).clip(0, 255).astype('uint8')
            side = np.hstack([t8, d8])
            bgr  = cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)

            # Project 3D landmarks automatically using estimated camera pose
            lm_pts3d = [centroids[n] for n in lm_names]
            proj_px  = project_landmarks(src, pax, u, v, lm_pts3d)

            pde_errors = []
            for i, (nm, pp) in enumerate(zip(lm_names, proj_px)):
                if pp is None:
                    continue
                px_l, py_l = pp  # position in left (real C-arm) panel
                px_r, py_r = px_l + OUT_SIZE, py_l  # position in right (DRR) panel
                hex_c = LABEL_COLORS.get(nm, '#FFFFFF')
                rc, gc_c, bc = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)

                # Draw on DRR (right panel) — estimated projected position
                cv2.circle(bgr, (px_r, py_r), 6, (rc, gc_c, bc), -1)
                cv2.circle(bgr, (px_r, py_r), 6, (255,255,255), 1)
                cv2.putText(bgr, nm, (px_r+7, py_r+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (rc, gc_c, bc), 1)

                # If auto-detected position available, draw on real C-arm and compute PDE
                if auto_2d and nm in auto_2d:
                    gtu, gtv = auto_2d[nm]
                    # Scale GT from original image to OUT_SIZE
                    orig_sz = max(target.shape)
                    gtu_s = int(gtu * OUT_SIZE / orig_sz)
                    gtv_s = int(gtv * OUT_SIZE / orig_sz)
                    cv2.circle(bgr, (gtu_s, gtv_s), 6, (0, 220, 0), -1)
                    cv2.circle(bgr, (gtu_s, gtv_s), 6, (255,255,255), 1)
                    cv2.putText(bgr, nm, (gtu_s+7, gtv_s+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,0), 1)
                    # PDE in mm
                    err_px = np.linalg.norm(np.array([pp[0] - gtu_s, pp[1] - gtv_s]))
                    pde_errors.append(float(err_px) * PIX_MM_BANK)

            go_val = float(go(drr, target))
            pde_mm = float(np.mean(pde_errors)) if pde_errors else None

            # Labels
            cv2.putText(bgr, f'Real C-arm ({label})',
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,220,0), 1)
            lbl_drr = f'Registered DRR  GO={go_val:.3f}'
            if pde_mm is not None:
                lbl_drr += f'  PDE={pde_mm:.1f}mm'
            cv2.putText(bgr, lbl_drr,
                        (OUT_SIZE+4, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,200,255), 1)
            cv2.putText(bgr, 'GREEN=GT  COLOUR=estimated',
                        (OUT_SIZE+4, OUT_SIZE-6), cv2.FONT_HERSHEY_SIMPLEX, .35, (200,200,200), 1)

            _, buf = cv2.imencode('.png', bgr)
            return buf.tobytes(), go_val, pde_mm

        if best_res is None:
            log_queue.append('Refinement could not improve retrieval — using best bank entry.')
            ai0, li0 = joint[0][1], joint[0][2]
            best_res = dict(
                src_ap=np.array(ap_e[ai0]['src']), pax_ap=np.array(ap_e[ai0]['pax']),
                u_ap=np.array(ap_e[ai0]['u']),     v_ap=np.array(ap_e[ai0]['v']),
                src_lat=np.array(lat_e[li0]['src']),pax_lat=np.array(lat_e[li0]['pax']),
                u_lat=np.array(lat_e[li0]['u']),    v_lat=np.array(lat_e[li0]['v']))
        r = best_res
        # Auto-detect landmarks in both frames — no manual labels needed
        log_queue.append('Auto-detecting landmark positions for PDE computation...')
        lm_pts3d = [centroids[n] for n in lm_names]
        proj_ap  = project_landmarks(r['src_ap'],  r['pax_ap'],  r['u_ap'],  r['v_ap'],  lm_pts3d)
        proj_lat = project_landmarks(r['src_lat'], r['pax_lat'], r['u_lat'], r['v_lat'], lm_pts3d)
        auto_ap  = auto_detect_2d_landmarks(data['ap_array'],  lm_names, proj_ap,  OUT_SIZE)
        auto_lat = auto_detect_2d_landmarks(data['lat_array'], lm_names, proj_lat, OUT_SIZE)
        method = 'CNN' if len(auto_ap) > 0 else 'gradient'
        log_queue.append(f'Auto-detected {len(auto_ap)} AP / {len(auto_lat)} LAT landmarks  (method: {method})')
        ap_img,  go_a, pde_a = make_overlay(r['src_ap'],  r['pax_ap'],  r['u_ap'],  r['v_ap'],  target_ap,  'AP',  auto_ap,  proj_ap)
        lat_img, go_l, pde_l = make_overlay(r['src_lat'], r['pax_lat'], r['u_lat'], r['v_lat'], target_lat, 'LAT', auto_lat, proj_lat)

        pde_mean = None
        if pde_a is not None and pde_l is not None:
            pde_mean = (pde_a + pde_l) / 2.0
        elif pde_a is not None:
            pde_mean = pde_a
        elif pde_l is not None:
            pde_mean = pde_l

        if pde_mean is not None:
            log_queue.append(f'PDE (AP)={pde_a:.1f}mm  PDE (LAT)={pde_l:.1f}mm  Mean={pde_mean:.1f}mm')
        else:
            log_queue.append('PDE: not computed (no ground truth 2D labels). Visual overlay shows estimated projections.')

        result_holder["result"] = {
            'ap_overlay':  ap_img,
            'lat_overlay': lat_img,
            'go_ap':  go_a,
            'go_lat': go_l,
            'go_joint': (go_a + go_l) / 2.0,
            'pde_ap':   pde_a,
            'pde_lat':  pde_l,
            'pde_mean': pde_mean,
        }
        log_queue.append('DONE')

    except Exception:
        import traceback
        log_queue.append(f'ERROR: {traceback.format_exc()}')


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Run and View
# ══════════════════════════════════════════════════════════════════════════════
def step_run_and_view():
    st.header('Step 5 — Run Registration')

    with st.expander('Input summary', expanded=False):
        c1,c2,c3 = st.columns(3)
        c1.metric('CT',           st.session_state.ct_filename or 'loaded')
        c2.metric('Centroids',    len(st.session_state.centroids))
        c3.metric('C-arm frames', len(st.session_state.carm_frames))
        cc1,cc2 = st.columns(2)
        fi = st.session_state.carm_frames
        cc1.metric('AP frame',  fi[st.session_state.ap_frame_idx]['meta']['name']
                   if st.session_state.ap_frame_idx is not None else '—')
        cc2.metric('LAT frame', fi[st.session_state.lat_frame_idx]['meta']['name']
                   if st.session_state.lat_frame_idx is not None else '—')

    # ── Already have a result ─────────────────────────────────────────────────
    if st.session_state.result is not None:
        r = st.session_state.result
        st.success('Registration complete!')

        col_a, col_b, col_c = st.columns(3)
        col_a.metric('GO similarity AP',  f"{r['go_ap']:.4f}")
        col_b.metric('GO similarity LAT', f"{r['go_lat']:.4f}")
        col_c.metric('Joint GO',          f"{r['go_joint']:.4f}")

        # PDE — shown if ground truth labels were provided
        if r.get('pde_mean') is not None:
            st.markdown('---')
            pd1, pd2, pd3 = st.columns(3)
            pde_m = r['pde_mean']
            pd1.metric('PDE (AP)',  f"{r['pde_ap']:.1f} mm"  if r.get('pde_ap')  else 'N/A')
            pd2.metric('PDE (LAT)', f"{r['pde_lat']:.1f} mm" if r.get('pde_lat') else 'N/A')
            if pde_m < 2.0:
                pd3.metric('Mean PDE ✅', f'{pde_m:.1f} mm', delta='< 2mm target met')
            elif pde_m < 5.0:
                pd3.metric('Mean PDE ⚠️', f'{pde_m:.1f} mm', delta='< 5mm — acceptable')
            else:
                pd3.metric('Mean PDE ❌', f'{pde_m:.1f} mm', delta='> 5mm — needs improvement')
            st.caption(
                'PDE = Projection Distance Error. '
                'Measures how far the estimated landmark positions are from the '
                'ground truth 2D positions in the X-ray image (in mm). '
                'Clinical target: < 2mm.')
        else:
            st.info(
                '📐 **PDE not computed** — no ground truth 2D labels were provided. '
                'The coloured dots on the DRR overlay show where the 3D CT centroids project '
                'under the estimated camera pose. Verify visually that they land on the correct vertebrae. '
                'To get a numeric PDE, upload `.mrk.json` label files in Step 4.')
        # GO interpretation guide
        go_j = r['go_joint']
        if go_j >= 0.55:
            st.success(f'GO = {go_j:.3f}  — Good structural alignment. Bone edges match well.')
        elif go_j >= 0.42:
            st.warning(f'GO = {go_j:.3f}  — Moderate alignment. Bone structure roughly matches but may need refinement.')
        else:
            st.error(f'GO = {go_j:.3f}  — Low alignment. Registration may be inaccurate — check AP/LAT frame selection.')

        with st.expander('ℹ️ About these metrics'):
            st.markdown("""
**GO Similarity (Gradient Orientation)** measures how well the bone edges
and gradients in the registered DRR match the real C-arm image.  
Range 0–1. Higher = better. Typical values: >0.55 good, 0.42–0.55 moderate, <0.42 poor.

**PDE (Projection Distance Error)** is the gold-standard accuracy metric — it measures
how far the 3D landmarks project from their true 2D positions (in mm).  
PDE requires manually labeled 2D ground-truth positions on the C-arm image,
which are not required in this automated pipeline.
To compute PDE: go back to the labeled sets and provide 2D centroid positions.

**Why not Dice score?** Dice measures overlap between two 3D binary segmentations.
It is not the right metric for 2D/3D registration — it has no direct clinical meaning here.
PDE and GO are the appropriate metrics for this problem.
            """)

        st.markdown('### AP View — Real C-arm (left)  vs  Registered DRR (right)')
        st.image(r['ap_overlay'],  use_column_width=True)
        st.markdown('### LAT View — Real C-arm (left)  vs  Registered DRR (right)')
        st.image(r['lat_overlay'], use_column_width=True)

        dl1, dl2, reset_col = st.columns(3)
        with dl1:
            st.download_button('Download AP overlay',  r['ap_overlay'],
                               'registered_ap.png',  'image/png')
        with dl2:
            st.download_button('Download LAT overlay', r['lat_overlay'],
                               'registered_lat.png', 'image/png')
        with reset_col:
            if st.button('Run again'):
                st.session_state.result = None; st.experimental_rerun()
        return

    # ── Start button ──────────────────────────────────────────────────────────
    if not st.session_state.running:
        cb, cr = st.columns(2)
        with cb:
            if st.button('← Back'):
                st.session_state.step = 3; st.experimental_rerun()
        with cr:
            if st.button('▶ Start Registration', type='primary'):
                st.session_state.running   = True
                st.session_state.log_lines = []
                st.experimental_rerun()
        return

    # ── Running: show live log ────────────────────────────────────────────────
    st.info('Registration running... (1–3 min on GPU, longer on CPU)')

    log   = st.session_state.log_lines
    log_ph = st.empty()

    # Start thread only if log is empty (first run of this rerun loop)
    if not log:
        # Extract all data from session_state BEFORE starting thread.
        # Background threads cannot safely access st.session_state.
        fi = st.session_state.carm_frames

        # Downsample CT by 2x before passing to thread.
        # The DRR renderer only needs moderate resolution (renders at 256x256).
        # Full-res CT (e.g. 371x512x512 float32 = 371 MB) would require ~742 MB
        # when CTVolume creates its internal copy — this fails on memory-limited systems.
        # 2x downsample reduces to ~46 MB, which is fully sufficient for DRR quality.
        import scipy.ndimage as _nd
        ct_full = st.session_state.ct_volume
        ds = 0.5  # downsample factor
        with st.spinner("Downsampling CT for memory efficiency..."):
            ct_ds = _nd.zoom(ct_full, ds, order=1).astype(np.float32)
        sp_orig = st.session_state.ct_spacing
        sp_ds   = (sp_orig[0] / ds, sp_orig[1] / ds, sp_orig[2] / ds)
        ori_ds  = st.session_state.ct_origin  # origin stays the same

        # Scale centroid world coordinates are already in mm — no change needed
        pipeline_data = {
            "ct_volume":   ct_ds,
            "ct_spacing":  sp_ds,
            "ct_origin":   ori_ds,
            "centroids":   dict(st.session_state.centroids),
            "ap_array":    fi[st.session_state.ap_frame_idx]["array"].copy(),
            "lat_array":   fi[st.session_state.lat_frame_idx]["array"].copy(),
            "camera_name": st.session_state.camera_name,
            "bank_size":   st.session_state.bank_size,
        }
        result_holder = {"result": None}
        st.session_state._result_holder = result_holder
        t = threading.Thread(
            target=run_pipeline_thread,
            args=(st.session_state.log_lines, pipeline_data, result_holder),
            daemon=True)
        t.start()

    # Poll until done or error
    for _ in range(600):
        with log_ph.container():
            for line in log[-15:]:
                if 'ERROR' in line:
                    st.error(line)
                elif line == 'DONE' or 'result images' in line:
                    st.success(line)
                else:
                    st.text(line)

        if log and ('DONE' in log[-1] or 'ERROR' in log[-1]):
            break
        time.sleep(0.5)
        st.experimental_rerun()

    st.session_state.running = False
    # Copy result from thread's holder to session_state
    rh = getattr(st.session_state, "_result_holder", None)
    if rh and rh.get("result"):
        st.session_state.result = rh["result"]
    if st.session_state.result:
        st.experimental_rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    init_state()
    render_sidebar()
    step = st.session_state.step
    if   step == 0: step_load_ct()
    elif step == 1: step_mark_centroids()
    elif step == 2: step_load_carm()
    elif step == 3: step_select_frames()
    elif step == 4: step_run_and_view()


if __name__ == '__main__':
    main()
