"""
annotate_2d_landmarks.py — Interactive 2D Landmark Annotator
=============================================================
Opens each selected Ramulamma DICOM frame in a matplotlib window.
You click on L1, L2, L3, L4, L5 vertebral body centres IN ORDER.
The script saves a Slicer-compatible mrk.json for each frame.

Controls:
  Left-click  : place next landmark
  Right-click : undo last point
  Enter / D   : done with this frame, move to next
  Q           : quit (saves progress so far)
  R           : redo current frame (clear and restart)

Output:
  data/Ramulamma intra op Dicom images/20251011/landmarks_2d/frame_<N>.mrk.json

Usage:
  python scripts/annotate_2d_landmarks.py
  python scripts/annotate_2d_landmarks.py --frames 40 50 60 70 80
  python scripts/annotate_2d_landmarks.py --frames 50  # annotate just one frame
  python scripts/annotate_2d_landmarks.py --list        # show available frames

Instructions:
  1. The X-ray image appears. Spine runs roughly vertically.
  2. Click the CENTRE of each vertebral body from BOTTOM to TOP:
        L5  (lowest) → L4 → L3 → L2 → L1 (highest)
  3. A coloured dot + label appears after each click.
  4. Press Enter when all 5 are placed to save and go to next frame.
  5. Right-click to undo the last point if you mis-clicked.

Tip: Zoom with matplotlib toolbar before clicking for accuracy.
     The more accurate your clicks, the better the registration.
"""

import sys, os, json, argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pydicom

# ── Paths ─────────────────────────────────────────────────────────────────
DICOM_DIR = Path('/home/supermicro/Documents/2D_3D_Raghu/data'
                 '/Ramulamma intra op Dicom images/20251011/0')
OUT_DIR   = DICOM_DIR.parent / 'landmarks_2d'

LANDMARK_NAMES = ['L5', 'L4', 'L3', 'L2', 'L1']   # click order: bottom → top
COLORS = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

SLICER_MRK_TEMPLATE = {
    "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/"
               "Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
    "markups": [{
        "type": "Fiducial",
        "coordinateSystem": "LPS",
        "locked": False,
        "labelFormat": "%N-%d",
        "controlPoints": [],
        "measurements": [],
        "display": {}
    }]
}


# ── DICOM helpers ──────────────────────────────────────────────────────────

def get_all_frames() -> dict:
    """Return {instance_number: path} for all DICOM files."""
    frames = {}
    for p in sorted(DICOM_DIR.iterdir()):
        if p.is_file():
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                inst = int(getattr(ds, 'InstanceNumber', -1))
                if inst > 0:
                    frames[inst] = p
            except Exception:
                pass
    return frames


def load_frame(path: Path) -> np.ndarray:
    """Load DICOM → float32 [0,1] 1024×1024."""
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.max() > 0:
        arr /= arr.max()
    return arr


def already_done(frame_num: int) -> bool:
    return (OUT_DIR / f'frame_{frame_num}.mrk.json').exists()


# ── Slicer JSON writer ────────────────────────────────────────────────────

def save_landmarks(frame_num: int, points: list):
    """Save [(label, u, v), ...] as Slicer mrk.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = json.loads(json.dumps(SLICER_MRK_TEMPLATE))   # deep copy
    for i, (label, u, v) in enumerate(points):
        cp = {
            "id": str(i + 1),
            "label": label,
            "description": "",
            "associatedNodeID": "",
            "position": [float(u), float(v), 0.0],
            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined"
        }
        doc['markups'][0]['controlPoints'].append(cp)
    out = OUT_DIR / f'frame_{frame_num}.mrk.json'
    with open(out, 'w') as f:
        json.dump(doc, f, indent=2)
    print(f"  ✓ Saved: {out}")


# ── Interactive annotator ─────────────────────────────────────────────────

def annotate_frame(frame_num: int, image: np.ndarray, skip_if_done: bool = True):
    """Annotate one frame interactively. Returns list of (label, u, v) or None."""
    out_path = OUT_DIR / f'frame_{frame_num}.mrk.json'
    if skip_if_done and out_path.exists():
        print(f"  Frame {frame_num}: already annotated — skipping "
              f"(delete {out_path} to re-annotate)")
        return None

    print(f"\n{'='*60}")
    print(f"  Frame {frame_num}:  click L5 → L4 → L3 → L2 → L1")
    print(f"  Right-click = undo | Enter/D = save & next | R = restart | Q = quit")
    print(f"{'='*60}")

    # State
    points = []     # [(label, u, v)]
    done   = [False]
    quit_  = [False]
    redo_  = [False]

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title(f'Annotate frame {frame_num}')
    ax.imshow(image, cmap='gray', vmin=0, vmax=1, origin='upper')
    ax.set_title(f'Frame {frame_num} — click: {LANDMARK_NAMES[0]}', fontsize=12)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    # Scatter and text handles
    markers = []
    texts   = []

    def _refresh_title():
        n = len(points)
        if n < len(LANDMARK_NAMES):
            ax.set_title(
                f'Frame {frame_num}  [{n}/{len(LANDMARK_NAMES)}]  '
                f'→ click  {LANDMARK_NAMES[n]}   '
                f'(R=restart  RightClick=undo  Enter=save)',
                fontsize=10, color='white'
            )
        else:
            ax.set_title(
                f'Frame {frame_num}  ALL DONE — press Enter to save & continue',
                fontsize=11, color='#2ecc71'
            )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.button is None:
            return
        if event.button == 3:   # right-click = undo
            if points:
                points.pop()
                m = markers.pop(); m.remove()
                t = texts.pop();   t.remove()
                _refresh_title()
            return
        if event.button != 1:
            return
        n = len(points)
        if n >= len(LANDMARK_NAMES):
            return
        label = LANDMARK_NAMES[n]
        u, v  = event.xdata, event.ydata
        points.append((label, u, v))
        col = COLORS[n]
        m, = ax.plot(u, v, 'o', color=col, markersize=10,
                     markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        t  = ax.text(u + 8, v - 8, label, color=col, fontsize=11,
                     fontweight='bold', zorder=6)
        markers.append(m)
        texts.append(t)
        _refresh_title()

    def on_key(event):
        if event.key in ('enter', 'd'):
            if len(points) == len(LANDMARK_NAMES):
                done[0] = True
                plt.close(fig)
            else:
                print(f"  Need all {len(LANDMARK_NAMES)} landmarks "
                      f"({len(points)} placed so far)")
        elif event.key == 'q':
            quit_[0] = True
            plt.close(fig)
        elif event.key == 'r':
            redo_[0] = True
            plt.close(fig)

    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0d0d1a')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    _refresh_title()
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

    if quit_[0]:
        print("  Quitting.")
        return 'quit'
    if redo_[0]:
        print("  Redoing frame.")
        return annotate_frame(frame_num, image, skip_if_done=False)
    if done[0]:
        save_landmarks(frame_num, points)
        return points
    return None


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Annotate 2D landmarks on Ramulamma C-arm DICOMs')
    parser.add_argument('--frames', nargs='+', type=int, default=None,
                        help='DICOM instance numbers to annotate (default: auto-select 8 evenly-spaced)')
    parser.add_argument('--list', action='store_true',
                        help='List all available frames and exit')
    parser.add_argument('--redo', action='store_true',
                        help='Re-annotate even if file already exists')
    args = parser.parse_args()

    all_frames = get_all_frames()
    if not all_frames:
        print(f"ERROR: No DICOM files found in {DICOM_DIR}")
        sys.exit(1)

    if args.list:
        annotated = {n for n in all_frames if already_done(n)}
        print(f"\nAvailable frames ({len(all_frames)} total):")
        for n in sorted(all_frames.keys()):
            status = '✓ annotated' if n in annotated else '  pending'
            print(f"  {n:>4}  {status}")
        print(f"\nAnnotated: {len(annotated)}/{len(all_frames)}")
        print(f"\nAnnotation output dir: {OUT_DIR}")
        return

    # Default: pick 8 evenly-spaced frames across the series
    if args.frames is None:
        all_nums = sorted(all_frames.keys())
        n_want = 8
        idxs = np.linspace(0, len(all_nums) - 1, n_want, dtype=int)
        frames_to_annotate = [all_nums[i] for i in idxs]
        print(f"Auto-selected {n_want} frames: {frames_to_annotate}")
        print("Use --frames 10 20 30 ... to pick specific frames.")
    else:
        frames_to_annotate = args.frames

    # Filter to existing frames
    missing = [f for f in frames_to_annotate if f not in all_frames]
    if missing:
        print(f"WARNING: frames not found in DICOM dir: {missing}")
    frames_to_annotate = [f for f in frames_to_annotate if f in all_frames]

    print(f"\nWill annotate {len(frames_to_annotate)} frames: {frames_to_annotate}")
    print(f"Output dir: {OUT_DIR}")
    print("\nFor each frame:")
    print("  Click vertebral body centres: L5 (bottom) → L4 → L3 → L2 → L1 (top)")
    print("  Right-click = undo,  Enter = save & next,  Q = quit\n")

    for frame_num in frames_to_annotate:
        image = load_frame(all_frames[frame_num])
        result = annotate_frame(frame_num, image,
                                skip_if_done=(not args.redo))
        if result == 'quit':
            break

    # Summary
    annotated = sorted(n for n in frames_to_annotate if already_done(n))
    print(f"\n{'='*60}")
    print(f"Annotation complete: {len(annotated)} frames saved")
    print(f"Saved to: {OUT_DIR}")
    if annotated:
        print(f"Annotated frames: {annotated}")
        print(f"\nNext step:")
        print(f"  python run_ramulamma.py")
    else:
        print("No frames annotated yet.")


if __name__ == '__main__':
    main()
