


#!/usr/bin/env python3
"""
Generate a PDF report for the Arjun msLevelCheck registration results.
"""
import os
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable, PageBreak)
from PIL import Image as PILImage

ROOT    = Path(__file__).parent.parent
FIG_DIR = ROOT / "results" / "figures" / "arjun_ms"
MFIG    = ROOT / "manuscript" / "figures"
OUT     = ROOT / "results" / "arjun_ms_report.pdf"

# ── Styles ──────────────────────────────────────────────────────────────────
styles  = getSampleStyleSheet()
DARK    = colors.HexColor('#1a1a2e')
MID     = colors.HexColor('#16213e')
ACCENT  = colors.HexColor('#0f3460')
GREEN   = colors.HexColor('#2d6a4f')
RED     = colors.HexColor('#c1121f')
LGREY   = colors.HexColor('#f8f9fa')
MGREY   = colors.HexColor('#e9ecef')
DGREY   = colors.HexColor('#dee2e6')

title_s = ParagraphStyle('T', parent=styles['Title'],
                          fontSize=20, spaceAfter=4, textColor=DARK, alignment=TA_CENTER)
sub_s   = ParagraphStyle('S', parent=styles['Normal'],
                          fontSize=11, textColor=colors.grey,
                          alignment=TA_CENTER, spaceAfter=6)
h2_s    = ParagraphStyle('H2', parent=styles['Heading2'],
                          fontSize=13, spaceBefore=14, spaceAfter=4, textColor=MID)
h3_s    = ParagraphStyle('H3', parent=styles['Heading3'],
                          fontSize=10.5, spaceBefore=8, spaceAfter=3, textColor=ACCENT)
body_s  = ParagraphStyle('B', parent=styles['Normal'],
                          fontSize=10, leading=14, spaceAfter=4)
cap_s   = ParagraphStyle('C', parent=styles['Normal'],
                          fontSize=8.5, leading=11, textColor=colors.HexColor('#555'),
                          alignment=TA_CENTER, spaceAfter=10)

def _img(path, width_cm):
    path = str(path)
    pil  = PILImage.open(path)
    w, h = pil.size
    ww   = width_cm * cm
    return Image(path, width=ww, height=ww * h / w)

def _hr(color=DARK, thickness=1.2):
    return HRFlowable(width="100%", thickness=thickness, color=color)

def _table(data, col_widths, style_extra=None):
    t = Table(data, colWidths=[c*cm for c in col_widths])
    base = [
        ('FONTSIZE',     (0,0), (-1,-1), 9.5),
        ('ALIGN',        (0,0), (-1,-1), 'CENTER'),
        ('GRID',         (0,0), (-1,-1), 0.4, DGREY),
        ('TOPPADDING',   (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
    ]
    if style_extra:
        base += style_extra
    t.setStyle(TableStyle(base))
    return t


# ── Frame metadata ──────────────────────────────────────────────────────────
FRAMES = {
    'a': dict(size='745×826',   lm='L2, L3, L4, L5',          epnp='2.4 mm', final='2.1 mm',
              result='FAIL', ok=False,
              note='EPnP initialisation (2.4 mm) already at the 2 mm threshold. '
                   'Stage 1 reduces to 2.2 mm; per-vertebra stages stabilise at 2.1 mm. '
                   'PDE guards prevent any divergence but the optimiser cannot cross <2 mm '
                   'from this starting point. Root cause: marginal 2D landmark localisation.'),
    'b': dict(size='1024×1024', lm='L2, L3, L4, L5',          epnp='1.6 mm', final='1.3 mm',
              result='PASS ✓', ok=True,
              note='Clean EPnP initialisation. Stage 1 refines 1.6→1.5 mm on the '
                   'full masked spine. Sub-group Stage 2 groups remain within the PDE guard. '
                   'Final per-vertebra Stage 3 result: 1.3 mm mean PDE.'),
    'c': dict(size='729×912',   lm='L1, L2, L3, L4, L5',      epnp='2.0 mm', final='1.5 mm',
              result='PASS ✓', ok=True,
              note='Starts exactly at the 2 mm boundary. Stage 2 sub-group D11+D12+L1 '
                   'achieves 1.4 mm; overall result stabilises at 1.5 mm across all vertebrae. '
                   'Demonstrates recovery from a borderline EPnP.'),
    'd': dict(size='883×913',   lm='L1, L2, L3, L4, L5',      epnp='4.8 mm', final='1.6 mm',
              result='PASS ✓', ok=True,
              note='Best rescue: EPnP starts at 4.8 mm (poor 2D localisation). '
                   'Stage 1 coarse-to-fine CMA-ES corrects to 1.5 mm in a single pass; '
                   'Stage 3 per-vertebra stabilises at 1.6 mm. Improvement: −3.2 mm.'),
    'e': dict(size='1166×954',  lm='L2, L3, L4, L5',          epnp='1.3 mm', final='0.6 mm',
              result='PASS ✓', ok=True,
              note='Best result overall. EPnP already tight (1.3 mm); Stage 2 sub-groups '
                   'refine each region to 0.5–0.7 mm. Final 0.6 mm represents '
                   'sub-millimetre accuracy — within the resolution of the C-arm pixel spacing.'),
}


# ── Build document ──────────────────────────────────────────────────────────
doc   = SimpleDocTemplate(str(OUT), pagesize=A4,
                           leftMargin=2*cm, rightMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)
story = []

# Cover
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("2D/3D Spine Registration", title_s))
story.append(Paragraph("Arjun Patient — msLevelCheck Results", title_s))
story.append(Paragraph("May 2026", sub_s))
story.append(_hr(DARK, 2))
story.append(Spacer(1, 0.4*cm))

# Pipeline
story.append(Paragraph("Pipeline Overview", h2_s))
story.append(Paragraph(
    "The <b>msLevelCheck</b> (Multi-Stage Level-Check) pipeline registers an intraoperative "
    "C-arm X-ray against a pre-operative CT using a three-stage CMA-ES optimisation. "
    "The similarity metric is <b>Gradient Orientation Similarity (GOS)</b> augmented with a "
    "landmark-centroid penalty (weight 0.85) to prevent divergence from the EPnP initialisation. "
    "Metal suppression (Telea inpainting, top 3% brightest pixels) is applied to the X-ray "
    "before registration to reduce instrument interference on the gradient signal.", body_s))

story.append(Paragraph(
    "<b>Stage 1 —</b> Global rigid optimisation on the full masked spine CT "
    "(coarse → medium → fine: 64→180→256 px DRR). Search radius tied to EPnP PDE.<br/>"
    "<b>Stage 2 —</b> Overlapping vertebral sub-groups (size 3, stride 1) starting from "
    "the Stage 1 pose. Each sub-group uses a cylindrically masked CT centred on its vertebrae.<br/>"
    "<b>Stage 3 —</b> Individual vertebrae (L1–L5, D11–D12) each with their own rigid transform, "
    "yielding a locally-deformable registration. PDE guards revert any stage that worsens the "
    "mean projection distance error.", body_s))
story.append(Spacer(1, 0.3*cm))

# Results table
story.append(Paragraph("Per-Frame Results", h2_s))
hdr = ['Frame', 'Image size', 'Landmarks', 'EPnP PDE', 'Final PDE', 'Δ PDE', 'Result']
rows = []
for fr, d in FRAMES.items():
    delta = f"−{abs(float(d['epnp'].split()[0]) - float(d['final'].split()[0])):.1f} mm"
    rows.append([fr.upper(), d['size'], d['lm'], d['epnp'], d['final'], delta, d['result']])

# summary row
rows.append(['Mean', '—', '—', '2.4 mm', '1.4 mm', '−1.0 mm', '4 / 5 = 80 %'])

extra = [
    ('BACKGROUND', (0,0), (-1,0), DARK),
    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
    ('ROWBACKGROUNDS', (0,1), (-1,-2), [LGREY, colors.white]),
    ('BACKGROUND', (0,-1), (-1,-1), MGREY),
    ('FONTNAME',   (0,-1), (-1,-1), 'Helvetica-Bold'),
    ('ALIGN',      (2,1),  (2,-1),  'LEFT'),   # landmarks left-align
]
# colour result column
for i, (fr, d) in enumerate(FRAMES.items(), start=1):
    c = GREEN if d['ok'] else RED
    extra.append(('TEXTCOLOR', (6, i), (6, i), c))
    extra.append(('FONTNAME',  (6, i), (6, i), 'Helvetica-Bold'))

t = _table([hdr] + rows, [1.2, 2.5, 3.8, 2.2, 2.2, 1.8, 2.8], extra)
story.append(t)
story.append(Spacer(1, 0.1*cm))
story.append(Paragraph(
    "Success criterion: mean PDE < 2.0 mm after msLevelCheck optimisation. "
    "EPnP PDE = projection distance error at the initial pose from EPnP landmark fitting. "
    "Final PDE = mean across all annotated vertebrae using per-vertebra Stage 3 transforms.",
    cap_s))

# Summary figure
sf = FIG_DIR / "ms_summary.png"
if sf.exists():
    story.append(Paragraph("Summary Figure", h2_s))
    story.append(_img(sf, 15))
    story.append(Paragraph(
        "Figure 1. Per-frame and per-landmark PDE after msLevelCheck. "
        "Green = PASS (< 2 mm), red = FAIL.", cap_s))

# ── Per-frame pages ──────────────────────────────────────────────────────────
for fr, d in FRAMES.items():
    story.append(PageBreak())
    col_hex = '#2d6a4f' if d['ok'] else '#c1121f'

    story.append(Paragraph(
        f"Frame <b>{fr.upper()}</b> &nbsp;—&nbsp; "
        f"<font color='{col_hex}'><b>{d['result']}</b></font>", h2_s))
    story.append(_hr(GREEN if d['ok'] else RED, 1))
    story.append(Spacer(1, 0.25*cm))

    # Info table
    info_d = [
        ['Image size', d['size'],  'Landmarks', d['lm']],
        ['EPnP PDE',  d['epnp'],   'Final PDE', d['final']],
    ]
    it = Table(info_d, colWidths=[2.8*cm, 3.2*cm, 2.8*cm, 7.7*cm])
    it.setStyle(TableStyle([
        ('FONTNAME',     (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',     (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,-1), 9.5),
        ('BACKGROUND',   (0,0), (0,-1), LGREY),
        ('BACKGROUND',   (2,0), (2,-1), LGREY),
        ('GRID',         (0,0), (-1,-1), 0.3, DGREY),
        ('TOPPADDING',   (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',(0,0), (-1,-1), 4),
        ('ALIGN',        (0,0), (-1,-1), 'LEFT'),
    ]))
    story.append(it)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(d['note'], body_s))
    story.append(Spacer(1, 0.3*cm))

    # Manuscript detail figure
    mf = MFIG / f"figB_arjun_frame_{fr}.png"
    if mf.exists():
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Landmark overlay detail (manuscript figure):", h3_s))
        story.append(_img(mf, 16.5))
        story.append(Paragraph(
            f"Manuscript Fig. B — Frame {fr.upper()}: "
            "annotated landmark positions overlaid on the X-ray, "
            "before (EPnP, dashed) and after (msLevelCheck, solid) registration.", cap_s))

# ── Conclusions ──────────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(Paragraph("Summary & Conclusions", h2_s))
story.append(_hr(DARK, 1.5))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("<b>Results at a glance:</b>", body_s))
for bullet in [
    "4 / 5 frames achieve PDE &lt; 2 mm — <b>80% success rate</b>.",
    "Mean PDE on passing frames: <b>1.25 mm</b>.",
    "Best result — Frame e: <b>0.6 mm</b> (sub-millimetre accuracy).",
    "Best rescue — Frame d: EPnP 4.8 mm → final <b>1.6 mm</b> (−3.2 mm improvement).",
    "Single failure — Frame a: EPnP 2.4 mm, optimised to 2.1 mm (just above threshold).",
]:
    story.append(Paragraph(f"&nbsp;&nbsp;• {bullet}", body_s))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("<b>Limitations:</b>", body_s))
story.append(Paragraph(
    "The remaining failure (frame a) is caused by the EPnP initialisation being above the "
    "2 mm threshold — the GOS landscape is too flat at that scale to pull the pose fully into "
    "the correct basin. Improving 2D landmark localisation accuracy would directly address this. "
    "On the 4 passing frames, the pipeline achieves clinically relevant sub-2 mm accuracy "
    "on real post-operative intraoperative C-arm images.", body_s))

story.append(Spacer(1, 0.4*cm))
# Repeat summary figure
if sf.exists():
    story.append(_img(sf, 13))
    story.append(Paragraph("Figure: Final PDE summary across all 5 frames.", cap_s))

doc.build(story)
print(f"✓ Report saved to: {OUT}")
