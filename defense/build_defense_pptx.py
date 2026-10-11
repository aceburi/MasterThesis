#!/usr/bin/env python3
"""Fill the HWAI / TUDa presentation template with the defense deck."""

from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
TEMPLATE = Path(r"c:\Users\User\Downloads\presentation_template (2).pptx")
OUT = Path(__file__).resolve().parent / "Defense_Cebrian_Urios.pptx"

FOOTER = "Master's Thesis Defense  /  Adrian Cebrian Urios"
DATE = "October 2026"

# TUDa theme
BLUE = RGBColor(0x00, 0x4E, 0x8A)
CYAN = RGBColor(0x00, 0x9C, 0xDA)
RED = RGBColor(0xE6, 0x00, 0x1A)
INK = RGBColor(0x00, 0x00, 0x00)
MUTED = RGBColor(0x53, 0x53, 0x53)
LINE = RGBColor(0xB5, 0xB5, 0xB5)
LIGHT = RGBColor(0xE7, 0xE6, 0xE6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xF4, 0xF4, 0xF4)

prs = Presentation(str(TEMPLATE))


def delete_all_slides():
    ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        rId = sldId.get(f"{ns}id")
        if rId:
            prs.part.drop_rel(rId)
        sldIdLst.remove(sldId)


def layout(name):
    for lay in prs.slide_layouts:
        if lay.name == name:
            return lay
    raise KeyError(name)


def get_ph(slide, idx):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            return ph
    return None


def set_run(run, text, size=None, bold=None, color=None, font=None):
    run.text = text
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    if font is not None:
        run.font.name = font


def _run(p):
    return p.runs[0] if p.runs else p.add_run()


def fill_tf(tf, lines, size=16, bold=False, color=INK, font="Arial", spacing=8, align=PP_ALIGN.LEFT):
    tf.clear()
    tf.word_wrap = True
    for i, item in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        p.space_before = Pt(0)
        p.level = 0
        if isinstance(item, tuple):
            text, kw = item
            set_run(_run(p), text, kw.get("size", size), kw.get("bold", bold),
                    kw.get("color", color), kw.get("font", font))
        else:
            set_run(_run(p), item, size, bold, color, font)


def set_ph_text(ph, text):
    """Keep the placeholder font (Arial Black on titles) and only change the words."""
    if ph is None:
        return
    tf = ph.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.text = text
    for extra in tf.paragraphs[1:]:
        for r in extra.runs:
            r.text = ""


def set_title(slide, text, size=24):
    title = slide.shapes.title
    if title is None:
        return
    set_ph_text(title, text)


def set_footer(slide):
    for ph in slide.placeholders:
        kind = ph.placeholder_format.type
        if kind == PP_PLACEHOLDER.FOOTER:
            fill_tf(ph.text_frame, [FOOTER], 10, False, MUTED, "Arial", 0)
        elif kind == PP_PLACEHOLDER.DATE:
            fill_tf(ph.text_frame, [DATE], 10, False, MUTED, "Arial", 0)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def add_slide(layout_name, title, title_size=24):
    s = prs.slides.add_slide(layout(layout_name))
    if title is not None and s.shapes.title is not None:
        set_title(s, title, title_size)
    set_footer(s)
    return s


def add_part(num, title, caption):
    s = add_slide("Section", title)
    sub = get_ph(s, 1)
    if sub is not None:
        fill_tf(sub.text_frame, [f"{num}  ·  {caption}"], 18, False, INK, "Arial", 0)
    notes(s, f"5 s. Name the part: {title}. Do not read the caption.")
    return s


def box(slide, l, t, w, h, fill=None, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sh.shadow.inherit = False
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill if fill else WHITE
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    return sh


def shfill(slide, kind, x, y, w, h, fill):
    sh = slide.shapes.add_shape(kind, x, y, w, h)
    sh.shadow.inherit = False
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    return sh


def connector(slide, x1, y1, x2, y2, color=LINE, width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    return c


def rbox(slide, l, t, w, h, fill=None, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    sh.shadow.inherit = False
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill if fill else WHITE
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    try:
        sh.adjustments[0] = 0.06
    except Exception:
        pass
    return sh


def tbox(slide, l, t, w, h, text, size=16, bold=False, color=INK,
         align=PP_ALIGN.LEFT, font="Arial"):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_after = Pt(0)
    set_run(_run(p), text, size, bold, color, font)
    return tb


def bullets(slide, l, t, w, h, items, size=16):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        set_run(_run(p), "•  " + item, size, False, INK, "Arial")
    return tb


def multi(slide, l, t, w, h, lines, size=15, spacing=8):
    tb = slide.shapes.add_textbox(l, t, w, h)
    fill_tf(tb.text_frame, lines, size, False, INK, "Arial", spacing)
    return tb


def add_img(slide, path, l, t, w=None, h=None):
    path = Path(path)
    if not path.exists():
        return None
    kw = {"left": l, "top": t}
    if w is not None:
        kw["width"] = w
    if h is not None:
        kw["height"] = h
    return slide.shapes.add_picture(str(path), **kw)


def arrow_right(slide, x, y, w=Inches(0.30), h=Inches(0.20), fill=RED):
    sh = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    sh.shadow.inherit = False
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    return sh


def add_img_fit(slide, path, top, max_w, max_h):
    """Place a figure, trim near-white margins, fit inside max_w × max_h, centred."""
    from io import BytesIO

    import numpy as np
    from PIL import Image

    path = Path(path)
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    a = np.array(im)
    mask = ~((a[:, :, 0] > 248) & (a[:, :, 1] > 248) & (a[:, :, 2] > 248))
    ys, xs = np.where(mask)
    if xs.size:
        pad = 28
        im = im.crop((
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(im.width, int(xs.max()) + pad),
            min(im.height, int(ys.max()) + pad),
        ))
    w_px, h_px = im.size
    w_in = max_w.inches
    h_in = w_in * h_px / w_px
    if h_in > max_h.inches:
        h_in = max_h.inches
        w_in = h_in * w_px / h_px
    left = Inches((prs.slide_width.inches - w_in) / 2)
    bio = BytesIO()
    im.save(bio, format="PNG")
    bio.seek(0)
    return slide.shapes.add_picture(bio, left, top, width=Inches(w_in), height=Inches(h_in))


def table(slide, l, t, w, h, rows, col_w=None):
    n_r, n_c = len(rows), len(rows[0])
    sh = slide.shapes.add_table(n_r, n_c, l, t, w, h)
    tbl = sh.table
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = cw
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = ""
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            try:
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
            except Exception:
                pass
            head = i == 0
            set_run(_run(p), str(val), 12 if head else 13, head,
                    WHITE if head else INK, "Arial")
            c.fill.solid()
            c.fill.fore_color.rgb = BLUE if head else (LIGHT if i % 2 == 0 else WHITE)
            _borders(c)
    return sh


def _borders(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    col = "B5B5B5"
    for edge in ("L", "R", "T", "B"):
        tcPr.append(etree.fromstring(
            f'<a:ln{edge} w="6350" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:solidFill><a:srgbClr val="{col}"/></a:solidFill></a:ln{edge}>'
        ))


def fill_body(slide, items, size=18):
    body = get_ph(slide, 1)
    if body is None:
        return
    fill_tf(body.text_frame, ["•  " + x for x in items], size, False, INK, "Arial", 12)


def fill_two(slide, left_items, right_items, size=16):
    left = get_ph(slide, 13)
    right = get_ph(slide, 14)
    if left:
        fill_tf(left.text_frame, left_items, size, False, INK, "Arial", 10)
    if right:
        fill_tf(right.text_frame, right_items, size, False, INK, "Arial", 10)


# ---------------------------------------------------------------------------
delete_all_slides()

# 1 Title
s = add_slide("Title Slide", None)
set_ph_text(get_ph(s, 0),
            "Conversion of Convolutional Neural Networks into Logic Flows for Efficient Execution on RISC-V")
fill_tf(get_ph(s, 1).text_frame, [
    "Adrian Cebrian Urios",
    "Master's Thesis Defense",
    "Examiner: Prof. Dr.-Ing. Li Zhang    ·    Supervisor: Daniel Stein, M.Sc.",
], 16, False, INK, "Arial", 6, PP_ALIGN.CENTER)
# emphasise name
p0 = get_ph(s, 1).text_frame.paragraphs[0]
set_run(_run(p0), "Adrian Cebrian Urios", 20, True, BLUE, "Arial")
set_footer(s)
notes(s, "30 s. Title, name, committee. One sentence: this thesis takes the NN2Logic hybrid from a fully connected occupancy network to tiny quantised CNNs and measures it on Spike, the RISC-V simulator, using the ISA Ibex implements.")

# 2 Agenda
s = add_slide("Title only", "Outline")
agenda = [
    ("01", "Motivation", "Local inference on a microcontroller, why CNNs still pay the same MACs, the ReLU tree, and NN2Logic on a fully connected network."),
    ("02", "Approach", "The pipeline, how the CNN is rewritten for NN2Logic, and how first-stage routes become a hybrid program."),
    ("03", "Evaluation", "Four networks, a matched looped reference. Cycles on Spike, the ISA that Ibex implements."),
    ("04", "Findings", "Results, then the limits of an early exit."),
]
for i, (num, tit, cap) in enumerate(agenda):
    y = Inches(2.15) + Inches(i * 1.15)
    rbox(s, Inches(0.39), y, Inches(12.55), Inches(1.05), PALE)
    tbox(s, Inches(0.55), y + Inches(0.22), Inches(1.0), Inches(0.6), num, 26, True, RED)
    tbox(s, Inches(1.7), y + Inches(0.12), Inches(10.9), Inches(0.38), tit, 20, True, BLUE)
    tbox(s, Inches(1.7), y + Inches(0.52), Inches(10.9), Inches(0.42), cap, 14, False, MUTED)
notes(s, "20 s. Four parts. Promise: not one speed-up figure, a characterisation of when the hybrid is a net win.")

add_part("01", "Motivation",
         "TinyML, CNNs, the ReLU tree, NN2Logic, then where the MACs sit.")

# 3 Motivation
s = add_slide("Title only", "TinyML setting")
apps = [
    ("Occupancy", "Always-on sensing in a building. The payload stays on the device."),
    ("Activity", "Wearable inertial units. The radio cannot dominate the budget."),
    ("Screening", "Medical images at the point of care. Latency has to be local."),
]
for i, (tit, body) in enumerate(apps):
    x = Inches(0.39) + Inches(i * 4.2)
    rbox(s, x, Inches(2.15), Inches(4.05), Inches(2.15), PALE)
    box(s, x, Inches(2.15), Inches(4.05), Inches(0.1), RED)
    tbox(s, x + Inches(0.25), Inches(2.4), Inches(3.6), Inches(0.45), tit, 20, True, BLUE)
    tbox(s, x + Inches(0.25), Inches(2.9), Inches(3.6), Inches(1.15), body, 14, False, INK)
rbox(s, Inches(0.39), Inches(4.5), Inches(12.55), Inches(2.25), BLUE)
tbox(s, Inches(0.65), Inches(4.65), Inches(12.1), Inches(0.35), "THE DEVICE", 12, True, CYAN)
chips = [
    ("TinyML", "A microcontroller.\nNo GPU."),
    ("CPU", "Control flow and\ninteger logic."),
    ("RISC-V", "A small open core.\nHere: Ibex."),
    ("Not a GPU", "No neural\naccelerator either."),
]
for i, (tit, body) in enumerate(chips):
    x = Inches(0.65) + Inches(i * 3.1)
    tbox(s, x, Inches(5.1), Inches(2.9), Inches(0.35), tit, 16, True, WHITE)
    tbox(s, x, Inches(5.5), Inches(2.9), Inches(1.05), body, 13, False, WHITE)
notes(s, "45 s. Three local tasks, then the device. Stop on Ibex. CNNs and the mismatch are the next slides.")

# 4b Tiny CNNs — thesis intro order
s = add_slide("Title only", "Tiny CNNs")
tbox(s, Inches(0.39), Inches(2.10), Inches(12.55), Inches(0.38),
     "CNNs are the default model for the visual and short time-series tasks in this setting.",
     16, False, INK)
cnn_fig = next((p for p in (FIG / "TinyCNN.jpeg", FIG / "TinyCNN.png", FIG / "TinyCNN.jpg") if p.exists()), None)
if cnn_fig is not None:
    add_img(s, cnn_fig, Inches(0.86), Inches(2.48), w=Inches(11.60))
rbox(s, Inches(0.39), Inches(5.62), Inches(12.55), Inches(0.68), PALE)
tbox(s, Inches(0.65), Inches(5.70), Inches(12.15), Inches(0.54),
     "Almost all of the arithmetic is sliding-window MACs. The TinyML literature makes those MACs cheaper: QAT, pruning, architecture search, scheduled loops.",
     15, False, INK)
rbox(s, Inches(0.39), Inches(6.38), Inches(12.55), Inches(0.72), BLUE)
tbox(s, Inches(0.65), Inches(6.50), Inches(12.15), Inches(0.50),
     "The kind of computation stays the same. Every sample still walks this block. That is a poor match for a microcontroller CPU.",
     15, False, WHITE)
notes(s, "50 s. Point at the figure left to right. Cheaper MACs, same block, then the hardware mismatch. No cycle figures.")

# 4c ReLU tree — thesis intro order
s = add_slide("Title only", "ReLU is a decision tree")
tbox(s, Inches(0.39), Inches(2.08), Inches(12.55), Inches(0.32),
     "Not a surrogate. A piecewise-linear network computes the same function as this tree.",
     16, False, INK)
rbox(s, Inches(0.39), Inches(2.48), Inches(5.85), Inches(3.38), PALE)
tbox(s, Inches(0.55), Inches(2.58), Inches(5.55), Inches(0.32), "THE NETWORK", 12, True, BLUE)
net = FIG / "stein_toy_network.png"
if net.exists():
    add_img(s, net, Inches(1.35), Inches(2.92), h=Inches(2.48))
rbox(s, Inches(7.10), Inches(2.48), Inches(5.85), Inches(3.38), PALE)
tbox(s, Inches(7.26), Inches(2.58), Inches(5.55), Inches(0.32), "THE TREE", 12, True, BLUE)
# diamonds and leaves
shfill(s, MSO_SHAPE.DIAMOND, Inches(8.95), Inches(2.95), Inches(2.15), Inches(0.72), CYAN)
tbox(s, Inches(8.95), Inches(3.10), Inches(2.15), Inches(0.42), "ReLU sign", 13, True, WHITE, PP_ALIGN.CENTER)
connector(s, Inches(9.55), Inches(3.67), Inches(8.35), Inches(4.05), BLUE, 2)
connector(s, Inches(10.50), Inches(3.67), Inches(11.55), Inches(4.05), BLUE, 2)
tbox(s, Inches(8.55), Inches(3.68), Inches(0.55), Inches(0.28), "T", 11, True, BLUE, PP_ALIGN.CENTER)
tbox(s, Inches(10.85), Inches(3.68), Inches(0.55), Inches(0.28), "F", 11, True, BLUE, PP_ALIGN.CENTER)
shfill(s, MSO_SHAPE.DIAMOND, Inches(7.35), Inches(4.02), Inches(2.00), Inches(0.68), CYAN)
tbox(s, Inches(7.35), Inches(4.16), Inches(2.00), Inches(0.40), "ReLU sign", 13, True, WHITE, PP_ALIGN.CENTER)
rbox(s, Inches(10.55), Inches(4.08), Inches(2.05), Inches(0.58), BLUE)
tbox(s, Inches(10.55), Inches(4.16), Inches(2.05), Inches(0.42), "class", 14, True, WHITE, PP_ALIGN.CENTER)
connector(s, Inches(7.95), Inches(4.70), Inches(7.55), Inches(5.08), BLUE, 2)
connector(s, Inches(8.85), Inches(4.70), Inches(9.25), Inches(5.08), BLUE, 2)
rbox(s, Inches(7.20), Inches(5.08), Inches(1.70), Inches(0.55), BLUE)
tbox(s, Inches(7.20), Inches(5.16), Inches(1.70), Inches(0.40), "class", 14, True, WHITE, PP_ALIGN.CENTER)
rbox(s, Inches(8.95), Inches(5.08), Inches(1.70), Inches(0.55), BLUE)
tbox(s, Inches(8.95), Inches(5.16), Inches(1.70), Inches(0.40), "class", 14, True, WHITE, PP_ALIGN.CENTER)
tbox(s, Inches(6.24), Inches(3.75), Inches(0.86), Inches(0.55), "=", 32, True, RED, PP_ALIGN.CENTER)
tbox(s, Inches(0.55), Inches(5.52), Inches(5.55), Inches(0.28),
     "Network figure: Stein et al., NN2Logic, 2026.", 10, False, MUTED)
chips = [
    ("Early exit, still MACs", "Usual early-exit skips layers. What remains is still neural arithmetic."),
    ("A path on this CPU", "Integer comparisons and a reduced set of MACs. Closer to what the core already does."),
]
for i, (tit, body) in enumerate(chips):
    x = Inches(0.39) + Inches(i * 6.71)
    rbox(s, x, Inches(6.00), Inches(5.85), Inches(0.95), BLUE if i == 1 else PALE)
    tbox(s, x + Inches(0.18), Inches(6.08), Inches(5.50), Inches(0.30), tit, 14, True, WHITE if i == 1 else BLUE)
    tbox(s, x + Inches(0.18), Inches(6.40), Inches(5.50), Inches(0.45), body, 13, False, WHITE if i == 1 else INK)
notes(s, "45 s. Point at network = tree. Early-exit still MACs. Path = comparisons + fewer MACs. Then NN2Logic.")

# 5 NN2Logic
s = add_slide("Title only", "NN2Logic")
tbox(s, Inches(0.39), Inches(2.12), Inches(12.55), Inches(0.40),
     "NN2Logic, from TU Darmstadt, turns that tree into a program for a quantized integer network.",
     15, False, INK)
steps = [
    ("Explore", "Walk the ReLU signs seen in training. Keep only branches a sample actually took."),
    ("Certify", "A leaf that is one class on the whole cell, not only on the samples that reached it."),
    ("Emit", "Those paths as C. A miss falls through to the original integer network."),
]
for i, (tit, body) in enumerate(steps):
    x = Inches(0.39) + Inches(i * 4.2)
    rbox(s, x, Inches(2.55), Inches(4.05), Inches(1.15), PALE)
    tbox(s, x + Inches(0.15), Inches(2.62), Inches(3.75), Inches(0.32), tit, 16, True, BLUE)
    tbox(s, x + Inches(0.15), Inches(2.96), Inches(3.75), Inches(0.65), body, 12, False, INK)
    if i < 2:
        tbox(s, x + Inches(3.90), Inches(2.85), Inches(0.35), Inches(0.45), "›", 22, True, RED, PP_ALIGN.CENTER)
if (FIG / "stein_hybrid_flow.png").exists():
    add_img(s, FIG / "stein_hybrid_flow.png", Inches(2.15), Inches(3.80), h=Inches(1.85))
tbox(s, Inches(2.15), Inches(5.62), Inches(9.0), Inches(0.22),
     "Figure: Stein et al., NN2Logic, 2026.", 10, False, MUTED, PP_ALIGN.CENTER)
cases = [
    ("Occupancy perceptron", "Shown in the published work. Hidden layers of similar width, so a short prefix can skip a large share."),
    ("Convolutional models", "Left open. Generated code size was not reported. That is the starting point of this thesis."),
]
for i, (tit, body) in enumerate(cases):
    x = Inches(0.39) + Inches(i * 6.4)
    rbox(s, x, Inches(5.88), Inches(6.2), Inches(0.97), PALE if i == 0 else BLUE)
    tbox(s, x + Inches(0.2), Inches(5.92), Inches(5.8), Inches(0.28), tit, 14, True, BLUE if i == 0 else WHITE)
    tbox(s, x + Inches(0.2), Inches(6.22), Inches(5.8), Inches(0.52), body, 12, False, INK if i == 0 else WHITE)
notes(s, "1 min 10. Explore / certify / emit, then the figure, then occupancy vs CNN. No 14.9% / 0.7%.")

# 6 CNN gap
s = add_slide("Title only", "Where the MACs sit")
tbox(s, Inches(0.39), Inches(2.15), Inches(12.55), Inches(0.40),
     "TinyML vision and sensing use convolutional models, not the small fully connected networks NN2Logic was shown on. On a CNN the cost sits here:",
     15, False, INK)
rbox(s, Inches(0.39), Inches(2.75), Inches(9.4), Inches(1.55), BLUE)
tbox(s, Inches(0.65), Inches(2.95), Inches(8.9), Inches(0.50), "Convolution and pool", 22, True, WHITE)
tbox(s, Inches(0.65), Inches(3.55), Inches(8.9), Inches(0.50), "Almost all of the MACs.", 16, False, WHITE)
rbox(s, Inches(10.0), Inches(2.75), Inches(2.95), Inches(1.55), LIGHT)
tbox(s, Inches(10.15), Inches(3.05), Inches(2.65), Inches(0.45), "Head", 22, True, BLUE, PP_ALIGN.CENTER)
tbox(s, Inches(10.15), Inches(3.55), Inches(2.65), Inches(0.50), "A short vector.", 14, False, INK, PP_ALIGN.CENTER)
exits = [
    ("Test in the convolution", "The rest of the map is not computed. This is the saving we want."),
    ("Test in the head", "The convolution has already run. Almost nothing is skipped."),
]
for i, (tit, body) in enumerate(exits):
    x = Inches(0.39) + Inches(i * 6.4)
    rbox(s, x, Inches(4.55), Inches(6.2), Inches(2.30), PALE)
    box(s, x, Inches(4.55), Inches(0.12), Inches(2.30), CYAN if i == 0 else RED)
    tbox(s, x + Inches(0.35), Inches(4.75), Inches(5.6), Inches(0.45), tit, 18, True, BLUE)
    tbox(s, x + Inches(0.35), Inches(5.35), Inches(5.6), Inches(1.20), body, 16, False, INK)
notes(s, "40 s. Early test saves, late test does not. This is the starting point of the thesis after NN2Logic.")

add_part("02", "Approach",
         "The pipeline, then the CNN rewrite, then trim, vet, remaining-work score.")

# 7 Pipeline
s = add_slide("Title only", "Pipeline")
pipe = FIG / "pipeline.jpeg"
if pipe.exists():
    add_img_fit(s, pipe, Inches(2.12), Inches(12.50), Inches(3.85))
legend = [
    (PALE, INK, BLUE, "Grey", "NN2Logic, called unchanged."),
    (BLUE, WHITE, CYAN, "Blue", "Thesis: export, route selection, emission."),
]
for i, (fill, ink, kicker_c, kicker, body) in enumerate(legend):
    x = Inches(0.39) + Inches(i * 6.40)
    rbox(s, x, Inches(6.12), Inches(6.20), Inches(0.48), fill)
    tbox(s, x + Inches(0.18), Inches(6.18), Inches(1.10), Inches(0.36), kicker, 14, True, kicker_c)
    tbox(s, x + Inches(1.35), Inches(6.18), Inches(4.65), Inches(0.36), body, 14, False, ink)
rbox(s, Inches(0.39), Inches(6.68), Inches(12.55), Inches(0.42), PALE)
tbox(s, Inches(0.55), Inches(6.72), Inches(12.25), Inches(0.34),
     "A constant leaf the certifier never returned cannot be recovered later.",
     14, False, MUTED)
notes(s, "50 s. Walk left to right. Grey is NN2Logic. Blue is this thesis. Do not say internal names.")

# 7b Quantized CNN — rewritten so NN2Logic can consume it
s = add_slide("Title only", "The quantized CNN")
tbox(s, Inches(0.39), Inches(2.08), Inches(12.55), Inches(0.36),
     "NN2Logic has no convolution operator. The CNN is rewritten so the unchanged tree builder can consume it.",
     15, False, INK)
adapt = [
    ("1", "Unfold", "GRID  →  ROW", "Each output position becomes one row of a dense matrix."),
    ("2", "Fuse", "CONV + BN + AVG-POOL", "Average pooling is affine, so it folds with batch-norm into the unfolded matrix. Max pooling would not. Dropout is training-only."),
    ("3", "ReLU after pool", "ONE TEST PER CELL", "The tree sees one sign after the pool, not after every convolution output. Trained that way."),
    ("4", "Recover geometry", "KERNEL · STRIDE · PAD", "Unfolding drops them. They sit beside the tree so the emitter can write a loop."),
]
for i, (n, tit, visual, body) in enumerate(adapt):
    x = Inches(0.39) + Inches((i % 2) * 6.40)
    y = Inches(2.52) + Inches((i // 2) * 1.70)
    rbox(s, x, y, Inches(6.20), Inches(1.58), PALE)
    tbox(s, x + Inches(0.16), y + Inches(0.12), Inches(0.45), Inches(0.40), n, 22, True, RED)
    tbox(s, x + Inches(0.70), y + Inches(0.16), Inches(5.25), Inches(0.32), tit, 16, True, BLUE)
    tbox(s, x + Inches(0.70), y + Inches(0.52), Inches(5.25), Inches(0.32), visual, 13, True, CYAN)
    tbox(s, x + Inches(0.70), y + Inches(0.90), Inches(5.25), Inches(0.55), body, 13, False, INK)
rbox(s, Inches(0.39), Inches(6.05), Inches(12.55), Inches(0.85), BLUE)
tbox(s, Inches(0.60), Inches(6.18), Inches(12.15), Inches(0.60),
     "Symmetric 8-bit QAT, no zero-points. NN2Logic consumes this integer network, not a float model.",
     15, False, WHITE)
notes(s, "50 s. Four rewrites, all of them. No figure that only shows ReLU after pool. Do not name evaluation workloads.")

# 9 Route selection overview
s = add_slide("Title only", "Route selection")
flow = [
    ("1  Trim", "Shorten the path\ninto the convolution."),
    ("2  Vet", "May this route\nenter the pool?"),
    ("3  Score", "Charge the work\nthat still remains."),
    ("4  Embed", "A short list.\nCheapest first."),
]
for i, (tit, body) in enumerate(flow):
    x = Inches(0.39) + Inches(i * 3.2)
    rbox(s, x, Inches(2.25), Inches(3.05), Inches(2.15), PALE)
    tbox(s, x + Inches(0.15), Inches(2.40), Inches(2.75), Inches(0.50), tit, 18, True, BLUE)
    tbox(s, x + Inches(0.15), Inches(3.00), Inches(2.75), Inches(1.15), body, 15, False, INK)
    if i < 3:
        tbox(s, x + Inches(2.90), Inches(2.95), Inches(0.35), Inches(0.5), "›", 24, True, RED, PP_ALIGN.CENTER)
rbox(s, Inches(0.39), Inches(4.65), Inches(12.55), Inches(2.15), BLUE)
tbox(s, Inches(0.65), Inches(4.80), Inches(12.1), Inches(1.85),
     "A route, in this talk: a path of ReLU-sign tests plus a class to return. That class is the integer network's own prediction, not the dataset label.\n\nThe next three slides unpack trim, vet, and score.",
     15, False, WHITE)
notes(s, "30 s. Define route. Then unpack 1, 2, 3. No C(R).")

# 10 Trimming
s = add_slide("Title only", "Path trimming")
tbox(s, Inches(0.39), Inches(2.15), Inches(12.5), Inches(0.3), "STEP 1  ·  EXPLORED PATH TO A CONSTANT LEAF", 12, True, BLUE)
full = [("t1", True), ("t2", True), ("t3", True), ("t4", False), ("t5", False), ("t6", False)]
for i, (lab, early) in enumerate(full):
    x = Inches(0.5) + Inches(i * 2.05)
    rbox(s, x, Inches(2.55), Inches(1.9), Inches(0.85), CYAN if early else LIGHT)
    tbox(s, x, Inches(2.55), Inches(1.9), Inches(0.5), lab, 16, True, WHITE if early else INK, PP_ALIGN.CENTER)
    tbox(s, x, Inches(3.05), Inches(1.9), Inches(0.3), "convolution" if early else "dense head", 11, False, WHITE if early else MUTED, PP_ALIGN.CENTER)
tbox(s, Inches(0.39), Inches(3.6), Inches(12.5), Inches(0.35), "↓  keep the shortest prefix that still separates the leaf, then drop redundant tests", 14, False, RED)
tbox(s, Inches(0.39), Inches(4.05), Inches(12.5), Inches(0.3), "TRIMMED ROUTE", 12, True, BLUE)
for i, lab in enumerate(("t1", "t2")):
    x = Inches(0.5) + Inches(i * 2.05)
    rbox(s, x, Inches(4.45), Inches(1.9), Inches(0.85), BLUE)
    tbox(s, x, Inches(4.45), Inches(1.9), Inches(0.5), lab, 16, True, WHITE, PP_ALIGN.CENTER)
    tbox(s, x, Inches(4.95), Inches(1.9), Inches(0.3), "kept", 11, False, WHITE, PP_ALIGN.CENTER)
rbox(s, Inches(4.7), Inches(4.45), Inches(7.85), Inches(0.85), PALE)
tbox(s, Inches(4.9), Inches(4.55), Inches(7.5), Inches(0.65),
     "t3–t6 dropped. The class is returned from the convolution. That is how a test in the head becomes a test that can skip the map.",
     14, False, INK)
rbox(s, Inches(0.39), Inches(5.55), Inches(12.55), Inches(1.20), PALE)
tbox(s, Inches(0.6), Inches(5.70), Inches(12.15), Inches(0.95),
     "The published certificate is the other origin: it is already minimal, so nothing is deleted from it. On these CNNs it still names a head test. It is an ablation, not the reported selector.",
     15, False, INK)
notes(s, "45 s. This is step 1 of the previous slide. Point at the deletion.")

# 11 Vetting
s = add_slide("Title only", "Vetting")
tbox(s, Inches(0.39), Inches(2.12), Inches(12.5), Inches(0.28),
     "STEP 2  ·  REPRODUCE THE INTEGER NETWORK ON A VETTING SET", 12, True, BLUE)
tbox(s, Inches(0.39), Inches(2.48), Inches(12.5), Inches(0.40),
     "A candidate may enter the pool only if it does.", 15, False, MUTED)
rules = [
    ("1", "No disagreements", "On every sample the route matches, its class must equal the integer network's class. One disagreement rejects it."),
    ("2", "Laplace support", "A handful of agreeing samples is not enough. A Laplace estimate of the next mismatch must stay small: about 50 clean matches."),
    ("3", "Held-out check", "The same route must still have no disagreements on a 15% slice it was not trimmed against."),
]
for i, (n, tit, body) in enumerate(rules):
    y = Inches(2.98) + Inches(i * 1.08)
    rbox(s, Inches(0.39), y, Inches(12.55), Inches(1.05), PALE)
    tbox(s, Inches(0.55), y + Inches(0.28), Inches(0.70), Inches(0.50), n, 22, True, RED)
    tbox(s, Inches(1.40), y + Inches(0.12), Inches(11.2), Inches(0.35), tit, 16, True, BLUE)
    tbox(s, Inches(1.40), y + Inches(0.50), Inches(11.2), Inches(0.45), body, 14, False, INK)
rbox(s, Inches(0.39), Inches(6.32), Inches(12.55), Inches(0.58), PALE)
tbox(s, Inches(0.60), Inches(6.42), Inches(12.15), Inches(0.42),
     "This is measured agreement, not a proof for every input. That residual is the tenth of a point reported later.",
     14, False, MUTED)
notes(s, "50 s. Three plain rules. Class = integer network, not the label. No formula.")

# 12 Remaining-work score
s = add_slide("Title only", "Remaining-work score")
tbox(s, Inches(0.39), Inches(2.12), Inches(12.5), Inches(0.28), "STEP 3  ·  CHARGE WHAT THE SAMPLE STILL HAS TO EXECUTE", 12, True, BLUE)
tbox(s, Inches(0.39), Inches(2.50), Inches(12.5), Inches(0.55),
     "NN2Logic ranks by multiplications an exit leaves undone. This thesis ranks remaining work: a head test then looks cheap even though the convolution has already run.",
     14, False, MUTED)
tbox(s, Inches(0.39), Inches(3.20), Inches(2.3), Inches(0.7), "A match", 16, True, BLUE)
rbox(s, Inches(2.8), Inches(3.25), Inches(3.4), Inches(0.6), CYAN)
tbox(s, Inches(2.8), Inches(3.25), Inches(3.4), Inches(0.6), "receptive field of the cheapest route", 12, True, WHITE, PP_ALIGN.CENTER)
tbox(s, Inches(6.4), Inches(3.25), Inches(6.4), Inches(0.6), "the rest of the map is not computed", 13, False, MUTED)
tbox(s, Inches(0.39), Inches(4.15), Inches(2.3), Inches(0.7), "A miss", 16, True, BLUE)
rbox(s, Inches(2.8), Inches(4.20), Inches(9.9), Inches(0.6), BLUE)
tbox(s, Inches(2.8), Inches(4.20), Inches(9.9), Inches(0.6), "full network   +   the cost of having tried the routes", 13, True, WHITE, PP_ALIGN.CENTER)
rbox(s, Inches(0.39), Inches(5.10), Inches(12.55), Inches(1.75), PALE)
tbox(s, Inches(0.60), Inches(5.30), Inches(12.15), Inches(1.40),
     "The score picks a short list of routes. It is a MAC count, not a Spike measurement. The emitter is then re-run on that fixed list.\n\nA published certificate that still names a head test scores almost a full pass. That is why it does not move the mean.",
     15, False, INK)
notes(s, "40 s. Remaining work, not skipped work. No C(R).")

# 13 Hybrid runtime
s = add_slide("Title only", "Hybrid code generation")
hyb_fig = next((p for p in (FIG / "generated_hybrid.jpeg", FIG / "generated_hybrid.png") if p.exists()), None)
if hyb_fig is not None:
    add_img_fit(s, hyb_fig, Inches(2.12), Inches(12.40), Inches(3.12))
hyb_cards = [
    ("HOT", CYAN, "A first-stage output a route reads. Geometry tells the emitter which pixels that window uses."),
    ("MATCH", BLUE, "Every sign agrees. Return the class after the hot block. The cold loop and the head do not run."),
    ("MISS", RED, "A sign disagrees. Finish remaining cells in one kernel loop, then the remaining layers. Same weights."),
]
for i, (tit, accent, body) in enumerate(hyb_cards):
    x = Inches(0.39) + Inches(i * 4.20)
    rbox(s, x, Inches(5.42), Inches(4.05), Inches(1.48), PALE)
    box(s, x, Inches(5.42), Inches(0.10), Inches(1.48), accent)
    tbox(s, x + Inches(0.28), Inches(5.52), Inches(3.62), Inches(0.30), tit, 12, True, accent)
    tbox(s, x + Inches(0.28), Inches(5.86), Inches(3.62), Inches(0.90), body, 13, False, INK)
notes(s, "1 min. Walk the figure left to right. Then the three cards: hot, match, miss.")

# 14 Hot loop figure
s = add_slide("Title only", "Hot-loop emission")
if (FIG / "hot_loop.png").exists():
    add_img(s, FIG / "hot_loop.png", Inches(0.3), Inches(2.15), w=Inches(8.15))
rbox(s, Inches(8.55), Inches(2.15), Inches(4.4), Inches(4.7), PALE)
tbox(s, Inches(8.75), Inches(2.3), Inches(4.05), Inches(0.35), "DEFAULT WRITING", 12, True, BLUE)
bullets(s, Inches(8.75), Inches(2.75), Inches(4.05), Inches(3.9), [
    "Blue cells are hot: the routes need them.",
    "Pale cells wait for a miss.",
    "The kernel is stored once and applied at those positions.",
    "Border positions stay unrolled: one body cannot shorten the window.",
    "The unrolled writing of the same routes is measured beside this.",
], 13)
notes(s, "1 min. Sharing buys bytes and delays the earliest return, because every hot neuron is computed before the first test. That trade is the emission slide in the evaluation.")

add_part("03", "Evaluation",
         "Four networks, a matched looped reference. Cycles counted on Spike.")

# 11 Setup
s = add_slide("Title only", "Workloads and reference")
nets = [
    ("Fashion-MNIST", "14×14, 1 ch  ·  T-shirt / Trouser",
     "Conv 3×3, 1 filter  ·  avg-pool 2×2\nLinear 49 → 4 → 2",
     "11 routes  ·  20 hot neurons", "Ref. acc. 97.65%"),
    ("PneumoniaMNIST", "14×14, 1 ch  ·  normal / pneumonia",
     "Same conv and linear stack",
     "12 routes  ·  25 hot neurons", "Ref. acc. 84.74%"),
    ("MNIST even/odd", "28×28, 1 ch  ·  even / odd",
     "Conv 5×5, 1 filter, stride 2  ·  avg-pool 2×2\nLinear 49 → 4 → 2",
     "6 routes  ·  15 hot neurons", "Ref. acc. 91.88%"),
    ("HAR, 3-class", "128×9, 1 ch  ·  static / walk / stairs",
     "Conv 5×3, 1 filter  ·  avg-pool 32×3\nLinear 12 → 8 → 3",
     "1 route  ·  2 hot neurons", "Ref. acc. 83.20%"),
]
for i, (name, inp, layers, routes, acc) in enumerate(nets):
    x = Inches(0.39) + Inches(i * 3.2)
    rbox(s, x, Inches(2.10), Inches(3.05), Inches(3.18), PALE)
    tbox(s, x + Inches(0.12), Inches(2.16), Inches(2.80), Inches(0.36), name, 13, True, BLUE)
    tbox(s, x + Inches(0.12), Inches(2.52), Inches(2.80), Inches(0.38), inp, 11, False, MUTED)
    tbox(s, x + Inches(0.12), Inches(2.90), Inches(2.80), Inches(0.78), layers, 12, False, INK)
    tbox(s, x + Inches(0.12), Inches(3.70), Inches(2.80), Inches(0.70), routes, 12, True, RED)
    tbox(s, x + Inches(0.12), Inches(4.42), Inches(2.80), Inches(0.70), acc, 12, False, INK)
tbox(s, Inches(0.39), Inches(5.32), Inches(12.55), Inches(0.28),
     "Arrows = linear layers. Hot = first-stage neurons the routes read. Ref. acc. = looped integer network on the test set.",
     13, False, MUTED)
rbox(s, Inches(0.39), Inches(5.68), Inches(6.20), Inches(1.22), BLUE)
tbox(s, Inches(0.55), Inches(5.76), Inches(5.90), Inches(0.26), "PLATFORM", 11, True, CYAN)
tbox(s, Inches(0.55), Inches(6.04), Inches(5.90), Inches(0.72),
     "Simulator: Spike. ISA: RV32IM, the integer + multiply set Ibex implements. Metric: architectural mcycle. No cache, no Flash wait states.",
     12, False, WHITE)
rbox(s, Inches(6.74), Inches(5.68), Inches(6.20), Inches(1.22), PALE)
tbox(s, Inches(6.90), Inches(5.76), Inches(5.90), Inches(0.26), "REFERENCE  =  HYBRID FALLBACK", 11, True, BLUE)
tbox(s, Inches(6.90), Inches(6.04), Inches(5.90), Inches(0.72),
     "Same weights, same loop, same 32-bit accumulators. The measured difference is the routes.",
     12, False, INK)
notes(s, "1 min 10. Channels, avg-pool, linear arrows, hot = first-stage neurons a route reads, ref accuracy. Spike is the simulator; Ibex is the ISA.")

# 12 Headline
s = add_slide("Title only", "Cycle results")
kpis = [
    ("Fashion-MNIST", "22.3%", "exit 36.7%", BLUE),
    ("PneumoniaMNIST", "8.4%", "exit 17.0%", CYAN),
    ("MNIST even/odd", "6.0%", "exit 12.2%", BLUE),
    ("HAR, 3-class", "7.9%", "exit 7.7%", RED),
]
for i, (name, val, ex, acc) in enumerate(kpis):
    x = Inches(0.39) + Inches(i * 3.2)
    rbox(s, x, Inches(2.15), Inches(3.05), Inches(1.85), PALE)
    tbox(s, x + Inches(0.12), Inches(2.22), Inches(2.8), Inches(0.28), name, 12, True, MUTED)
    tbox(s, x + Inches(0.12), Inches(2.5), Inches(2.8), Inches(0.7), val, 28, True, acc)
    tbox(s, x + Inches(0.12), Inches(3.25), Inches(2.8), Inches(0.55), "mean cycle reduction\n" + ex, 12, False, INK)
tbox(s, Inches(0.39), Inches(4.02), Inches(12.55), Inches(0.22),
     "Measured on Spike. ISA: RV32IM, the integer + multiply set Ibex implements.",
     12, False, MUTED)
rows = [
    ["Workload", "Ref. mean", "Hyb. mean", "Hyb. min", "Max red.", "dacc"],
    ["Fashion", "5170", "4019", "662", "87.2%", "-0.05 pp"],
    ["Pneumonia", "5278", "4834", "923", "82.5%", "-0.12 pp"],
    ["MNIST", "15274", "14356", "2656", "82.6%", "-0.08 pp"],
    ["HAR", "14384", "13253", "903", "93.7%", "0.00 pp"],
]
table(s, Inches(0.39), Inches(4.26), Inches(12.55), Inches(2.50), rows,
      [Inches(2.15), Inches(2.08), Inches(2.08), Inches(2.08), Inches(2.08), Inches(2.08)])
notes(s, "2 min. Average includes misses. A match returns in 82–94% fewer cycles. Fashion moves the mean because 36.7% match. HAR: few exits but the route skips 93.9% of the MACs.")

# 13 Distributions
s = add_slide("Title only", "Cycle distributions")
if (FIG / "ibex_fashion_cycles.png").exists():
    add_img(s, FIG / "ibex_fashion_cycles.png", Inches(0.35), Inches(2.15), h=Inches(4.7))
rbox(s, Inches(7.55), Inches(2.15), Inches(5.35), Inches(4.7), PALE)
tbox(s, Inches(7.75), Inches(2.3), Inches(5.0), Inches(0.3), "FASHION-MNIST", 12, True, RED)
tbox(s, Inches(7.75), Inches(2.62), Inches(5.0), Inches(0.70),
     "Spike mcycle (RV32IM / Ibex ISA). Low-cycle mode: a match. Mass near the reference: a miss that has already paid the dispatcher.",
     13, False, INK)
nums = [
    ("Fashion-MNIST", "22.3% mean   ·   exit 36.7%"),
    ("PneumoniaMNIST", "8.4% mean   ·   exit 17.0%"),
    ("MNIST even/odd", "6.0% mean   ·   exit 12.2%"),
    ("HAR, 3-class", "7.9% mean   ·   exit 7.7%"),
]
for i, (name, line) in enumerate(nums):
    y = Inches(3.40) + Inches(i * 0.82)
    tbox(s, Inches(7.75), y, Inches(5.0), Inches(0.30), name, 14, True, BLUE if i else RED)
    tbox(s, Inches(7.75), y + Inches(0.30), Inches(5.0), Inches(0.38), line, 13, False, INK if i else INK)
notes(s, "1 min 20. On HAR every sample can be faster than the reference while only 7.7% take the tree route: partial convolution shortens the hot path that every sample starts.")

# 14 Code size
s = add_slide("Title only", "Code size")
rows = [
    ["Workload", "vs. looped reference"],
    ["Fashion-MNIST", "2.16x"],
    ["PneumoniaMNIST", "2.15x"],
    ["MNIST even/odd", "2.11x"],
    ["HAR", "1.85x"],
]
table(s, Inches(0.39), Inches(2.15), Inches(5.5), Inches(2.2), rows, [Inches(3.3), Inches(2.2)])
tbox(s, Inches(0.39), Inches(4.45), Inches(5.5), Inches(0.3), "SAME ROUTES, THREE WRITINGS  ·  MNIST", 11, True, BLUE)
rows_e = [
    ["Hot path", "Red.", "Bytes"],
    ["Unrolled", "7.4%", "12238"],
    ["Loop, two in front", "6.9%", "9792"],
    ["Loop, cut at zero", "6.0%", "6414"],
]
table(s, Inches(0.39), Inches(4.8), Inches(5.5), Inches(1.95), rows_e,
      [Inches(2.6), Inches(1.4), Inches(1.5)])
if (FIG / "flash_hot_macs.png").exists():
    add_img(s, FIG / "flash_hot_macs.png", Inches(6.15), Inches(2.15), w=Inches(6.7))
notes(s, "1 min 30. Object code = .text + .rodata. Sharing the hot path keeps the image models near 2x. Linear in hot first-stage MACs: about 15 bytes per MAC.")

# 15 Ablations
s = add_slide("Title only", "Certificate and original selector")
tbox(s, Inches(0.39), Inches(2.12), Inches(6.15), Inches(0.32), "IRREDUCIBLE CERTIFICATE  (IIS)", 13, True, BLUE)
tbox(s, Inches(0.39), Inches(2.42), Inches(6.15), Inches(0.55),
     "Smallest proving subset. On these CNNs it still names a head neuron, so the convolution runs in full.",
     12, False, MUTED)
table(s, Inches(0.39), Inches(3.05), Inches(6.15), Inches(2.55), [
    ["Workload", "Reduction", "dacc", "Code"],
    ["PneumoniaMNIST", "+0.6%", "0", "2906 / 2854 B"],
    ["Fashion-MNIST", "+0.5%", "0", "2700 / 2652 B"],
    ["MNIST even/odd", "0.0%", "0", "3154 / 3042 B"],
    ["HAR", "+0.7%", "0", "3528 / 3486 B"],
], [Inches(2.15), Inches(1.25), Inches(1.0), Inches(1.75)])
tbox(s, Inches(6.75), Inches(2.12), Inches(6.2), Inches(0.32), "ORIGINAL SELECTOR", 13, True, BLUE)
tbox(s, Inches(6.75), Inches(2.42), Inches(6.2), Inches(0.55),
     "Skipped-multiplication score, no embed file, same emission. Tests stay out of the convolution.",
     12, False, MUTED)
table(s, Inches(6.75), Inches(3.05), Inches(6.2), Inches(2.35), [
    ["Workload", "Original", "This score", "dacc"],
    ["Fashion-MNIST", "+0.5%, 1.03x", "+22.3%, 2.16x", "0"],
    ["PneumoniaMNIST", "-3.4%, 1.04x", "+8.4%, 2.15x", "0"],
    ["MNIST even/odd", "0.0%, 1.04x", "+6.0%, 2.11x", "0"],
    ["HAR, 3-class", "n/a", "+7.9%, 1.85x", "—"],
], [Inches(2.0), Inches(1.65), Inches(1.65), Inches(0.9)])
tbox(s, Inches(6.75), Inches(5.42), Inches(6.2), Inches(0.38),
     "HAR n/a: the original selector is binary; HAR is 3-class.",
     12, False, RED)
rbox(s, Inches(0.39), Inches(5.88), Inches(12.55), Inches(0.95), PALE)
tbox(s, Inches(0.6), Inches(6.02), Inches(12.15), Inches(0.68),
     "Accuracy stays exact. Neither IIS nor the original selector has an early-exit mode. The remaining-work score is the column that moves the mean.",
     14, False, INK)
notes(s, "1 min 20. Two different things. Point at HAR n/a: original selector is binary. Remaining-work score moves the mean. Pneumonia original is slower than the reference.")

add_part("04", "Findings",
         "Results, then the limits of an early exit.")

# 16 Conclusions
s = add_slide("Title only", "Conclusions")
points = [
    ("The hybrid is faster", "Mean Ibex cycles fall by 22.3%, 8.4%, 6.0%, and 7.9% against a matched looped reference."),
    ("The saving is an early exit", "The receptive field of a few first-stage neurons, not a shortened run of the whole network. The mean tracks the exit mass."),
    ("The byte side is about 2x", "Sharing the hot path keeps the image models near twice the looped object code."),
    ("Accuracy stays within a tenth of a point", "The encoding is exact. The gap is the price of trimming tests into the first stage."),
]
for i, (tit, body) in enumerate(points):
    y = Inches(2.15) + Inches(i * 1.15)
    rbox(s, Inches(0.39), y, Inches(12.55), Inches(1.05), PALE)
    box(s, Inches(0.39), y, Inches(0.1), Inches(1.05), RED)
    tbox(s, Inches(0.7), y + Inches(0.1), Inches(12.0), Inches(0.35), tit, 16, True, BLUE)
    tbox(s, Inches(0.7), y + Inches(0.48), Inches(12.0), Inches(0.48), body, 13, False, INK)
notes(s, "1 min. Four results. Limits and what to extend are the next slide.")

# 17 Limitations
s = add_slide("Title only", "Limitations")
lims = [
    ("Unchanged certifier", "Constant leaves come from the published pass. Several networks never yielded a first-stage route."),
    ("No IIS on the routes", "The embedded paths are not an IIS certificate. They can lose a fraction of a point of accuracy against the reference."),
    ("One convolutional channel", "Geometry recovery and the loop are written for a single first-stage channel."),
    ("What to extend", "A multi-channel front end under the same paired comparison. Vetting is not a proof. Cycles are architectural."),
]
for i, (tit, body) in enumerate(lims):
    x = Inches(0.39) + Inches((i % 2) * 6.40)
    y = Inches(2.18) + Inches((i // 2) * 2.40)
    rbox(s, x, y, Inches(6.20), Inches(2.22), PALE)
    box(s, x, y, Inches(6.20), Inches(0.10), RED if i < 3 else CYAN)
    tbox(s, x + Inches(0.28), y + Inches(0.28), Inches(5.65), Inches(0.50), tit, 18, True, BLUE)
    tbox(s, x + Inches(0.28), y + Inches(0.88), Inches(5.65), Inches(1.10), body, 15, False, INK)
notes(s, "45 s. Four cards. Embedded routes have no IIS, so a small accuracy gap vs the reference is possible. No borders.")

# 18 Thank you
s = add_slide("Title Slide", None)
set_ph_text(get_ph(s, 0), "Thank you.")
fill_tf(get_ph(s, 1).text_frame, [
    "Questions",
    "Adrian Cebrian Urios",
    "Conversion of Convolutional Neural Networks into Logic Flows for Efficient Execution on RISC-V",
], 16, False, INK, "Arial", 8, PP_ALIGN.CENTER)
set_run(_run(get_ph(s, 1).text_frame.paragraphs[0]), "Questions", 22, True, RED, "Arial")
set_footer(s)
notes(s, "Stop. Backup slides are ready for IIS numbers, the tree survey, MACs, accuracy, and limits.")

# ========================= BACKUP =========================

s = add_slide("Section", "Backup")
fill_tf(get_ph(s, 1).text_frame, ["Tables and limits for questions"], 18, False, INK, "Arial", 0)
notes(s, "Backup divider.")

s = add_slide("Title only", "Accuracy")
table(s, Inches(1.2), Inches(2.3), Inches(10.9), Inches(2.8), [
    ["Workload", "Reference", "Hybrid", "dacc"],
    ["Fashion-MNIST", "97.65%", "97.60%", "-0.05 pp"],
    ["PneumoniaMNIST", "84.74%", "84.62%", "-0.12 pp"],
    ["MNIST even/odd", "91.88%", "91.80%", "-0.08 pp"],
    ["HAR", "83.20%", "83.20%", "0.00 pp"],
], [Inches(3.3), Inches(2.5), Inches(2.5), Inches(2.6)])
tbox(s, Inches(1.2), Inches(5.3), Inches(10.9), Inches(1.4),
     "The gap is a property of placing the tests in the first stage. The looped reference computes the same function as the unrolled one. QAT error is outside this comparison.",
     14, False, MUTED)

s = add_slide("Title only", "MAC breakdown")
table(s, Inches(0.6), Inches(2.2), Inches(12.1), Inches(2.8), [
    ["Task", "Input", "Total", "First stage", "Head"],
    ["MNIST even/odd", "28x28", "2313", "2116  (91.5%)", "197"],
    ["Fashion binary", "14x14", "879", "676  (76.9%)", "203"],
    ["PneumoniaMNIST", "14x14", "879", "676  (76.9%)", "203"],
    ["HAR, 3-class", "128x9", "1938", "1820  (93.9%)", "118"],
], [Inches(2.8), Inches(1.8), Inches(1.8), Inches(3.1), Inches(2.6)])
tbox(s, Inches(0.6), Inches(5.2), Inches(12.1), Inches(1.4),
     "Counts are the non-zero products after the convolution and the pool are folded into one matrix. Folded windows: 4x4 on Fashion / Pneumonia, 7x7 on MNIST, 36x5 on HAR.",
     14, False, MUTED)

s = add_slide("Title only", "All cycle distributions")
plots = [
    ("Fashion-MNIST  22.3%", "ibex_fashion_cycles.png"),
    ("PneumoniaMNIST  8.4%", "ibex_pneumonia_cycles.png"),
    ("MNIST even/odd  6.0%", "ibex_mnist_cycles.png"),
    ("HAR  7.9%", "ibex_har_cycles.png"),
]
for i, (cap, fn) in enumerate(plots):
    col, row = i % 2, i // 2
    x = Inches(0.4) + Inches(col * 6.45)
    y = Inches(2.15) + Inches(row * 2.4)
    pth = FIG / fn
    if pth.exists():
        add_img(s, pth, x + Inches(1.2), y, h=Inches(1.95))
    tbox(s, x, y + Inches(1.95), Inches(6.2), Inches(0.3), cap, 12, True, BLUE, PP_ALIGN.CENTER)

s = add_slide("Title only", "Full cycle table")
table(s, Inches(0.25), Inches(2.3), Inches(12.8), Inches(3.4), [
    ["Workload", "Ref avg", "Ref min", "Ref max", "Hyb avg", "Hyb min", "Hyb max", "Max%", "Avg%", "Exit"],
    ["Fashion", "5170", "5170", "5172", "4019", "662", "6013", "87.2", "22.3", "36.7%"],
    ["Pneumonia", "5278", "5278", "5290", "4834", "923", "5848", "82.5", "8.4", "17.0%"],
    ["MNIST", "15274", "15274", "15277", "14356", "2656", "15971", "82.6", "6.0", "12.2%"],
    ["HAR", "14384", "14384", "14387", "13253", "903", "14292", "93.7", "7.9", "7.7%"],
])

s = add_slide("Title only", "MNIST emission swap")
table(s, Inches(0.8), Inches(2.2), Inches(11.7), Inches(2.4), [
    ["Hot path", "Cycles", "Reduction", "Bytes", "vs. reference"],
    ["Unrolled", "14148", "7.4%", "12238", "4.02x"],
    ["Shared loop, two routes in front", "14223", "6.9%", "9792", "3.22x"],
    ["Shared loop, cut at zero", "14356", "6.0%", "6414", "2.11x"],
], [Inches(4.3), Inches(1.7), Inches(1.9), Inches(1.8), Inches(2.0)])
tbox(s, Inches(0.8), Inches(4.85), Inches(11.7), Inches(1.8),
     "Reference: 15274 cycles, 3042 B. Accuracy 91.80%, exit 12.2% on all three rows. The loop does not reduce arithmetic; it removes copies.",
     14, False, MUTED)

s = add_slide("Title only", "Leaf survey")
table(s, Inches(1.8), Inches(2.15), Inches(9.7), Inches(4.85), [
    ["Tree", "Leaves", "Constant"],
    ["Fashion binary, fused", "5351", "3846  (71.9%)"],
    ["MNIST even/odd, fused", "44513", "18834  (42.3%)"],
    ["PneumoniaMNIST, fused", "2366", "1336  (56.5%)"],
    ["Fashion binary, unfused", "11144", "1709  (15.3%)"],
    ["Fashion binary, BN unfused", "11379", "737  (6.5%)"],
    ["Fashion-10", "8894", "0  (0.0%)"],
    ["Fashion-3, footwear merge", "13688", "1940  (14.2%)"],
    ["Occupancy FC", "261", "85  (32.6%)"],
    ["Occupancy CNN", "268", "90  (33.6%)"],
    ["HAR, 3-class", "588", "227  (38.6%)"],
], [Inches(4.5), Inches(2.2), Inches(3.0)])

s = add_slide("Title Content", "Limits")
fill_body(s, [
    "Geometry recovery reads one input channel. Shapes are supplied when the layer is not square (HAR).",
    "Border positions of the first stage stay unrolled. A padded copy built before any test costs more cycles than running the network.",
    "A route is accepted from measured agreement on a vetting set, not a proof covering every input.",
    "Constant leaves are a product of the unchanged certifier. The networks were fused and chosen so that it would return more of them.",
    "Every network carried through has a single convolutional channel in the first stage.",
    "Cycle counts are architectural. They do not include Flash wait states or a cache.",
], 15)

s = add_slide("Title Content", "Contributions")
fill_body(s, [
    "CNN-capable pipeline: export, geometry recovery, route selection, hybrid C. Tree construction and certification remain the published passes.",
    "Hybrid strategies for convolution-dominated networks: score a route by the arithmetic that remains when it fires, return from inside the first stage, complete the map only on a miss.",
    "Code-size-aware convolution emission: loops that hold one copy of the kernel, with the unrolled writing measured beside them. Cycles reported together with generated code size.",
], 16)

s = add_slide("Title Content", "Geometry recovery")
body = get_ph(s, 1)
fill_tf(body.text_frame, [
    "Once the convolution is unfolded, the emitter sees a weight matrix. Nothing in that matrix says its rows come from sliding one kernel.",
    "The recovery stage enumerates kernel sizes, strides, and paddings that could produce a matrix of those dimensions, and accepts a candidate only after rebuilding the whole matrix and matching every row.",
    "A layer that no convolutional geometry reproduces is recorded as dense. The result is a satellite file beside the tree.",
    "Recovering the geometry keeps the export path and the tree file untouched. A tree takes hours to build.",
], 16, False, INK, "Arial", 12)

# Master footer default
for sh in prs.slide_master.shapes:
    if sh.has_text_frame and "TITEL" in sh.text_frame.text:
        fill_tf(sh.text_frame, [FOOTER], 10, False, MUTED, "Arial", 0)

for candidate in (
    OUT,
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_TUDa.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_new.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v3.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v4.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v5.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v6.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v7.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v8.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v9.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v10.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v11.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v12.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v13.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v14.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v15.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v16.pptx",
    Path(__file__).resolve().parent / "Defense_Cebrian_Urios_v17.pptx",
):
    try:
        prs.save(candidate)
        OUT = candidate
        break
    except PermissionError:
        continue
else:
    raise PermissionError("Close the open PowerPoint files and rebuild.")
print(f"Wrote {OUT}  ({len(prs.slides)} slides) from {TEMPLATE.name}")
