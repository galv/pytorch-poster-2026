"""Build poster v2 (36 x 24 inch) on the PTC 2026 PowerPoint template.

The slide is generated into a copy of the template (poster.pptx), exported to
poster.pdf with LibreOffice, and checked with poppler-utils. The only Python
dependency is pygments.
"""

import argparse
import ast
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Number, String


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXAMPLES = HERE / "examples.py"
TEMPLATE = ROOT / "PyTorch Conf NA 2026 Poster Template 36x24_horizontal (1).pptx"
if not TEMPLATE.exists():
    TEMPLATE = HERE / "templates" / TEMPLATE.name
PPTX = HERE / "poster.pptx"
PDF = HERE / "poster.pdf"

TITLE = "Shape-Stable Dynamic Control Flow in PyTorch CUDA Graphs"
AUTHORS = (("Daniel Galvez", "NVIDIA"), ("Thomas Ortner", "IBM Research"))

# All layout is in points, measured from the top-left corner of the page.
WIDTH, HEIGHT = 36 * 72, 24 * 72
HEADER = 248  # Bottom of the template's title band.
MARGIN, GAP = 48, 42
COL = (WIDTH - 2 * MARGIN - 2 * GAP) / 3
XS = [MARGIN + i * (COL + GAP) for i in range(3)]
SANS, MONO = "Arial", "Liberation Mono"
MONO_ADVANCE = 0.6  # Liberation Mono glyph width in em; metric-compatible with Courier New.
INK = "222222"
MUTED = "727070"
RULE = "D5D4D5"
TEAL = "1A6E6F"
BLUE = "295897"
ORANGE = "C44918"
HIGHLIGHT = "FEF1C8"
WHITE = "FFFFFF"
GREEN = "2E7D32"
STRIPE = "E8710A"
SHADE = "F4F4F4"
KERNEL_FILL = "F2F2F2"
KERNEL_LINE = "9E9E9E"
HANDLE_FILL = "E3EFEF"
NODE_FILL = "FFF7E8"
HOPS = re.compile(r"torch\.(?:cond|while_loop)\b|\bswitch(?=\()")
PULL = "https://github.com/pytorch/pytorch/pull/"
OPEN_WORK = (
    ("#186056", PULL + "186056", "input mutation in control flow (in-place optimizer step)"),
    ("#189462", PULL + "189462", "a single IF/ELSE conditional node (CUDA 12.8)"),
    ("#189461", PULL + "189461", "torch.switch() to a SWITCH conditional node (CUDA 12.8)"),
    ("#192098", PULL + "192098", "reuse memory blocks across conditional streams"),
    ("#191681", "https://github.com/pytorch/pytorch/issues/191681", "NCCL collectives inside conditional bodies"),
)


SGD = """
def safe_sgd(param, grad, lr):
    finite = torch.isfinite(grad).all()

    def update():
        param.sub_(lr * grad)

    torch.cond(finite, update, lambda: None, ())
"""

CAPTURE = """
step = torch.compile(safe_sgd, backend="cudagraphs")
step(param, grad, lr)  # replays one graph
"""

BAG = """
def embedding_bag(table, ids, n):
    caps = (32, 128, 512, 4096)
    bucket = sum((n > c).to(torch.int32)
                 for c in caps[:-1])

    def reduce(cap):
        pos = torch.arange(cap, device=ids.device)
        rows = table[ids[:cap]]
        return (rows * (pos < n)[:, None]).sum(0)

    branches = [lambda c=c: reduce(c) for c in caps]
    return switch(bucket, branches, ())
"""

SWITCH_IMPORT = "from torch._higher_order_ops import switch"

MOE_ROUNDS = """
total_rounds = sync_all_reduce_max(local_rounds)
def cond(iteration, done, output):
    return iteration < total_rounds
"""

MOE_GEMM = """
counts = (
    (packed_experts[:, None]
     == local_expert_ids[None, :])
    & packed_valid[:, None]
).sum(0)
offsets = torch.cumsum(counts, 0, dtype=torch.int32)
packed_output = torch.nn.functional.grouped_mm(
    packed_tokens,
    local_weights.transpose(-2, -1),
    offs=offsets,
)
"""

MOE_LOOP = """
iteration, _, output = torch.while_loop(
    cond, body, (iteration, done, output))
return output, iteration
"""


def verify_code():
    """Require displayed functions to match examples.py and displayed lines to appear in it."""
    source = EXAMPLES.read_text()
    functions = {
        node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
    }
    for snippet in (SGD, BAG):
        shown = ast.parse(snippet).body[0]
        if ast.dump(shown) != ast.dump(functions[shown.name]):
            raise ValueError(f"Poster code differs from examples.py: {shown.name}")
    lines = {line.strip() for line in source.splitlines()}
    for snippet in (CAPTURE, SWITCH_IMPORT):
        for line in snippet.strip().splitlines():
            if line.split("#")[0].strip() not in lines:
                raise ValueError(f"Poster line is not in examples.py: {line}")
    original = ast.parse((HERE / "ep8_moe_backpressure.py").read_text())
    original_nodes = {ast.dump(node) for node in ast.walk(original)}
    for snippet in (MOE_ROUNDS, MOE_GEMM, MOE_LOOP):
        for node in ast.parse(snippet).body:
            if ast.dump(node) not in original_nodes:
                raise ValueError("Displayed MoE excerpt differs from original source")
    print("Displayed code matches examples.py and original ep8_moe_backpressure.py.")


def emu(points):
    return round(points * 12700)


def xfrm(x, top, width, height, flip=""):
    return (
        f'<a:xfrm{flip}><a:off x="{emu(x)}" y="{emu(top)}"/>'
        f'<a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
    )


def fill_xml(color):
    return "<a:noFill/>" if color is None else f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def line_xml(color, width=1, cap=""):
    if color is None:
        return "<a:ln><a:noFill/></a:ln>"
    return f'<a:ln w="{emu(width)}"{cap}>{fill_xml(color)}<a:round/></a:ln>'


def clip(polygon, a, b, c):
    """Keep the part of a convex polygon where a * u + b * v + c >= 0."""
    kept = []
    for p, q in zip(polygon, polygon[1:] + polygon[:1]):
        fp, fq = a * p[0] + b * p[1] + c, a * q[0] + b * q[1] + c
        if fp >= 0:
            kept.append(p)
        if fp * fq < 0:
            t = fp / (fp - fq)
            kept.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return kept


class Slide:
    """Accumulates DrawingML shapes plus the text extents used for verification."""

    def __init__(self):
        self.shapes = []
        self.links = []
        # (name, x0, top, x1, bottom): every rendered word must lie in one of these.
        self.boxes = []
        # (code line, slot top, slot bottom) for each highlighted control-flow call.
        self.highlights = []

    def next_id(self):
        return len(self.shapes) + 2

    def run(self, value, size, font=SANS, color=INK, bold=False, url=None):
        link = ""
        if url is not None:
            self.links.append(url)
            link = f'<a:hlinkClick r:id="rId{len(self.links) + 2}"/>'
        return (
            f'<a:r><a:rPr lang="en-US" sz="{round(size * 100)}" b="{int(bold)}" dirty="0">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="{font}"/><a:cs typeface="{font}"/>{link}</a:rPr>'
            f"<a:t>{escape(value)}</a:t></a:r>"
        )

    def shape(self, name, x, top, width, height, geometry, fill=None, line=None, line_width=1, text=""):
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{self.next_id()}" name="{escape(name)}"/>'
            f"<p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>{xfrm(x, top, width, height)}"
            f"{geometry}{fill_xml(fill)}{line_xml(line, line_width)}</p:spPr>{text}</p:sp>"
        )

    def rect(self, name, x, top, width, height, color):
        self.shape(name, x, top, width, height, '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>', color)

    def rule(self, x, top, width, color=RULE, thickness=1):
        self.rect("Rule", x, top, width, thickness, color)

    def paragraphs_xml(self, paragraphs, size, leading=None, align="l"):
        spacing = "" if leading is None else f'<a:lnSpc><a:spcPts val="{round(leading * 100)}"/></a:lnSpc>'
        return "".join(
            f'<a:p><a:pPr marL="0" indent="0" algn="{align}">{spacing}<a:buNone/></a:pPr>'
            f'{"".join(runs)}<a:endParaRPr lang="en-US" sz="{round(size * 100)}" dirty="0"/></a:p>'
            for runs in paragraphs
        )

    def textbox(self, name, x, top, width, height, paragraphs, size, leading=None, align="l", anchor="t", wrap="none"):
        """Text box with zero insets and optional word wrapping."""
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{self.next_id()}" name="{escape(name)}"/>'
            f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>{xfrm(x, top, width, height)}'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="{wrap}" lIns="0" tIns="0" rIns="0" bIns="0" anchor="{anchor}">'
            f"<a:noAutofit/></a:bodyPr><a:lstStyle/>{self.paragraphs_xml(paragraphs, size, leading, align)}"
            "</p:txBody></p:sp>"
        )

    def text(self, name, x, top, value, size, width=COL, align="l", **style):
        """One line of text; value is a string or a list of runs from run()."""
        runs = [self.run(value, size, **style)] if isinstance(value, str) else value
        height = 1.25 * size
        self.textbox(name, x, top, width, height, [runs], size, align=align)
        self.boxes.append((name, x, top, x + width, top + height))

    def node(self, name, x, top, width, height, value, size, fill=KERNEL_FILL, line=KERNEL_LINE,
             line_width=1.5, radius=8, font=MONO, color=INK, bold=False):
        """Rounded box with one centered line of text, as used for graph nodes."""
        runs = [self.run(value, size, font, color, bold)] if isinstance(value, str) else value
        adj = round(100000 * radius / min(width, height))
        body = (
            '<p:txBody><a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="ctr">'
            f'<a:noAutofit/></a:bodyPr><a:lstStyle/>{self.paragraphs_xml([runs], size, align="ctr")}</p:txBody>'
        )
        geometry = f'<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val {adj}"/></a:avLst></a:prstGeom>'
        self.shape(name, x, top, width, height, geometry, fill, line, line_width, body if value else "")
        if value:
            self.boxes.append((name, x, top, x + width, top + height))

    def container(self, name, x, top, width, height, title, handle):
        """Conditional-node frame: orange outline, title at the top left, handle at the top right."""
        self.node(name, x, top, width, height, "", 0, NODE_FILL, ORANGE, 3, 14)
        self.text(f"{name} title", x + 18, top + 12, title, 22, width / 2, color=ORANGE, bold=True)
        self.text(f"{name} handle", x + width / 2, top + 14, handle, 19, width / 2 - 18, "r", font=MONO, color=TEAL)

    def arrow(self, x1, y1, x2, y2, color=MUTED, width=2.5, head=True):
        flip = (' flipH="1"' if x2 < x1 else "") + (' flipV="1"' if y2 < y1 else "")
        tail = '<a:tailEnd type="triangle" w="lg" len="lg"/>' if head else ""
        self.shapes.append(
            f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{self.next_id()}" name="Arrow"/><p:cNvCxnSpPr/>'
            f"<p:nvPr/></p:nvCxnSpPr><p:spPr>{xfrm(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1), flip)}"
            '<a:prstGeom prst="straightConnector1"><a:avLst/></a:prstGeom>'
            f'<a:ln w="{emu(width)}">{fill_xml(color)}<a:round/>{tail}</a:ln></p:spPr></p:cxnSp>'
        )

    def path(self, name, x, top, width, height, polygons, fill=None, line=None, line_width=1, closed=True):
        """Custom geometry; polygons are point lists relative to (x, top)."""
        paths = ""
        for points in polygons:
            (u0, v0), *rest = points
            commands = f'<a:moveTo><a:pt x="{emu(u0)}" y="{emu(v0)}"/></a:moveTo>' + "".join(
                f'<a:lnTo><a:pt x="{emu(u)}" y="{emu(v)}"/></a:lnTo>' for u, v in rest
            )
            mode = "" if fill else ' fill="none"'
            paths += f'<a:path w="{emu(width)}" h="{emu(height)}"{mode}>{commands}{"<a:close/>" if closed else ""}</a:path>'
        geometry = (
            '<a:custGeom><a:avLst/><a:gdLst/><a:ahLst/><a:cxnLst/><a:rect l="0" t="0" r="r" b="b"/>'
            f"<a:pathLst>{paths}</a:pathLst></a:custGeom>"
        )
        cap = ' cap="rnd"' if line else ""
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{self.next_id()}" name="{name}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f"<p:spPr>{xfrm(x, top, width, height)}{geometry}{fill_xml(fill)}{line_xml(line, line_width, cap)}"
            "</p:spPr></p:sp>"
        )

    def tick(self, cx, cy, size=30):
        x, top = cx - size / 2, cy - size / 2
        points = [(0.1 * size, 0.55 * size), (0.4 * size, 0.85 * size), (0.92 * size, 0.18 * size)]
        self.path("Tick", x, top, size, size, [points], line=GREEN, line_width=0.15 * size, closed=False)

    def cross(self, cx, cy, size=30):
        x, top = cx - size / 2, cy - size / 2
        strokes = [
            [(0.2 * size, 0.2 * size), (0.8 * size, 0.8 * size)],
            [(0.2 * size, 0.8 * size), (0.8 * size, 0.2 * size)],
        ]
        self.path("Unsupported X", x, top, size, size, strokes,
                  line="C62828", line_width=0.15 * size, closed=False)

    def barrier(self, cx, cy, size=30):
        """Construction barrier: striped board on two legs."""
        width, board, legs = 1.4 * size, 0.5 * size, 0.32 * size
        x, top = cx - width / 2, cy - (board + legs) / 2
        for u in (0.18 * width, 0.82 * width - 0.1 * size):
            self.rect("Barrier leg", x + u, top + board, 0.1 * size, legs, "3A3A3A")
        self.rect("Barrier board", x, top, width, board, WHITE)
        stripe = width / 6
        panel = [(0, 0), (width, 0), (width, board), (0, board)]
        stripes = []
        for k in range(-1, 6):
            start = 2 * k * stripe
            band = clip(clip(panel, 1, 1, -start), -1, -1, start + stripe)
            if len(band) >= 3:
                stripes.append(band)
        self.path("Barrier stripes", x, top, width, board, stripes, fill=STRIPE)
        self.path("Barrier outline", x, top, width, board, [panel], line="3A3A3A", line_width=1.5)

    def code(self, name, source, x, top, size, leading, width=COL):
        lines = source.strip("\n").splitlines()
        paragraphs = []
        for index, line in enumerate(lines):
            if len(line) * MONO_ADVANCE * size > width:
                raise ValueError(f"Code exceeds column: {line}")
            spans = [match.span() for match in HOPS.finditer(line)]
            if spans:
                slot = top + index * leading
                self.rect(f"{name} highlight", x - 8, slot - 1, width + 16, leading + 1, HIGHLIGHT)
                self.rect(f"{name} marker", x - 8, slot - 1, 3, leading + 1, ORANGE)
                self.highlights.append((line, slot, slot + leading))
            styled = []
            offset = 0
            for token, value in lex(line, PythonLexer(stripnl=False, ensurenl=False)):
                value = value.rstrip("\n")
                color, bold = INK, False
                if token in Comment:
                    color = MUTED
                elif token in Keyword:
                    color, bold = BLUE, True
                elif token in Number or token in String:
                    color = TEAL
                if any(start <= offset < end for start, end in spans):
                    color, bold = ORANGE, True
                if styled and styled[-1][1:] == (color, bold):
                    styled[-1][0] += value
                else:
                    styled.append([value, color, bold])
                offset += len(value)
            paragraphs.append(
                [self.run(value, size, MONO, color, bold) for value, color, bold in styled if value]
            )
        height = len(lines) * leading
        self.textbox(name, x, top, width, height, paragraphs, size, leading)
        self.boxes.append((name, x, top, x + width, top + height + 0.4 * size))
        return top + height

    def heading(self, x, top, number, title, subline):
        runs = [self.run(title, 30, bold=True)]
        if number:
            runs.insert(0, self.run(number + "  ", 30, color=TEAL, bold=True))
        self.text(f"Heading {title}", x, top, runs, 30)
        self.text(f"Subline {title}", x, top + 42, subline, 21, color=MUTED)

    def bullets(self, x, top, items, size=21, leading=27, gap=14, width=COL):
        """Items are (bold lead, first line, *continuation lines); returns the bottom."""
        for index, (lead, first, *rest) in enumerate(items):
            self.rect("Bullet", x, top + 0.42 * size, 9, 9, TEAL)
            runs = [self.run(lead + " ", size, bold=True), self.run(first, size)]
            self.text(f"Bullet {index + 1}", x + 24, top, runs, size, width - 24)
            for line in rest:
                top += leading
                self.text(f"Bullet {index + 1} cont.", x + 24, top, line, size, width - 24)
            top += leading + gap
        return top

    def title(self):
        """Fill the template's title placeholder with the title and author line."""
        authors = []
        for index, (name, affiliation) in enumerate(AUTHORS):
            separator = "        " if index else ""
            authors.append(self.run(f"{separator}{name}", 32, color=WHITE, bold=True))
            authors.append(self.run(f", {affiliation}", 32, color=WHITE))
        title = self.run(TITLE, 52, color=WHITE, bold=True)
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{self.next_id()}" name="Title"/>'
            '<p:cNvSpPr txBox="1"/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>'
            '<p:spPr><a:xfrm><a:off x="6485809" y="0"/><a:ext cx="20855700" cy="3149700"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
            '<p:txBody><a:bodyPr wrap="none" lIns="102450" tIns="53275" rIns="102450" bIns="53275" anchor="ctr">'
            "<a:noAutofit/></a:bodyPr><a:lstStyle/>"
            f'<a:p><a:pPr algn="l"><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr>{title}</a:p>'
            '<a:p><a:pPr algn="l"><a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
            f'<a:spcBef><a:spcPts val="1600"/></a:spcBef></a:pPr>{"".join(authors)}</a:p>'
            "</p:txBody></p:sp>"
        )
        left = 6485809 / 12700 + 102450 / 12700
        self.boxes.append(("Title", left, 0, (6485809 + 20855700 - 102450) / 12700, HEADER))

    def slide_xml(self):
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
            ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
            ' xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/>'
            '</p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            f'{"".join(self.shapes)}</p:spTree></p:cSld>'
            "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
        )

    def rels_xml(self):
        base = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        links = "".join(
            f'<Relationship Id="rId{index + 3}" Type="{base}/hyperlink" Target="{escape(url)}" TargetMode="External"/>'
            for index, url in enumerate(self.links)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{base}/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
            f'<Relationship Id="rId2" Type="{base}/notesSlide" Target="../notesSlides/notesSlide1.xml"/>'
            f"{links}</Relationships>"
        )


def support_matrix(s, x, top):
    label_width = 290
    cell = (COL - label_width) / 3
    header, row = 76, 58
    ops = (("cond", "IF / ELSE"), ("while_loop", "WHILE"), ("switch", "SWITCH"))
    rows = (
        ("PyTorch 2.14", "yes", "yes", "yes"),
        ("CUDA graph support", "yes", "yes", "wip"),
        ("In-place input mutation*", "yes", "yes", "yes"),
        ("Minimum CUDA", "12.4", "12.4", "12.8"),
        ("Inductor backend", "no", "no", "no"),
    )
    for j, (op, node) in enumerate(ops):
        left = x + label_width + j * cell
        s.text(f"Matrix op {op}", left, top + 6, op, 23, cell, "ctr", font=MONO, color=ORANGE, bold=True)
        s.text(f"Matrix node {op}", left, top + 40, node, 18, cell, "ctr", color=TEAL, bold=True)
    s.rule(x, top + header, COL, INK, 2)
    for i, (label, *cells) in enumerate(rows):
        row_top = top + header + 2 + i * row
        if i % 2:
            s.rect("Matrix shade", x, row_top, COL, row, SHADE)
        s.text(f"Matrix row {label}", x + 10, row_top + (row - 28) / 2, label, 23, label_width - 10)
        for j, value in enumerate(cells):
            cx, cy = x + label_width + (j + 0.5) * cell, row_top + row / 2
            if value == "yes":
                s.tick(cx, cy, 36)
            elif value == "no":
                s.cross(cx, cy, 36)
            elif value == "wip":
                s.barrier(cx, cy, 34)
            else:
                s.text(f"Matrix cell {label} {j}", cx - cell / 2, cy - 14, value, 23, cell, "ctr", bold=True)
    bottom = top + header + 2 + len(rows) * row
    s.rule(x, bottom, COL, INK, 1.5)

    legend = bottom + 22
    used = {value for _, *cells in rows for value in cells}
    left = x
    if "yes" in used:
        s.tick(left + 18, legend + 12, 26)
        s.text("Legend done", left + 42, legend, "supported in PyTorch 2.14", 20, 260)
        left += 320
    if "wip" in used:
        s.barrier(left + 18, legend + 12, 26)
        s.text("Legend open", left + 48, legend, "PR open, not yet merged", 20, x + COL - left - 48)
    notes = (
        "* Inputs mutated in place under torch.no_grad() or torch.inference_mode().",
        "switch is torch._higher_order_ops.switch; CUDA graph capture needs #189461.",
        "Inductor runs the control flow on the host; it emits no conditional nodes yet.",
        "Checked on PyTorch main (91a0eda); every feature above is in 2.14.",
    )
    for index, note in enumerate(notes):
        s.text(f"Matrix note {index + 1}", x, legend + 38 + index * 25, note, 18, color=MUTED)


def cond_graph(s, x, top):
    center = x + COL / 2
    node_width = 470
    s.node("Kernel: isfinite", center - node_width / 2, top, node_width, 54, "isfinite(grad).all()", 21)
    s.text("Note: kernels", center + node_width / 2 + 14, top + 15, "kernel nodes", 18, x + COL - center - node_width / 2 - 14, color=MUTED)
    s.arrow(center, top + 54, center, top + 84)
    s.node("Kernel: set handle", center - node_width / 2, top + 84, node_width, 54, "set_conditional(handle, finite)", 19, HANDLE_FILL, TEAL)
    s.text("Note: handle", center + node_width / 2 + 14, top + 99, "1-thread kernel", 18, x + COL - center - node_width / 2 - 14, color=MUTED)
    s.arrow(center, top + 138, center, top + 168)

    frame_top, frame_height = top + 168, 252
    s.container("IF/ELSE node", x + 8, frame_top, COL - 16, frame_height, "IF / ELSE conditional node", "handle = finite")
    body_width = (COL - 16 - 3 * 18) / 2
    for k, (label, kernel) in enumerate((("child graph 0: runs if finite", "param.sub_(lr * grad)"), ("child graph 1: runs otherwise", "no-op"))):
        left = x + 8 + 18 + k * (body_width + 18)
        s.node(f"Body {k}", left, frame_top + 58, body_width, 172, "", 0, WHITE, KERNEL_LINE, 1.25, 6)
        s.text(f"Body {k} label", left + 16, frame_top + 72, label, 19, body_width - 32, bold=True)
        s.node(f"Body {k} kernel", left + 16, frame_top + 120, body_width - 32, 56, kernel, 19)
        s.text(f"Body {k} note", left + 16, frame_top + 190, "update param in place" if k == 0 else "leave param unchanged", 17, body_width - 32, color=MUTED)
    s.arrow(center, frame_top + frame_height, center, frame_top + frame_height + 30)
    s.node("Parameter buffer", center - 300, frame_top + frame_height + 30, 600, 54, "param: same buffer  →  next kernels", 20, font=SANS)


def bucket_graph(s, x, top):
    s.container("Embedding SWITCH", x + 8, top, COL - 16, 126,
                "SWITCH: token-count bucket", "n = 80")
    gap = 12
    cell = (COL - 16 - 36 - 3 * gap) / 4
    for i, cap in enumerate((32, 128, 512, 4096)):
        left = x + 26 + i * (cell + gap)
        selected = cap == 128
        s.node(f"Embedding bucket {cap}", left, top + 51, cell, 44,
               f"{cap} tokens", 20, ORANGE if selected else WHITE,
               ORANGE if selected else KERNEL_LINE, font=SANS,
               color=WHITE if selected else MUTED, bold=selected)
    s.text("Embedding bucket result", x + 26, top + 101,
           "Every branch returns [D]; 80 tokens use the 128-token branch.",
           18, COL - 52, "ctr", color=MUTED)


def switch_graph(s, x, top):
    s.node("Kernel: router", x + 8, top, 330, 52, "argmax(x @ router)", 19)
    s.arrow(x + 338, top + 26, x + 372, top + 26)
    s.node("Kernel: set index", x + 372, top, COL - 380, 52, "set_conditional(handle, expert)", 18, HANDLE_FILL, TEAL)
    s.arrow(x + (372 + COL - 8) / 2, top + 52, x + (372 + COL - 8) / 2, top + 82)
    frame_top = top + 82
    s.container("SWITCH node", x + 8, frame_top, COL - 16, 176, "SWITCH conditional node", "handle = expert")
    count, chosen, spacing = 8, 5, 12
    width = (COL - 16 - 36 - (count - 1) * spacing) / count
    for e in range(count):
        left = x + 26 + e * (width + spacing)
        if e == chosen:
            s.node(f"Expert {e}", left, frame_top + 58, width, 70, f"e{e}", 22, ORANGE, ORANGE, font=SANS, color=WHITE, bold=True)
        else:
            s.node(f"Expert {e}", left, frame_top + 58, width, 70, f"e{e}", 22, WHITE, KERNEL_LINE, font=SANS, color=MUTED)
    s.text("SWITCH label", x + 26, frame_top + 140, "one body per expert; only body[expert] runs at replay", 17, COL - 52, "ctr", color=MUTED)


def build_slide():
    s = Slide()
    s.title()
    width = WIDTH - 2 * MARGIN
    lead = [
        s.run("Write control flow with torch.cond(), torch.while_loop(), or torch.switch(); PyTorch captures it as CUDA graph conditional nodes automatically.", 25),
    ]
    s.text("Lead", MARGIN, 266, lead, 25, width)

    # Row 1: past/present context, torch.cond, and its captured graph.
    row1, content1 = 318, 404
    past = (
        "CUDA graphs could not capture any data-dependent control flow in PyTorch "
        "because that involved synchronizing with the host. Though CUDA graph "
        "conditional nodes can run data-dependent control flow fully on the GPU, "
        "they were not yet expressible via PyTorch. Users had to split their "
        "workload into multiple CUDA graphs, which is intrusive and hard to maintain."
    )
    present = (
        "Control flow expressed with torch.cond(), torch.while_loop(), or "
        "torch.switch() can now be lowered to a CUDA graph. Outputs must keep "
        "a fixed shape across branches and loop iterations. Within that constraint, "
        "data-dependent control flow is useful for real workloads, as the examples "
        "here demonstrate."
    )
    for name, top, height, value, color in (
        ("Past", row1, 310, past, MUTED),
        ("Present", row1 + 336, 278, present, TEAL),
    ):
        paragraphs = [
            [s.run(name, 30, bold=True, color=color)],
            [s.run(value, 25)],
        ]
        s.textbox(name, XS[0], top, COL, height, paragraphs, 25,
                  leading=34, wrap="square")
        s.boxes.append((name, XS[0], top, XS[0] + COL, top + height))

    s.heading(XS[1], row1, "01", "Skip optimizer if gradient is non-finite", "torch.cond()  →  IF/ELSE conditional node")
    bottom = s.code("Code: safe_sgd", SGD, XS[1], content1, 23, 30)
    s.text("Capture label", XS[1], bottom + 26, "Capture once, then replay:", 21, bold=True)
    bottom = s.code("Code: capture", CAPTURE, XS[1], bottom + 62, 21, 27)
    s.text("Note: cond 1", XS[1], bottom + 28, "The predicate never leaves the GPU: no .item(), no CPU sync.", 21)
    s.text("Note: cond 2", XS[1], bottom + 60, "The same graph updates param or leaves it unchanged on each replay.", 21)
    s.text("Note: cond 3", XS[1], bottom + 92, "Before: a Python if on this flag forces a CPU sync and breaks capture.", 21, color=MUTED)

    s.heading(XS[2], row1, "", "The captured CUDA graph", "What safe_sgd becomes: one graph, GPU-side branch")
    cond_graph(s, XS[2], content1)

    for x in XS[1:]:
        s.rect("Divider", x - GAP / 2, row1, 1, 932 - row1, RULE)
    s.rule(MARGIN, 952, width, TEAL, 2)

    # Row 2: support matrix, embedding bags, and lossless distributed MoE.
    row2, content2 = 972, 1058
    s.heading(XS[0], row2, "", "Support matrix", "Control-flow ops and the CUDA graph node each lowers to")
    support_matrix(s, XS[0], content2)

    s.heading(XS[1], row2, "02", "Embedding bags without maximum padding",
              "torch.switch()  →  SWITCH conditional node")
    x = XS[1]
    motivation = (
        "Recommender system requests vary widely in token count, but each embedding "
        "bag returns a fixed-width vector. Select a token-count bucket on the GPU "
        "and pad only to that bucket's capacity."
    )
    s.textbox("Embedding motivation", x, content2, COL, 100,
              [[s.run(motivation, 23)]], 23, leading=29, wrap="square")
    s.boxes.append(("Embedding motivation", x, content2, x + COL, content2 + 100))
    s.text("Import: embedding switch", x, content2 + 110, SWITCH_IMPORT, 18, font=MONO, color=MUTED)
    bottom = s.code("Code: embedding_bag", BAG, x, content2 + 146, 20, 24)
    bucket_graph(s, x, bottom + 20)
    s.text("Embedding input contract", x, bottom + 158,
           "ids: fixed 4,096-slot buffer; 0 ≤ n ≤ 4,096; output: [D].", 20, color=MUTED)

    s.heading(XS[2], row2, "03", "Lossless MoE routing with software backpressure",
              "ep8_moe: torch.while_loop() + grouped_mm()")
    x = XS[2]
    s.text("MoE capacity", x, content2, "EP=8; 4,096 tokens/rank; 512 slots/source; zero tokens dropped.", 20)
    s.text("MoE rounds heading", x, content2 + 36, "1. All ranks agree on the number of rounds", 21, bold=True)
    bottom = s.code("Code: MoE rounds", MOE_ROUNDS, x, content2 + 68, 19, 23)
    s.text("MoE GEMM heading", x, bottom + 20, "2. Group packed tokens by expert; run grouped GEMM", 21, bold=True)
    bottom = s.code("Code: MoE grouped GEMM", MOE_GEMM, x, bottom + 54, 19, 23)
    s.text("MoE loop heading", x, bottom + 20, "3. Repeat bounded all-to-all rounds on the GPU", 21, bold=True)
    bottom = s.code("Code: MoE while loop", MOE_LOOP, x, bottom + 54, 19, 23)
    s.text("MoE excerpt note", x, bottom + 20, "Excerpts: routing, packing, returns, and capture helpers omitted.", 18, color=MUTED)

    for x in XS[1:]:
        s.rect("Divider", x - GAP / 2, row2, 1, 1700 - row2, RULE)
    return s


def write_pptx(slide):
    creator = "; ".join(name for name, _ in AUTHORS)
    with zipfile.ZipFile(TEMPLATE) as src, zipfile.ZipFile(PPTX, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "ppt/slides/slide1.xml":
                data = slide.slide_xml().encode()
            elif item.filename == "ppt/slides/_rels/slide1.xml.rels":
                data = slide.rels_xml().encode()
            elif item.filename == "ppt/theme/theme2.xml":
                # LibreOffice draws hyperlinks in the theme color, ignoring the run color.
                data = data.replace(b'<a:hlink><a:srgbClr val="CCCCFF"/>', f'<a:hlink><a:srgbClr val="{BLUE}"/>'.encode())
            elif item.filename == "docProps/core.xml":
                data = re.sub(rb"<dc:title>.*?</dc:title>", b"", data)
                data = re.sub(
                    rb"<dc:creator(?:\s*/>|>.*?</dc:creator>)",
                    f"<dc:title>{escape(TITLE)}</dc:title><dc:creator>{escape(creator)}</dc:creator>".encode(),
                    data,
                )
            dst.writestr(item, data)
    print(f"PPTX: {PPTX}")


def export_pdf():
    with tempfile.TemporaryDirectory() as tmp:
        # A private profile avoids clashing with a running LibreOffice instance.
        subprocess.run(
            [
                "soffice",
                f"-env:UserInstallation=file://{tmp}/profile",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                tmp,
                str(PPTX),
            ],
            check=True,
            capture_output=True,
        )
        shutil.move(Path(tmp) / f"{PPTX.stem}.pdf", PDF)


def poppler(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def verify_pdf(slide):
    info = poppler("pdfinfo", str(PDF))
    if not re.search(r"^Pages:\s+1$", info, re.M) or f"{WIDTH} x {HEIGHT} pts" not in info:
        raise ValueError(f"The poster must be one 36 x 24 inch page:\n{info}")
    fonts = poppler("pdffonts", str(PDF)).splitlines()[2:]
    if not fonts or any(line.split()[-5] != "yes" for line in fonts):
        raise ValueError("Font not embedded:\n" + "\n".join(fonts))

    words = [
        (float(x0), float(y0), float(x1), float(y1), value)
        for x0, y0, x1, y1, value in re.findall(
            r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>',
            poppler("pdftotext", "-bbox", str(PDF), "-"),
        )
    ]
    for x0, y0, x1, y1, value in words:
        candidates = [b for b in slide.boxes if b[1] - 2 <= x0 <= b[3] and b[2] - 8 <= y0 and y1 <= b[4] + 8]
        owner = max(candidates, key=lambda b: b[1], default=None)
        if owner is None:
            raise ValueError(f"Text outside every layout box: {value!r} at {x0:.0f}, {y0:.0f}")
        if x1 > owner[3] + 1:
            raise ValueError(f"Text overflows {owner[0]}: {value!r} ends at {x1:.0f} > {owner[3]:.0f}")
        if y1 > HEIGHT - 24:
            raise ValueError(f"Text too close to the bottom edge: {value!r}")

    for line, slot_top, slot_bottom in slide.highlights:
        hop = HOPS.search(line).group(0)
        centers = [(y0 + y1) / 2 for x0, y0, x1, y1, value in words if hop in value]
        if not any(slot_top <= center <= slot_bottom for center in centers):
            raise ValueError(f"Highlight is not behind its code line: {line.strip()}")

    text = poppler("pdftotext", str(PDF), "-")
    for required in ("torch.cond", "torch.while_loop", "grouped_mm", "ep8_moe", TITLE, *(a for author in AUTHORS for a in author)):
        if required not in text:
            raise ValueError(f"Missing poster content: {required}")
    links = set(re.findall(rb"/URI\s*\(([^)]*)\)", PDF.read_bytes()))
    for url in slide.links:
        if url.encode() not in links:
            raise ValueError(f"Missing PDF link: {url}")

    poppler("pdftoppm", "-r", "150", "-png", "-singlefile", str(PDF), str(PDF.with_name(PDF.stem + "_150dpi")))
    poppler("pdftoppm", "-r", "54", "-png", "-singlefile", str(PDF), str(PDF.with_name(PDF.stem + "_preview")))
    print(f"Verified: 1 page, 36 x 24 inches, embedded fonts, {len(words)} words inside their boxes, links.")
    print(f"PDF: {PDF}")


def main():
    global PPTX, PDF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", default="poster", help="Output basename, without extension")
    args = parser.parse_args()
    if Path(args.output_stem).name != args.output_stem or args.output_stem in ("", ".", ".."):
        parser.error("--output-stem must be a basename without directories")
    PPTX = HERE / (args.output_stem + ".pptx")
    PDF = HERE / (args.output_stem + ".pdf")
    verify_code()
    slide = build_slide()
    write_pptx(slide)
    export_pdf()
    verify_pdf(slide)


if __name__ == "__main__":
    main()
