# -*- coding: utf-8 -*-
"""Mishmish the Curious Cat — 3D animated story with Arabic narration."""
import math, os, subprocess, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import uharfbuzz as hb
import freetype

ASSETS = "/workspace/story_assets"
W, H = 1280, 720
FPS = 30
XFADE = 1.0
FONT_TITLE = os.path.join(ASSETS, "BalooBhaijaan2.ttf")
FONT_CAP = os.path.join(ASSETS, "Tajawal-Bold.ttf")

FF = "/home/appuser/.local/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v9.0.2"

# ================= text shaping (HarfBuzz + FreeType) =================

_hb_cache = {}
def _get_fonts(path):
    if path not in _hb_cache:
        blob = hb.Blob.from_file_path(path)
        face = hb.Face(blob)
        hbf = hb.Font(face)
        ftf = freetype.Face(path)
        _hb_cache[path] = (face, hbf, ftf)
    return _hb_cache[path]

def shape_text(text, font_path):
    face, hbf, ftf = _get_fonts(font_path)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hbf, buf, {"kern": True, "liga": True})
    return buf.direction, list(buf.glyph_infos), list(buf.glyph_positions)

def draw_arabic(img, text, size, cx, cy, fill, font_path=FONT_CAP, anchor="mm", stroke=None):
    direction, infos, poss = shape_text(text, font_path)
    face, hbf, ftf = _get_fonts(font_path)
    scale_x, scale_y = hbf.scale
    factor = size / scale_x
    ftf.set_pixel_sizes(0, max(8, int(size * 1.15)))
    total = sum(p.x_advance for p in poss) * factor
    pen = total if direction == "rtl" else 0.0
    asc = size * 0.85
    x0 = cx - total / 2.0
    y0 = cy + asc / 2.0 if anchor == "mm" else cy
    canvas = np.zeros((int(size * 2.4), int(total + size * 2)), dtype=np.float32)
    ox = size
    oy = int(size * 1.2)
    for info, p in zip(infos, poss):
        if direction == "rtl":
            pen -= p.x_advance * factor
        gx = ox + pen + p.x_offset * factor
        gy = oy - p.y_offset * factor
        ftf.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING)
        g = ftf.glyph
        bmp = g.bitmap
        if bmp.width > 0 and bmp.rows > 0:
            arr = np.array(bmp.buffer, dtype=np.float32).reshape(bmp.rows, bmp.width) / 255.0
            bx = int(round(gx + g.bitmap_left))
            by = int(round(gy - g.bitmap_top))
            h_, w_ = arr.shape
            if by < 0:
                arr = arr[-by:]; by = 0; h_ = arr.shape[0]
            if bx < 0:
                arr = arr[:, -bx:]; bx = 0; w_ = arr.shape[1]
            eh, ew = canvas.shape
            y2, x2 = min(by + h_, eh), min(bx + w_, ew)
            if y2 > by and x2 > bx:
                canvas[by:y2, bx:x2] = np.maximum(canvas[by:y2, bx:x2], arr[:y2 - by, :x2 - bx])
        if direction != "rtl":
            pen += p.x_advance * factor
    ys, xs = np.nonzero(canvas > 0.02)
    if len(xs) == 0:
        return
    ty0, ty1 = ys.min(), ys.max() + 1
    tx0, tx1 = xs.min(), xs.max() + 1
    alpha = canvas[ty0:ty1, tx0:tx1]
    px = int(round(x0 + tx0 - ox))
    py = int(round(y0 + ty0 - oy))
    ah, aw = alpha.shape
    px = max(0, min(px, img.width - aw))
    py = max(0, min(py, img.height - ah))
    region = np.asarray(img.crop((px, py, px + aw, py + ah)), dtype=np.float32)
    col = np.array(fill[:3], dtype=np.float32)
    if stroke is not None:
        scol = np.array(stroke[:3], dtype=np.float32)
        pad = 2
        big = np.zeros((ah + pad * 2, aw + pad * 2), dtype=np.float32)
        a2 = np.zeros_like(big)
        a2[pad:pad + ah, pad:pad + aw] = alpha
        for dy in (-pad, 0, pad):
            for dx in (-pad, 0, pad):
                big = np.maximum(big, np.roll(np.roll(a2, dy, 0), dx, 1))
        ring = np.clip(big[pad:pad + ah, pad:pad + aw] - alpha, 0, 1)
        region[...] = region * (1 - ring[..., None]) + scol * ring[..., None]
    a = alpha[..., None]
    region[...] = region * (1 - a) + col * a
    img.paste(Image.fromarray(region.astype(np.uint8), "RGB"), (px, py))

# ================= tiny 3D engine =================

def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def rot_xyz(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

class Obj:
    def __init__(self, verts, quads, color, emissive=False, ground=False):
        self.verts = np.asarray(verts, dtype=float)
        self.quads = np.asarray(quads, dtype=int)
        self.color = color
        self.emissive = emissive
        self.ground = ground

def place_parts(parts, pos=(0, 0, 0), rot_y=0.0, scale=1.0):
    R = rot_xyz(0, rot_y, 0)
    out = []
    for p in parts:
        v = p.verts * scale
        v = v @ R.T + np.asarray(pos, dtype=float)
        out.append(Obj(v, p.quads, p.color, p.emissive))
    return out

def place(obj, pos=(0, 0, 0), scale=(1, 1, 1), rot=(0, 0, 0)):
    R = rot_xyz(*rot)
    v = obj.verts * np.asarray(scale)
    v = v @ R.T + np.asarray(pos)
    return Obj(v, obj.quads, obj.color, obj.emissive)

def uv_sphere(r=1.0, nu=10, nv=7):
    verts = []
    for j in range(nv + 1):
        phi = math.pi * j / nv
        for i in range(nu):
            th = 2 * math.pi * i / nu
            verts.append((math.sin(phi) * math.cos(th), math.cos(phi), math.sin(phi) * math.sin(th)))
    verts = np.array(verts) * r
    quads = []
    for j in range(nv):
        for i in range(nu):
            a = j * nu + i
            b = j * nu + (i + 1) % nu
            c = (j + 1) * nu + (i + 1) % nu
            d = (j + 1) * nu + i
            quads.append((a, b, c, d))
    return Obj(verts, quads, (255, 255, 255))

def box(w, h, d):
    x, y, z = w / 2, h / 2, d / 2
    verts = [(-x, 0, -z), (x, 0, -z), (x, 0, z), (-x, 0, z),
             (-x, h, -z), (x, h, -z), (x, h, z), (-x, h, z)]
    quads = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7), (3, 2, 1, 0)]
    return Obj(np.array(verts, dtype=float), quads, (255, 255, 255))

def pyramid(r, h, seg=4):
    verts = [(0, 0, 0)]
    for i in range(seg):
        a = 2 * math.pi * i / seg + math.pi / seg
        verts.append((r * math.cos(a), 0, r * math.sin(a)))
    verts.append((0, h, 0))
    quads = []
    for i in range(seg):
        quads.append((0, 1 + i, 1 + (i + 1) % seg, seg + 1))
    return Obj(np.array(verts, dtype=float), quads, (255, 255, 255))

def cone(r, h, seg=8):
    verts = [(0, 0, 0)]
    for i in range(seg):
        a = 2 * math.pi * i / seg
        verts.append((r * math.cos(a), 0, r * math.sin(a)))
    verts.append((0, h, 0))
    quads = []
    for i in range(seg):
        quads.append((0, 1 + i, 1 + (i + 1) % seg, seg + 1))
    return Obj(np.array(verts, dtype=float), quads, (255, 255, 255))

def flat_quad(w, d, color=(255, 255, 255), emissive=False):
    verts = [(-w / 2, 0, -d / 2), (w / 2, 0, -d / 2), (w / 2, 0, d / 2), (-w / 2, 0, d / 2)]
    return Obj(np.array(verts, dtype=float), [(0, 1, 2, 3)], color, emissive)

class Camera:
    def __init__(self, pos, target, fov=50):
        self.pos = np.asarray(pos, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.fov = fov

def render3d(objects, cam, sky_img, sun_dir, ambient, fog_color, fog_start, fog_end,
             shadows=None, sun_pos=None, sun_r=40, glows=None):
    f = H / (2 * math.tan(math.radians(cam.fov) / 2))
    fwd = norm(cam.target - cam.pos)
    right = norm(np.cross(fwd, [0, 1, 0]))
    up = np.cross(right, fwd)

    frame = sky_img.copy()
    d = ImageDraw.Draw(frame, "RGBA")

    if sun_pos is not None:
        rel = np.asarray(sun_pos) - cam.pos
        z = rel @ fwd
        if z > 1:
            sx = W / 2 + (rel @ right) / z * f
            sy = H / 2 - (rel @ up) / z * f
            r = max(8, sun_r / z * f)
            for gr, ga in ((r * 5, 60), (r * 2.6, 90), (r * 1.4, 130)):
                d.ellipse([sx - gr, sy - gr, sx + gr, sy + gr], fill=(255, 240, 190, ga))
            d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(255, 246, 210, 255))

    for ob in objects:
        rel = ob.verts - cam.pos
        x = rel @ right
        y = rel @ up
        z = rel @ fwd
        ok = z > 0.06
        if not ok.any():
            continue
        sx = W / 2 + np.where(ok, x / np.maximum(z, 1e-6), 0) * f
        sy = H / 2 - np.where(ok, y / np.maximum(z, 1e-6), 0) * f
        ob._sx, ob._sy, ob._z, ob._ok = sx, sy, z, ok
        ob._projected = True
    L = norm(np.asarray(sun_dir, dtype=float))
    faces = []
    gfaces = []
    for ob in objects:
        if not getattr(ob, "_projected", False):
            continue
        v = ob.verts
        for q in ob.quads:
            if not (ob._ok[q].all()):
                continue
            p0, p1, p2, p3 = v[q[0]], v[q[1]], v[q[2]], v[q[3]]
            n = np.cross(p1 - p0, p3 - p0)
            ln = np.linalg.norm(n)
            if ln < 1e-9:
                continue
            n = n / ln
            center = (p0 + p1 + p2 + p3) / 4
            if n @ (center - cam.pos) > 0:
                continue
            depth = (ob._z[q].mean())
            if ob.emissive:
                inten = 1.0
                col = np.array(ob.color, dtype=float)
            else:
                inten = ambient + (1 - ambient) * max(0.0, float(n @ (-L)))
                inten = min(1.0, 0.25 + 0.85 * inten)
                col = np.array(ob.color, dtype=float) * inten
            fogt = clamp((depth - fog_start) / max(1e-6, fog_end - fog_start), 0, 1) ** 1.1
            col = col * (1 - fogt) + np.array(fog_color, dtype=float) * fogt
            col = np.clip(col, 0, 255).astype(int)
            pts = [(float(ob._sx[i]), float(ob._sy[i])) for i in q]
            area = abs((pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) - (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1]))
            if area < 1.5:
                continue
            (gfaces if ob.ground else faces).append((depth, pts, tuple(col)))
    gfaces.sort(key=lambda t: -t[0])
    for _, pts, col in gfaces:
        d.polygon(pts, fill=col)
    if shadows:
        for sp in shadows:
            rel = np.asarray(sp["pos"]) - cam.pos
            z = rel @ fwd
            if z <= 0.3:
                continue
            sx = W / 2 + (rel @ right) / z * f
            sy = H / 2 - (rel @ up) / z * f
            r = sp["r"] / z * f
            d.ellipse([sx - r, sy - r * 0.32, sx + r, sy + r * 0.32], fill=(20, 25, 20, sp.get("a", 70)))
    faces.sort(key=lambda t: -t[0])
    for _, pts, col in faces:
        d.polygon(pts, fill=col)

    if glows:
        for gp in glows:
            rel = np.asarray(gp["pos"]) - cam.pos
            z = rel @ fwd
            if z <= 0.3:
                continue
            sx = W / 2 + (rel @ right) / z * f
            sy = H / 2 - (rel @ up) / z * f
            r = gp["r"] / z * f
            glow2d(frame, sx, sy, r, gp.get("color", (255, 230, 160)), gp.get("a", 110))
    return frame

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def glow2d(frame, x, y, r, color, a):
    x, y, r = float(x), float(y), float(r)
    if r < 2:
        return
    size = int(r * 2)
    if size < 4 or x < -size or x > W + size or y < -size or y > H + size:
        return
    g = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(g)
    gd.ellipse([0, 0, size - 1, size - 1], fill=color + (int(a),))
    g = g.filter(ImageFilter.GaussianBlur(max(1, r * 0.4)))
    frame.alpha_composite(g, (int(x - r), int(y - r)))

def sky_grad(top, bot):
    t = np.linspace(0, 1, H)[:, None, None]
    top = np.array(top, dtype=float)
    bot = np.array(bot, dtype=float)
    arr = top[None, None, :] * (1 - t) + bot[None, None, :] * t
    return Image.fromarray(np.repeat(arr, W, axis=1).astype(np.uint8), "RGB").convert("RGBA")

def ease(t):
    return t * t * (3 - 2 * t)

def lerp(a, b, t):
    return a + (b - a) * t

# ================= shared meshes =================

SPH_HI = uv_sphere(1.0, 12, 8)
SPH_MD = uv_sphere(1.0, 8, 6)
SPH_LO = uv_sphere(1.0, 6, 4)
CONE4 = pyramid(1.0, 1.0, 4)
CONE8 = cone(1.0, 1.0, 8)

ORANGE = (243, 156, 74)
ORANGE_D = (208, 116, 46)
CREAM = (255, 233, 200)
PINK = (255, 150, 165)
BLACK = (40, 35, 38)
GREEN = (96, 168, 84)
GREEN_D = (70, 130, 64)
WOOD = (146, 100, 62)
GRAY = (168, 176, 188)

def make_cat(t, pose="stand", scale=1.0, color=ORANGE):
    parts = []
    bob = math.sin(t * 9) * 0.03 if pose == "walk" else math.sin(t * 2.5) * 0.015
    body_tilt = 0.0 if pose == "walk" else -0.35
    if pose == "sit":
        body_pos = (0, 0.42 + bob, 0)
        body_scale = (0.44, 0.55, 0.4)
        haunch = [((0.05, 0.35 + bob, 0), (0.5, 0.42, 0.42))]
    else:
        body_pos = (0, 0.5 + bob, 0)
        body_scale = (0.6, 0.42, 0.4)
        haunch = []
    body = place(SPH_MD, pos=body_pos, scale=body_scale, rot=(0, 0, body_tilt))
    body.color = color
    parts.append(body)
    for off, sc in haunch:
        p = place(SPH_MD, pos=off, scale=sc)
        p.color = color
        parts.append(p)
    chest = place(SPH_MD, pos=(0.28, 0.32 + bob, 0), scale=(0.26, 0.3, 0.28))
    chest.color = CREAM
    parts.append(chest)
    head_pos = (0.52, 0.88 + bob, 0)
    head = place(SPH_MD, pos=head_pos, scale=(0.36, 0.34, 0.34))
    head.color = color
    parts.append(head)
    snout = place(SPH_LO, pos=(0.82, 0.8 + bob, 0), scale=(0.14, 0.11, 0.13))
    snout.color = CREAM
    parts.append(snout)
    for zi in (-1, 1):
        ear = place(CONE4, pos=(0.5, 1.12 + bob, zi * 0.16), scale=(0.14, 0.24, 0.14), rot=(0, 0, -0.25))
        ear.color = color
        parts.append(ear)
        eye = place(SPH_LO, pos=(0.8, 0.95 + bob, zi * 0.14), scale=(0.05, 0.06, 0.05))
        eye.color = BLACK
        parts.append(eye)
    nose = place(SPH_LO, pos=(0.94, 0.82 + bob, 0), scale=(0.045, 0.035, 0.045))
    nose.color = PINK
    parts.append(nose)
    if pose == "walk":
        for li, (lx, lz, ph) in enumerate([(0.3, 0.2, 0), (0.3, -0.2, math.pi), (-0.3, 0.2, math.pi), (-0.3, -0.2, 0)]):
            swing = math.sin(t * 9 + ph) * 0.14
            leg = place(SPH_LO, pos=(lx + swing, 0.16, lz), scale=(0.1, 0.34, 0.1))
            leg.color = color
            parts.append(leg)
    else:
        for lx, lz in [(0.28, 0.16), (0.28, -0.16)]:
            leg = place(SPH_LO, pos=(lx, 0.14, lz), scale=(0.09, 0.3, 0.09))
            leg.color = color
            parts.append(leg)
    tail_sway = math.sin(t * 2.4 + 1) * 0.5
    for i in range(5):
        u = i / 4
        tp = (-0.55 - u * 0.3, 0.45 + u * 0.45, tail_sway * u * 0.35)
        tr = 0.09 - u * 0.04
        seg = place(SPH_LO, pos=tp, scale=(tr, tr * 1.4, tr))
        seg.color = ORANGE_D if i > 2 else color
        parts.append(seg)
    if scale != 1.0:
        return [Obj(p.verts * scale, p.quads, p.color, p.emissive) for p in parts]
    return parts

def make_butterfly(t, scale=1.0):
    parts = []
    flap = math.sin(t * 14)
    body = place(SPH_LO, pos=(0, 0, 0), scale=(0.09, 0.07, 0.07))
    body.color = (70, 55, 65)
    parts.append(body)
    for zi, col in ((1, (90, 165, 255)), (-1, (255, 205, 80))):
        wing = place(SPH_LO, pos=(0, 0.02, zi * 0.16 * abs(flap) + zi * 0.05), scale=(0.2, 0.03, 0.16))
        wing.color = col
        parts.append(wing)
    return [Obj(p.verts * scale, p.quads, p.color, p.emissive) for p in parts]

def make_pigeon(t):
    parts = []
    flap = math.sin(t * 4) * 0.25
    body = place(SPH_MD, pos=(0, 0.3, 0), scale=(0.36, 0.27, 0.24))
    body.color = GRAY
    parts.append(body)
    head = place(SPH_LO, pos=(0.34, 0.5, 0), scale=(0.17, 0.16, 0.16))
    head.color = (205, 212, 222)
    parts.append(head)
    beak = place(CONE4, pos=(0.5, 0.46, 0), scale=(0.06, 0.14, 0.06), rot=(0, 0, -math.pi / 2))
    beak.color = (240, 150, 60)
    parts.append(beak)
    eye = place(SPH_LO, pos=(0.42, 0.54, 0.1), scale=(0.035, 0.035, 0.035))
    eye.color = BLACK
    parts.append(eye)
    for zi in (-1, 1):
        wing = place(SPH_LO, pos=(0, 0.36 + flap, zi * 0.26), scale=(0.3, 0.05, 0.14), rot=(0, 0, flap * zi))
        wing.color = (190, 198, 210)
        parts.append(wing)
    tail = place(box(0.3, 0.04, 0.18), pos=(-0.4, 0.34, 0), rot=(0, 0, 0.3))
    tail.color = (150, 158, 172)
    parts.append(tail)
    return parts

def make_girl(t):
    parts = []
    dress = place(CONE8, pos=(0, 0.05, 0), scale=(0.5, 1.05, 0.42))
    dress.color = (226, 120, 130)
    parts.append(dress)
    head = place(SPH_MD, pos=(0, 1.32, 0), scale=(0.3, 0.32, 0.3))
    head.color = (250, 214, 180)
    parts.append(head)
    hair = place(SPH_MD, pos=(0, 1.44, -0.06), scale=(0.34, 0.3, 0.34))
    hair.color = (90, 60, 40)
    parts.append(hair)
    for zi in (-1, 1):
        eye = place(SPH_LO, pos=(0.24, 1.36, zi * 0.11), scale=(0.04, 0.05, 0.04))
        eye.color = BLACK
        parts.append(eye)
    for zi in (-1, 1):
        arm = place(SPH_LO, pos=(0.18, 0.85, zi * 0.28), scale=(0.3, 0.09, 0.1), rot=(0, 0.5 * zi, 0.9))
        arm.color = (226, 120, 130)
        parts.append(arm)
    return parts

def make_house(x, z, w=2.2, h=2.0, d=2.2, body=(226, 135, 120), roof=(150, 90, 80), lit=False):
    parts = []
    b = place(box(w, h, d), pos=(x, 0, z))
    b.color = body
    parts.append(b)
    rf = place(CONE4, pos=(x, h, z), scale=(w * 0.78, w * 0.4, d * 0.78), rot=(0, math.pi / 4, 0))
    rf.color = roof
    parts.append(rf)
    door = place(box(0.5, 0.95, 0.06), pos=(x, 0, z + d / 2 + 0.02))
    door.color = (110, 76, 50)
    parts.append(door)
    wc = (255, 236, 150) if lit else (200, 230, 245)
    for wx in (-w * 0.25, w * 0.25):
        win = place(box(0.5, 0.5, 0.05), pos=(x + wx, h * 0.55, z + d / 2 + 0.02))
        win.color = wc
        win.emissive = lit
        parts.append(win)
    return parts

def make_tree(x, z, s=1.0):
    parts = []
    trunk = place(box(0.24 * s, 1.1 * s, 0.24 * s), pos=(x, 0, z))
    trunk.color = WOOD
    parts.append(trunk)
    for off, sc in [((0, 1.4 * s, 0), (0.85 * s, 0.7 * s, 0.85 * s)), ((0.3 * s, 1.05 * s, 0.1 * s), (0.55 * s, 0.45 * s, 0.55 * s)), ((-0.3 * s, 1.1 * s, -0.1 * s), (0.5 * s, 0.42 * s, 0.5 * s))]:
        fo = place(SPH_MD, pos=(x + off[0], off[1], z + off[2]), scale=sc)
        fo.color = GREEN
        parts.append(fo)
    return parts

def make_cloud(x, y, z, s=1.0):
    parts = []
    for off, sc in [((0, 0, 0), (1, 0.55, 0.7)), ((0.9, 0.15, 0.1), (0.7, 0.45, 0.55)), ((-0.85, 0.05, -0.05), (0.65, 0.4, 0.5))]:
        c = place(SPH_LO, pos=(x + off[0] * s, y + off[1] * s, z + off[2] * s), scale=(sc[0] * s, sc[1] * s, sc[2] * s))
        c.color = (255, 255, 255)
        c.emissive = True
        parts.append(c)
    return parts

def ground_quads(color, half=60, step=10):
    verts = []
    quads = []
    idx = 0
    for i in range(int(-half / step), int(half / step)):
        for j in range(int(-half / step), int(half / step)):
            x0, x1 = i * step, (i + 1) * step
            z0, z1 = j * step, (j + 1) * step
            verts += [(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)]
            quads.append((idx, idx + 1, idx + 2, idx + 3))
            idx += 4
    o = Obj(np.array(verts, dtype=float), quads, color)
    o.ground = True
    return [o]

def flower3d(x, z, col):
    parts = []
    stem = place(box(0.05, 0.5, 0.05), pos=(x, 0, z))
    stem.color = (70, 140, 70)
    parts.append(stem)
    head = place(SPH_LO, pos=(x, 0.55, z), scale=(0.16, 0.12, 0.16))
    head.color = col
    parts.append(head)
    core = place(SPH_LO, pos=(x, 0.62, z), scale=(0.07, 0.06, 0.07))
    core.color = (255, 220, 90)
    parts.append(core)
    return parts

# ================= scenes =================

NARR = [
    "في قطة صغيرة اسمها مشمش، كانت عايشة في بيت دافي، وقلبها مليان فضول.",
    "في يوم من الأيام، شافت فراشة ملونة، وقررت تتبعها على طول.",
    "الفراشة طارت بعيد، ومشمش لقيت نفسها ضايعة في شوارع غريبة.",
    "هناك قابلت حمامة طيبة، دلته على طريق البيت.",
    "ومشت ورا قلبها، لحد ما شافت نور البيت من بعيد.",
    "صاحبتها عملت لها حضن دافي، وعرفت إن البيت أحلى مكان في الدنيا.",
]
NARR_DUR = [8.56, 7.68, 7.41, 5.61, 5.85, 7.34]
PAD = 3.4
SCENE_DUR = [max(8.5, nd + PAD) for nd in NARR_DUR]
N_SCENES = 6

def cam_path(p0, t0, p1, t1, u):
    e = ease(u)
    return (np.array(p0) * (1 - e) + np.array(p1) * e, np.array(t0) * (1 - e) + np.array(t1) * e)

def scene1(cv, tsec, dur):
    u = tsec / dur
    cam_pos, cam_tgt = cam_path((2.6, 1.9, 5.8), (0.3, 1.35, 0), (1.2, 1.5, 4.2), (0.1, 1.2, 0), u)
    cam = Camera(cam_pos, cam_tgt, fov=55)
    objs = []
    wall_col = (240, 214, 180)
    wx, wz = 0, -1.0
    ww, wh, wt = 6.0, 4.0, 0.3
    op_w, op_h = 2.6, 2.0
    op_y = 1.1
    for px, py, pw, ph in [(wx - ww / 2 - (-ww / 2 + (ww - op_w) / 2) / 2 + 0, 0, 0, 0)]:
        pass
    segs = [
        (wx - ww / 2, 0, (ww - op_w) / 2, wh),
        (wx + op_w / 2 + (ww - op_w) / 4, 0, (ww - op_w) / 2, wh),
        (wx - op_w / 2 - (ww - op_w) / 4, 0, (ww - op_w) / 2, wh),
    ]
    segs = [
        (-(op_w / 2 + (ww - op_w) / 4), 0, (ww - op_w) / 2, wh),
        (+(op_w / 2 + (ww - op_w) / 4), 0, (ww - op_w) / 2, wh),
    ]
    for sx_, sy_, sw_, sh_ in segs:
        b = place(box(sw_, sh_, wt), pos=(sx_, sy_, wz))
        b.color = wall_col
        objs.append(b)
    top = place(box(op_w, wh - op_h - op_y, wt), pos=(0, op_y + op_h, wz))
    top.color = wall_col
    objs.append(top)
    bot = place(box(op_w, op_y, wt), pos=(0, 0, wz))
    bot.color = wall_col
    objs.append(bot)
    sill = place(box(op_w + 0.4, 0.12, 0.7), pos=(0, op_y - 0.06, wz + 0.35))
    sill.color = (255, 250, 240)
    objs.append(sill)
    frame_c = (255, 252, 245)
    for fx in (-op_w / 2, op_w / 2):
        fpost = place(box(0.12, op_h, 0.36), pos=(fx, op_y, wz - 0.02))
        fpost.color = frame_c
        objs.append(fpost)
    ftop = place(box(op_w, 0.12, 0.36), pos=(0, op_y + op_h, wz - 0.02))
    ftop.color = frame_c
    objs.append(ftop)
    objs += [place(o, pos=(0, 0, 0)) for o in []]
    floor = ground_quads((196, 152, 108), half=30, step=6)
    objs += floor
    rug = place(flat_quad(2.6, 1.8), pos=(0, 0.012, 2.2))
    rug.color = (180, 90, 90)
    objs.append(rug)
    cat = make_cat(tsec, pose="sit")
    objs += place_parts(cat, pos=(0, op_y + 0.06, wz + 0.12), rot_y=-math.pi / 2 + 0.4, scale=0.62)
    sky_img = sky_grad((255, 214, 165), (168, 216, 240))
    sun3 = place(SPH_LO, pos=(4.5, 7.5, -14), scale=(1.6, 1.6, 1.6))
    sun3.color = (255, 230, 150)
    sun3.emissive = True
    objs.append(sun3)
    for i, (cx, cy, cs, sp) in enumerate([(-5, 5.2, 1.2, 0.5), (3.5, 6.4, 1.0, 0.35), (7, 4.6, 0.8, 0.42)]):
        objs += make_cloud(cx + tsec * sp, cy, -13, cs)
    tree_far = make_tree(-6, -12, 2.2)
    for p in tree_far:
        p.color = GREEN_D
    objs += tree_far
    flowers = []
    for fx, fc in [(-4.5, (235, 90, 90)), (-3.6, (250, 200, 80)), (-2.8, (190, 120, 220)), (5.2, (235, 90, 90)), (6.1, (250, 200, 80))]:
        flowers += flower3d(fx, -11, fc)
    objs += flowers
    frame = render3d(objs, cam, sky_img, sun_dir=(-0.4, 0.8, 0.45), ambient=0.55,
                     fog_color=(255, 230, 200), fog_start=14, fog_end=30,
                     shadows=[{"pos": (0, 0.01, 2.2), "r": 1.1, "a": 45}],
                     sun_pos=(4.5, 7.5, -14), sun_r=1.6)
    return frame

def scene2(cv, tsec, dur):
    u = tsec / dur
    walk = ease(clamp(u * 1.25, 0, 1))
    cat_x = -4.5 + 8.0 * walk
    cam_pos, cam_tgt = cam_path((cat_x - 2.5, 2.2, 7.0), (cat_x + 1.5, 1.0, 0),
                                (cat_x - 1.2, 1.8, 5.2), (cat_x + 2.0, 0.9, 0), u)
    cam = Camera(cam_pos, cam_tgt, fov=55)
    objs = ground_quads((116, 178, 92), half=60, step=10)
    objs += make_tree(-7.5, -2.5, 1.4)
    objs += make_tree(8.0, -3.5, 1.7)
    for fx, fz, fc in [(-6, 1.5, (235, 90, 90)), (-3, 2.2, (250, 200, 80)), (0, 1.8, (190, 120, 220)),
                       (3, 2.4, (240, 130, 160)), (6, 1.6, (250, 200, 80)), (9, 2.2, (235, 90, 90)),
                       (-1, 3.2, (235, 90, 90)), (4.5, 3.4, (190, 120, 220))]:
        objs += flower3d(fx, fz, fc)
    bx = cat_x + 1.2 + 2.5 * math.sin(u * 3.2)
    by = 1.6 + 0.5 * math.sin(u * 7.5) + 0.3 * math.sin(u * 3.1)
    bz = 0.4 * math.sin(u * 5.0)
    objs += place_parts(make_butterfly(tsec), pos=(bx, by, bz), scale=2.1)
    cat = make_cat(tsec, pose="walk")
    objs += place_parts(cat, pos=(cat_x, 0, 0), scale=0.85)
    sky_img = sky_grad((120, 190, 245), (205, 236, 250))
    sun3 = place(SPH_LO, pos=(9, 9, -18), scale=(1.8, 1.8, 1.8))
    sun3.color = (255, 235, 160)
    sun3.emissive = True
    objs.append(sun3)
    for cx, cy, cs, sp in [(-8, 7.5, 1.4, 0.6), (2, 8.5, 1.1, 0.45), (10, 6.8, 0.9, 0.5)]:
        objs += make_cloud(cx + tsec * sp, cy, -16, cs)
    frame = render3d(objs, cam, sky_img, sun_dir=(0.5, 0.85, 0.3), ambient=0.6,
                     fog_color=(205, 236, 250), fog_start=18, fog_end=42,
                     shadows=[{"pos": (cat_x, 0.01, 0), "r": 0.9, "a": 60}],
                     sun_pos=(9, 9, -18), sun_r=1.8)
    return frame

def scene3(cv, tsec, dur):
    u = tsec / dur
    cam_pos, cam_tgt = cam_path((0.0, 2.4, 9.0), (0, 1.2, 0), (0.6, 1.7, 6.4), (0, 1.1, 0), u)
    cam = Camera(cam_pos, cam_tgt, fov=55)
    objs = ground_quads((112, 116, 124), half=60, step=10)
    for i in range(4):
        objs += make_house(-7.5 + i * 5.0, -4.5, body=[(226, 135, 120), (140, 170, 200), (220, 180, 120), (170, 190, 165)][i], roof=(150, 90, 80), lit=(i == 1))
    for i in range(3):
        objs += make_house(-5.0 + i * 5.5, 5.5, body=[(190, 155, 145), (160, 180, 190), (200, 170, 140)][i], roof=(120, 80, 75))
    road = place(flat_quad(4.0, 60), pos=(0, 0.012, 0))
    road.color = (90, 94, 102)
    objs.append(road)
    for i in range(8):
        dash = place(flat_quad(1.6, 0.35), pos=(0, 0.02, -14 + i * 4))
        dash.color = (230, 230, 220)
        objs.append(dash)
    bx = 2.0 + 6.0 * u
    by = 2.6 + 2.0 * u + 0.25 * math.sin(u * 8)
    objs += place_parts(make_butterfly(tsec), pos=(bx, by, -1.0), scale=1.4)
    cat = make_cat(tsec, pose="sit")
    objs += place_parts(cat, pos=(-0.8, 0, 0.5), rot_y=-0.9, scale=0.8)
    sky_img = sky_grad((150, 175, 205), (205, 215, 225))
    for cx, cy, cs, sp in [(-6, 8.0, 1.3, 0.3), (5, 9.0, 1.0, 0.25)]:
        objs += make_cloud(cx + tsec * sp, cy, -18, cs, )
    frame = render3d(objs, cam, sky_img, sun_dir=(0.2, 0.7, 0.5), ambient=0.52,
                     fog_color=(205, 215, 225), fog_start=16, fog_end=40,
                     shadows=[{"pos": (-0.8, 0.01, 0.5), "r": 0.85, "a": 60}])
    return frame

def scene4(cv, tsec, dur):
    u = tsec / dur
    cam_pos, cam_tgt = cam_path((-2.8, 2.6, 6.0), (0.4, 1.2, 0), (-1.4, 2.0, 4.4), (0.5, 1.1, 0), u)
    cam = Camera(cam_pos, cam_tgt, fov=55)
    objs = ground_quads((74, 62, 98), half=60, step=10)
    for i, (hx, hh) in enumerate([(-6, 2.2), (-1.5, 2.8), (3.5, 2.4), (8, 3.0)]):
        b = place(box(4.0, hh, 4.0), pos=(hx, 0, -2.0))
        b.color = (86 + i * 6, 72 + i * 5, 112 + i * 6)
        objs.append(b)
        for wx in (-1.0, 1.0):
            win = place(box(0.7, 0.7, 0.08), pos=(hx + wx, hh * 0.55, 0.05))
            win.color = (255, 220, 140)
            win.emissive = True
            objs.append(win)
    ledge = place(box(3.2, 0.25, 1.6), pos=(0.8, 2.0, 1.2))
    ledge.color = (96, 80, 124)
    objs.append(ledge)
    objs += place_parts(make_pigeon(tsec), pos=(1.7, 2.12, 1.2), rot_y=math.pi + 0.3)
    cat = make_cat(tsec, pose="sit")
    objs += place_parts(cat, pos=(-0.6, 2.12, 1.2), rot_y=1.25, scale=0.8)
    sky_img = sky_grad((88, 76, 140), (240, 150, 110))
    moon = place(SPH_LO, pos=(6, 8.5, -16), scale=(1.2, 1.2, 1.2))
    moon.color = (255, 244, 214)
    moon.emissive = True
    objs.append(moon)
    frame = render3d(objs, cam, sky_img, sun_dir=(-0.3, 0.6, 0.5), ambient=0.5,
                     fog_color=(150, 110, 130), fog_start=14, fog_end=36,
                     sun_pos=(6, 8.5, -16), sun_r=1.2)
    stars(frame, tsec, 40, seed=5, ymax=0.55)
    return frame

def scene5(cv, tsec, dur):
    u = tsec / dur
    run_x = -5.5 + 9.5 * ease(clamp(u * 1.2, 0, 1))
    cam_pos, cam_tgt = cam_path((run_x - 1.5, 2.0, 6.5), (run_x + 2.0, 1.2, 0),
                                (run_x + 0.5, 1.6, 5.0), (run_x + 2.5, 1.0, 0), u)
    cam = Camera(cam_pos, cam_tgt, fov=55)
    objs = ground_quads((128, 96, 96), half=60, step=10)
    road = place(flat_quad(4.0, 60), pos=(0, 0.012, 0))
    road.color = (108, 82, 88)
    objs.append(road)
    for i in range(4):
        objs += make_house(-8.0 + i * 5.4, -4.8, body=[(150, 90, 90), (170, 110, 90), (140, 85, 95), (175, 120, 95)][i], roof=(110, 65, 70), lit=(i == 3))
    home = make_house(run_x + 4.2, -1.2, body=(210, 150, 120), roof=(140, 80, 80), lit=True)
    objs += home
    sun3 = place(SPH_LO, pos=(2, 1.2, -30), scale=(3.2, 3.2, 3.2))
    sun3.color = (255, 200, 110)
    sun3.emissive = True
    objs.append(sun3)
    cat = make_cat(tsec, pose="walk")
    objs += place_parts(cat, pos=(run_x, 0, 0), scale=0.85)
    sky_img = sky_grad((250, 150, 100), (255, 210, 140))
    for cx, cy, cs in [(-7, 7.5, 1.2), (6, 8.5, 1.0)]:
        objs += make_cloud(cx, cy, -20, cs)
    frame = render3d(objs, cam, sky_img, sun_dir=(0.0, 0.35, 0.9), ambient=0.58,
                     fog_color=(255, 190, 130), fog_start=14, fog_end=38,
                     shadows=[{"pos": (run_x, 0.01, 0), "r": 0.9, "a": 55}],
                     sun_pos=(2, 1.2, -30), sun_r=3.2)
    return frame

def scene6(cv, tsec, dur):
    u = tsec / dur
    cam_pos, cam_tgt = cam_path((0.6, 1.9, 6.2), (0, 1.2, 0), (0.0, 1.5, 4.4), (0, 1.1, 0), u)
    cam = Camera(cam_pos, cam_tgt, fov=55)
    objs = ground_quads((172, 130, 92), half=30, step=6)
    wall_col = (120, 92, 110)
    ww, wh, wt = 7.0, 4.2, 0.3
    wz = -1.0
    for sx_, sw_ in [(-(2.2 + 1.3), 2.6), (2.2 + 1.3, 2.6)]:
        b = place(box(sw_, wh, wt), pos=(sx_, 0, wz))
        b.color = wall_col
        objs.append(b)
    top = place(box(2.2, 1.0, wt), pos=(0, 3.2, wz))
    top.color = wall_col
    objs.append(top)
    bot = place(box(2.2, 2.2, wt), pos=(0, 0, wz))
    bot.color = wall_col
    objs.append(bot)
    sill = place(box(2.6, 0.12, 0.7), pos=(0, 2.14, wz + 0.35))
    sill.color = (200, 170, 150)
    objs.append(sill)
    rug = place(flat_quad(3.4, 2.2), pos=(0, 0.012, 1.6))
    rug.color = (150, 80, 80)
    objs.append(rug)
    lamp_x = -2.6
    pole = place(box(0.1, 1.9, 0.1), pos=(lamp_x, 0, 0.6))
    pole.color = (110, 80, 55)
    objs.append(pole)
    shade = place(CONE8, pos=(lamp_x, 1.9, 0.6), scale=(0.5, 0.45, 0.5), rot=(math.pi, 0, 0))
    shade.color = (250, 210, 130)
    shade.emissive = True
    objs.append(shade)
    objs += place_parts(make_girl(tsec), pos=(0.35, 0, 1.5), rot_y=-math.pi / 2 + 0.25)
    objs += place_parts(make_cat(tsec + 3, pose="sit"), pos=(0.32, 0.66, 1.05), rot_y=-math.pi / 2 + 0.25, scale=0.55)
    sky_img = sky_grad((30, 34, 66), (70, 55, 90))
    moon = place(SPH_LO, pos=(1.2, 7.0, -14), scale=(1.1, 1.1, 1.1))
    moon.color = (255, 244, 214)
    moon.emissive = True
    objs.append(moon)
    frame = render3d(objs, cam, sky_img, sun_dir=(-0.4, 0.7, 0.4), ambient=0.42,
                     fog_color=(70, 55, 90), fog_start=12, fog_end=30,
                     shadows=[{"pos": (0.35, 0.01, 1.35), "r": 1.0, "a": 60}],
                     glows=[{"pos": (lamp_x, 2.0, 0.6), "r": 2.2, "color": (255, 220, 140), "a": 120}])
    stars(frame, tsec, 26, seed=9, ymax=0.4)
    return frame

def stars(frame, tsec, n, seed=1, ymax=0.5):
    rng = np.random.default_rng(seed)
    d = ImageDraw.Draw(frame, "RGBA")
    for i in range(n):
        x = rng.uniform(0, W)
        y = rng.uniform(0, H * ymax)
        a = int(120 + 120 * math.sin(tsec * 1.8 + i * 1.7))
        r = rng.uniform(1, 2.4)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 225, max(40, a)))

SCENES = [scene1, scene2, scene3, scene4, scene5, scene6]

# ================= overlays =================

def vignette(frame):
    v = Image.new("L", (W, H), 0)
    dv = ImageDraw.Draw(v)
    dv.ellipse([-W * 0.3, -H * 0.3, W * 1.3, H * 1.3], fill=255)
    v = v.filter(ImageFilter.GaussianBlur(190))
    arr = np.asarray(v).astype(float) / 255.0
    alpha = ((1 - arr) * 80).astype(np.uint8)
    dark = Image.new("RGBA", (W, H), (15, 12, 22, 0))
    dark.putalpha(Image.fromarray(alpha, "L"))
    frame.alpha_composite(dark)

def draw_caption(frame, idx, tsec, dur):
    tp = tsec / dur
    a = clamp(tp / 0.1, 0, 1) * clamp((1 - tp) / 0.1, 0, 1)
    if a <= 0.03:
        return
    band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    bd.rounded_rectangle([90, H - 148, W - 90, H - 58], radius=26, fill=(25, 20, 32, int(150 * a)))
    frame.alpha_composite(band)
    rgb = frame.convert("RGB")
    draw_arabic(rgb, NARR[idx], 40, W / 2, H - 103, (255, 250, 240), FONT_CAP)
    frame.paste(rgb, (0, 0))

def draw_title(frame, tsec):
    if tsec > 4.6:
        return
    a = clamp(tsec / 0.9, 0, 1) * clamp((4.6 - tsec) / 0.9, 0, 1)
    if a <= 0.03:
        return
    band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    bd.rounded_rectangle([W / 2 - 330, 36, W / 2 + 330, 210], radius=34, fill=(60, 38, 20, int(140 * a)))
    frame.alpha_composite(band)
    rgb = frame.convert("RGB")
    draw_arabic(rgb, "مشمش القطة الفضولية", 66, W / 2, 100, (255, 240, 210), FONT_TITLE, stroke=(90, 55, 25))
    draw_arabic(rgb, "قصة قصيرة بالصوت والصورة", 34, W / 2, 172, (255, 220, 180), FONT_CAP)
    frame.paste(rgb, (0, 0))

def draw_end(frame, tsec, dur):
    st = dur - 3.4
    if tsec < st:
        return
    u = (tsec - st) / 3.4
    a = clamp(u / 0.45, 0, 1)
    rgb = frame.convert("RGB")
    ov = Image.new("RGBA", (W, H), (20, 15, 30, int(120 * a)))
    frame.alpha_composite(ov)
    rgb = frame.convert("RGB")
    draw_arabic(rgb, "النهاية", 100, W / 2, H * 0.36, (255, 246, 225), FONT_TITLE, stroke=(60, 40, 20))
    draw_arabic(rgb, "البيت أحلى مكان", 44, W / 2, H * 0.52, (255, 220, 170), FONT_CAP)
    frame.paste(rgb, (0, 0))

# ================= audio =================

def make_audio(total_dur, out_path):
    sr = 44100
    total = int(sr * total_dur)
    mix = np.zeros(total)

    def add_tone(freq, start, length, amp, timbre=1.0):
        i0 = int(start * sr)
        n = min(int(length * sr), total - i0)
        if n <= 0:
            return
        tt = np.arange(n) / sr
        env = np.clip(np.minimum(tt / 0.06, 1.0) * np.minimum((length - tt) / 0.2, 1.0), 0, 1)
        sig = (np.sin(2 * np.pi * freq * tt) + timbre * 0.3 * np.sin(4 * np.pi * freq * tt)) * env * amp
        mix[i0:i0 + n] += sig

    C, E, G, A, B, D, F = 261.63, 329.63, 392.0, 440.0, 493.88, 587.33, 349.23
    prog = [[C * 0.5, E, G], [A * 0.5, C, E], [F * 0.5, A, C], [G * 0.5, B, D]]
    bar = 4.0
    for bidx in range(int(total_dur / bar) + 1):
        chord = prog[bidx % 4]
        for fn in chord:
            add_tone(fn, bidx * bar, bar * 1.05, 0.028)
        add_tone(chord[0] * 0.5, bidx * bar, bar * 0.95, 0.034)
    melody = [E, G, A, G, E, D, C, D, E, G, A, C * 2, A, G, E, G]
    for i, mn in enumerate(melody * 6):
        if i * 1.5 >= total_dur - 2:
            break
        add_tone(mn, i * 1.5, 1.4, 0.022, 0.2)
    mix /= max(1e-9, np.max(np.abs(mix)))
    mix *= 0.30

    starts = []
    t0 = 0.0
    for i, dur in enumerate(SCENE_DUR):
        starts.append(t0)
        t0 += dur - (XFADE if i < N_SCENES - 1 else 0)
    for i in range(N_SCENES):
        raw = subprocess.run([FF, "-y", "-i", f"{ASSETS}/narration/n{i}.mp3", "-ar", str(sr), "-ac", "1", "-f", "f32le", "-"],
                             capture_output=True)
        narr = np.frombuffer(raw.stdout, dtype=np.float32).astype(np.float64)
        ns = int((starts[i] + 0.7) * sr)
        end = min(ns + len(narr), total)
        mix[ns:end] += narr[:end - ns] * 1.9
    fade = int(sr * 1.5)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-int(sr * 2.2):] *= np.linspace(1, 0, int(sr * 2.2))
    st = np.zeros((total, 2))
    st[:, 0] = mix
    shift = int(sr * 0.012)
    st[shift:, 1] = mix[:-shift]
    pcm = (np.clip(st, -1, 1) * 32767).astype("<i2")
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return starts

# ================= main =================

def render_scene_frame(idx, tsec):
    dur = SCENE_DUR[idx]
    cv = SCENES[idx](None, tsec, dur)
    return cv

def main():
    total_dur = sum(SCENE_DUR) - XFADE * (N_SCENES - 1)
    print("total duration:", round(total_dur, 2), flush=True)
    audio_path = os.path.join(ASSETS, "story3d.wav")
    out_path = os.path.join(ASSETS, "mishmish_story_3d.mp4")
    starts = make_audio(total_dur, audio_path)
    print("audio done", flush=True)

    cmd = [FF, "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", audio_path,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart",
           out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total_frames = int(total_dur * FPS)
    scene_starts = starts
    for fi in range(total_frames):
        tsec = fi / FPS
        idx = 0
        for i in range(N_SCENES):
            if tsec >= scene_starts[i]:
                idx = i
        local = tsec - scene_starts[idx]
        if idx < N_SCENES - 1 and local > SCENE_DUR[idx] - XFADE:
            u = (local - (SCENE_DUR[idx] - XFADE)) / XFADE
            fa = render_scene_frame(idx, local)
            fb = render_scene_frame(idx + 1, local - (SCENE_DUR[idx] - XFADE))
            frame = Image.blend(fa, fb, ease(u))
            cap_idx = idx if u < 0.5 else idx + 1
            cap_local = local if u < 0.5 else local - (SCENE_DUR[idx] - XFADE)
        else:
            frame = render_scene_frame(idx, local)
            cap_idx, cap_local = idx, local
        vignette(frame)
        draw_caption(frame, cap_idx, clamp(cap_local, 0, SCENE_DUR[cap_idx]), SCENE_DUR[cap_idx])
        if idx == 0:
            draw_title(frame, local)
        if idx == N_SCENES - 1:
            draw_end(frame, local, SCENE_DUR[idx])
        proc.stdin.write(frame.convert("RGB").tobytes())
        if fi % 150 == 0:
            print(f"frame {fi}/{total_frames}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("video done", proc.returncode, flush=True)

if __name__ == "__main__":
    main()
