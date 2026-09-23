#!/usr/bin/env python3
"""Minimal stand-in for the dataviz skill's validate_palette.js (which is absent
from the bundled-skill dir this session). Implements the two hard checks the skill
text specifies for a categorical palette:

  * adjacent-pair separation under CVD:  OKLab ΔE×100 >= 8 target (6-8 floor)
  * normal-vision separation:            OKLab ΔE×100 >= 15 (below = hard FAIL)

CVD simulation uses the Machado et al. (2009) severity-1.0 matrices for
deuteranopia / protanopia / tritanopia, applied in linear-sRGB. OKLab per
Björn Ottosson. This is a faithful reconstruction of the *rules*, enough to
choose a 4th hue by computation instead of taste.
"""
import itertools
import math
import sys

MACHADO = {  # severity 1.0
    "deuteranopia": [[0.367322, 0.860646, -0.227968],
                     [0.280085, 0.672501, 0.047413],
                     [-0.011820, 0.042940, 0.968881]],
    "protanopia": [[0.152286, 1.052583, -0.204868],
                   [0.114503, 0.786281, 0.099216],
                   [-0.003882, -0.048116, 1.051998]],
    "tritanopia": [[1.255528, -0.076749, -0.178779],
                   [-0.078411, 0.930809, 0.147602],
                   [0.004733, 0.691367, 0.303900]],
}


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def srgb2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin2srgb(c):
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def apply_cvd(rgb, kind):
    lin = [srgb2lin(c) for c in rgb]
    m = MACHADO[kind]
    out = [sum(m[r][k] * lin[k] for k in range(3)) for r in range(3)]
    return tuple(lin2srgb(c) for c in out)


def rgb2oklab(rgb):
    r, g, b = (srgb2lin(c) for c in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def de(c1, c2):
    a, b = rgb2oklab(c1), rgb2oklab(c2)
    return 100 * math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def check(palette):
    cols = [hex2rgb(h) for h in palette]
    ok = True
    print("normal-vision pairwise ΔE (floor 15):")
    for (i, a), (j, b) in itertools.combinations(enumerate(cols), 2):
        d = de(a, b)
        flag = "FAIL" if d < 15 else "ok"
        if d < 15:
            ok = False
        print(f"  {palette[i]}–{palette[j]}: {d:5.1f}  {flag}")
    print("adjacent-pair CVD ΔE (target 8, floor 6):")
    for k in MACHADO:
        worst = min(de(apply_cvd(cols[i], k), apply_cvd(cols[i + 1], k))
                    for i in range(len(cols) - 1))
        allmin = min(de(apply_cvd(a, k), apply_cvd(b, k))
                     for a, b in itertools.combinations(cols, 2))
        flag = "FAIL" if allmin < 6 else ("WARN" if allmin < 8 else "PASS")
        if allmin < 6:
            ok = False
        print(f"  {k:13} adjacent-min {worst:5.1f}   all-pairs-min {allmin:5.1f}  {flag}")
    print("=> " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    pal = sys.argv[1].split(",")
    check(pal)
