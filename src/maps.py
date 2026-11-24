"""Choropleth maps of English upper-tier local authorities (no GIS dependencies).

Boundaries: ONS Counties and Unitary Authorities (December 2023) BUC, England only
(data/raw/..._BUC_England.geojson). Longitude is scaled by cos(52.5 deg) so that the map has
approximately the right shape at English latitudes.
"""
import json

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

from common import BOUNDARIES
from plotstyle import AXIS, INK2

COS = np.cos(np.radians(52.5))
MISSING = "#d9d8d2"
# Areas published combined with a neighbour: draw them with the neighbour's value
COMBINED_WITH = {"E06000053": "E06000052", "E09000001": "E09000012"}
LONDON_BOX = (-0.56, 0.36, 51.27, 51.71)  # lon min, lon max, lat min, lat max


def _ring_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def load_polygons():
    """code -> list of exterior rings (projected); drawn largest-first so enclaves sit on top."""
    g = json.loads(BOUNDARIES.read_text())
    out = []
    for f in g["features"]:
        code = f["properties"]["CTYUA23CD"]
        geom = f["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings = [np.column_stack([np.array(p[0])[:, 0] * COS, np.array(p[0])[:, 1]]) for p in parts]
        out.append((code, rings, sum(_ring_area(r) for r in rings)))
    out.sort(key=lambda t: -t[2])
    return [(c, r) for c, r, _ in out]


POLYGONS = None


def draw(ax, values, cmap, vmin, vmax, london_inset=True, inset_rect=(0.70, 0.585, 0.30, 0.29)):
    """values: dict la_code -> number (NaN/missing drawn grey). Returns a ScalarMappable."""
    global POLYGONS
    if POLYGONS is None:
        POLYGONS = load_polygons()

    def collection():
        verts, colours = [], []
        for code, rings in POLYGONS:
            v = values.get(COMBINED_WITH.get(code, code), np.nan)
            c = MISSING if (v is None or not np.isfinite(v)) else cmap((v - vmin) / (vmax - vmin))
            for r in rings:
                verts.append(r)
                colours.append(c)
        return PolyCollection(verts, facecolors=colours, edgecolors="white", linewidths=0.25)

    ax.add_collection(collection())
    ax.set_xlim(-5.8 * COS, 3.2 * COS)  # extra sea to the east holds the London inset
    ax.set_ylim(49.85, 55.85)
    ax.set_aspect("equal")
    ax.axis("off")
    if london_inset:
        ins = ax.inset_axes(inset_rect)
        ins.add_collection(collection())
        x0, x1, y0, y1 = LONDON_BOX
        ins.set_xlim(x0 * COS, x1 * COS)
        ins.set_ylim(y0, y1)
        ins.set_aspect("equal")
        ins.set_xticks([])
        ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_color(AXIS)
            sp.set_linewidth(0.6)
        ins.set_title("London", fontsize=6, color=INK2, pad=2)
        ax.add_patch(Rectangle((x0 * COS, y0), (x1 - x0) * COS, y1 - y0, fill=False, ec=AXIS, lw=0.6))
    sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap)
    return sm
