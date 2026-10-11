"""Where is a river big enough to keep its old name? A map of the threshold.

At every point of a grid, a logistic curve of old name on log catchment area
is fitted to nearby rivers only, weighted by a Gaussian kernel of distance
(geographically weighted logistic regression). The size at which that local
curve crosses 50% is the local threshold. Low thresholds (amber) mean even
small streams kept their old names; high ones (blue) mean only giant rivers
did, or none.

US: all classified named streams, 1 degree grid, 175 km kernel.
England: Ekwall's labelled rivers, 10 km grid, 45 km kernel.

Output: figures/threshold_surface.png, data/derived/surface_*.parquet
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from pyproj import Transformer
from scipy.spatial import cKDTree

from common import DERIVED, FIGURES, RAW

BG, TEXT, SUBTLE = "#0a0d12", "#e8e4da", "#8a8f98"
CMAP = LinearSegmentedColormap.from_list("threshold", ["#f6b73c", "#e8885a", "#c9b98f", "#6f8fae", "#2c4a6e", "#18263a"])
CAP = 1e7  # thresholds beyond the largest rivers are shown at the cap


def wlogit(x, y, w, iters=30):
    """Weighted logistic regression of y on [1, x] by Newton's method."""
    b = np.zeros(2)
    X = np.c_[np.ones_like(x), x]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (w * (y - p))
        h = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def surface(xy, larea, old, grid, bw, min_old=25):
    tree = cKDTree(xy)
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        idx = tree.query_ball_point(g, 3 * bw)
        if len(idx) < 200:
            continue
        idx = np.array(idx)
        d = np.hypot(*(xy[idx] - g).T)
        w = np.exp(-0.5 * (d / bw) ** 2)
        if (w * old[idx]).sum() < min_old * 0.3:
            continue
        b0, b1 = wlogit(larea[idx], old[idx], w)
        if b1 <= 0:
            out[i] = np.log10(CAP)
        else:
            out[i] = min(-b0 / b1, np.log10(CAP))
    return out


def us_surface():
    u = pd.read_parquet(DERIVED / "us_analysis.parquet")
    u = u[u.old.notna()]
    to = Transformer.from_crs(4326, 5070, always_xy=True)
    x, y = to.transform(u.lon.to_numpy(), u.lat.to_numpy())
    xy = np.c_[x, y]
    gx, gy = np.meshgrid(np.arange(-2.4e6, 2.3e6, 40_000), np.arange(2.5e5, 3.2e6, 40_000))
    grid = np.c_[gx.ravel(), gy.ravel()]
    z = surface(xy, u.larea.to_numpy(), u.old.to_numpy(), grid, bw=175_000)
    states = gpd.read_file(f"zip://{RAW / 'ne_10m_admin_1_states_provinces.zip'}", columns=["iso_a2", "name"])
    land = states[states.iso_a2.eq("US") & ~states.name.isin(["Alaska", "Hawaii"])].to_crs(5070).union_all()
    near = gpd.GeoSeries(gpd.points_from_xy(grid[:, 0], grid[:, 1]), crs=5070).distance(land).to_numpy() < 80_000
    z[~near] = np.nan
    pd.DataFrame({"x": grid[:, 0], "y": grid[:, 1], "log_x50": z}).to_parquet(DERIVED / "surface_us.parquet")
    return gx, gy, z.reshape(gx.shape), land


def england_surface():
    o = pd.read_parquet(DERIVED / "england_analysis.parquet")
    ek = o[o.source.isin(["ekwall", "ekwall-generic"]) & o.old.notna()]
    xy = np.c_[ek.mid_x, ek.mid_y]
    gx, gy = np.meshgrid(np.arange(130_000, 660_000, 8_000), np.arange(0, 680_000, 8_000))
    grid = np.c_[gx.ravel(), gy.ravel()]
    tree = cKDTree(xy)
    z = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        idx = np.array(tree.query_ball_point(g, 135_000))
        if len(idx) < 60:
            continue
        d = np.hypot(*(xy[idx] - g).T)
        w = np.exp(-0.5 * (d / 45_000) ** 2)
        if w.sum() < 25:
            continue
        b0, b1 = wlogit(ek.larea.to_numpy()[idx], ek.old.to_numpy()[idx], w)
        z[i] = np.log10(CAP) if b1 <= 0 else min(-b0 / b1, np.log10(CAP))
    mu = gpd.read_file(f"zip://{RAW / 'ne_10m_admin_0_map_units.zip'}").to_crs(27700)
    a1 = gpd.read_file(f"zip://{RAW / 'ne_10m_admin_1_states_provinces.zip'}")
    corn = a1[a1.name.isin(["Cornwall", "Isles of Scilly"])].to_crs(27700).union_all()
    land = mu[mu.NAME.eq("England")].union_all().difference(corn)
    near = gpd.GeoSeries(gpd.points_from_xy(grid[:, 0], grid[:, 1]), crs=27700).distance(land).to_numpy() < 30_000
    z[~near] = np.nan
    pd.DataFrame({"x": grid[:, 0], "y": grid[:, 1], "log_x50": z}).to_parquet(DERIVED / "surface_england.parquet")
    return gx, gy, z.reshape(gx.shape), land


def clip_to(geom, ax):
    """A patch following the land outline, used to clip the surface at the coast."""
    verts, codes = [], []
    for g in getattr(geom, "geoms", [geom]):
        for ring in [g.exterior, *g.interiors]:
            xy = np.asarray(ring.coords)
            verts.append(xy)
            codes.append([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
    patch = PathPatch(Path(np.concatenate(verts), np.concatenate(codes)), transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    return patch


def outline(ax, geom, **kw):
    for g in getattr(geom, "geoms", [geom]):
        ax.plot(*g.exterior.xy, **kw)


def draw_rivers(ax, which):
    if which == "us":
        d = pd.read_pickle(DERIVED / "us_map_segments.pkl")
        keep = np.flatnonzero(d["da"] >= 3000)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection([d["segs"][i] for i in keep],
                                         linewidths=0.15 + 0.12 * np.log10(d["da"][keep]), colors="#0a0d12", alpha=0.55, zorder=3))
    else:
        links = gpd.read_parquet(DERIVED / "gb_links.parquet", columns=["up_m", "in_england", "geometry"])
        links = links[links.in_england & (links.up_m > 60_000)]
        links.plot(ax=ax, color="#0a0d12", linewidth=0.15 + 0.25 * np.log10(links.up_m / 1000) - 0.3, alpha=0.6, zorder=3)


def main():
    norm = LogNorm(10, CAP)
    fig = plt.figure(figsize=(16, 7.6), dpi=250, facecolor=BG)
    ax1 = fig.add_axes([0.02, 0.03, 0.62, 0.78])
    ax2 = fig.add_axes([0.66, 0.03, 0.26, 0.78])
    for ax, (gx, gy, z, land), which, title in (
            (ax1, us_surface(), "us", "United States"), (ax2, england_surface(), "england", "England")):
        ax.set_facecolor(BG)
        im = ax.pcolormesh(gx, gy, np.ma.masked_invalid(10 ** z), cmap=CMAP, norm=norm, shading="gouraud", zorder=1)
        im.set_clip_path(clip_to(land, ax))
        outline(ax, land, color=SUBTLE, lw=0.5, zorder=4)
        draw_rivers(ax, which)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.0, 1.0, title, transform=ax.transAxes, color=TEXT, fontsize=13, va="top")
    ax1.set_xlim(-2.4e6, 2.3e6)
    ax1.set_ylim(2.5e5, 3.2e6)
    ax2.set_xlim(130_000, 660_000)
    ax2.set_ylim(0, 680_000)
    cax = fig.add_axes([0.935, 0.2, 0.012, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([10, 100, 1e3, 1e4, 1e5, 1e6, 1e7])
    cb.set_ticklabels(["10 km²", "100", "1,000", "10,000", "100,000", "1,000,000", "never"])
    cb.ax.tick_params(colors=SUBTLE, labelsize=8)
    cb.outline.set_visible(False)
    fig.text(0.02, 0.95, "How big must a river be to keep its old name?", color=TEXT, fontsize=17, fontweight="bold")
    fig.text(0.02, 0.91, "Catchment area at which half of nearby river names are old (local logistic fit). "
             "Amber: even small streams kept old names. Blue: only giant rivers did, or none.",
             color=SUBTLE, fontsize=9.5)
    fig.savefig(FIGURES / "threshold_surface.png", facecolor=BG)
    print("wrote threshold_surface.png")


if __name__ == "__main__":
    main()
