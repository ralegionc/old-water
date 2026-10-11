"""Hero maps: every river drawn by size, coloured by the age of its name.

Old names (pre-English in England, indigenous in the US) glow amber; newcomer
names are a cool blue; names the classifier could not read are grey; unnamed
channels form a faint lace underneath.

Output: figures/england_map.png, figures/us_map.png
"""
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import shapely
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import statsmodels.formula.api as smf

from common import DERIVED, FIGURES, RAW, RESULTS

BG = "#0a0d12"
OLD, NEW, UNK, LACE = "#f6b73c", "#4f86b8", "#7b8290", "#2b333f"
TEXT, SUBTLE = "#e8e4da", "#8a8f98"
NEW_LABEL = "#a9cdee"  # lighter blue so labels read against blue rivers
FONT = "DejaVu Sans"


def segments(geoms):
    """Shapely lines -> list of (n, 2) coordinate arrays, one per part."""
    parts = shapely.get_parts(np.asarray(geoms))
    coords, idx = shapely.get_coordinates(parts, return_index=True)
    cut = np.flatnonzero(np.diff(idx)) + 1
    return np.split(coords, cut), np.repeat(np.arange(len(geoms)), shapely.get_num_geometries(np.asarray(geoms)))


def draw(ax, segs, widths, color, alpha=1.0, glow=False, z=1):
    if not len(segs):
        return
    if glow:
        ax.add_collection(LineCollection(segs, linewidths=np.asarray(widths) * 3.5, colors=color,
                                         alpha=0.05, capstyle="round", zorder=z))
        ax.add_collection(LineCollection(segs, linewidths=np.asarray(widths) * 1.8, colors=color,
                                         alpha=0.14, capstyle="round", zorder=z + 0.1))
    ax.add_collection(LineCollection(segs, linewidths=widths, colors=color, alpha=alpha,
                                     capstyle="round", zorder=z + 0.2))


def legend(ax, x, y, labels, size=9):
    handles = [Line2D([], [], color=c, lw=lw) for c, lw, _ in labels]
    leg = ax.legend(handles, [t for *_, t in labels], loc="lower left", bbox_to_anchor=(x, y),
                    frameon=False, fontsize=size, handlelength=2.2, labelcolor=TEXT)
    return leg


ENGLAND_LABELS = ["Thames", "Severn", "Trent", "Tyne", "Tees", "Eden", "Exe", "Tamar", "Wye",
                  "Mersey", "Arun", "Waveney", "Medway", "Great Ouse", "Ribble", "Swale", "Avon"]
ZONES = [("England west (<2.5W)", "west of 2.5°W", OLD), ("England middle", "middle", "#c9b98f"),
         ("England east (>1.25W)", "east of 1.25°W", NEW)]


def label_rivers(ax, links, o):
    halo = [pe.withStroke(linewidth=2.2, foreground=BG)]
    for name in ENGLAND_LABELS:
        cand = o[o.name.eq(f"River {name}")].sort_values("upstream_km", ascending=False)
        if cand.empty:
            continue
        r = cand.iloc[0]
        geom = links.loc[links.river_id.eq(r.river_id), "geometry"].union_all()
        line = shapely.line_merge(geom)
        if line.geom_type == "MultiLineString":
            line = max(line.geoms, key=lambda g: g.length)
        pt = line.interpolate(0.45, normalized=True)
        col = OLD if r.old == 1 else NEW_LABEL if r.old == 0 else UNK
        ax.annotate(name, (pt.x, pt.y), xytext=(4, 4), textcoords="offset points", fontsize=6.5,
                    style="italic", color=col, path_effects=halo, zorder=10)


def zone_inset(fig, rect):
    """Share of pre-English names against size, west vs east (Ekwall labels)."""
    curves = pd.read_csv(RESULTS / "curves.csv")
    o = pd.read_parquet(DERIVED / "england_analysis.parquet")
    ek = o[o.source.isin(["ekwall", "ekwall-generic"]) & o.old.notna()]
    zone = pd.cut(ek.lon, [-6, -2.5, -1.25, 2], labels=[z for z, *_ in ZONES])
    ax = fig.add_axes(rect, facecolor="none")
    xs = np.linspace(0, 4.2, 100)
    for key, text, col in ZONES:
        c = curves[curves.curve.eq(key)]
        g = ek[zone.eq(key)]
        m = smf.logit("old ~ larea", g).fit(disp=0)
        ax.plot(10 ** xs, m.predict(pd.DataFrame({"larea": xs})), color=col, lw=1.6, label=text)
        ax.scatter(c["mid"], c["share"], s=np.clip(c["n"] / 3, 6, 40), color=col, alpha=0.85, lw=0, zorder=3)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", labelcolor=[col for *_, col in ZONES],
              handlelength=1.2, borderaxespad=0.2)
    ax.set_xscale("log")
    ax.set_xlim(1, 10 ** 4.2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "50%", "100%"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(SUBTLE)
    ax.tick_params(colors=SUBTLE, labelsize=6.5)
    ax.set_xlabel("catchment area, km²", color=SUBTLE, fontsize=7)
    ax.set_title("Share of names that are pre-English", color=TEXT, fontsize=8, loc="left")
    return ax


def england_map():
    links = gpd.read_parquet(DERIVED / "gb_links.parquet")
    o_full = pd.read_parquet(DERIVED / "england_origin.parquet")
    o = o_full[["river_id", "old", "source"]]
    links = links[~links.is_canal]
    links = links.merge(o, on="river_id", how="left")
    eng = links.in_england | links.river_id.isin(o.river_id)
    up = links.up_m.to_numpy() / 1000
    w = 0.12 + 0.55 * np.log10(1 + up) ** 1.15

    fig, ax = plt.subplots(figsize=(9, 11.5), dpi=300, facecolor=BG)
    ax.set_facecolor(BG)
    ctx = ~eng  # Wales and Scotland as context
    draw(ax, segments(links.geometry[ctx])[0], w[ctx] * 0.6, LACE, alpha=0.5, z=0)
    for mask, col, kw in [
        (eng & links.river_id.isna(), LACE, dict(alpha=0.9)),
        (eng & links.river_id.notna() & links.old.isna(), UNK, dict(alpha=0.55)),
        (eng & links.old.eq(0), NEW, dict(alpha=0.8)),
        (eng & links.old.eq(1), OLD, dict(glow=True)),
    ]:
        segs, part_of = segments(links.geometry[mask])
        draw(ax, segs, w[mask][part_of], col, z=2, **kw)
    ax.set_xlim(130_000, 660_000)
    ax.set_ylim(0, 680_000)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.07, 0.955, "Old Water", color=TEXT, fontsize=26, fontweight="bold", family=FONT)
    fig.text(0.07, 0.928, "England's rivers, coloured by the age of their names", color=SUBTLE, fontsize=11.5)
    label_rivers(ax, links, o_full)
    legend(ax, 0.70, 0.64, [(OLD, 2.5, "pre-English name\n(British or older)"),
                            (NEW, 2.0, "English or\nScandinavian name"),
                            (UNK, 1.5, "not classified"),
                            (LACE, 1.2, "unnamed channel")], size=8)
    zone_inset(fig, [0.64, 0.07, 0.25, 0.12])
    fig.text(0.07, 0.025, "Width: upstream channel length. Labels: Ekwall, English River-Names (1928), "
             "extended by a transparency test. Rivers: OS Open Rivers.", color=SUBTLE, fontsize=6.5)
    fig.savefig(FIGURES / "england_map.png", facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote england_map.png")


US_LABELS = ["Mississippi River", "Missouri River", "Ohio River", "Columbia River", "Colorado River",
             "Rio Grande", "Arkansas River", "Snake River", "Platte River", "Yellowstone River",
             "Tennessee River", "Susquehanna River", "Sacramento River", "Potomac River", "Hudson River",
             "Connecticut River", "Gila River", "Pecos River", "Red River", "Chattahoochee River",
             "Kansas River", "Willamette River", "Delaware River", "Illinois River"]
US_REGIONS = [("Colonial East (Atlantic slope)", "Colonial East", OLD),
              ("Ohio-Great Lakes-Upper Mississippi", "Ohio and Great Lakes", "#c9b98f"),
              ("Mountain and Pacific West", "Mountain and Pacific West", NEW)]


def us_segments(chunk=400_000):
    """Read, simplify and classify NHDPlus flowlines once; cache the result."""
    cache = DERIVED / "us_map_segments.pkl"
    if cache.exists():
        return classify_segments(pd.read_pickle(cache))
    atts = pd.read_parquet(RAW / "enhd_nhdplusatts.parquet",
                           columns=["comid", "gnis_id", "gnis_name", "streamorde", "totdasqkm", "ftype"])
    atts = atts[atts.ftype.isin(["StreamRiver", "ArtificialPath", "Connector"])]
    origin = pd.read_parquet(DERIVED / "us_origin.parquet", columns=["gnis_id", "name", "old", "da_km2"])
    targets = (origin[origin.name.isin(US_LABELS)].sort_values("da_km2", ascending=False)
               .drop_duplicates("name").set_index("gnis_id").name)
    atts["named"] = atts.gnis_name.fillna("").str.strip().ne("")
    atts = atts.merge(origin[["gnis_id", "old"]], on="gnis_id", how="left")
    atts.loc[~atts.named, "old"] = np.nan
    atts = atts[atts.named | (atts.streamorde >= 3)]
    atts["comid"] = atts.comid.astype(np.int64)
    atts = atts.set_index("comid")

    layer = pyogrio.list_layers(RAW / "enhd_nhdplus_flowlines.gpkg")[0][0]
    total = pyogrio.read_info(RAW / "enhd_nhdplus_flowlines.gpkg", layer=layer)["features"]
    out = {"segs": [], "da": [], "gnis": [], "named": []}
    label_parts = {gid: [] for gid in targets.index}
    for start in range(0, total, chunk):
        g = pyogrio.read_dataframe(RAW / "enhd_nhdplus_flowlines.gpkg", layer=layer, columns=["comid"],
                                   skip_features=start, max_features=chunk)
        g["comid"] = g.comid.astype(np.int64)
        g = g[g.comid.isin(atts.index)].to_crs(5070)
        g["geometry"] = g.geometry.simplify(250)
        a = atts.loc[g.comid]
        segs, part_of = segments(g.geometry.to_numpy())
        out["segs"].extend(segs)
        out["da"].extend(a.totdasqkm.to_numpy()[part_of])
        out["gnis"].extend(a.gnis_id.fillna("").to_numpy()[part_of])
        out["named"].extend(a.named.to_numpy()[part_of])
        for gid, geom in zip(a.gnis_id.to_numpy(), g.geometry.to_numpy()):
            if gid in label_parts:
                label_parts[gid].append(geom)
        print(f"  read {min(start + chunk, total):,} of {total:,}", flush=True)
    labels = []
    for gid, parts in label_parts.items():
        if not parts:
            continue
        line = shapely.line_merge(shapely.union_all(parts))
        if line.geom_type == "MultiLineString":
            line = max(line.geoms, key=lambda x: x.length)
        pt = line.interpolate(0.42, normalized=True)
        labels.append((targets[gid].replace(" River", ""), pt.x, pt.y, gid))
    res = dict(segs=out["segs"], da=np.array(out["da"]), gnis=np.array(out["gnis"], dtype=object),
               named=np.array(out["named"]), labels=labels)
    pd.to_pickle(res, cache)
    return classify_segments(res)


def classify_segments(res):
    """Colour classes and labels from the current classification and the cached segments."""
    origin = pd.read_parquet(DERIVED / "us_origin.parquet", columns=["gnis_id", "name", "old", "da_km2"])
    old = origin.set_index("gnis_id").old
    o = pd.Series(res["gnis"]).map(old).to_numpy()
    res["cls"] = np.where(~res["named"], "lace", np.where(o == 1, "old", np.where(o == 0, "new", "unk")))
    targets = (origin[origin.name.isin(US_LABELS)].sort_values("da_km2", ascending=False)
               .drop_duplicates("name").set_index("gnis_id").name)
    gnis = pd.Series(res["gnis"])
    labels = []
    for gid, name in targets.items():
        idx = np.flatnonzero(gnis.eq(gid).to_numpy())
        if not len(idx):
            continue
        line = shapely.line_merge(shapely.MultiLineString([res["segs"][i] for i in idx if len(res["segs"][i]) > 1]))
        if line.geom_type == "MultiLineString":
            line = max(line.geoms, key=lambda x: x.length)
        pt = line.interpolate(0.42, normalized=True)
        labels.append((name.replace(" River", ""), pt.x, pt.y, old.get(gid, np.nan)))
    res["labels"] = labels
    return res


def us_inset(fig, rect):
    us = pd.read_parquet(DERIVED / "us_analysis.parquet")
    us = us[us.old.notna()]
    from analyze import REGIONS
    ax = fig.add_axes(rect, facecolor="none")
    xs = np.linspace(0, 6.4, 200)
    for key, text, col in US_REGIONS:
        g = us[REGIONS[key](us.lon, us.lat)]
        m = smf.logit("old ~ larea", g).fit(disp=0)
        ax.plot(10 ** xs, m.predict(pd.DataFrame({"larea": xs})), color=col, lw=1.6, label=text)
    ax.set_xscale("log")
    ax.set_xlim(1, 10 ** 6.4)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "50%", "100%"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(SUBTLE)
    ax.tick_params(colors=SUBTLE, labelsize=7)
    ax.set_xlabel("drainage area, km²", color=SUBTLE, fontsize=7.5)
    ax.set_title("Share of names that are indigenous", color=TEXT, fontsize=8.5, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left", labelcolor=[c for *_, c in US_REGIONS],
              handlelength=1.2, borderaxespad=0.2)


def us_map():
    d = us_segments()
    w = 0.04 + 0.22 * np.log10(1 + d["da"]) ** 1.25
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300, facecolor=BG)
    ax.set_facecolor(BG)
    for k, col, kw in [("lace", LACE, dict(alpha=0.7, z=0)), ("unk", UNK, dict(alpha=0.5, z=1)),
                       ("new", NEW, dict(alpha=0.75, z=2)), ("old", OLD, dict(glow=True, z=3))]:
        m = d["cls"] == k
        segs = [s for s, keep in zip(d["segs"], m) if keep]
        draw(ax, segs, w[m] * (0.8 if k == "lace" else 1), col, **kw)
    halo = [pe.withStroke(linewidth=2.4, foreground=BG)]
    for name, x, y, old in d["labels"]:
        col = OLD if old == 1 else NEW_LABEL if old == 0 else UNK
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7.5,
                    style="italic", color=col, path_effects=halo, zorder=10)
    ax.set_xlim(-2_400_000, 2_300_000)
    ax.set_ylim(250_000, 3_200_000)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.06, 0.93, "Old Water", color=TEXT, fontsize=30, fontweight="bold", family=FONT)
    fig.text(0.06, 0.902, "Rivers of the United States, coloured by the age of their names",
             color=SUBTLE, fontsize=13)
    legend(ax, 0.0, 0.02, [(OLD, 2.5, "indigenous name"), (NEW, 2.0, "English, French, Spanish, Dutch or personal name"),
                           (UNK, 1.5, "name not classified"), (LACE, 1.2, "unnamed channel")], size=10)
    us_inset(fig, [0.845, 0.30, 0.12, 0.15])
    fig.text(0.06, 0.04, "Width: drainage area. Rivers: USGS NHDPlus V2 (E2NHDPlusV2). Names classified with "
             "Wikipedia's lists of Native-origin place names, word lists and a character model.",
             color=SUBTLE, fontsize=7.5)
    fig.savefig(FIGURES / "us_map.png", facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote us_map.png")


def australasia_map():
    """Australia and New Zealand side by side, named rivers only (no unnamed lace)."""
    fig = plt.figure(figsize=(16, 9.5), dpi=300, facecolor=BG)
    panels = [("australia_geom.parquet", 3577, [0.02, 0.06, 0.6, 0.8], (-2.0e6, 2.15e6), (-4.95e6, -1.0e6),
               "Australia: Aboriginal names"),
              ("nz_geom.parquet", 2193, [0.62, 0.06, 0.36, 0.8], (1.05e6, 2.12e6), (4.75e6, 6.2e6),
               "New Zealand: Maori names")]
    for file, crs, rect, xl, yl, title in panels:
        ax = fig.add_axes(rect)
        ax.set_facecolor(BG)
        g = gpd.read_parquet(DERIVED / file).to_crs(crs)
        # width: the capped catchment; where the cap removed it (pieces of a long river,
        # floodplain channels) the smaller of the snapped area and what the drawn length
        # supports by Hack's law (A = (L / 1.4) ** (1 / 0.6))
        hack = (g.geometry.length.to_numpy() / 1000 / 1.4) ** (1 / 0.6)
        raw = g["raw_area"].fillna(0).to_numpy() if "raw_area" in g else np.zeros(len(g))
        size = np.where(g["area"].notna(), g["area"], np.minimum(raw, hack))
        g = g.assign(size=np.maximum(size, 1)).sort_values("size")
        w = 0.05 + 0.2 * np.log10(1 + g["size"].to_numpy()) ** 1.25
        for cls, col, kw in [(None, UNK, dict(alpha=0.55)), (0.0, NEW, dict(alpha=0.9)), (1.0, OLD, dict(glow=True))]:
            m = g.old.isna().to_numpy() if cls is None else g.old.eq(cls).to_numpy()
            segs, part_of = segments(g.geometry.to_numpy()[m])
            draw(ax, segs, w[m][part_of], col, z=2, **kw)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color=TEXT, fontsize=12, loc="left")
    fig.text(0.03, 0.95, "Old Water", color=TEXT, fontsize=28, fontweight="bold", family=FONT)
    fig.text(0.03, 0.915, "Australia and New Zealand: named rivers, coloured by the age of their names",
             color=SUBTLE, fontsize=12)
    leg_ax = fig.axes[0]
    legend(leg_ax, 0.0, 0.0, [(OLD, 2.5, "indigenous name"), (NEW, 2.0, "English or personal name"),
                              (UNK, 1.5, "not classified")], size=9.5)
    fig.text(0.03, 0.025, "Width: catchment area (HydroRIVERS). Rivers: Geoscience Australia; OpenStreetMap contributors "
             "(LINZ). Names: Wikipedia list of Aboriginal place names, NZ Gazetteer, word lists and a character model.",
             color=SUBTLE, fontsize=7.5)
    fig.savefig(FIGURES / "australasia_map.png", facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote australasia_map.png")


if __name__ == "__main__":
    import sys
    if not {"us", "australasia"} & set(sys.argv[1:]):
        england_map()
    if "england" not in sys.argv[1:] and "australasia" not in sys.argv[1:]:
        us_map()
    if not {"us", "england"} & set(sys.argv[1:]) or "australasia" in sys.argv[1:]:
        australasia_map()
