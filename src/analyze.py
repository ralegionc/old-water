"""Do big rivers keep old names? Retention curves in four countries.

For each river: old = 1 if its name predates the newcomer language (pre-English
in England, indigenous in the US, Aboriginal in Australia, Maori in New
Zealand), 0 if it is a newcomer name. Size is catchment area: HydroRIVERS in
England, Australia and New Zealand (snap.py), NHDPlus drainage area in the US.

Every interval accounts for clustering: rivers in one basin were often named
by the same people. England uses a basin bootstrap; the US, Australia and New
Zealand use cluster-robust errors by HUC4 subregion or HydroRIVERS basin.
Naive intervals are printed alongside.

Output: results/summary.txt, results/curves.csv,
        data/derived/england_analysis.parquet, us_analysis.parquet
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from common import DERIVED, LABELS, RESULTS
from stats import binned, describe, fit

LINES = []
NRFA = (0.653, 0.799)  # log10 area = a + b log10 upstream_km (validate_size.py)
REGIONS = {  # outlet longitude/latitude boxes, coarse on purpose
    "Colonial East (Atlantic slope)": lambda lon, lat: (lon > -80.5) & (lat > 31),
    "Southeast interior": lambda lon, lat: (lon <= -80.5) & (lon > -91) & (lat <= 37),
    "Ohio-Great Lakes-Upper Mississippi": lambda lon, lat: (lon <= -80.5) & (lon > -97) & (lat > 37),
    "Plains": lambda lon, lat: (lon <= -97) & (lon > -104),
    "Spanish Southwest": lambda lon, lat: (lon <= -104) & (lat <= 37.5) | ((lon <= -97) & (lon > -104) & (lat <= 31)),
    "Mountain and Pacific West": lambda lon, lat: (lon <= -104) & (lat > 37.5),
}
EDGES = [0, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 1e7]


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def channel_density(eng, radius_m=10_000):
    links = gpd.read_parquet(DERIVED / "gb_links.parquet", columns=["is_canal", "length", "geometry"])
    links = links[~links.is_canal]
    mids = gpd.GeoDataFrame(links[["length"]], geometry=links.geometry.interpolate(0.5, normalized=True),
                            crs=27700)
    pts = gpd.GeoDataFrame(eng[["river_id"]], geometry=gpd.points_from_xy(eng.mid_x, eng.mid_y), crs=27700)
    j = gpd.sjoin(pts.assign(geometry=pts.buffer(radius_m)), mids, predicate="contains")
    km = j.groupby("river_id")["length"].sum() / 1000
    return eng.river_id.map(km).fillna(0) / (np.pi * (radius_m / 1000) ** 2)


def england(rows):
    o = pd.read_parquet(DERIVED / "england_origin.parquet")
    o = o.merge(pd.read_parquet(DERIVED / "england_area.parquet")[["river_id", "area_km2", "area_source"]],
                on="river_id", how="left")
    o = o[(o.upstream_km > 0) & (o.area_km2 > 0)].copy()
    o["lsize"] = np.log10(o.upstream_km)
    o["larea"] = np.log10(o.area_km2)
    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    o["lon"], o["lat"] = to_ll.transform(o.mid_x.to_numpy(), o.mid_y.to_numpy())
    o.to_parquet(DERIVED / "england_analysis.parquet", index=False)
    say("ENGLAND (size = catchment area; intervals from a basin bootstrap)")
    ek = o[o.source.isin(["ekwall", "ekwall-generic"]) & o.old.notna()].copy()
    e = fit(ek, x="larea", cluster="basin", boot=500)
    say(describe("Ekwall labels (county lists + generic rule)", e, "km2"))
    rows.append(dict(country="England", curve="all", **{k: e[k] for k in ("n", "slope", "x50")},
                     lo=e["x50_ci"][0], hi=e["x50_ci"][1]))
    say(describe("Ekwall county lists only", fit(ek[ek.source.eq("ekwall")], x="larea", cluster="basin", boot=500), "km2"))
    say(describe("size as upstream channel length instead", fit(ek, x="lsize", cluster="basin", boot=500), "km"))
    binned(ek, "area_km2", EDGES, "England, Ekwall", CURVES)
    allr = o[o.source.ne("no specific")].copy()
    for bound, fill in (("lower", 0.0), ("upper", 1.0)):
        a = allr.assign(old=allr.old.fillna(fill))
        say(describe(f"all {len(a):,} named rivers, opaque counted as {'newcomer' if fill == 0 else 'old'} ({bound} bound)",
                     fit(a, x="larea", cluster="basin", boot=200), "km2"))
    zones = pd.cut(ek.lon, [-6, -2.5, -1.25, 2], labels=["west (<2.5W)", "middle", "east (>1.25W)"])
    for z, g in ek.groupby(zones, observed=True):
        e = fit(g, x="larea", cluster="basin", boot=400)
        say(describe(f"zone {z}", e, "km2"))
        binned(g, "area_km2", [0, 10, 30, 100, 300, 1000, 1e5], f"England {z}", CURVES)
    return ek


def united_states(rows):
    u = pd.read_parquet(DERIVED / "us_origin.parquet")
    u = u[u.origin.ne("no specific") & (u.da_km2 > 0)].copy()
    u["larea"] = np.log10(u.da_km2)
    u["lsize"] = np.log10(u.upstream_km.clip(lower=0.1))
    u.to_parquet(DERIVED / "us_analysis.parquet", index=False)
    say()
    say("UNITED STATES (size = NHDPlus drainage area; intervals clustered by HUC4 subregion)")
    us = u[u.old.notna()].copy()
    e = fit(us, x="larea", cluster="huc4")
    say(describe("all named streams", e, "km2"))
    say(f"    clustering widens the slope's standard error {e['se_ratio']:.1f}x")
    rows.append(dict(country="United States", curve="all", **{k: e[k] for k in ("n", "slope", "x50")},
                     lo=e["x50_ci"][0], hi=e["x50_ci"][1]))
    noeth = us.assign(old=np.where(us.ethnonym_word | us.origin.eq("transferred"), 0.0, us.old))
    say(f"  tribe names among old names: {(us.ethnonym_word & us.old.eq(1)).sum():,}; "
        f"state-name transfers: {us.origin.eq('transferred').sum():,}")
    variants = [
        ("tribe names and state-name transfers counted as newcomer", noeth),
        ("model-labelled names dropped (list and tribe names only)", us[us.origin.ne("native (model)")]),
        ("list ignored: every name judged by the character model", us.assign(
            old=np.where(us.origin.isin(["native (list)", "native (model)"]), (us.p_native >= 0.5).astype(float), us.old))),
        ("opaque counted as newcomer (lower bound)", u.assign(old=u.old.fillna(0.0))),
        ("opaque counted as indigenous (upper bound)", u.assign(old=u.old.fillna(1.0))),
    ]
    for name, d in variants:
        say(describe(name, fit(d, x="larea", cluster="huc4"), "km2"))
    binned(us, "da_km2", EDGES, "US", CURVES)
    say("  by region:")
    for name, f in REGIONS.items():
        g = us[f(us.lon, us.lat)]
        e = fit(g, x="larea", cluster="huc4")
        say(describe(name, e, "km2"))
        binned(g, "da_km2", [0, 10, 100, 1000, 1e4, 1e7], f"US {name}", CURVES)
        top = g.sort_values("da_km2", ascending=False).head(12)
        say("      biggest: " + ", ".join(f"{n} ({'old' if o == 1 else 'new'})" for n, o in zip(top.name, top.old)))
    return us


def australasia(rows):
    for country, file in (("Australia", "australia_origin.parquet"), ("New Zealand", "nz_origin.parquet")):
        if not (DERIVED / file).exists():
            say(f"{country}: not built yet (run australasia.py)")
            continue
        r = pd.read_parquet(DERIVED / file)
        r["basin"] = r.basin.astype("Int64").astype(str)
        say()
        say(f"{country.upper()} (size = HydroRIVERS area; intervals clustered by HydroRIVERS basin)")
        if country == "Australia":
            # the 150 largest rivers were checked by hand (data/labels/au_top150_labels.csv)
            top = pd.read_csv(LABELS / "au_top150_sample.csv").reset_index()
            top["rank"] = top["index"] + 1
            top = top.merge(pd.read_csv(LABELS / "au_top150_labels.csv")[["rank", "truth"]], on="rank")
            hand = top.set_index("group").truth.map({"old": 1.0, "new": 0.0})
            r0 = r[r.old.notna() & (r.area > 0)].assign(larea=lambda d: np.log10(d.area))
            say(describe("classifier labels only", fit(r0, x="larea", cluster="basin"), "km2"))
            in_top = r.group.isin(hand.index)
            r.loc[in_top, "old"] = r.loc[in_top, "group"].map(hand)
            say(f"  hand labels applied to the {in_top.sum()} largest rivers ({hand.isna().sum()} left unclear)")
        r = r[r.old.notna() & (r.area > 0)].copy()
        r["larea"] = np.log10(r.area)
        AUSTRALASIA[country] = r
        e = fit(r, x="larea", cluster="basin")
        say(describe("named rivers" + (" (largest 150 hand-checked)" if country == "Australia" else ""), e, "km2"))
        rows.append(dict(country=country, curve="all", **{k: e[k] for k in ("n", "slope", "x50")},
                         lo=e["x50_ci"][0], hi=e["x50_ci"][1]))
        binned(r, "area", EDGES, country, CURVES)
        for iv, g in r.groupby(pd.cut(r.area, [0, 100, 1000, 1e4, 1e7]), observed=True):
            say(f"    area {str(iv):22} n={len(g):6,}  old {g.old.mean():.1%}")
        top = r.sort_values("area", ascending=False).head(15)
        say("      biggest: " + ", ".join(f"{n} ({'old' if o == 1 else 'new'})" for n, o in zip(top.name, top.old)))


def large_rivers():
    """The explorer-naming prediction: no size gradient among Australia's large rivers."""
    import warnings
    warnings.simplefilter("ignore")
    o = pd.read_parquet(DERIVED / "england_analysis.parquet")
    data = [("England", o[o.source.isin(["ekwall", "ekwall-generic"]) & o.old.notna()], "basin", 300)]
    us = pd.read_parquet(DERIVED / "us_analysis.parquet")
    us = us[us.old.notna()]
    data += [("United States", us, "huc4", 0),
             ("  Colonial East", us[REGIONS["Colonial East (Atlantic slope)"](us.lon, us.lat)], "huc4", 0),
             ("  Mountain and Pacific West", us[REGIONS["Mountain and Pacific West"](us.lon, us.lat)], "huc4", 0)]
    for country, d in AUSTRALASIA.items():
        data.append((country, d, "basin", 0))
    say()
    say("LARGE RIVERS ONLY (odds ratio per tenfold area among rivers above the cut)")
    for name, d, cl, boot in data:
        for cut in (300, 1000):
            g = d[d.larea >= np.log10(cut)]
            e = fit(g, x="larea", cluster=cl, boot=boot)
            lo, hi = np.exp(e["slope_ci"])
            say(f"  {name:28} above {cut:5,} km2: n={len(g):6,}, old {g.old.mean():4.0%}, "
                f"odds ratio {np.exp(e['slope']):.2f} [{lo:.2f}, {hi:.2f}]")


AUSTRALASIA = {}
CURVES = []


def main():
    rows = []
    england(rows)
    united_states(rows)
    australasia(rows)
    large_rivers()
    say()
    say("COMPARISON (odds ratio per tenfold catchment area; size at which half of names are old)")
    for r in rows:
        say(f"  {r['country']:14} n={r['n']:7,}  odds ratio {np.exp(r['slope']):.2f}  "
            f"half old at {10 ** r['x50']:,.0f} km2 [{10 ** r['lo']:,.0f}, {10 ** r['hi']:,.0f}]")
    pd.DataFrame(rows).to_csv(RESULTS / "countries.csv", index=False)
    pd.DataFrame(CURVES).to_csv(RESULTS / "curves.csv", index=False)
    (RESULTS / "summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
