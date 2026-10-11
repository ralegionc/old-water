"""Why do big rivers keep old names? Tests of the mechanism, not just the pattern.

1. United States, contact time. For each river outlet, the year the first
   county was organised there (Newberry Library, Atlas of Historical County
   Boundaries, every county dated to the day since 1629), ignoring nominal
   units (purchases, unorganised territory, counties over 20,000 square miles). If names pass on
   through contact with speakers, earlier organised places should keep old
   names down to smaller rivers, also within each region.
2. England, conquest date. An approximate date of English political control
   for each of Ekwall's county groups, from the standard chronology
   (data/labels/england_conquest.csv). A national gazetteer of early
   Anglo-Saxon cemeteries does not exist in open form; Historic England's
   scheduled monuments list only 94 Anglo-Saxon sites, many of them later
   churches, so the documentary chronology is used instead. Longitude is the
   cross-check.
3. England, density (Ekwall's second explanation), with basin-clustered
   intervals.
4. England, renaming. Ekwall's dictionary entries record replaced names
   ('An earlier name was Bramber'). Is a recorded replacement less likely for
   bigger rivers?

Output: results/mechanisms.txt, results/mechanism_curves.csv,
        data/derived/us_county_year.parquet, data/derived/ekwall_entries.parquet
"""
import glob
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from common import DERIVED, LABELS, RAW, RESULTS
from ekwall import core
from stats import binned, coefs, describe, fit

LINES = []
COUNTY_ABBR = {  # Ekwall's county abbreviations -> approximate centre (lat, lon)
    "Bd": (52.1, -0.45), "Bk": (51.8, -0.8), "Brk": (51.45, -1.25), "C": (52.3, 0.1), "Ch": (53.2, -2.55),
    "Cu": (54.65, -3.05), "D": (50.75, -3.8), "Db": (53.1, -1.6), "Do": (50.8, -2.3), "Du": (54.7, -1.8),
    "Ess": (51.8, 0.55), "Gl": (51.8, -2.2), "Ha": (51.05, -1.3), "He": (52.1, -2.75), "Hrt": (51.85, -0.25),
    "Hu": (52.35, -0.2), "K": (51.2, 0.75), "L": (53.15, -0.3), "La": (53.9, -2.65), "Le": (52.65, -1.1),
    "Mx": (51.55, -0.35), "Nb": (55.2, -2.0), "Nf": (52.65, 0.95), "Np": (52.3, -0.9), "Nt": (53.15, -1.0),
    "O": (51.8, -1.3), "R": (52.65, -0.65), "Sa": (52.6, -2.75), "Sf": (52.2, 1.05), "So": (51.1, -2.95),
    "Sr": (51.25, -0.4), "St": (52.8, -2.0), "Sx": (50.95, -0.35), "W": (51.3, -1.95), "Wa": (52.3, -1.6),
    "We": (54.45, -2.65), "Wo": (52.2, -2.2), "Y": (54.0, -1.3),
}


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def table(name, tab, unit=""):
    say(f"  {name}")
    for t, (b, lo, hi, p) in tab.items():
        say(f"    {t:28} {b:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.2g}{unit}")


# --------------------------------------------------------------------------- US

def us_county_year():
    path = DERIVED / "us_county_year.parquet"
    if path.exists():
        return pd.read_parquet(path)
    shp = glob.glob(str(RAW / "v2" / "newberry" / "**" / "US_HistCounties_Gen001.shp"), recursive=True)[0]
    c = gpd.read_file(shp, columns=["NAME", "STATE_TERR", "START_N", "CNTY_TYPE", "AREA_SQMI"])
    c = c[~c.STATE_TERR.isin(["Alaska", "Hawaii"]) & c.START_N.notna()]
    # The atlas also maps nominal units as counties: 'Louisiana Purchase', 'Unorganized
    # Federal Territory', non-county areas ('MO NCA 1'), and colonial counties that on
    # paper reached the Mississippi. Dropping placeholders and anything over 20,000
    # square miles keeps counties that reflect actual settlement.
    placeholder = c.NAME.str.contains(r"NCA|Unorg|Purchase|Cession|Territory", case=False, regex=True)
    c = c[~placeholder & (c.AREA_SQMI <= 20_000)]
    r = pd.read_parquet(DERIVED / "us_rivers.parquet", columns=["gnis_id", "lon", "lat"])
    pts = gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r.lon, r.lat), crs=4326)
    j = gpd.sjoin(pts, c, predicate="within")
    first = j.groupby("gnis_id").START_N.min()
    out = pd.DataFrame({"gnis_id": first.index, "county_year": (first // 10000).astype(int).to_numpy()})
    out.to_parquet(path, index=False)
    return out


def united_states(rows):
    from analyze import REGIONS
    u = pd.read_parquet(DERIVED / "us_analysis.parquet")
    u = u[u.old.notna()].merge(us_county_year(), on="gnis_id", how="inner")
    u["cent"] = (u.county_year - 1800) / 100  # centuries after 1800
    u["region"] = "other"
    for name, f in REGIONS.items():
        u.loc[f(u.lon, u.lat), "region"] = name
    say("UNITED STATES: contact time (year the first county was organised at the outlet)")
    say(f"  rivers with a county date: {len(u):,}; median year {u.county_year.median():.0f}")
    t, _ = coefs(u, "old ~ larea + cent", cluster="huc4")
    table("old ~ log area + centuries after 1800", t)
    t, _ = coefs(u, "old ~ larea + cent + C(region)", cluster="huc4", terms=["larea", "cent"])
    table("... within regions (region fixed effects)", t)
    t, _ = coefs(u, "old ~ larea * cent", cluster="huc4", terms=["larea", "cent", "larea:cent"])
    table("... with interaction (does the size slope change with contact time?)", t)
    bins = [1600, 1750, 1800, 1840, 1880, 2001]
    labels = ["before 1750", "1750-1799", "1800-1839", "1840-1879", "1880 or later"]
    u["period"] = pd.cut(u.county_year, bins, right=False, labels=labels)
    for lab, g in u.groupby("period", observed=True):
        e = fit(g, x="larea", cluster="huc4")
        say(describe(f"first county {lab}", e, "km2"))
        rows.append(dict(country="US", group=lab, order=labels.index(lab), x50=e["x50"],
                         lo=e["x50_ci"][0], hi=e["x50_ci"][1], slope=e["slope"], n=e["n"]))
        binned(g, "da_km2", [0, 10, 100, 1000, 1e4, 1e7], f"US county {lab}", CURVES)


# ---------------------------------------------------------------------- England

def england_sample():
    o = pd.read_parquet(DERIVED / "england_analysis.parquet")
    ek = o[o.source.isin(["ekwall", "ekwall-generic"]) & o.old.notna()].copy()
    conq = pd.read_csv(LABELS / "england_conquest.csv").set_index("county").year
    ek["conquest"] = ek.county.map(conq)
    return ek


def england(rows):
    from analyze import channel_density
    ek = england_sample()
    ek["west"] = -ek.lon
    say()
    say("ENGLAND: conquest date and longitude (Ekwall labels, basin bootstrap)")
    d = ek.dropna(subset=["conquest"]).copy()
    d["conq_c"] = (d.conquest - 500) / 100  # centuries after 500
    say(f"  rivers with a county conquest date: {len(d):,} (generic-rule names have no county)")
    t1, m1 = coefs(d, "old ~ larea + conq_c", cluster="basin", boot=400)
    table("old ~ log area + centuries of English rule after 500", t1)
    t2, m2 = coefs(d, "old ~ larea + west", cluster="basin", boot=400)
    table("old ~ log area + degrees west", t2)
    t3, _ = coefs(d, "old ~ larea + conq_c + west", cluster="basin", boot=400)
    table("both together", t3)
    say(f"  AIC: conquest date {m1.aic:.1f}, longitude {m2.aic:.1f} (lower fits better)")
    bins = [0, 501, 572, 621, 800]
    labels = ["by 500", "520-571", "577-620", "640-700"]
    d["period"] = pd.cut(d.conquest, bins, right=False, labels=labels)
    for lab, g in d.groupby("period", observed=True):
        e = fit(g, x="larea", cluster="basin", boot=300)
        say(describe(f"English rule from {lab}", e, "km2"))
        rows.append(dict(country="England", group=lab, order=labels.index(lab), x50=e["x50"],
                         lo=e["x50_ci"][0], hi=e["x50_ci"][1], slope=e["slope"], n=e["n"]))
        binned(g, "area_km2", [0, 10, 30, 100, 300, 1000, 1e5], f"England conquest {lab}", CURVES)

    ek["ld"] = np.log(channel_density(ek))
    say()
    say("ENGLAND: Ekwall's density explanation (coefficient on log channel density, basin bootstrap)")
    for name, dd in [("all", ek), ("Scandinavian names dropped", ek[ek.category.ne("scand")]),
                     ("south of 53.5N (Ekwall's own comparison)", ek[ek.lat < 53.5])]:
        t, _ = coefs(dd, "old ~ larea + west + lat + ld", cluster="basin", boot=400, terms=["ld"])
        b, lo, hi, p = t["ld"]
        say(f"    {name:42} {b:+.2f} [{lo:+.2f}, {hi:+.2f}]  bootstrap two-sided p~{p:.2g}")


# ----------------------------------------------------------- Ekwall dictionary

ABBR = "|".join(sorted(COUNTY_ABBR, key=len, reverse=True))
HEAD = re.compile(rf"^([A-Z][a-z'’\-]+(?: [A-Z][a-z'’\-]+)*(?: or [A-Za-z' ]+)?) ((?:(?:{ABBR}), )*(?:{ABBR}))\s*$")
MILES = re.compile(r"\b(?:c\. ?)?(\d{1,3})(?:\s*\|?\s*)m\.")
RENAMED = re.compile(r"\b(old|earlier|former|older|original) name (?:was|seems to have been|may have been|is|of the river was)"
                     r"|formerly called|was formerly|an earlier name", re.I)
NOT_YEAR = re.compile(r"^(BCS|KCD|Ass|m|f|p|ib\.?|pp?\.)$")


def parse_dictionary():
    text = (RAW / "ekwall_1928_djvu.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
    start = next(i for i, l in enumerate(text) if i > 6500 and HEAD.match(l.strip()))
    entries, cur = [], None
    for line in text[start:]:
        m = HEAD.match(line.strip())
        if m:
            if cur:
                entries.append(cur)
            cur = dict(name=m.group(1).split(" or ")[0].strip(), counties=m.group(2).split(", "), body=[])
        elif cur is not None:
            cur["body"].append(line)
    if cur:
        entries.append(cur)
    out = []
    for e in entries:
        body = " ".join(e["body"])
        head = " ".join(e["body"][:6])
        miles = MILES.search(head) if "Rises" in head or "rises" in head or "flows" in head else None
        years = []
        for line in e["body"]:
            toks = line.replace("(", " ( ").replace(")", " ) ").replace(";", " ; ").split()
            for a, b in zip([""] + toks, toks):
                if b.isdigit() and 600 <= int(b) <= 1600 and not NOT_YEAR.match(a):
                    years.append(int(b))
        out.append(dict(name=e["name"], counties=",".join(e["counties"]),
                        miles=float(miles.group(1)) if miles else np.nan,
                        first_year=min(years) if years else np.nan,
                        renamed=bool(RENAMED.search(body)),
                        renamed_text=RENAMED.search(body).group(0) if RENAMED.search(body) else ""))
    return pd.DataFrame(out)


def renaming(rows):
    ent = parse_dictionary()
    rivers = pd.read_parquet(DERIVED / "england_analysis.parquet")
    rivers["core"] = rivers.name.map(core)
    by_core = rivers.groupby("core").indices
    to_ll = Transformer.from_crs(4326, 27700, always_xy=True)
    matched = []
    for e in ent.itertuples():
        cands = by_core.get(core(e.name), [])
        if not len(cands):
            continue
        sub = rivers.iloc[cands]
        best = None
        for c in e.counties.split(","):
            lat, lon = COUNTY_ABBR[c]
            x, y = to_ll.transform(lon, lat)
            dist = np.hypot(sub.mid_x - x, sub.mid_y - y) / 1000
            if dist.min() <= 70 and (best is None or dist.min() < best[1]):
                best = (sub.iloc[int(np.argmin(dist.to_numpy()))], dist.min())
        if best is not None:
            r = best[0]
            matched.append(dict(**e._asdict(), river_id=r.river_id, larea=r.larea, area_km2=r.area_km2,
                                basin=r.basin, old=r.old))
    m = pd.DataFrame(matched).drop(columns="Index").drop_duplicates("river_id")
    m["renamed"] = m.renamed.astype(float)
    m.to_parquet(DERIVED / "ekwall_entries.parquet", index=False)
    say()
    say("ENGLAND: Ekwall's dictionary entries")
    say(f"  headword entries parsed: {len(ent):,}; matched to OS rivers: {len(m):,}; "
        f"with a length in miles: {m.miles.notna().sum():,}; with an attested year: {m.first_year.notna().sum():,}")
    ok = m.dropna(subset=["miles", "larea"])
    say(f"  Ekwall's river length vs HydroRIVERS area: log-log r = {np.corrcoef(np.log10(ok.miles), ok.larea)[0, 1]:.2f} "
        f"(n={len(ok)}; an independent check on the size measure)")
    say(f"  entries recording a replaced name: {m.renamed.sum():,} of {len(m):,}")
    t, _ = coefs(m, "renamed ~ larea", cluster="basin", boot=400)
    table("renamed ~ log area", t)
    for iv, g in m.groupby(pd.cut(m.area_km2, [0, 30, 100, 300, 1e5]), observed=True):
        say(f"    area {str(iv):18} n={len(g):4}  share with a recorded earlier name {g.renamed.mean():.1%}")
    d = m.dropna(subset=["first_year"])
    t, _ = coefs(d.assign(early=(d.first_year < 1100).astype(float)), "early ~ larea + old", cluster="basin", boot=400)
    table("first recorded before 1100 ~ log area + old name", t)
    say("  examples of recorded replacements: " + "; ".join(
        f"{n} ('{t}')" for n, t in m[m.renamed.eq(1)].sort_values("area_km2", ascending=False)[["name", "renamed_text"]].head(12).to_numpy()))


CURVES = []


def main():
    rows = []
    united_states(rows)
    england(rows)
    renaming(rows)
    pd.DataFrame(rows).to_csv(RESULTS / "mechanism_thresholds.csv", index=False)
    pd.DataFrame(CURVES).to_csv(RESULTS / "mechanism_curves.csv", index=False)
    (RESULTS / "mechanisms.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
