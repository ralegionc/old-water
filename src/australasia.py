"""Australia and New Zealand: the cases that could break the pattern.

If old names survive where newcomers learned them from speakers, colonies
mapped by expedition should not show the size gradient. In Australia the
trunk rivers were named by explorers after patrons and officials (Murray,
Darling, Lachlan, Macquarie, Fitzroy), so the prediction, stated before
looking, is a flat or reversed curve. New Zealand, settled through decades of
dealings with Maori communities, should keep Maori names widely.

Rivers:
  Australia: Geoscience Australia Surface Hydrology Lines (National), named
             watercourses grouped by name and shared endpoints
  New Zealand: OpenStreetMap named waterways (largely LINZ imports)
Size: HydroRIVERS upstream area. Each river piece is snapped to HydroRIVERS at
  three interior points; a piece counts only when two of the three agree on a
  reach, so a tributary does not inherit its parent's area at the confluence.
  The river's size is the largest matched area; its basin is that reach's
  MAIN_RIV, the clustering unit.
Names:
  Australia: Wikipedia's list of place names of Aboriginal origin, English and
             personal-name readings, Aboriginal loanwords counted as English
             (Coolabah, Brigalow), and a character model for the rest
  New Zealand: the NZ Gazetteer's official Maori-name flag where the name is
             listed, otherwise Maori spelling rules. Dual names restored in
             recent decades ('Clutha River / Mata-Au') are read by their first,
             historical part.

Output: data/derived/australia_origin.parquet, nz_origin.parquet,
        results/australasia_report.txt
"""
import json
import re
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

from classify import native_model
from common import DERIVED, RAW, RESULTS, fold, split_name
from lexicon import OUT as LEXDIR, load
from snap import group_rivers, hydrorivers, snap_area
from transparency import transparent

V2 = RAW / "v2"
REPORT = []
MAORI = re.compile(r"^((ng|wh|[hkmnprtw])?[aeiou])+$")
SURNAME_FORM = re.compile(r"^(mac|mc|fitz)[a-z]{3,}$")


def say(*lines):
    for line in lines:
        print(line, flush=True)
        REPORT.append(line)


def rivers_table(lines, hydro):
    lines = lines.reset_index(drop=True)
    sp = lines["name"].map(split_name)
    lines["key"] = [f"{a}|{b}" for a, b in sp]
    lines["group"] = group_rivers(lines)
    lines["len_km"] = lines.geometry.length / 1000
    lines["hr_area"], lines["basin"] = snap_area(lines, hydro)  # not 'area': GeoDataFrame.area
    # A piece snapped beside a bigger river (an anabranch, or a tributary's last
    # stretch) picks up that river's catchment. Hack's law ties catchment to river
    # length (L ~ 1.4 A^0.6, km and km2); pieces claiming more than 20 times the
    # area the river's own length supports are ignored.
    glen = lines.groupby("group").len_km.transform("sum")
    limit = 20 * (glen / 1.4) ** (1 / 0.6)
    lines["raw_area"] = lines.hr_area
    lines.loc[lines.hr_area > limit, ["hr_area", "basin"]] = np.nan
    mid = shapely.line_interpolate_point(lines.geometry.to_numpy(), 0.5, normalized=True)
    ll = gpd.GeoSeries(mid, crs=lines.crs).to_crs(4326)
    lines["lon"], lines["lat"] = ll.x.to_numpy(), ll.y.to_numpy()
    lines = lines.sort_values("hr_area", ascending=False, na_position="last")
    r = lines.groupby("group", sort=False).agg(name=("name", "first"), area=("hr_area", "max"),
                                               raw_area=("raw_area", "max"), length_km=("len_km", "sum"),
                                               basin=("basin", "first"), lon=("lon", "first"),
                                               lat=("lat", "first"), pieces=("name", "size")).reset_index()
    r["inherits_parent"] = r.raw_area.notna() & r.area.isna()
    say(f"  rivers whose only snapped area belonged to a bigger river (dropped): {r.inherits_parent.sum():,}; "
        f"rivers with some pieces capped: {(r.raw_area > r.area).sum():,}")
    return r, lines


def classify_au(r):
    abo = pd.read_csv(LEXDIR / "aboriginal_wikipedia.csv").specific
    english_loans = load("au_loanwords") | load("force_english")
    drop = load("surnames_very_common") | load("force_english")
    abo_set = {s for s in abo if not set(s.split()) <= drop}
    specs = pd.Series(r.specific.unique())
    tr = {s: transparent(s, "us") for s in specs}
    pos = sorted({s for s in abo_set if tr.get(s, transparent(s, "us"))[0] is None})
    with zipfile.ZipFile(RAW / "lexicon" / "census_surnames_2010.zip") as z:
        sn = pd.read_csv(z.open("Names_2010Census.csv"), keep_default_na=False, na_values=["(S)"])
    rng = np.random.default_rng(0)
    rare = sn[sn["count"].between(300, 3000)].name.str.lower()
    rare = [w for w in rare if w not in abo_set]
    eng = [w for w in rng.choice(sorted(load("english")), 3000, replace=False) if len(w) >= 6]
    neg = sorted(set(rng.choice(rare, 5000, replace=False)) | set(eng))
    model, p_cv, auc, p_all, y = native_model(pos, neg)
    scored = [s for s in specs if s and (tr[s][0] is None or tr[s][2]) and s not in abo_set]
    p = dict(zip(scored, model.predict_proba(scored)[:, 1])) if scored else {}

    def lab(s):
        if not s:
            return "no specific", np.nan
        words = s.split()
        if all(w in english_loans or tr[s][0] for w in words) and any(w in english_loans for w in words):
            return "english (loanword)", 0.0
        cls, _, weak = tr[s]
        if s in abo_set:
            return ("english", 0.0) if cls == "english" and not weak else ("aboriginal (list)", 1.0)
        if cls and not (weak and p.get(s, 0) >= 0.9):
            return cls, 0.0
        if all(SURNAME_FORM.match(w) or w in en_proper or tr.get(w, (None,))[0] for w in words):
            return "personal", 0.0  # Macquarie, Fitzroy, Lachlan: surnames and given names of officials
        return ("aboriginal (model)", 1.0) if p.get(s, 0) >= 0.5 else ("opaque", np.nan)

    en_proper = load("english_proper") | load("given")
    for w in {w for s in specs for w in s.split()}:
        tr.setdefault(w, transparent(w, "us"))
    labs = {s: lab(s) for s in specs}
    r["origin"] = r.specific.map(lambda s: labs[s][0])
    r["old"] = r.specific.map(lambda s: labs[s][1])
    say(f"  Aboriginal answer key: {len(abo_set):,} place-name words; character model AUC {auc:.3f}, "
        f"recall {np.mean(p_cv >= 0.5):.0%}, false positive rate {np.mean(p_all[len(pos):] >= 0.5):.0%}")
    return r


def classify_nz(r):
    gaz = pd.read_csv(V2 / "nz_gazetteer.csv", dtype=str, encoding="utf-8-sig")
    gaz = gaz[gaz.feat_type.isin(["Stream", "River"])]
    # dual names ('Clutha River / Mata-Au') were restored recently; the flag refers to the
    # pair, so the historical first part is judged by spelling instead
    gaz = gaz[gaz.maori_name.isin(["Yes", "No"]) & ~gaz.name.str.contains(" / ", regex=False)]
    flag = gaz.assign(key=gaz.name.map(fold)).groupby("key").maori_name.agg(lambda s: s.mode().iat[0])

    def lab(name, spec):
        if not spec:
            return "no specific", np.nan
        f = flag.get(fold(name), None)
        if f == "Yes":
            return "maori (gazetteer)", 1.0
        if f == "No":
            return "english (gazetteer)", 0.0
        word = spec.replace(" ", "").replace("-", "").replace("'", "")
        cls = transparent(spec, "us")[0]
        if MAORI.match(word) and not cls:
            return "maori (spelling)", 1.0
        if cls:
            return cls, 0.0
        # Maori spelling rules are strict: a name that breaks them is not Maori
        return ("maori (spelling)", 1.0) if MAORI.match(word) else ("not maori (spelling)", 0.0)

    labs = [lab(n, s) for n, s in zip(r.name, r.specific)]
    r["origin"] = [a for a, _ in labs]
    r["old"] = [b for _, b in labs]
    hits = r.name.map(lambda n: fold(n) in flag.index)
    say(f"  names found in the NZ Gazetteer: {hits.mean():.0%}")
    return r


def clean(name):
    name = re.sub(r"\(.*?\)", " ", str(name)).split(" / ")[0]
    return re.sub(r"\s+", " ", name).strip()


def australia():
    say("AUSTRALIA")
    gdb = V2 / "ga_hydro" / "SurfaceHydrologyLinesNational.gdb"
    lines = pyogrio.read_dataframe(gdb, layer="HydroLines", columns=["NAME", "FEATURETYPE"],
                                   where="NAME <> '' AND FEATURETYPE = 'Watercourse'")
    lines = lines.to_crs(3577)
    lines["name"] = lines.NAME.map(clean).str.title()
    say(f"  named watercourse lines: {len(lines):,}")
    hydro = hydrorivers("au", 3577, (112, -44.5, 154.5, -9))
    r, lines = rivers_table(lines[["name", "geometry"]], hydro)
    r["specific"] = r.name.map(lambda n: split_name(n)[0])
    r = classify_au(r)
    save_geometry(r, lines, DERIVED / "australia_geom.parquet", min_area=0, tol=300)
    say(f"  named rivers: {len(r):,}; with a HydroRIVERS area: {r.area.notna().sum():,}",
        "  origins: " + ", ".join(f"{k} {v:,}" for k, v in r.origin.value_counts().items()),
        "  largest: " + ", ".join(f"{n} ({'old' if o == 1 else 'new' if o == 0 else '?'})"
                                  for n, o in r.sort_values("area", ascending=False).head(25)[["name", "old"]].to_numpy()))
    r.to_parquet(DERIVED / "australia_origin.parquet", index=False)


def new_zealand():
    say("", "NEW ZEALAND")
    els = json.load(open(V2 / "nz_osm_waterways.json", encoding="utf-8"))["elements"]
    geoms, names = [], []
    for e in els:
        if e.get("type") != "way" or "geometry" not in e or e["tags"].get("waterway") == "canal":
            continue
        xy = [(p["lon"], p["lat"]) for p in e["geometry"]]
        if len(xy) < 2:
            continue
        geoms.append(shapely.LineString(xy))
        names.append(clean(e["tags"]["name"]))
    lines = gpd.GeoDataFrame({"name": names}, geometry=geoms, crs=4326).to_crs(2193)
    say(f"  named waterway lines: {len(lines):,}")
    hydro = hydrorivers("au", 2193, (165, -48, 179.5, -34))
    r, lines = rivers_table(lines, hydro)
    r["specific"] = r.name.map(lambda n: split_name(n)[0])
    r = classify_nz(r)
    save_geometry(r, lines, DERIVED / "nz_geom.parquet", min_area=0, tol=100)
    say(f"  named rivers: {len(r):,}; with a HydroRIVERS area: {r.area.notna().sum():,}",
        "  origins: " + ", ".join(f"{k} {v:,}" for k, v in r.origin.value_counts().items()),
        "  largest: " + ", ".join(f"{n} ({'old' if o == 1 else 'new' if o == 0 else '?'})"
                                  for n, o in r.sort_values("area", ascending=False).head(25)[["name", "old"]].to_numpy()))
    r.to_parquet(DERIVED / "nz_origin.parquet", index=False)


def save_geometry(r, lines, path, min_area, tol):
    """Simplified line geometry of every named river, for maps and the web map.
    raw_area (before the plausibility cap) sets line widths only; analyses use area."""
    keep = r.loc[(r.raw_area.fillna(0) >= min_area) | r.area.notna(), "group"]
    g = lines[lines.group.isin(keep)][["group", "geometry"]].copy()
    g["geometry"] = g.geometry.simplify(tol)
    g = g.dissolve("group").reset_index().to_crs(4326)
    g.merge(r[["group", "name", "area", "raw_area", "old", "origin"]], on="group").to_parquet(path)


def reclassify():
    """Rerun only the name classification on the saved river tables."""
    for file, geom, fn in (("australia_origin.parquet", "australia_geom.parquet", classify_au),
                           ("nz_origin.parquet", "nz_geom.parquet", classify_nz)):
        say("", file)
        r = fn(pd.read_parquet(DERIVED / file).drop(columns=["origin", "old"]))
        say("  origins: " + ", ".join(f"{k} {v:,}" for k, v in r.origin.value_counts().items()))
        r.to_parquet(DERIVED / file, index=False)
        g = gpd.read_parquet(DERIVED / geom).drop(columns=["old", "origin"], errors="ignore")
        g.merge(r[["group", "old", "origin"]], on="group").to_parquet(DERIVED / geom)


def main():
    import sys
    if "classify" in sys.argv[1:]:
        reclassify()
        return
    australia()
    new_zealand()
    (RESULTS / "australasia_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
