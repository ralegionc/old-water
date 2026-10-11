"""Data for the interactive map (docs/index.html): rivers, classes, evidence, curves.

Each country gets one GeoJSON of simplified named rivers with:
  n  name     c  class (old / new / unk)     a  catchment area, km2
  e  the evidence behind the class (Ekwall's county list, the Wikipedia list,
     the word reading, the model score, the GNIS history)
Coordinates are rounded to about 100 m to keep files small.

Output: docs/data/{us,england,australia,nz}.json, docs/data/curves.json, docs/index.html
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from common import DERIVED, RESULTS, ROOT

OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CLS = {1.0: "old", 0.0: "new"}


def write(gdf, name, tol_deg):
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = shapely.set_precision(gdf.geometry.simplify(tol_deg).to_numpy(), 0.001)
    gdf = gdf[~gdf.geometry.is_empty]
    feats = [{"type": "Feature", "id": i, "properties": p, "geometry": shapely.geometry.mapping(g)}
             for i, (p, g) in enumerate(zip(gdf.drop(columns="geometry").to_dict("records"), gdf.geometry))]
    path = OUT / f"{name}.json"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")),
                    encoding="utf-8")
    print(f"  {name}: {len(feats):,} rivers, {path.stat().st_size / 1e6:.1f} MB")


def us_evidence(r):
    how = r.how if isinstance(r.how, str) else ""
    base = {
        "native (list)": "On Wikipedia's lists of place names of Native American origin",
        "native (model)": f"Spelling looks indigenous to the character model ({r.p_native:.0%})",
        "ethnonym": "A tribe's name",
        "transferred": "A state's name, given to the creek after the state",
        "opaque": "Not classified: no reading and no list entry",
    }.get(r.origin, f"Reads as {r.origin}: {how.replace('+', ' + ')}" if how else f"Reads as {r.origin}")
    if isinstance(r.gnis_history, str):
        base += f". GNIS: {r.gnis_history[:130]}"
    return base


def united_states():
    d = pd.read_pickle(DERIVED / "us_map_segments.pkl")
    u = pd.read_parquet(DERIVED / "us_analysis.parquet")
    gold = pd.read_parquet(DERIVED / "gnis_gold.parquet", columns=["gnis_id", "history"]).drop_duplicates("gnis_id")
    u["gnis_history"] = u.gnis_id.astype(str).map(gold.set_index("gnis_id").history)
    u = u[u.da_km2 >= 250]
    keep = set(u.gnis_id)
    gn = pd.Series(d["gnis"])
    idx = np.flatnonzero(gn.isin(keep).to_numpy())
    parts = pd.DataFrame({"gnis_id": gn.iloc[idx].to_numpy(), "i": idx})
    geoms = parts.groupby("gnis_id").i.apply(
        lambda ix: shapely.line_merge(shapely.MultiLineString([d["segs"][k] for k in ix if len(d["segs"][k]) > 1])))
    g = gpd.GeoDataFrame({"gnis_id": geoms.index}, geometry=geoms.to_numpy(), crs=5070).to_crs(4326)
    g = g.merge(u, on="gnis_id")
    out = gpd.GeoDataFrame({
        "n": g.name, "c": g.old.map(CLS).fillna("unk"), "a": g.da_km2.round(0).astype(int),
        "e": [us_evidence(r) for r in g.itertuples()]}, geometry=g.geometry, crs=4326)
    write(out.sort_values("a"), "us", 0.005)


def england():
    o = pd.read_parquet(DERIVED / "england_analysis.parquet")
    o = o[(o.area_km2 >= 15) | o.source.isin(["ekwall", "ekwall-generic"])]
    links = gpd.read_parquet(DERIVED / "gb_links.parquet", columns=["river_id", "is_canal", "geometry"])
    links = links[links.river_id.isin(o.river_id) & ~links.is_canal]
    geom = links.dissolve("river_id").geometry.map(shapely.line_merge)
    geom = gpd.GeoSeries(geom, crs=27700).to_crs(4326)

    def ev(r):
        if r.source == "ekwall":
            word = {"pre": "pre-English", "eng": "English", "scand": "Scandinavian",
                    "back": "a back-formation from a place-name", "doubt": "doubtful"}[r.category]
            return f"Ekwall 1928 lists it as {word} ({r.county})"
        if r.source == "ekwall-generic":
            return "Ekwall 1928: names that simply mean 'stream' (Bourne, Burn, Eye, Rea, Yeo) are English"
        if r.source == "transparent":
            return f"Not in Ekwall's lists; reads as English: {str(r.how).replace('+', ' + ')}"
        return "Not in Ekwall's lists and not readable as English: not classified"
    o = o[o.river_id.isin(geom.index)]
    out = gpd.GeoDataFrame({
        "n": o.name, "c": o.old.map(CLS).fillna("unk"), "a": o.area_km2.round(0).astype(int),
        "e": [ev(r) for r in o.itertuples()]}, geometry=geom.loc[o.river_id].to_numpy(), crs=4326)
    write(out.sort_values("a"), "england", 0.0012)


def australasia():
    texts = {
        "aboriginal (list)": "On Wikipedia's list of Australian place names of Aboriginal origin",
        "aboriginal (model)": "Spelling looks Aboriginal to the character model",
        "english (loanword)": "An Aboriginal word that English had borrowed (Coolabah, Brigalow)",
        "maori (gazetteer)": "The NZ Gazetteer flags it as a Maori name",
        "english (gazetteer)": "The NZ Gazetteer flags it as not Maori",
        "maori (spelling)": "Follows Maori spelling rules",
        "opaque": "Not classified",
    }
    for name, file, tol, cut in (("australia", "australia_geom.parquet", 0.007, 600),
                                 ("nz", "nz_geom.parquet", 0.0015, 15)):
        g = gpd.read_parquet(DERIVED / file)
        g["geometry"] = g.geometry.map(shapely.line_merge)
        size = g.raw_area.fillna(g.area) if "raw_area" in g else g.area
        g = g.assign(size=size)[size >= cut]
        out = gpd.GeoDataFrame({
            "n": g.name, "c": g.old.map(CLS).fillna("unk"), "a": g["size"].round(0).astype(int),
            "e": [texts.get(o, f"Reads as {o}") for o in g.origin]}, geometry=g.geometry, crs=4326)
        write(out.sort_values("a"), name, tol)


def curves():
    c = pd.read_csv(RESULTS / "curves.csv")
    k = pd.read_csv(RESULTS / "countries.csv")
    keys = {"us": "US", "england": "England, Ekwall", "australia": "Australia", "nz": "New Zealand"}
    names = {"us": "United States", "england": "England", "australia": "Australia", "nz": "New Zealand"}
    out = {}
    for key, curve in keys.items():
        pts = c[c.curve.eq(curve)]
        row = k[k.country.eq(names[key])].iloc[0]
        out[key] = dict(points=pts[["mid", "share", "n"]].round(4).to_dict("records"),
                        slope=round(float(row.slope), 4), x50=round(float(row.x50), 4),
                        lo=round(float(row.lo), 3), hi=round(float(row.hi), 3), n=int(row.n))
    (OUT / "curves.json").write_text(json.dumps(out), encoding="utf-8")


def page():
    """docs/index.html for GitHub Pages: map_body.html inside a document skeleton.
    (The Artifact version publishes map_body.html directly; the host adds the skeleton.)"""
    body = (ROOT / "docs" / "map_body.html").read_text(encoding="utf-8")
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    (ROOT / "docs" / "index.html").write_text(html, encoding="utf-8")


def main():
    page()
    united_states()
    england()
    australasia()
    curves()


if __name__ == "__main__":
    main()
