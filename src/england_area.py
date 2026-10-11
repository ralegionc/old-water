"""Catchment area for England's named rivers, from HydroRIVERS (elevation-derived).

Every OS link of an English named river is snapped to HydroRIVERS with the
two-of-three rule (snap.py); a river's area is the largest matched area along
it. Rivers too small for HydroRIVERS (under ~10 km2) fall back to the
channel-length conversion fitted on NRFA gauges in validate_size.py.

Checked link by link against NRFA gauged catchment areas.
Output: data/derived/england_area.parquet, results/size_validation.txt (appended)
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from common import DERIVED, RAW, RESULTS
from ekwall import core
from snap import hydrorivers, snap_area

NRFA = (0.653, 0.799)


def main():
    rivers = pd.read_parquet(DERIVED / "england_rivers.parquet")
    links = gpd.read_parquet(DERIVED / "gb_links.parquet")
    links = links[links.river_id.isin(rivers.river_id) & ~links.is_canal].reset_index(drop=True)
    hydro = hydrorivers("eu", 27700, (-6.5, 49.8, 2.0, 56.0))
    # 'hr_area', not 'area': GeoDataFrame.area is the geometric area of the shapes
    links["hr_area"], links["hr_basin"] = snap_area(links, hydro, max_dist=500)
    best = links.sort_values("hr_area", ascending=False).dropna(subset=["hr_area"]).drop_duplicates("river_id")
    out = rivers[["river_id", "upstream_km"]].merge(best[["river_id", "hr_area", "hr_basin"]], on="river_id", how="left")
    conv = 10 ** (NRFA[0] + NRFA[1] * np.log10(out.upstream_km.clip(lower=0.01)))
    # A channel beside an estuary or a big river (marsh sewers, dykes) can snap to
    # it and inherit its catchment. The OS network measures each river's own
    # upstream length, so the snapped area is kept only when the two agree within
    # a factor of 5; otherwise the network-based conversion is used.
    ok = out.hr_area.notna() & (out.hr_area / conv).between(0.2, 5)
    out["area_source"] = np.where(ok, "hydrorivers", np.where(out.hr_area.notna(), "channel length (snap rejected)",
                                                              "channel length"))
    out["area_km2"] = np.where(ok, out.hr_area, conv)
    out.to_parquet(DERIVED / "england_area.parquet", index=False)

    # link-level check against NRFA gauges: same-name link within 500 m carrying most flow
    st = pd.DataFrame(json.load(open(RAW / "nrfa_stations.json"))["data"]).dropna(
        subset=["catchment-area", "easting", "northing", "river"])
    st = st[st.northing < 1_250_000]
    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.easting, st.northing), crs=27700)
    pts["core"] = pts.river.map(core)
    lk = links.dropna(subset=["hr_area"]).copy()
    lk["core"] = lk.watercourse_name.map(core)
    j = gpd.sjoin(pts.assign(geometry=pts.buffer(500)), lk[["core", "hr_area", "up_m", "geometry"]],
                  predicate="intersects", lsuffix="st", rsuffix="ln")
    j = j[j.core_st == j.core_ln].sort_values("up_m").drop_duplicates("id", keep="last")
    x, y = np.log10(j["hr_area"]), np.log10(j["catchment-area"])
    ratio = j["hr_area"] / j["catchment-area"]
    lines = [
        "",
        "HydroRIVERS area (England rivers, two-of-three snapping):",
        f"  rivers with a HydroRIVERS area: {out.hr_area.notna().sum():,} of {len(out):,}; kept {ok.sum():,}, "
        f"rejected as inherited from a bigger river {(out.hr_area.notna() & ~ok).sum():,}; the rest use the "
        f"channel-length conversion",
        f"  NRFA stations checked: {len(j)}; log-log r = {np.corrcoef(x, y)[0, 1]:.3f}; "
        f"median area ratio {ratio.median():.2f}, within a factor of 1.5 for {np.mean((ratio > 2 / 3) & (ratio < 1.5)):.0%}",
    ]
    print("\n".join(lines))
    with open(RESULTS / "size_validation.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
