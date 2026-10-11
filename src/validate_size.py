"""Check network-derived upstream length against measured catchment areas.

The UK National River Flow Archive (NRFA) gives catchment area for ~1,600
gauging stations. Each station is snapped to the nearest OS link carrying
the same river name (within 500 m), and its upstream length compared with
the gauged area. A tight log-log relation means upstream length is a
usable size measure; the slope converts it to km2 for comparison with the
US, where NHDPlus gives drainage area directly.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from common import DERIVED, RAW, RESULTS
from ekwall import core


def main():
    st = pd.DataFrame(json.load(open(RAW / "nrfa_stations.json"))["data"])
    st = st.dropna(subset=["catchment-area", "easting", "northing", "river"])
    st = st[st.northing < 1_250_000]  # GB grid; drop Northern Ireland (Irish grid)
    links = gpd.read_parquet(DERIVED / "gb_links.parquet")
    links = links[links.watercourse_name.notna()].copy()
    links["core"] = links.watercourse_name.map(core)
    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.easting, st.northing), crs=27700)
    pts["core"] = pts.river.map(core)
    # every same-name link within 500 m; the main channel is the one carrying most
    buf = pts.assign(geometry=pts.buffer(500))
    j = gpd.sjoin(buf, links[["core", "up_m", "in_england", "geometry"]],
                  predicate="intersects", lsuffix="st", rsuffix="ln")
    j = j[j.core_st == j.core_ln].sort_values("up_m").drop_duplicates("id", keep="last")
    j["d"] = 0.0
    j["up_km"] = j.up_m / 1000
    x, y = np.log10(j.up_km), np.log10(j["catchment-area"])
    r = np.corrcoef(x, y)[0, 1]
    slope, icpt = np.polyfit(x, y, 1)
    eng = j[j.in_england]
    re_ = np.corrcoef(np.log10(eng.up_km), np.log10(eng["catchment-area"]))[0, 1]
    ratio = (eng["catchment-area"] / eng.up_km)
    lines = [
        f"stations matched: {len(j)} of {len(st)} (England {len(eng)})",
        f"log10 area vs log10 upstream length: r = {r:.3f} (England {re_:.3f})",
        f"fit: log10(area) = {icpt:.3f} + {slope:.3f} log10(upstream_km)",
        f"England area per km of channel: median {ratio.median():.2f} km2, IQR {ratio.quantile(.25):.2f} to {ratio.quantile(.75):.2f}",
    ]
    print("\n".join(lines))
    (RESULTS / "size_validation.txt").write_text("\n".join(lines) + "\n")
    j[["id", "name", "river", "catchment-area", "up_km", "in_england", "d"]].to_csv(
        DERIVED / "nrfa_matched.csv", index=False)
    bad = j.assign(ratio=j["catchment-area"] / j.up_km)
    print(bad.sort_values("ratio").iloc[np.r_[0:6, -6:0]][["name", "catchment-area", "up_km", "ratio"]])


if __name__ == "__main__":
    main()
