"""Named rivers of the conterminous US from NHDPlus V2 (USGS E2NHDPlusV2 v3.0).

One row per GNIS feature (gnis_id). Size is the drainage area at the
feature's outlet (max totdasqkm over its flowlines) and the arbolate sum
(total upstream channel length), the same measure used for England.

Output: data/derived/us_rivers.parquet
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from common import DERIVED, RAW, split_name

ATTS = RAW / "enhd_nhdplusatts.parquet"
LINES = RAW / "enhd_nhdplus_flowlines.gpkg"


def main():
    cols = ["comid", "gnis_name", "gnis_id", "ftype", "totdasqkm", "arbolatesu", "lengthkm",
            "streamorde", "vpuid", "hydroseq", "terminalpa", "reachcode"]
    a = pd.read_parquet(ATTS, columns=cols)
    a = a[a.gnis_name.fillna("").str.strip().ne("") & a.ftype.isin(["StreamRiver", "ArtificialPath", "Connector"])]
    print(f"{len(a):,} named flowlines, {a.gnis_id.nunique():,} GNIS features")

    a = a.sort_values("totdasqkm", ascending=False)
    out = a.groupby("gnis_id", sort=False).agg(
        name=("gnis_name", "first"), outlet_comid=("comid", "first"),
        da_km2=("totdasqkm", "max"), upstream_km=("arbolatesu", "max"),
        length_km=("lengthkm", "sum"), n_lines=("comid", "size"),
        order=("streamorde", "max"), vpuid=("vpuid", "first"), terminal=("terminalpa", "first"),
        reachcode=("reachcode", "first"),
    ).reset_index()
    out["huc4"] = out.reachcode.str[:4]  # NHD subregion of the outlet, the clustering unit
    sp = out.name.map(split_name)
    out["specific"] = [s for s, _ in sp]
    out["generic"] = [g for _, g in sp]

    # outlet coordinates: downstream end of the outlet flowline
    comids = set(out.outlet_comid.astype(np.int64))
    info = pyogrio.read_info(LINES)
    layer = pyogrio.list_layers(LINES)[0][0]
    print("reading outlet geometries from", layer, info["fields"][:8])
    idcol = next(c for c in info["fields"] if c.lower() in ("comid", "featureid"))
    g = pyogrio.read_dataframe(LINES, layer=layer, columns=[idcol])
    g = g[g[idcol].astype(np.int64).isin(comids)].to_crs(4326)
    ends = g.geometry.apply(lambda geom: (geom.geoms[-1] if geom.geom_type == "MultiLineString" else geom).coords[-1])
    pos = pd.DataFrame({"outlet_comid": g[idcol].astype(np.int64).to_numpy(),
                        "lon": [p[0] for p in ends], "lat": [p[1] for p in ends]})
    out["outlet_comid"] = out.outlet_comid.astype(np.int64)
    out = out.merge(pos, on="outlet_comid", how="left")
    out.to_parquet(DERIVED / "us_rivers.parquet", index=False)
    print(out.sort_values("da_km2", ascending=False).head(15)[["name", "specific", "da_km2", "lat", "lon"]])
    print(f"missing coordinates: {out.lat.isna().sum():,}")


if __name__ == "__main__":
    main()
