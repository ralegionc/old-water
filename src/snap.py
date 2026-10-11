"""Catchment area from HydroRIVERS, shared by England, Australia and New Zealand.

HydroRIVERS (Lehner and Grill 2013) is a global river network derived from
the HydroSHEDS elevation model. Every reach carries UPLAND_SKM, the area
draining through it, and MAIN_RIV, the id of its sea outlet. It only maps
reaches draining 10 km2 or more.

A named river's lines are snapped to HydroRIVERS at three interior points per
line (25%, 50%, 75% along). A line counts only when two of the three points
land on the same reach, so the last stretch of a tributary does not pick up
its parent river's area at the confluence. The river's area is the largest
matched area along it.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import RAW

V2 = RAW / "v2"


def hydrorivers(region, crs, bbox=None):
    path = f"zip://{V2 / f'HydroRIVERS_v10_{region}_shp.zip'}!HydroRIVERS_v10_{region}_shp/HydroRIVERS_v10_{region}.shp"
    h = gpd.read_file(path, bbox=bbox, columns=["HYRIV_ID", "MAIN_RIV", "UPLAND_SKM"], engine="pyogrio")
    return h.to_crs(crs)


def group_rivers(lines, tol=25, gap=1000):
    """Union lines with the same name key that share an endpoint (to tol metres),
    then join same-key pieces lying within gap metres of each other ('River
    Murray' and 'Murray River' share the key murray|river)."""
    g = lines.geometry.to_numpy()
    coords, idx = shapely.get_coordinates(g, return_index=True)
    n = len(lines)
    first = coords[np.searchsorted(idx, np.arange(n))]
    last = coords[np.searchsorted(idx, np.arange(n), side="right") - 1]
    names = lines["key"].to_numpy() if "key" in lines else lines["name"].to_numpy()

    def keys(xy):
        q = np.round(xy / tol).astype(np.int64)
        return [f"{a}|{x}|{y}" for a, x, y in zip(names, q[:, 0], q[:, 1])]
    k = pd.factorize(pd.Series(keys(first) + keys(last)))[0]
    rows = np.r_[np.arange(n), np.arange(n)]
    size = n + k.max() + 1
    m = coo_matrix((np.ones(2 * n), (rows, n + k)), shape=(size, size))
    _, lab = connected_components(m, directed=False)
    lab = lab[:n]
    if gap:
        lab = merge_near(lines.geometry, pd.Series(names), lab, gap)
    return lab


def merge_near(geoms, keys, lab, gap):
    """Union components with the same key whose shapes come within gap metres."""
    comp = gpd.GeoDataFrame({"key": keys.to_numpy(), "lab": lab}, geometry=geoms.to_numpy(), crs=geoms.crs)
    comp = comp.dissolve("lab", aggfunc="first").reset_index()
    multi = comp[comp.duplicated("key", keep=False)].reset_index(drop=True)
    if multi.empty:
        return lab
    tree = shapely.STRtree(multi.geometry.to_numpy())
    left, right = tree.query(multi.geometry.to_numpy(), predicate="dwithin", distance=gap)
    same = multi.key.to_numpy()[left] == multi.key.to_numpy()[right]
    left, right = left[same], right[same]
    k = len(multi)
    m = coo_matrix((np.ones(len(left)), (left, right)), shape=(k, k))
    _, merged = connected_components(m, directed=False)
    remap = dict(zip(multi.lab.to_numpy(), multi.lab.to_numpy()[pd.Series(np.arange(k)).groupby(merged).transform("first")]))
    return np.array([remap.get(v, v) for v in lab])


def snap_area(lines, hydro, max_dist=300):
    """Upstream area and basin for each line, by the two-of-three rule."""
    pts = []
    for f in (0.25, 0.5, 0.75):
        p = shapely.line_interpolate_point(lines.geometry.to_numpy(), f, normalized=True)
        pts.append(gpd.GeoDataFrame({"line": np.arange(len(lines))}, geometry=p, crs=lines.crs))
    pts = pd.concat(pts, ignore_index=True)
    j = gpd.sjoin_nearest(pts, hydro[["HYRIV_ID", "geometry"]], max_distance=max_dist, how="inner")
    votes = j.groupby(["line", "HYRIV_ID"]).size().reset_index(name="n")
    votes = votes[votes.n >= 2].drop_duplicates("line")
    info = hydro.set_index("HYRIV_ID")[["UPLAND_SKM", "MAIN_RIV"]]
    area = np.full(len(lines), np.nan)
    basin = np.full(len(lines), np.nan)
    area[votes.line.to_numpy()] = info.loc[votes.HYRIV_ID, "UPLAND_SKM"].to_numpy()
    basin[votes.line.to_numpy()] = info.loc[votes.HYRIV_ID, "MAIN_RIV"].to_numpy()
    return area, basin
