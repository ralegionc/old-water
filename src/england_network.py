"""Named rivers of England from OS Open Rivers, each with its upstream channel length.

Size is the arbolate sum at the river's outlet: the total length of every
channel that drains through it. It is computed on the network itself, so no
DEM is needed. Canals stay in the graph so canalised river navigations do not
cut rivers in two, but they add no length and take only a token share of the
flow where they leave a river. Checked against NRFA gauged catchment areas
in validate_size.py.

Output: data/derived/england_rivers.parquet (one row per named river)
        data/derived/gb_links.parquet       (every link, for maps)
"""
import time

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyogrio

from common import DERIVED, RAW, split_name

GPKG = RAW / "oprvrs" / "Data" / "oprvrs_gb.gpkg"
MERGE_GAP_M = 25000  # rejoin same-name pieces split by unnamed gaps or tidal reaches
ESTUARY_WORDS = {"reach", "hope", "roads", "falls", "haven", "harbour", "pool", "channel"}


def load_links():
    links = pyogrio.read_dataframe(GPKG, layer="watercourse_link")
    links = links.copy()
    links["is_canal"] = links.form == "canal"
    # one canonical name per link: the English form where OS gives a Welsh primary
    welsh = links.watercourse_name.str.startswith("Afon ", na=False) & links.watercourse_name_alternative.notna()
    links.loc[welsh, ["watercourse_name", "watercourse_name_alternative"]] =         links.loc[welsh, ["watercourse_name_alternative", "watercourse_name"]].values
    # tidal reach names ("Woolwich Reach", "Lower Hope") are estuary segments, not rivers
    last = links.watercourse_name.str.split().str[-1].str.lower()
    links.loc[last.isin(ESTUARY_WORDS) & (links.form == "tidalRiver"), "watercourse_name"] = None
    rev = links.flow_direction == "in opposite direction"
    links.loc[rev, ["start_node", "end_node"]] = links.loc[rev, ["end_node", "start_node"]].values
    return links.reset_index(drop=True)


def arbolate(links):
    """Upstream channel length (m) carried through each link.

    Two passes over a topological order of nodes (Kahn's algorithm; the few
    braid cycles are broken by forcing the node with the fewest unresolved
    inputs). Pass one measures the longest route from every node to the sea.
    Pass two accumulates upstream length. At a split the
    upstream total is shared in proportion to each branch's route onward:
    braids that rejoin get the whole total back, and an intake that ends at a
    works or reservoir after a few hundred metres takes almost nothing.
    """
    nodes = pd.Index(pd.unique(np.concatenate([links.start_node, links.end_node])))
    s = nodes.get_indexer(links.start_node)
    e = nodes.get_indexer(links.end_node)
    length = links["length"].to_numpy(float)
    # canals add no channel length of their own and take only a token share
    # at junctions, but still carry a river through canalised navigations
    canal = links.is_canal.to_numpy()
    own = np.where(canal, 0.0, length)
    n = len(nodes)
    by_start = np.argsort(s, kind="stable")
    first = np.searchsorted(s[by_start], np.arange(n + 1))

    indeg = np.bincount(e, minlength=n)
    done = np.zeros(n, bool)
    queue = list(np.flatnonzero(indeg == 0))
    topo, forced = [], 0
    while len(topo) < n:
        if not queue:
            left = np.flatnonzero(~done)
            queue = [left[np.argmin(indeg[left])]]
            forced += 1
        v = queue.pop()
        if done[v]:
            continue
        done[v] = True
        topo.append(v)
        for k in by_start[first[v]:first[v + 1]]:
            indeg[e[k]] -= 1
            if indeg[e[k]] == 0 and not done[e[k]]:
                queue.append(e[k])
    print(f"forced {forced} nodes out of cycles", flush=True)

    onward = np.zeros(n)  # longest route to a terminal node
    for v in reversed(topo):
        outs = by_start[first[v]:first[v + 1]]
        if len(outs):
            onward[v] = max(length[k] + onward[e[k]] for k in outs)

    up = np.zeros(n)
    carried = np.zeros(len(links))
    for v in topo:
        outs = by_start[first[v]:first[v + 1]]
        if not len(outs):
            continue
        w = (length[outs] + onward[e[outs]]) * np.where(canal[outs], 0.01, 1.0)
        carried[outs] = up[v] * w / w.sum()
        for k in outs:
            up[e[k]] += carried[k] + own[k]
    return carried + own


def main_successor(links):
    """For each node, the outgoing link that carries the most flow."""
    order = np.argsort(links.up_m.to_numpy())  # ascending, so the largest wins
    return dict(zip(links.start_node.to_numpy()[order], links.index.to_numpy()[order]))


def named_rivers(links):
    up = links.up_m.to_numpy()
    length = links["length"].to_numpy(float)
    end_node = links.end_node.to_numpy()
    succ = main_successor(links)
    named = links[links.watercourse_name.notna() & ~links.is_canal]
    rows = []
    for name, grp in named.groupby("watercourse_name"):
        ug = nx.Graph()
        for i, s, e in zip(grp.index, grp.start_node, grp.end_node):
            ug.add_edge(s, e, i=i)
        comps = [set(d["i"] for _, _, d in ug.subgraph(c).edges(data=True))
                 for c in nx.connected_components(ug)]
        # merge pieces when walking downstream from one outlet meets the same name
        parent = list(range(len(comps)))
        owner = {i: k for k, c in enumerate(comps) for i in c}

        def find(k):
            while parent[k] != k:
                k = parent[k]
            return k
        if len(comps) > 1:
            for k, c in enumerate(comps):
                node, walked, seen = end_node[max(c, key=up.__getitem__)], 0.0, set()
                while node in succ and walked < MERGE_GAP_M and node not in seen:
                    seen.add(node)
                    nxt = succ[node]
                    if nxt in c:
                        break
                    if nxt in owner:
                        parent[find(k)] = find(owner[nxt])
                        break
                    walked += length[nxt]
                    node = end_node[nxt]
        groups = {}
        for k, c in enumerate(comps):
            groups.setdefault(find(k), set()).update(c)
        for ids in groups.values():
            ids = sorted(ids)
            rows.append((name, ids, max(ids, key=up.__getitem__)))

    mid = links.geometry.interpolate(0.5, normalized=True)
    mx, my = mid.x.to_numpy(), mid.y.to_numpy()
    eng, cor, wal = (links[c].to_numpy(float) for c in ("in_england", "in_cornwall", "in_wales"))
    alt_all = links.watercourse_name_alternative.to_numpy()
    out, member = [], {}
    for rid, (name, ids, outlet) in enumerate(rows):
        ids = np.array(ids)
        w = length[ids]
        tot = w.sum()
        alts = [a for a in alt_all[ids] if a]
        spec, gen = split_name(name)
        ox, oy = links.geometry.iat[outlet].coords[-1]
        out.append(dict(
            river_id=rid, name=name, alt_name=max(set(alts), key=alts.count) if alts else None,
            outlet_link=int(outlet),
            specific=spec, generic=gen, n_links=len(ids),
            length_km=tot / 1000, upstream_km=up[outlet] / 1000,
            frac_england=(w * eng[ids]).sum() / tot, frac_cornwall=(w * cor[ids]).sum() / tot,
            frac_wales=(w * wal[ids]).sum() / tot,
            outlet_x=ox, outlet_y=oy,
            mid_x=(w * mx[ids]).sum() / tot, mid_y=(w * my[ids]).sum() / tot,
        ))
        member.update(dict.fromkeys(ids.tolist(), rid))
    return pd.DataFrame(out), member


def side_channels(links, rivers):
    """Flag named reaches that take their water from one named river and hand
    it back to the same river: mill streams, backwaters, named pools on the
    main channel. A renamed continuation (Ure becoming Ouse) or an estuary
    (Humber) is not flagged, because the parent name never reappears below.
    """
    rid = links.river_id.to_numpy(dtype=float, na_value=np.nan)
    nm = links.watercourse_name.to_numpy()
    up = links.up_m.to_numpy()
    length = links["length"].to_numpy(float)
    start = links.start_node.to_numpy()
    end = links.end_node.to_numpy()
    into = {}
    for i, e in enumerate(end):
        into.setdefault(e, []).append(i)
    succ = main_successor(links)
    members = {}
    for i, r in enumerate(rid):
        if r == r:
            members.setdefault(int(r), []).append(i)
    flags = np.zeros(len(rivers), bool)
    for pos, (river, name, upkm) in enumerate(zip(rivers.river_id, rivers.name, rivers.upstream_km)):
        ids = members.get(river, [])
        mine = set(ids)
        inflow = [j for i in ids for j in into.get(start[i], []) if j not in mine]
        if not inflow:
            continue
        j = max(inflow, key=up.__getitem__)
        parent = nm[j]
        if parent is None or parent == name or up[j] < 0.3 * upkm * 1000:
            continue
        node, walked, seen = end[max(ids, key=up.__getitem__)], 0.0, set()
        while node in succ and walked < 5000 and node not in seen:
            seen.add(node)
            k = succ[node]
            if nm[k] == parent:
                flags[pos] = True
                break
            walked += length[k]
            node = end[k]
    return flags


def terminal_basin(links, rivers):
    """Id of the sea outlet each river drains to, following the main flow path.

    Rivers sharing a basin were often named by the same communities, so they
    are not independent; analyze.py resamples whole basins.
    """
    succ = main_successor(links)
    end = links.end_node.to_numpy()
    memo = {}

    def terminal(node):
        path, seen = [], set()
        while node in succ and node not in seen and node not in memo:
            seen.add(node)
            path.append(node)
            node = end[succ[node]]
        t = memo.get(node, node)
        for n in path:
            memo[n] = t
        return t
    return [terminal(end[o]) for o in rivers.outlet_link]


def merge_pieces(links, rivers, gap_m=1000):
    """Join same-name pieces that lie within gap_m of each other into one river.

    The downstream walk in named_rivers misses pieces split by tidal reaches,
    braids or network gaps (the Thames came out as three rivers). Pieces of a
    common name far apart (two Black Brooks in different counties) stay apart.
    """
    geom = links.loc[links.river_id.notna()].groupby("river_id").geometry.apply(lambda g: g.union_all())
    rivers = rivers.copy()
    rivers["merged_into"] = rivers.river_id
    for name, grp in rivers[~rivers.is_feature].groupby("name"):
        if len(grp) < 2:
            continue
        ids = grp.sort_values("upstream_km", ascending=False).river_id.tolist()
        parent = {i: i for i in ids}

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for a_i, a in enumerate(ids):
            for b in ids[a_i + 1:]:
                if find(a) != find(b) and geom[a].distance(geom[b]) <= gap_m:
                    parent[find(b)] = find(a)
        for i in ids:
            rivers.loc[rivers.river_id.eq(i), "merged_into"] = find(i)
    agg = rivers.groupby("merged_into").agg(
        length_km=("length_km", "sum"), upstream_km=("upstream_km", "max"), n_links=("n_links", "sum"))
    keep = rivers[rivers.river_id.eq(rivers.merged_into)].drop(columns=["length_km", "upstream_km", "n_links"])
    keep = keep.merge(agg, left_on="river_id", right_index=True)
    # England share recomputed over the merged pieces, weighted by length
    w = rivers.assign(we=rivers.frac_england * rivers.length_km, wc=rivers.frac_cornwall * rivers.length_km)
    fr = w.groupby("merged_into")[["we", "wc", "length_km"]].sum()
    keep["frac_england"] = keep.river_id.map(fr.we / fr.length_km)
    keep["frac_cornwall"] = keep.river_id.map(fr.wc / fr.length_km)
    remap = dict(zip(rivers.river_id, rivers.merged_into))
    print(f"merged {len(rivers) - len(keep):,} same-name pieces into neighbours")
    return keep.reset_index(drop=True), remap


def main():
    links = load_links()
    print(f"{len(links):,} links ({links.is_canal.sum():,} canal)", flush=True)
    t0 = time.time()
    links["up_m"] = arbolate(links)
    print(f"arbolate {time.time() - t0:.0f}s", flush=True)

    mu = gpd.read_file(f"zip://{RAW / 'ne_10m_admin_0_map_units.zip'}").to_crs(27700)
    a1 = gpd.read_file(f"zip://{RAW / 'ne_10m_admin_1_states_provinces.zip'}")
    corn = a1[a1.name.isin(["Cornwall", "Isles of Scilly"])].to_crs(27700).union_all()
    eng, wal = (mu[mu.NAME == n].union_all() for n in ("England", "Wales"))
    mid = gpd.GeoSeries(links.geometry.interpolate(0.5, normalized=True), crs=27700)
    links["in_cornwall"] = mid.within(corn).to_numpy()
    links["in_england"] = mid.within(eng).to_numpy() & ~links.in_cornwall.to_numpy()
    links["in_wales"] = mid.within(wal).to_numpy()
    # coastal links can fall just outside the generalised coastline
    near = mid.distance(eng) < 2000
    links["in_england"] |= (near & ~mid.within(wal) & ~links.in_cornwall
                            & (mid.distance(corn) > 2000)).to_numpy()

    print(f"regions {time.time() - t0:.0f}s", flush=True)
    rivers, member = named_rivers(links)
    print(f"named rivers {time.time() - t0:.0f}s", flush=True)
    links["river_id"] = links.index.map(member).astype("Int64")
    rivers["is_branch"] = side_channels(links, rivers)
    rivers["basin"] = terminal_basin(links, rivers)
    print(f"side channels {time.time() - t0:.0f}s", flush=True)
    # short named spots on a big river (fishing pools, weirs, tidal reaches)
    # inherit its whole catchment; they are features, not rivers
    rivers["is_feature"] = rivers.is_branch | (
        (rivers.length_km < 2) & (rivers.upstream_km > 10 * rivers.length_km + 20))
    print(f"dropped {rivers.is_feature.sum():,} side channels, pools and reaches")
    rivers, remap = merge_pieces(links, rivers)
    links["river_id"] = links.river_id.map(lambda r: remap.get(r, r) if pd.notna(r) else r).astype("Int64")
    keep = ["id", "form", "is_canal", "length", "watercourse_name", "up_m", "in_england", "in_wales",
            "in_cornwall", "river_id", "geometry"]
    gpd.GeoDataFrame(links[keep], crs=27700).to_parquet(DERIVED / "gb_links.parquet")
    rivers.to_parquet(DERIVED / "gb_rivers.parquet", index=False)
    eng_r = rivers[(rivers.frac_england >= 0.25) & (rivers.frac_cornwall < 0.5) & ~rivers.is_feature]
    eng_r.to_parquet(DERIVED / "england_rivers.parquet", index=False)
    print(f"{len(rivers):,} named rivers in GB, {len(eng_r):,} in England (excl. Cornwall)")
    print(eng_r.sort_values("upstream_km", ascending=False)
          .head(25)[["name", "specific", "length_km", "upstream_km", "frac_england"]].to_string())


if __name__ == "__main__":
    main()
