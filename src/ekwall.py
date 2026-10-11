"""Match Ekwall's 1928 origin labels to the named rivers of England.

A label matches a river when their cores agree. The core is the folded name
with generic words removed (River, Brook, Beck, Burn, ...), so Ekwall's
"Kielder" matches OS "Kielder Burn" while "Avon" does not match "Little Avon
River". Names shared by many rivers (Avon, Stour, Frome, Rother) are tied to
Ekwall's county: a label applies to same-core rivers within RADIUS_KM of the
county's centre (distance to the nearest point of the river's course), and
where two counties claim one river the nearer wins.

Output: data/derived/england_labels.parquet, results/ekwall_unmatched.txt
"""
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

from common import DERIVED, GENERIC_SUFFIX, GENERIC_PREFIX, LABELS, RESULTS, fold

RADIUS_KM = 70
NEWCOMER = {"eng", "scand", "back"}
GENERIC = (GENERIC_SUFFIX | GENERIC_PREFIX) - {"rio", "lake"}


def core(name: str) -> str:
    words = [w for w in fold(name).replace("'", "").split() if w not in GENERIC]
    return " ".join(words)


def parse(path=LABELS / "ekwall_1928_ch4.txt"):
    rows, group = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"\[(.+)\]\s+(-?[\d.]+)\s+(-?[\d.]+)", line)
        if m:
            group = (m.group(1), float(m.group(2)), float(m.group(3)))
            continue
        cat, names = line.split(":", 1)
        for raw in names.split(","):
            raw = raw.strip()
            lost, unsure = raw.startswith("*"), raw.endswith("?")
            raw = raw.strip("*?").strip()
            forms = [f.strip() for f in raw.split("=")]
            rows.append(dict(county=group[0], lat=group[1], lon=group[2], category=cat.strip(),
                             name=forms[0], forms=forms, lost=lost, uncertain=unsure))
    return pd.DataFrame(rows)


def main():
    lab = parse()
    print(f"{len(lab)} label rows, {lab.lost.sum()} lost names, "
          f"{lab.category.value_counts().to_dict()}")
    rivers = pd.read_parquet(DERIVED / "england_rivers.parquet")
    rivers["core"] = rivers.name.map(core)
    by_core = rivers.groupby("core").indices

    links = gpd.read_parquet(DERIVED / "gb_links.parquet", columns=["river_id", "geometry"])
    links = links[links.river_id.notna()]
    shape = links.dissolve("river_id").geometry  # one multiline per river
    to_bng = Transformer.from_crs(4326, 27700, always_xy=True)
    hits, unmatched = [], []
    for r in lab[~lab.lost].itertuples():
        cands = sorted({i for f in r.forms for i in by_core.get(core(f), [])})
        if not cands:
            unmatched.append(f"{r.county}: {r.name} ({r.category})")
            continue
        sub = rivers.iloc[cands]
        if r.county == "ANY":
            dist = np.zeros(len(sub))
        else:
            centre = Point(*to_bng.transform(r.lon, r.lat))
            dist = shape.loc[sub.river_id].distance(centre).to_numpy() / 1000
            keep = dist <= RADIUS_KM
            if not keep.any():
                unmatched.append(f"{r.county}: {r.name} ({r.category}) nearest {dist.min():.0f} km")
                continue
            sub, dist = sub[keep], dist[keep]
        for (_, s), d in zip(sub.iterrows(), dist):
            hits.append(dict(river_id=s.river_id, os_name=s["name"], label_name=r.name,
                             county=r.county, category=r.category, uncertain=r.uncertain,
                             dist_km=d, generic_rule=r.county == "ANY"))
    hits = pd.DataFrame(hits)
    # county labels beat the generic rule; among counties the nearest centre wins
    hits = (hits.sort_values(["river_id", "generic_rule", "dist_km"])
                .drop_duplicates("river_id"))
    hits["origin"] = np.where(hits.category == "pre", "pre-English",
                              np.where(hits.category.isin(NEWCOMER), "newcomer", "doubtful"))
    hits.to_parquet(DERIVED / "england_labels.parquet", index=False)
    (RESULTS / "ekwall_unmatched.txt").write_text("\n".join(unmatched) + "\n", encoding="utf-8")

    print(f"labelled {len(hits):,} OS rivers "
          f"({(~hits.generic_rule).sum():,} from county lists, {hits.generic_rule.sum():,} by generic rule)")
    print(hits.groupby(["generic_rule", "origin"]).size())
    print(f"{len(unmatched)} current names not found; see results/ekwall_unmatched.txt")


if __name__ == "__main__":
    main()
