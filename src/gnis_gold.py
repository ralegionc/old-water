"""An external answer key for US names, read from GNIS feature histories.

The USGS Geographic Names Information System keeps a free-text history for
about 55,000 features ("Named for Peter Osterried, who had mining claims...",
"Choctaw word meaning 'big creek'"). Each named stream with a history is
labelled by rules on that text:

  indigenous   an indigenous language or 'Indian' next to word / name / term /
               meaning / means / derived, or 'said to mean'
  newcomer     'named for / after / in honor of' with no indigenous reference;
               'Spanish for', 'a French name'; a translation of an indigenous
               name; or an indigenous variant or original name mentioned for a
               river whose current name reads as English, French, Spanish or a
               personal name ('Ohaha is the original Indian name for Salt River')
  ethnonym     named for a tribe, band, chief or village: an indigenous word,
               usually a newcomer's act of naming (kept apart, as in classify.py)
  unknown      anything else ('Local name reported in 1954', 'Founded in 1835')

The rules are deliberately strict; 'unknown' is the price of precision.
Output: data/derived/gnis_gold.parquet, results/gnis_check.txt
"""
import re
import zipfile

import numpy as np
import pandas as pd

from common import DERIVED, RAW, RESULTS

V2 = RAW / "v2"
LANGS = r"""algonqui\w*|iroquoi\w*|siouan|muskogee|muscogee|muskogean|choctaw|cherokee|chickasaw|seminole|
navajo|navaho|hopi|apache|paiute|piute|shoshon\w*|\bute\b|comanche|kiowa|pawnee|osage|omaha|ponca|dakota|
lakota|sioux|ojibw\w*|chippewa|potawatomi|menominee|winnebago|ho-chunk|miami|shawnee|lenape|munsee|unami|
abenaki|penobscot|passamaquoddy|maliseet|mohegan|mohican|narragansett|wampanoag|massachusett|nipmuc|
mohawk|seneca|oneida|onondaga|cayuga|tuscarora|wyandot|huron|catawba|yuchi|natchez|caddo|wichita|
tonkawa|karankawa|salish\w*|chinook\w*|chinuk|klickitat|yakima|yakama|nez perce|cayuse|umatilla|klamath|
modoc|yurok|karuk|karok|hupa|pomo|miwok|maidu|wintu\w*|ohlone|costanoan|yokuts|chumash|luiseno|cahuilla|
mojave|mohave|quechan|tohono|papago|pima|zuni|tewa|keres\w*|tiwa|towa|washo\w*|arapaho|cheyenne|
blackfoot|blackfeet|assiniboine|gros ventre|mandan|hidatsa|arikara|kootenai|kutenai|spokane|colville|
okanogan|wenatchi|lummi|skagit|snohomish|nisqually|puyallup|quinault|makah|timucua|calusa|apalachee|
koasati|coushatta|biloxi|tunica|chitimacha|atakapa|houma|mobilian|powhatan|nanticoke|piscataway|
pamunkey|delaware indian|creek indian|indian|native american|aboriginal""".replace("\n", "")
WORD = r"(word|name|term|meaning|means|meant|derived|derivation|origin|for|signif\w*)"
INDIGENOUS = [
    re.compile(rf"\b({LANGS})\b.{{0,60}}\b{WORD}\b", re.I),
    re.compile(rf"\b(word|name|term)\b.{{0,40}}\b({LANGS})\b", re.I),
    re.compile(r"\bsaid to mean\b", re.I),
]
TRANSLATION = re.compile(r"\btranslat\w*\b", re.I)
ETHNONYM = re.compile(r"\bnamed (for|after|from|in honou?r of)\b.{0,40}\b(indians?|tribe|band|chief|village|people|nation)\b", re.I)
NAMED_FOR = re.compile(r"\bnamed (for|after|in honou?r of|by)\b", re.I)
EUROPEAN = re.compile(r"\b(spanish|french|german|dutch|swedish|latin|norwegian|welsh|gaelic) (word|name|term|for)\b"
                      r"|\bis (spanish|french|german|dutch|latin) for\b", re.I)
# the history mentions an indigenous name that is not the current one
VARIANT = re.compile(r"\bvariant\b|\b(original|former|old|earlier) (indian |native )?name\b"
                     r"|\boriginally (known|called|named)\b|\bformerly\b|\bwas renamed\b|\bindian meaning\b", re.I)


def label(text, readable=False):
    """readable: the current name reads as English, French, Spanish or a personal name."""
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    if ETHNONYM.search(text):
        return "ethnonym"
    indigenous = any(p.search(text) for p in INDIGENOUS)
    if indigenous and TRANSLATION.search(text):
        return "newcomer"  # an English rendering of an indigenous name
    if indigenous and readable and VARIANT.search(text):
        return "newcomer"  # 'Ohaha is the original Indian name for Salt River'
    if indigenous:
        return "indigenous"
    if EUROPEAN.search(text) or NAMED_FOR.search(text):
        return "newcomer"
    return "unknown"


def main():
    with zipfile.ZipFile(V2 / "FeatureDescriptionHistory_National_Text.zip") as z:
        h = pd.read_csv(z.open("Text/FeatureDescriptionHistory_National.txt"), sep="|", dtype=str,
                        encoding="utf-8-sig", on_bad_lines="skip")
    h = h[h.history.notna()].copy()
    u = pd.read_parquet(DERIVED / "us_origin.parquet")
    u["gnis_id"] = u.gnis_id.astype(str)
    g = u.merge(h.rename(columns={"feature_id": "gnis_id"}), on="gnis_id", how="inner")
    readable = ~g.origin.isin(["native (list)", "native (model)", "ethnonym", "transferred", "opaque", "no specific"])
    g["gold"] = [label(t, r) for t, r in zip(g.history, readable)]
    g.to_parquet(DERIVED / "gnis_gold.parquet", index=False)

    lines = [f"named streams with a GNIS history: {len(g):,} of {len(u):,}",
             "gold labels: " + ", ".join(f"{k} {v:,}" for k, v in g.gold.value_counts().items())]
    ev = g[g.gold.isin(["indigenous", "newcomer", "ethnonym"]) & g.old.notna()].copy()
    ev["truth"] = ev.gold.isin(["indigenous", "ethnonym"]).astype(float)
    lines.append(f"evaluated (gold known and name classified): {len(ev):,}")
    for name, d in [("all", ev)] + list(ev.groupby(pd.cut(ev.da_km2, [0, 10, 100, 1000, 1e8]), observed=True)):
        tp = ((d.old == 1) & (d.truth == 1)).sum()
        fp = ((d.old == 1) & (d.truth == 0)).sum()
        fn = ((d.old == 0) & (d.truth == 1)).sum()
        tn = ((d.old == 0) & (d.truth == 0)).sum()
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        lines.append(f"  {str(name):22} n={len(d):6,}  truly old {d.truth.mean():6.1%}  "
                     f"precision {prec:6.1%}  recall {rec:6.1%}  (tp {tp}, fp {fp}, fn {fn}, tn {tn})")
    fps = ev[(ev.old == 1) & (ev.truth == 0)].head(12)
    fns = ev[(ev.old == 0) & (ev.truth == 1)].head(12)
    lines.append("  false old: " + "; ".join(f"{n} [{h[:70]}]" for n, h in zip(fps.name, fps.history)))
    lines.append("  missed old: " + "; ".join(f"{n} [{h[:70]}]" for n, h in zip(fns.name, fns.history)))
    print("\n".join(lines))
    (RESULTS / "gnis_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
