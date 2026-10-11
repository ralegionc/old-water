"""Give every named river an origin: old (pre-newcomer) or newcomer.

England, in order of trust:
  1. Ekwall's county lists (pre-English / English / Scandinavian / back-formation)
  2. Ekwall's rule that Bourne, Burn, Eye, Ray, Rea, Yeo are English anywhere
  3. transparent in English or Old English / Old Norse elements -> newcomer
  4. otherwise opaque (unknown)

United States, in order:
  0. a short list of documented overrides (data/labels/us_overrides.csv)
  1. on a Wikipedia list of place names of Native American origin -> native,
     unless the listed name is plain English ('Licking', 'Raccoon', 'Red') or
     a common French or Spanish word ('Platte', 'Loup'): those are
     translations, and the original word was replaced -> newcomer
  2. an opaque word that is a tribe's name (Cheyenne, Kansa) -> ethnonym
  3. transparent in English, a personal name, Spanish, French, Dutch, German
     -> newcomer, unless the reading is weak (an English proper noun matched
     only as a surname or rare foreign word, like Tennessee) and the model
     below gives p >= 0.9
  4. otherwise scored by a character n-gram model trained on the Wikipedia
     list against rare surnames and foreign words; p >= 0.5 -> native

Ethnonyms are kept apart: the word is indigenous but the act of naming a
river after a people was usually a European one.

Output: data/derived/england_origin.parquet, us_origin.parquet,
        results/classification_report.txt
"""
import zipfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from common import DERIVED, LABELS, RAW, RESULTS
from lexicon import OUT as LEXDIR, load
from wordfreq import zipf_frequency

from transparency import LANGS, is_english, transparent

REPORT = []
STATES = {s.lower() for s in """Alabama Alaska Arizona Arkansas California Colorado Connecticut Dakota
Delaware Florida Georgia Idaho Illinois Indiana Iowa Kansas Kentucky Massachusetts Michigan Minnesota
Mississippi Missouri Montana Nebraska Nevada Ohio Oklahoma Oregon Tennessee Texas Utah Wisconsin
Wyoming""".split()}


def say(*lines):
    for line in lines:
        print(line)
        REPORT.append(line)


def england():
    r = pd.read_parquet(DERIVED / "england_rivers.parquet")
    lab = pd.read_parquet(DERIVED / "england_labels.parquet")
    r = r.merge(lab[["river_id", "origin", "category", "county", "generic_rule", "uncertain"]],
                on="river_id", how="left")
    tr = r.specific.map(lambda s: transparent(s, "en")[:2])
    r["transparent"] = [t[0] is not None for t in tr]
    r["how"] = [t[1] for t in tr]
    src = np.select([r.origin.notna() & ~r.generic_rule.fillna(False).astype(bool),
                     r.origin.notna(), r.transparent],
                    ["ekwall", "ekwall-generic", "transparent"], "opaque")
    r["source"] = src
    r["old"] = np.select([r.origin.eq("pre-English"), r.origin.eq("newcomer"),
                          r.origin.isna() & r.transparent], [1.0, 0.0, 0.0], np.nan)
    r.loc[r.specific.eq(""), ["old", "source"]] = [np.nan, "no specific"]

    ek = r[r.source.eq("ekwall") & r.origin.isin(["pre-English", "newcomer"])]
    t_pre = ek.loc[ek.origin.eq("pre-English"), "transparent"].mean()
    t_new = ek.loc[ek.origin.eq("newcomer"), "transparent"].mean()
    say("ENGLAND",
        f"  named rivers: {len(r):,}",
        *[f"  {k:15} {v:,}" for k, v in r.source.value_counts().items()],
        f"  transparency check on Ekwall's county labels:",
        f"    newcomer names read as transparent: {t_new:.0%} (sensitivity)",
        f"    pre-English names read as transparent: {t_pre:.0%} (false newcomer rate)",
        "    pre-English names that look English: " + ", ".join(
            ek.loc[ek.origin.eq("pre-English") & ek.transparent, "specific"].drop_duplicates().head(25)))
    r.to_parquet(DERIVED / "england_origin.parquet", index=False)
    return r


def native_model(native, negatives):
    x = list(native) + list(negatives)
    y = np.r_[np.ones(len(native)), np.zeros(len(negatives))]
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
                          LogisticRegression(C=4, max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    model.fit(x, y)
    return model, pd.Series(p[:len(native)], index=list(native)), roc_auc_score(y, p), p, y


def united_states():
    r = pd.read_parquet(DERIVED / "us_rivers.parquet")
    nat = pd.read_csv(LEXDIR / "native_wikipedia.csv")
    nat = nat[nat.specific.str.split().str.len() <= 2]
    # the lists include English names given in honour of Native people (Logan,
    # Jefferson) and words like 'indian'; drop names made only of English naming
    # words and very common surnames (20,000+ bearers: Logan, Jefferson, but not
    # Santee or Dee), and not given names: Tennessee and Missouri were both
    drop = load("force_english") | load("surnames_very_common")
    native_set = {s for s in nat.specific if not set(s.split()) <= drop}
    eth = load("ethnonyms")
    en_proper = load("english_proper")

    def ethnonym(spec):
        for w in spec.split():
            if len(w) >= 4 and any(w == e or (len(e) >= 5 and (
                    (w.startswith(e) and len(w) - len(e) <= 2) or (e.startswith(w) and len(e) - len(w) <= 1)))
                    for e in eth):
                return True
        return False

    specs = pd.Series(r.specific.unique())
    tr = {s: transparent(s, "us") for s in specs}
    is_eth = {s: ethnonym(s) for s in specs if tr[s][0] is None or tr[s][2]}

    # training data for the character model
    pos = sorted({s for s in native_set if transparent(s, "us")[0] is None})
    with zipfile.ZipFile(RAW / "lexicon" / "census_surnames_2010.zip") as z:
        sn = pd.read_csv(z.open("Names_2010Census.csv"), keep_default_na=False, na_values=["(S)"])
    rng = np.random.default_rng(0)
    rare_sn = sn[(sn["count"].between(300, 3000)) & (sn.pctaian.fillna(0) < 5)].name.str.lower()
    rare_sn = [w for w in rare_sn if w not in native_set and w not in eth]
    foreign = [w for l in ("es", "fr", "nl", "de") for w in rng.choice(sorted(load(l)), 1500, replace=False)
               if len(w) >= 5]
    english = [w for w in rng.choice(sorted(load("english")), 3000, replace=False) if len(w) >= 6]
    neg = sorted(set(rng.choice(rare_sn, 6000, replace=False)) | set(foreign) | set(english))
    neg = [w for w in neg if w not in native_set]
    model, p_pos_cv, auc, p_all, y_all = native_model(pos, neg)
    neg_cv = p_all[len(pos):]

    # score every word the lexicon could not read, or read only weakly
    scored = [s for s in specs if s and (tr[s][0] is None or tr[s][2]) and s not in native_set]
    p = dict(zip(scored, model.predict_proba(scored)[:, 1])) if scored else {}
    over = pd.read_csv(LABELS / "us_overrides.csv").set_index("specific")

    def label(s):
        if not s:
            return "no specific", np.nan
        if s in over.index:
            return over.at[s, "origin"], float(over.at[s, "old"])
        cls, _, weak = tr[s]
        if s in native_set:
            # plain English ('Bear', 'Licking', 'Red') or a common French or Spanish word
            # ('Platte', 'Loup', 'Brule') is listed as a translation: the word was replaced
            if cls == "english" and not weak:
                return "english", 0.0
            if cls in LANGS and all(zipf_frequency(w, LANGS[cls]) >= 3.5 for w in s.split()
                                    if not is_english(w)):
                return cls, 0.0
            return "native (list)", 1.0
        if cls is None and is_eth.get(s):
            return "ethnonym", 1.0
        if cls and not (weak and p.get(s, 0) >= 0.9):
            return cls, 0.0
        if p[s] >= 0.5:
            return "native (model)", 1.0
        # proper nouns the model does not find indigenous: Egypt, Antioch, Georgetown
        if all(w in en_proper or transparent(w, "en")[0] for w in s.split()):
            return "english", 0.0
        return "opaque", np.nan

    lab = {s: label(s) for s in specs}
    r["origin"] = r.specific.map(lambda s: lab[s][0])
    r["old"] = r.specific.map(lambda s: lab[s][1])
    # a state's name on a creek ('Texas Creek') was given after the state; only the
    # largest river carrying a state's name (Ohio, Missouri, Tennessee) can be its source
    st = r.specific.isin(STATES) & r.old.eq(1)
    namesake = r[st].sort_values("da_km2", ascending=False).drop_duplicates("specific").index
    r.loc[st & ~r.index.isin(namesake), "origin"] = "transferred"
    # tribe names flagged whether or not they are on the list (Missouri, Kansas, Kickapoo)
    r["ethnonym_word"] = r.specific.map(lambda s: bool(s) and ethnonym(s))
    r["p_native"] = r.specific.map(lambda s: p.get(s, p_pos_cv.get(s, np.nan)))
    r["how"] = r.specific.map(lambda s: tr[s][1])

    # how would the automatic steps (2-4) have labelled the listed names?
    listed = r[r.origin.eq("native (list)")].drop_duplicates("specific")
    def automatic(s):
        cls, _, weak = tr[s]
        if cls and not (weak and p_pos_cv.get(s, 0) >= 0.9):
            return cls
        if ethnonym(s):
            return "ethnonym"
        return "native (model)" if p_pos_cv.get(s, 0) >= 0.5 else "opaque"
    auto = listed.specific.map(automatic)
    say("",
        "UNITED STATES",
        f"  named rivers: {len(r):,} ({r.specific.nunique():,} distinct specifics)",
        *[f"  {k:15} {v:,}" for k, v in r.origin.value_counts().items()],
        f"  character model: {len(pos):,} listed native words vs {len(neg):,} surnames and foreign words",
        f"    cross-validated AUC {auc:.3f}; at p >= 0.5, recall {np.mean(p_pos_cv >= 0.5):.0%}, "
        f"false positive rate {np.mean(neg_cv >= 0.5):.0%}",
        f"  listed native names found on rivers: {len(listed):,}; without the list the automatic steps give:",
        *[f"    {k:15} {v:.0%}" for k, v in auto.value_counts(normalize=True).items()],
        "  listed names the lexicon calls European: " + ", ".join(
            f"{s} ({tr[s][0]})" for s in listed.specific if tr[s][0])[:600],
        "  model-native examples (largest rivers): " + ", ".join(
            r[r.origin.eq("native (model)")].sort_values("da_km2", ascending=False).name.head(30)),
        "  opaque examples (largest rivers): " + ", ".join(
            r[r.origin.eq("opaque")].sort_values("da_km2", ascending=False).name.head(30)))
    r.to_parquet(DERIVED / "us_origin.parquet", index=False)
    return r


def main():
    england()
    united_states()
    (RESULTS / "classification_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
