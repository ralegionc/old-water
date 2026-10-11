"""How accurate are the US labels, and what does the curve look like corrected?

Two checks:
  1. GNIS histories (gnis_gold.py): external text, read by rules.
  2. A stratified audit of 400 names: 40 in each of 5 size bands x {labelled
     old, labelled newcomer}. Each name was read by the assistant that built
     this project (Claude, an AI model), using the GNIS history where one
     exists and general etymological knowledge otherwise; 'unclear' was
     allowed and is excluded. These are not expert labels. Their agreement
     with GNIS, where both exist, is reported as a check on the reader.

Correction: in each size band x label cell of the audit, P(truly old) is
estimated with a Jeffreys prior. Every river then gets a probability of truly being old,
and the retention curve is refitted to those probabilities (fractional
logit), clustered by HUC4 subregion. The audit is bootstrapped within its
strata to carry its own uncertainty into the corrected curve.

Usage: python audit.py sample    (writes data/labels/us_audit_sample.csv)
       python audit.py           (evaluates data/labels/us_audit_labels.csv)
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from common import DERIVED, LABELS, RAW, RESULTS
from stats import wilson

BANDS = [0, 10, 100, 1000, 10_000, 1e8]
BAND_NAMES = ["<10", "10-100", "100-1k", "1k-10k", ">10k"]
SAMPLE = LABELS / "us_audit_sample.csv"
LABELLED = LABELS / "us_audit_labels.csv"
RNG = np.random.default_rng(7)
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def sample(per_cell=40):
    u = pd.read_parquet(DERIVED / "us_origin.parquet")
    u = u[u.old.notna() & (u.da_km2 > 0)].copy()
    u["band"] = pd.cut(u.da_km2, BANDS, labels=BAND_NAMES)
    states = gpd.read_file(f"zip://{RAW / 'ne_10m_admin_1_states_provinces.zip'}", columns=["name", "iso_a2"])
    states = states[states.iso_a2.eq("US")]
    pts = gpd.GeoDataFrame(u[["gnis_id"]], geometry=gpd.points_from_xy(u.lon, u.lat), crs=4326)
    st = gpd.sjoin_nearest(pts, states[["name", "geometry"]], max_distance=0.2).drop_duplicates("gnis_id")
    u["state"] = u.gnis_id.map(st.set_index("gnis_id")["name"])
    gold = pd.read_parquet(DERIVED / "gnis_gold.parquet", columns=["gnis_id", "history"])
    u["gnis_history"] = u.gnis_id.astype(str).map(gold.drop_duplicates("gnis_id").set_index("gnis_id").history)
    picks = []
    for (band, old), g in u.groupby(["band", "old"], observed=True):
        picks.append(g.sample(min(per_cell, len(g)), random_state=int(RNG.integers(1 << 31))))
    s = pd.concat(picks).sample(frac=1, random_state=3).reset_index(drop=True)
    s.insert(0, "audit_id", np.arange(1, len(s) + 1))
    cols = ["audit_id", "name", "state", "da_km2", "band", "origin", "old", "how", "p_native", "gnis_history", "gnis_id"]
    s[cols].to_csv(SAMPLE, index=False)
    print(f"wrote {len(s)} names to {SAMPLE}")


def corrected_fit(u, q):
    X = sm.add_constant(u[["larea"]])
    m = sm.GLM(q, X, family=sm.families.Binomial()).fit(cov_type="cluster",
                                                       cov_kwds={"groups": pd.factorize(u.huc4)[0]})
    b0, b1 = m.params["const"], m.params["larea"]
    return b1, -b0 / b1


def evaluate():
    s = pd.read_csv(SAMPLE)
    lab = pd.read_csv(LABELLED)
    a = s.merge(lab, on="audit_id")
    say(f"audit names: {len(a)}; unclear: {(a.truth == 'unclear').sum()}")
    a = a[a.truth.isin(["old", "new"])].copy()
    a["truth"] = a.truth.eq("old").astype(float)
    a["pred"] = a.old.astype(float)
    a["larea"] = np.log10(a.da_km2)

    say("  share truly old, by size band and label (Wilson 95% CI):")
    for band in BAND_NAMES:
        for pred, word in ((1.0, "labelled old"), (0.0, "labelled newcomer")):
            g = a[a.band.eq(band) & a.pred.eq(pred)]
            if len(g):
                lo, hi = wilson(g.truth.sum(), len(g))
                say(f"    {band:7} {word:18} n={len(g):3}  truly old {g.truth.mean():6.1%} [{lo:.0%}, {hi:.0%}]")
    ppv = a[a.pred.eq(1)].truth.mean()
    fo = a[a.pred.eq(0)].truth.mean()
    say(f"  overall: labelled-old names truly old {ppv:.0%}; labelled-newcomer names truly old {fo:.1%}")

    gold = pd.read_parquet(DERIVED / "gnis_gold.parquet", columns=["gnis_id", "gold"]).drop_duplicates("gnis_id")
    a["gnis"] = a.gnis_id.astype(str).map(gold.set_index("gnis_id").gold)
    both = a[a.gnis.isin(["indigenous", "newcomer", "ethnonym"])]
    if len(both):
        agree = (both.truth == both.gnis.isin(["indigenous", "ethnonym"]).astype(float)).mean()
        say(f"  audit labels agree with the GNIS history rule on {agree:.0%} of {len(both)} names where both exist")

    u = pd.read_parquet(DERIVED / "us_analysis.parquet")
    u = u[u.old.notna()].copy()
    u["pred"] = u.old.astype(float)
    u["band"] = pd.cut(u.da_km2, BANDS, labels=BAND_NAMES).astype(str)
    a["band"] = a.band.astype(str)

    def cell_rates(d, prior=0.5):
        """P(truly old) in each size band x label cell, with a Jeffreys prior so a
        cell with no truly old names in the sample is not treated as exactly zero."""
        g = d.groupby(["band", "pred"]).truth.agg(["sum", "size"])
        return ((g["sum"] + prior) / (g["size"] + 2 * prior)).to_dict()

    def q_for(d, rates):
        return np.array([rates[(b, p)] for b, p in zip(d.band, d.pred)])

    rates = cell_rates(a)
    q = q_for(u, rates)
    b1, x50 = corrected_fit(u, q)
    raw = smf.logit("old ~ larea", u).fit(disp=0)
    boot = []
    cells = a.groupby(["band", "pred"], observed=True).indices
    for _ in range(300):
        take = np.concatenate([RNG.choice(ix, len(ix), replace=True) for ix in cells.values()])
        boot.append(corrected_fit(u, q_for(u, cell_rates(a.iloc[take]))))
    boot = np.array(boot)
    # pessimistic about misses: P(truly old | labelled newcomer) taken from the GNIS
    # histories, where indigenous names are over-represented (their stories get written down)
    gold = pd.read_parquet(DERIVED / "gnis_gold.parquet")
    gold = gold[gold.gold.isin(["indigenous", "newcomer", "ethnonym"]) & gold.old.eq(0)].copy()
    gold["band"] = pd.cut(gold.da_km2, BANDS, labels=BAND_NAMES).astype(str)
    miss = gold.groupby("band").gold.apply(lambda s: (s.isin(["indigenous", "ethnonym"]).sum() + 0.5) / (len(s) + 1))
    pess = {k: (miss.get(k[0], v) if k[1] == 0.0 else v) for k, v in rates.items()}
    b1p, x50p = corrected_fit(u, q_for(u, pess))

    say("  P(truly old) used for the correction (audit cells, Jeffreys prior):")
    for band in BAND_NAMES:
        say(f"    {band:7} labelled old {rates[(band, 1.0)]:.2f}   labelled newcomer {rates[(band, 0.0)]:.3f}"
            f"   (GNIS-based miss rate {miss.get(band, np.nan):.3f})")
    say("  retention curve, all US rivers (cluster-robust by HUC4):")
    say(f"    as labelled:            odds ratio per 10x {np.exp(raw.params['larea']):.2f}, "
        f"half old at {10 ** (-raw.params['Intercept'] / raw.params['larea']):,.0f} km2")
    say(f"    corrected for accuracy: odds ratio per 10x {np.exp(b1):.2f} "
        f"[{np.exp(np.percentile(boot[:, 0], 2.5)):.2f}, {np.exp(np.percentile(boot[:, 0], 97.5)):.2f}], "
        f"half old at {10 ** x50:,.0f} km2 [{10 ** np.percentile(boot[:, 1], 2.5):,.0f}, "
        f"{10 ** np.percentile(boot[:, 1], 97.5):,.0f}] (interval from resampling the audit)")
    say(f"    corrected, misses at GNIS rates (pessimistic): odds ratio per 10x {np.exp(b1p):.2f}, "
        f"half old at {10 ** x50p:,.0f} km2")
    from analyze import REGIONS
    say("  corrected, by region (audit rates / GNIS miss rates):")
    for name, f in REGIONS.items():
        g = u[f(u.lon, u.lat)]
        b, x = corrected_fit(g, q_for(g, rates))
        bp, xp = corrected_fit(g, q_for(g, pess))
        say(f"    {name:36} odds ratio {np.exp(b):.2f} / {np.exp(bp):.2f}, half old at {10 ** x:,.0f} / {10 ** xp:,.0f} km2")
    u["q_true_old"] = q
    u[["gnis_id", "q_true_old"]].to_parquet(DERIVED / "us_corrected.parquet", index=False)
    (RESULTS / "audit.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sample() if "sample" in sys.argv[1:] else evaluate()
