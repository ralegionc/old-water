"""Retention-curve fits with intervals that respect clustering.

Streams in one basin were often named by the same communities, so they are
not independent draws. Two ways to account for it:

  cluster bootstrap   resample whole basins with replacement and refit
                      (used for England's 1,024 Ekwall-labelled rivers)
  cluster-robust SE   sandwich covariance grouped by basin, delta method for
                      the half-old size (used for the large US, Australian and
                      New Zealand samples, where refitting is slow)

Every fit also records the naive interval so the effect of clustering is visible.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RNG = np.random.default_rng(1)


def _x50(params, x):
    return -params["Intercept"] / params[x]


def fit(df, x="larea", cluster=None, boot=0, formula=None, y="old"):
    """Logit of y on x. Returns slope, half-old point x50 (log10 units) and CIs."""
    formula = formula or f"{y} ~ {x}"
    naive = smf.logit(formula, df).fit(disp=0)
    est = dict(n=len(df), share_old=df[y].mean(), slope=naive.params[x], x50=_x50(naive.params, x),
               model=naive, naive_slope_ci=naive.conf_int().loc[x].to_numpy())
    groups = pd.factorize(df[cluster])[0] if cluster else None
    est["clusters"] = int(groups.max() + 1) if cluster else len(df)
    if cluster and boot:
        ids = np.unique(groups)
        rows = pd.Series(np.arange(len(df))).groupby(groups).apply(np.array)
        bs = []
        for _ in range(boot):
            take = np.concatenate(rows.loc[RNG.choice(ids, len(ids), replace=True)].to_numpy())
            try:
                p = smf.logit(formula, df.iloc[take]).fit(disp=0).params
                bs.append((p[x], _x50(p, x)))
            except Exception:
                continue
        bs = np.array(bs)
        est["slope_ci"] = np.percentile(bs[:, 0], [2.5, 97.5])
        est["x50_ci"] = np.percentile(bs[:, 1], [2.5, 97.5])
        est["method"] = f"basin bootstrap, {len(ids)} basins"
        return est
    m = smf.logit(formula, df).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups}) if cluster else naive
    b0, b1 = m.params["Intercept"], m.params[x]
    v = m.cov_params().loc[["Intercept", x], ["Intercept", x]].to_numpy()
    g = np.array([-1 / b1, b0 / b1 ** 2])
    se50, se1 = np.sqrt(g @ v @ g), np.sqrt(v[1, 1])
    est["slope_ci"] = b1 + np.array([-1.96, 1.96]) * se1
    est["x50_ci"] = est["x50"] + np.array([-1.96, 1.96]) * se50
    est["se_ratio"] = se1 / naive.bse[x]
    est["method"] = f"cluster-robust, {est['clusters']} basins" if cluster else "naive"
    return est


def coefs(df, formula, cluster=None, boot=0, terms=None):
    """Coefficient table for a multi-term logit, clustered by basin."""
    groups = pd.factorize(df[cluster])[0] if cluster else None
    if cluster and boot:
        m = smf.logit(formula, df).fit(disp=0)
        ids = np.unique(groups)
        rows = pd.Series(np.arange(len(df))).groupby(groups).apply(np.array)
        bs = []
        for _ in range(boot):
            take = np.concatenate(rows.loc[RNG.choice(ids, len(ids), replace=True)].to_numpy())
            try:
                bs.append(smf.logit(formula, df.iloc[take]).fit(disp=0).params)
            except Exception:
                continue
        bs = pd.DataFrame(bs)
        terms = terms or [t for t in m.params.index if t != "Intercept"]
        return {t: (m.params[t], *np.percentile(bs[t], [2.5, 97.5]), np.mean(np.sign(bs[t]) != np.sign(m.params[t])) * 2)
                for t in terms}, m
    m = smf.logit(formula, df).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups}) if cluster \
        else smf.logit(formula, df).fit(disp=0)
    terms = terms or [t for t in m.params.index if t != "Intercept"]
    ci = m.conf_int()
    return {t: (m.params[t], ci.loc[t, 0], ci.loc[t, 1], m.pvalues[t]) for t in terms}, m


def describe(name, e, unit, scale=lambda v: 10 ** v):
    lo, hi = e["x50_ci"]
    s_lo, s_hi = np.exp(e["slope_ci"])
    n_lo, n_hi = np.exp(e["naive_slope_ci"])
    return (f"  {name}: n={e['n']:,}, old share {e['share_old']:.1%}, odds ratio per 10x size "
            f"{np.exp(e['slope']):.2f} [{s_lo:.2f}, {s_hi:.2f}] (naive [{n_lo:.2f}, {n_hi:.2f}]), "
            f"half of names old at {scale(e['x50']):,.0f} {unit} [{scale(lo):,.0f}, {scale(hi):,.0f}]; {e['method']}")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binned(df, size, edges, label, rows, y="old"):
    for iv, g in df.groupby(pd.cut(df[size], edges), observed=True):
        k, n = g[y].sum(), len(g)
        lo, hi = wilson(k, n)
        rows.append(dict(curve=label, lo_edge=iv.left, hi_edge=iv.right, mid=g[size].median(),
                         n=n, share=k / n, ci_lo=lo, ci_hi=hi))
