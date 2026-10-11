"""Curve figures.

figures/retention_curves.png  England by zone, US by region, Australia and New Zealand
figures/contact.png           size needed for even odds, by US county founding period
                              and by date of English conquest
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analyze import REGIONS
from common import DERIVED, FIGURES, RESULTS

BG, TEXT, SUBTLE, GRID = "#0a0d12", "#e8e4da", "#8a8f98", "#1c222b"
ENG = [("west of 2.5°W", lambda d: d.lon < -2.5, "#f6b73c"),
       ("middle", lambda d: (d.lon >= -2.5) & (d.lon < -1.25), "#c9b98f"),
       ("east of 1.25°W", lambda d: d.lon >= -1.25, "#4f86b8")]
US = {"Colonial East (Atlantic slope)": ("Colonial East", "#f6b73c"),
      "Southeast interior": ("Southeast interior", "#e8885a"),
      "Ohio-Great Lakes-Upper Mississippi": ("Ohio, Great Lakes, upper Mississippi", "#c9b98f"),
      "Spanish Southwest": ("Spanish Southwest", "#c780a0"),
      "Plains": ("Plains", "#8597ab"),
      "Mountain and Pacific West": ("Mountain and Pacific West", "#4f86b8")}


def curve(ax, d, col, label, edges, area="area"):
    d = d.assign(larea=np.log10(d[area]))
    m = smf.logit("old ~ larea", d).fit(disp=0)
    xs = np.linspace(np.log10(max(d[area].quantile(0.01), 1)), np.log10(d[area].max()), 200)
    ax.plot(10 ** xs, m.predict(pd.DataFrame({"larea": xs})), color=col, lw=2, label=label, zorder=3)
    g = d.groupby(pd.cut(d[area], edges), observed=True).agg(x=(area, "median"), y=("old", "mean"), n=("old", "size"))
    g = g[g.n >= 15]
    ax.scatter(g.x, g.y, s=np.clip(g.n / 25, 8, 70), color=col, alpha=0.8, lw=0, zorder=4)


def binned_line(ax, d, col, label, edges, area="area"):
    """Binned shares with Wilson intervals, no curve shape imposed (used where the
    logistic misfits: Australia's share stops rising among the largest rivers)."""
    from stats import wilson
    g = d.groupby(pd.cut(d[area], edges), observed=True).agg(x=(area, "median"), k=("old", "sum"), n=("old", "size"))
    g = g[g.n >= 15]
    lo, hi = wilson(g.k, g.n)
    ax.fill_between(g.x, lo, hi, color=col, alpha=0.15, lw=0)
    ax.plot(g.x, g.k / g.n, color=col, lw=2, label=label, zorder=3)
    ax.scatter(g.x, g.k / g.n, s=np.clip(g.n / 25, 8, 70), color=col, alpha=0.9, lw=0, zorder=4)


def style(ax, title, ylabel=False):
    ax.set_facecolor(BG)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(color=GRID, lw=0.6)
    ax.axhline(0.5, color=SUBTLE, lw=0.6, ls=(0, (3, 3)))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=SUBTLE, labelsize=9)
    ax.set_xlabel("catchment area, km²", color=SUBTLE, fontsize=10)
    if ylabel:
        ax.set_ylabel("share of rivers with an old name", color=SUBTLE, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=12, loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for t, h in zip(leg.get_texts(), leg.legend_handles):
        t.set_color(h.get_color())


def au_with_hand_labels(au):
    from common import LABELS
    top = pd.read_csv(LABELS / "au_top150_sample.csv").reset_index()
    top["rank"] = top["index"] + 1
    top = top.merge(pd.read_csv(LABELS / "au_top150_labels.csv")[["rank", "truth"]], on="rank")
    hand = top.set_index("group").truth.map({"old": 1.0, "new": 0.0})
    au = au.copy()
    m = au.group.isin(hand.index)
    au.loc[m, "old"] = au.loc[m, "group"].map(hand)
    return au


def retention():
    o = pd.read_parquet(DERIVED / "england_analysis.parquet")
    ek = o[o.source.isin(["ekwall", "ekwall-generic"]) & o.old.notna()].rename(columns={"area_km2": "area"})
    us = pd.read_parquet(DERIVED / "us_analysis.parquet")
    us = us[us.old.notna()].rename(columns={"da_km2": "area"})
    au = pd.read_parquet(DERIVED / "australia_origin.parquet")
    nz = pd.read_parquet(DERIVED / "nz_origin.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=220, facecolor=BG, sharey=True)
    for label, f, col in ENG:
        curve(axes[0], ek[f(ek)], col, label, [0, 10, 30, 100, 300, 1000, 1e4, 1e5])
    style(axes[0], "England: pre-English names", ylabel=True)
    for key, f in REGIONS.items():
        label, col = US[key]
        curve(axes[1], us[f(us.lon, us.lat)], col, label, [0, 10, 100, 1e3, 1e4, 1e5, 1e7])
    style(axes[1], "United States: indigenous names")
    au = au_with_hand_labels(au)
    for d, label, col in ((nz, "New Zealand (Maori)", "#f6b73c"), (au, "Australia (Aboriginal)", "#4f86b8")):
        d = d[d.old.notna() & (d.area > 0)]
        binned_line(axes[2], d, col, label, [0, 10, 30, 100, 300, 1000, 3000, 1e4, 1e5, 1e7])
    style(axes[2], "Australia and New Zealand (binned shares)")
    fig.text(0.05, 0.985, "Big rivers keep old names, but how big depends on how the land changed hands",
             color=TEXT, fontsize=15, fontweight="bold", va="top")
    fig.text(0.05, 0.935, "England and US: logistic fits and binned shares; Australia and New Zealand: binned shares with 95% bands "
             "(dot size: number of rivers). England: Ekwall's 1928 labels. "
             "Catchment area from HydroRIVERS (England, Australia, New Zealand) and NHDPlus (US).",
             color=SUBTLE, fontsize=8.5, va="top")
    fig.tight_layout(rect=(0.03, 0, 1, 0.9))
    fig.savefig(FIGURES / "retention_curves.png", facecolor=BG)
    plt.close(fig)
    print("wrote retention_curves.png")


def contact():
    t = pd.read_csv(RESULTS / "mechanism_thresholds.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=220, facecolor=BG)
    for ax, country, title, xlabel, note in (
            (axes[0], "US", "United States: by year the first county was organised", "first county at the river's outlet",
             "earlier settlement: longer coexistence with Native speakers before removal"),
            (axes[1], "England", "England: by date of English conquest", "approximate start of English rule",
             "later conquest: the British population lasted longer and merged with the English (Ekwall)")):
        d = t[t.country.eq(country)].sort_values("order")
        cap = 7.0
        y = np.minimum(d.x50, cap)
        lo, hi = np.minimum(d.lo, cap), np.minimum(d.hi, cap)
        ax.vlines(range(len(d)), 10 ** lo, 10 ** hi, color=SUBTLE, lw=1.4)
        ax.scatter(range(len(d)), 10 ** y, s=60, color="#f6b73c", zorder=3)
        for i, (v, n) in enumerate(zip(d.x50, d.n)):
            ax.annotate(f"{10 ** v:,.0f} km²" if v < cap else "beyond any river", (i, 10 ** min(v, cap)),
                        xytext=(8, 0), textcoords="offset points", color=TEXT, fontsize=8.5, va="center")
            ax.annotate(f"n={n:,}", (i, 10 ** min(v, cap)), xytext=(8, -12), textcoords="offset points",
                        color=SUBTLE, fontsize=7.5, va="center")
        ax.set_yscale("log")
        ax.set_ylim(5, 2e7)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d.group, color=SUBTLE, fontsize=9)
        ax.set_xlim(-0.5, len(d) - 0.2)
        ax.set_facecolor(BG)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=SUBTLE, labelsize=9)
        ax.set_xlabel(xlabel, color=SUBTLE, fontsize=10)
        ax.set_title(title, color=TEXT, fontsize=12, loc="left", pad=22)
        ax.text(0, 1.02, note, transform=ax.transAxes, color=SUBTLE, fontsize=8.5, va="bottom")
    axes[0].set_ylabel("catchment at which half of names are old, km²", color=SUBTLE, fontsize=10)
    fig.text(0.05, 0.98, "The longer newcomers lived beside the old language, the smaller the rivers that kept it",
             color=TEXT, fontsize=14, fontweight="bold", va="top")
    fig.text(0.05, 0.925, "Points: size at which half of river names are old; bars: 95% interval, clustered by basin.",
             color=SUBTLE, fontsize=8.5, va="top")
    fig.tight_layout(rect=(0.02, 0, 1, 0.9))
    fig.savefig(FIGURES / "contact.png", facecolor=BG)
    plt.close(fig)
    print("wrote contact.png")


if __name__ == "__main__":
    retention()
    contact()
