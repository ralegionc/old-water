"""Known-answer checks for the parts of the pipeline that are easy to get wrong.

Run from the project root:  python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import split_name  # noqa: E402
from ekwall import core  # noqa: E402


@pytest.mark.parametrize("name, specific", [
    ("River Little Ouse", "ouse"),
    ("Glaze Brook", "glaze"),
    ("North Fork Kentucky River", "kentucky"),
    ("Raystown Branch Juniata River", "raystown"),
    ("East Fork of the Little Miami River", "miami"),
    ("Burn of Norwick", "norwick"),
    ("Bayou Teche", "teche"),
    ("Bogue Chitto", "bogue chitto"),
    ("Bourne Eau", "bourne"),
    ("Double Mountain Fork Brazos River", "double mountain"),
    ("Coondoomea Creek", "coondoomea"),
    ("Clearys Gully", "clearys"),
])
def test_split_name(name, specific):
    assert split_name(name)[0] == specific


def test_core_matches_ekwall_forms():
    assert core("Kielder Burn") == core("Kielder")
    assert core("Devil's Water") == core("Devils Water")
    assert core("Little Avon River") != core("River Avon")


@pytest.fixture(scope="module")
def transparent():
    from transparency import transparent
    return transparent


@pytest.mark.parametrize("spec, country, expect", [
    ("ravensbourne", "en", "english"),
    ("blackwater", "en", "english"),
    ("avon", "en", None),
    ("susquehanna", "us", None),
    ("chattahoochee", "us", None),
    ("des moines", "us", "french"),
    ("brazos", "us", "spanish"),
    ("lavaca", "us", "spanish"),
    ("brushy", "us", "english"),
    ("indian", "us", "english"),
    ("jefferson", "us", "personal"),
])
def test_transparent(transparent, spec, country, expect):
    assert transparent(spec, country)[0] == expect


def test_proper_nouns_never_read_as_common_words(transparent):
    # Ohio is a proper noun in English and German dictionaries; it must stay opaque
    assert transparent("ohio", "us")[0] is None
    assert transparent("mississippi", "us")[0] is None


def test_maori_spelling_rule():
    from australasia import MAORI
    for w in ("waikato", "whanganui", "rangitikei", "waimakariri"):
        assert MAORI.match(w)
    for w in ("clutha", "buller", "grey", "selwyn"):
        assert not MAORI.match(w)


def test_weighted_logit_recovers_known_curve():
    from surface import wlogit
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 5, 20_000)
    y = (rng.random(20_000) < 1 / (1 + np.exp(-(-3 + 1.2 * x)))).astype(float)
    b0, b1 = wlogit(x, y, np.ones_like(x))
    assert abs(b0 + 3) < 0.15 and abs(b1 - 1.2) < 0.06


def test_group_rivers_joins_touching_same_name_lines():
    import geopandas as gpd
    from shapely.geometry import LineString
    from snap import group_rivers
    lines = gpd.GeoDataFrame({"name": ["A", "A", "A", "B"]}, geometry=[
        LineString([(0, 0), (100, 0)]), LineString([(100, 0), (200, 0)]),
        LineString([(5000, 0), (5100, 0)]), LineString([(100, 0), (100, 100)])], crs=3577)
    g = group_rivers(lines)
    assert g[0] == g[1] and g[2] != g[0] and g[3] != g[0]


def test_cluster_bootstrap_is_wider_than_naive_for_clustered_data():
    import pandas as pd
    from stats import fit
    rng = np.random.default_rng(1)
    n_basins, per = 60, 50
    basin = np.repeat(np.arange(n_basins), per)
    shift = rng.normal(0, 1.5, n_basins)[basin]  # shared naming history within a basin
    x = rng.uniform(0, 5, n_basins * per)
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-(-2 + 0.8 * x + shift)))).astype(float)
    df = pd.DataFrame({"larea": x, "old": y, "basin": basin})
    e = fit(df, x="larea", cluster="basin", boot=150)
    assert np.diff(e["slope_ci"])[0] > np.diff(e["naive_slope_ci"])[0]
