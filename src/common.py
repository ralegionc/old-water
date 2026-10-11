"""Paths and name parsing shared by every step."""
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
LABELS = ROOT / "data" / "labels"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
for d in (DERIVED, LABELS, FIGURES, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# Words that say "this is a stream" and carry no naming information.
# The specific is what is left once these are removed.
GENERIC_SUFFIX = {
    # England
    "river", "brook", "beck", "burn", "water", "stream", "drain", "gill", "ghyll",
    "sike", "syke", "dike", "dyke", "ditch", "pool", "cut", "lode", "rife", "rithe",
    "eau", "leat", "fleet", "sewer", "gutter", "channel", "rhyne", "rhine", "reen",
    "lane", "goit", "race",
    # United States
    "creek", "branch", "run", "fork", "wash", "bayou", "swamp", "slough", "arroyo",
    "kill", "prong", "outlet", "canal", "inlet", "coulee", "draw", "gulch", "canyon",
    "hollow", "spring", "springs", "lateral", "bottom", "valley",
    "riviere", "rio", "lake",
    # Australia and New Zealand
    "gully", "rivulet", "anabranch", "watercourse", "billabong", "waterhole",
}
GENERIC_PREFIX = {"river", "afon", "the", "rio", "arroyo", "bayou", "riviere", "canada"}
MODIFIERS = {
    "little", "big", "great", "north", "south", "east", "west", "middle", "upper",
    "lower", "new", "old", "back", "main", "first", "second", "third", "dry", "fork",
    "branch", "of", "northeast", "northwest", "southeast", "southwest", "mid",
}


def fold(s: str) -> str:
    """Lower-case and strip accents, keep letters, apostrophes and spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_name(name: str):
    """Return (specific, generic) for a watercourse name.

    'River Little Ouse' -> ('ouse', 'river'); 'Glaze Brook' -> ('glaze', 'brook');
    'North Fork Kentucky River' -> ('kentucky', 'river'); 'Black Brook' -> ('black', 'brook').
    Returns specific '' when nothing is left (e.g. 'The River').
    """
    words = fold(name).split()
    generic = ""
    # "X River of the North", "Fork of X": drop trailing "of ..." qualifiers
    if "of" in words[1:]:
        i = words.index("of", 1)
        tail = words[i + 1:]
        head = words[:i]
        # "Burn of Norwick" style: the specific comes after "of"
        if head and head[-1] in GENERIC_SUFFIX and all(h in GENERIC_SUFFIX | MODIFIERS for h in head):
            # "Burn of Norwick", "East Fork of the Little Miami River"
            generic, words = head[-1], [w for w in tail if w != "the"]
        else:
            words = head
    # "Raystown Branch Juniata River": the stream's own name is Raystown;
    # "North Fork Humboldt River": only modifiers before Fork, so it is the Humboldt
    for i, w in enumerate(words[1:-1], 1):
        if w in ("fork", "branch", "prong"):
            head = words[:i]
            words = words[i + 1:] if all(h in MODIFIERS for h in head) else head
            generic = generic or w
            break
    while words and words[0] in GENERIC_PREFIX:
        generic = generic or words[0]
        words = words[1:]
    while len(words) > 1 and words[-1] in GENERIC_SUFFIX:
        generic = generic or words[-1]
        words = words[:-1]
    while len(words) > 1 and words[0] in MODIFIERS:
        words = words[1:]
    while len(words) > 1 and words[-1] in GENERIC_SUFFIX | MODIFIERS:
        words = words[:-1]
    spec = " ".join(words)
    if spec in GENERIC_SUFFIX:
        spec = ""
    return spec, generic


if __name__ == "__main__":
    for n in ["River Little Ouse", "Glaze Brook", "North Fork Kentucky River", "Black Brook",
              "Devils Water", "Burn of Norwick", "Red River of the North", "Bayou Teche",
              "Rio Grande", "East Branch Delaware River", "Mill Creek", "The River",
              "Bogue Chitto", "Little Colorado River", "River Avon", "Bourne Eau",
              "Raystown Branch Juniata River", "North Fork Humboldt River",
              "Double Mountain Fork Brazos River", "East Fork of the Little Miami River"]:
        print(f"{n!r:35} -> {split_name(n)}")
