"""Is a river name readable as words of a newcomer language?

transparent('ravensbourne', 'en') -> ('english', 'raven+s+bourne', False)
transparent('johnson', 'us')      -> ('personal', 'johnson', False)
transparent('des moines', 'us')   -> ('french', 'des moines', False)
transparent('susquehanna', 'us')  -> (None, '', False)

The third value flags a weak reading: a word that English dictionaries list
as a proper noun (Tennessee, Missouri) and that matched only as a surname,
a given name or a rare foreign word. classify.py lets the character model
overrule weak readings when it is confident the word is indigenous.

English and French lookups go through full hunspell morphology (spylls), so
'forked', 'mussels', 'moines' are found. Lookups are case-sensitive, so a
proper noun such as 'ohio' never matches a lower-case query. Spanish and
French also get an accent-free stem index with plural, gender, diminutive and
glued-article rules, since US toponyms drop accents (Alamito, Lavaca).
Compound pieces must be everyday words (wordfreq zipf >= 3) or listed
place-name elements, so a Native name does not dissolve into rare words.
"""
from functools import lru_cache

from spylls.hunspell import Dictionary
from wordfreq import zipf_frequency

from common import RAW
from lexicon import load

HUN = {l: Dictionary.from_files(str(RAW / "lexicon" / f"hunspell_{l}"))
       for l in ("en", "en-GB", "fr", "es", "nl", "de")}
ELEMENTS = load("elements")
SURNAMES = load("surnames")
HISPANIC = load("surnames_hispanic")
GIVEN = load("given")
SURNAMES_COMMON = load("surnames_common")
GIVEN_COMMON = load("given_common")
FORCE_ENGLISH = load("force_english")
SAINTS = load("saints")
EN_PROPER = load("english_proper")
# accent-folded stems for the Romance languages; German and Dutch use only
# case-sensitive lookups (the German list carries lower-case US state names)
STEMS = {l: load(l) for l in ("fr", "es")}
LANGS = {"french": "fr", "spanish": "es", "dutch": "nl", "german": "de"}  # tie order
ARTICLES = ("los", "las", "les", "des", "aux", "la", "le", "el", "du", "au")


@lru_cache(maxsize=None)
def is_english(w: str) -> bool:
    return HUN["en"].lookup(w) or HUN["en-GB"].lookup(w)


@lru_cache(maxsize=None)
def piece_ok(p: str) -> bool:
    if p in ELEMENTS:
        return len(p) >= 2
    floor = 2.5 if len(p) >= 5 else 3.0  # longer pieces are rarely accidental
    return len(p) >= 3 and zipf_frequency(p, "en") >= floor and is_english(p)


@lru_cache(maxsize=None)
def segment(word: str, depth: int = 2):
    """Split word into at most `depth` English pieces; allow a linking s or e."""
    if piece_ok(word):
        return word
    if depth == 1:
        return None
    for i in range(len(word) - 2, 1, -1):  # longest first piece first
        head, tail = word[:i], word[i:]
        if not piece_ok(head):
            continue
        for link in ("", "s", "e"):
            if link and not tail.startswith(link):
                continue
            rest = segment(tail[len(link):], depth - 1) if len(tail) > len(link) else None
            if rest:
                return f"{head}+{link + '+' if link else ''}{rest}"
    return None


def english_word(w: str):
    w = w.replace("'", "")
    if w in ELEMENTS or w in FORCE_ENGLISH or is_english(w):
        return w
    if w.endswith("s") and is_english(w[:-1]):
        return w
    # adjectives in -y / -ey: brushy, glady, briery, rooty, clifty
    for suf in ("ey", "y", "ie"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[: -len(suf)]
            if is_english(stem) or (stem[-1] == stem[-2] and is_english(stem[:-1])):
                return w
    return segment(w)


def variants(w: str, lang: str):
    """Accent-free inflectional variants to try against the stem index."""
    out = {w}
    for suf, rep in (("es", ""), ("s", ""), ("x", "")):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            out.add(w[: -len(suf)] + rep)
    if lang == "es":
        for v in list(out):
            if v.endswith("a"):
                out.add(v[:-1] + "o")
            for suf, rep in (("ito", "o"), ("ita", "a"), ("cito", ""), ("cita", ""),
                             ("illo", "o"), ("illa", "a"), ("osa", "oso"), ("ita", "o")):
                if v.endswith(suf) and len(v) > len(suf) + 2:
                    out.add(v[: -len(suf)] + rep)
    return out


@lru_cache(maxsize=None)
def foreign_word(w: str, lang: str, glued: bool = True) -> bool:
    if HUN[lang].lookup(w) or (lang == "fr" and "oe" in w and HUN["fr"].lookup(w.replace("oe", "œ"))):
        return True
    if lang in STEMS and any(v in STEMS[lang] for v in variants(w, lang)):
        return True
    if glued and lang in ("es", "fr"):  # 'lavaca' = la vaca, 'auglaize' = au glaise
        for a in ARTICLES:
            if w.startswith(a) and len(w) - len(a) >= 4 and foreign_word(w[len(a):], lang, False):
                return True
    return False


def personal(w: str):
    w = w.replace("'", "")
    for cand in (w, w[:-1] if w.endswith(("s", "y")) else None, w[:-2] if w.endswith("ie") else None):
        if cand and (cand in SURNAMES or cand in GIVEN):
            return "hispanic" if cand in HISPANIC else "personal"
    return None


def transparent(spec: str, country: str):
    """Return (class, how, weak) or (None, '', False) if the name is opaque."""
    words = spec.split()
    if not words:
        return None, "", False
    how, classes, weak = [], [], False
    for w in words:
        e = english_word(w)
        if e:
            classes.append("english"), how.append(e)
            continue
        if country != "us":
            return None, "", False
        if w in SAINTS:
            classes.append("saint"), how.append(w)
            continue
        hit = next((n for n, l in LANGS.items() if foreign_word(w, l)), None)
        if hit:
            classes.append(hit), how.append(w)
            if w in EN_PROPER and zipf_frequency(w, LANGS[hit]) < 3.0:
                weak = True  # e.g. a rare foreign word that is also a US place name
            continue
        p = personal(w)
        if p:
            classes.append(p), how.append(w)
            if w in EN_PROPER and w not in GIVEN_COMMON and w not in SURNAMES_COMMON:
                weak = True  # Tennessee, Missouri: rare surnames, but also place names
            continue
        return None, "", False
    foreign = [c for c in classes if c not in ("english", "personal", "hispanic", "saint")]
    if "saint" in classes and not foreign:
        foreign = ["spanish"]  # San/Santa/St with a saint's name
    if foreign:
        # 'rio blanco', 'des moines': a foreign word sets the class
        order = list(LANGS)
        return max(set(foreign), key=lambda c: (foreign.count(c), -order.index(c))), " ".join(how), weak
    for c in ("hispanic", "personal"):
        if c in classes and "english" not in classes:
            return c, " ".join(how), weak
    return "english", " ".join(how), weak


if __name__ == "__main__":
    for s, c in [("ravensbourne", "en"), ("mill", "en"), ("avon", "en"), ("dove", "en"),
                 ("susquehanna", "us"), ("johnson", "us"), ("des moines", "us"), ("bear", "us"),
                 ("pocomoke", "us"), ("chattahoochee", "us"), ("blanco", "us"), ("beaverdam", "us"),
                 ("forked deer", "us"), ("mattawoman", "us"), ("brazos", "us"), ("thames", "en"),
                 ("nueces", "us"), ("lavaca", "us"), ("alamito", "us"), ("atascosa", "us"),
                 ("ohio", "us"), ("tennessee", "us"), ("cheyenne", "us"), ("wabash", "us"),
                 ("marais des cygnes", "us"), ("boeuf", "us"), ("auglaize", "us"), ("musselshell", "us"),
                 ("republican", "us"), ("jefferson", "us"), ("mississippi", "us"), ("missouri", "us")]:
        print(f"{s:18} {c}  {transparent(s, c)}")
