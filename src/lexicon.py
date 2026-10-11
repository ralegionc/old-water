"""Word lists used to decide whether a river name is transparent.

A name is transparent in a language when it can be read as words of that
language: 'Mill Brook', 'Ravensbourne' (raven + s + bourne), 'Bear Creek',
'Rio Blanco'. Sources:

  hunspell dictionaries (wooorm/dictionaries): en, en-GB, fr, es, nl, de.
      Lower-case entries are common words; capitalised entries are proper
      nouns and are kept apart, so 'Mississippi' or 'Ohio' never count as
      English just because a dictionary lists them.
  wordfreq: word frequencies, used only to keep compound pieces to everyday
      words (frequency lists are full of proper nouns, so they never add words).
  US Census 2010 surnames: personal names (Johnson Creek). Surnames with
      20% or more American Indian / Alaska Native bearers are left out.
  US given names 1880-2008 (hadley/data-baby-names): only names already in
      use by 1920, so 'Dakota' or 'Cheyenne' as modern baby names do not
      make a Native word look European.
  Old English and Old Norse place-name elements, listed below.

Output: data/derived/lexicon/*.txt and native_wikipedia.csv
"""
import json
import re
import zipfile

import pandas as pd
from wordfreq import top_n_list, zipf_frequency

from common import DERIVED, RAW, fold, split_name

LEX = RAW / "lexicon"
OUT = DERIVED / "lexicon"
OUT.mkdir(exist_ok=True)

# Elements of English (Old English, Middle English) and Scandinavian place- and
# stream-names, from the standard element lists (Smith, English Place-Name
# Elements; Ekwall). Short forms are allowed as compound pieces.
ELEMENTS = """
brook broc burn bourne borne born beck bec well wall welle wella ey ea eau fleet flete
lade lake lode mere pool pol pole rithe rife sike syke sitch sich gill ghyll gil wath
water wash dike dyke ditch gote goit ford forth mouth spring keld kell tarn foss force
grain slack lane race leat cut drain sewer stream bach batch bache
ton tun ham hamm hamp ley leigh lee lea legh field feld worth wardine wick wich wych wyke
bury borough burgh brough by bie thorpe thorp throp thwaite toft stead sted stede stow
stoke cot cote cott cotes den dene dean dale combe coombe comb cumb hope hurst holt wood
wold shaw shot ridge rigg don down dun hill low law how howe haugh hale halgh ing ings
ington inton minster church kirk chester caster cester sett side end gate green heath
moor more marsh fen carr holme hulme ness naze wade bridge brig mill ash oak ac aac elm
thorn birch alder aller ald hol holl hollow crook broad brad long lang short great mickle
much north south east west sut sud nor middle nether upper over under high hey hay heigh
black blake white whit red rad read green grey gray cold cald cal warm sweet sour fair
foul ful bright deep swift slow still bar barn bark bank bottom clough cleugh dean dene
shaw scar scale scales fell garth garths lund ling skel skell mire myre moss stang stain
stone stan crag craig knott knot pike hag hagg pit pitt wor worf wer weir pen penn hop
hope sel sele sal salt wad wid wil wel wyn win wig ston sand gravel lin linn dub hag
cross gill hirst hyrst hurst wald weald bold bolt botl hythe hithe stath staithe
""".split()

# Spanish and French saints' and given names common in US toponyms (San Juan,
# Santa Ynez, St. Croix). Foreign dictionaries' proper-noun lists are not used:
# they include US place names such as Ohio and Cheyenne.
SAINTS = """
san santa santo sainte saint st ste juan jose pedro pablo francisco antonio miguel gabriel
diego luis lucas mateo marcos felipe fernando carlos jacinto juana maria ana clara rosa cruz
isabel teresa ines lucia barbara catalina margarita elena inez ynez gertrudis domingo benito
lorenzo rafael ramon vicente andres tomas simon martin joaquin emigdio gregorio jeronimo
bernardino buenaventura ysidro isidro dimas tadeo nicolas agustin patricio sebastian esteban
jacinto mateo felix cristobal xavier javier ignacio anselmo leandro saturnino timoteo
pierre jean jacques louis joseph francois marie anne claire charles philippe michel laurent
etienne denis andre croix esprit genevieve germain honore regis gabriel baptiste
""".split()

GENERIC_PLACE = {"county", "city", "town", "township", "village", "borough", "parish", "lake",
                 "mount", "mountain", "mountains", "valley", "bay", "island", "falls", "park",
                 "hills", "hill", "ridge", "springs", "spring", "state", "forest", "national",
                 "reservoir", "pond", "point", "beach", "gap", "of", "the", "fort", "cape",
                 "sound", "harbor", "peak", "butte", "mesa", "pass", "canyon", "bluff",
                 "reservation", "heights", "junction", "station", "rapids", "glen", "trail"}


def hunspell(lang):
    common, proper = set(), set()
    with open(LEX / f"hunspell_{lang}.dic", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            w = re.split(r"[/\t ]", line.strip(), maxsplit=1)[0]
            if not w or not w[0].isalpha():
                continue
            (proper if w[0].isupper() else common).add(fold(w))
    common.discard("")
    proper.discard("")
    return common, proper


def wordfreq_common(lang, proper, n=80000):
    return {w for w in top_n_list(lang, n) if w.isalpha() and len(w) >= 3 and w not in proper}


def surnames():
    with zipfile.ZipFile(LEX / "census_surnames_2010.zip") as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(z.open(name), na_values=["(S)"], keep_default_na=False)
    df = df[df["count"] >= 300]
    df = df[df.pctaian.fillna(0) < 20]
    hisp = set(df.loc[df.pcthispanic.fillna(0) >= 50, "name"].str.lower())
    common = set(df.loc[df["count"] >= 2000, "name"].str.lower())
    very_common = set(df.loc[df["count"] >= 20000, "name"].str.lower())
    return set(df.name.str.lower()) - {""}, hisp, common, very_common


def given_names():
    """Names in use by 1920, and the subset that was common (1 in 3,000 births)."""
    df = pd.read_csv(LEX / "baby_names_1880_2008.csv")
    early = df[df.year <= 1920]
    common = early.groupby("name").percent.max()
    return {fold(n) for n in early.name.unique()}, {fold(n) for n in common[common >= 1 / 3000].index}


def native_wikipedia():
    """Specifics of place names listed on Wikipedia as of Native American origin."""
    pages = json.load(open(RAW / "wikipedia" / "native_origin_lists.json", encoding="utf-8"))
    rows = []
    for title, text in pages.items():
        for line in text.splitlines():
            if not line.startswith("*"):
                continue
            m = re.search(r"\[\[([^\]|#]+)(?:\|([^\]]+))?\]\]", line) or re.search(r"'''([^']+)'''", line)
            if not m:
                continue
            name = (m.group(2) if m.lastindex and m.lastindex >= 2 and m.group(2) else m.group(1))
            name = re.sub(r"\(.*?\)", "", name).split(",")[0].strip()
            words = [w for w in fold(name).split() if w not in GENERIC_PLACE]
            spec, _ = split_name(" ".join(words)) if words else ("", "")
            if spec and len(spec) >= 3:
                rows.append(dict(specific=spec, source=title, entry=name))
    df = pd.DataFrame(rows).drop_duplicates("specific")
    return df


# Words that are English naming acts even though dictionaries list them only as
# proper nouns, or that English absorbed from Native languages and Chinook
# Jargon (the trade pidgin settlers used) and then used as ordinary words:
# Indian Creek, Spanish Fork, Pisgah Creek, Chinquapin Branch, Skookum Creek.
# The same logic already counts Hickory, Raccoon and Moose as English.
FORCE_ENGLISH = """
indian indians american mexican spanish french dutch dutchman frenchman english englishman irish
irishman scotch scotchman german italian swede swedes norwegian chinese chinaman negro
yankee welsh cornish portuguese canadian russian polish
pisgah jericho antioch egypt canaan jordan goshen bethel zion shiloh babylon hebron gilead
lebanon sharon carmel moriah nebo calvary galilee jerusalem bethlehem nazareth ebenezer
chinquapin chinkapin wickiup wikiup skookum siwash tillicum kanaka camas camass quamash wapato
kinnikinnick kinnikinic muskeg tamarack catalpa cushaw quahog menhaden terrapin papoose sachem
squaw tepee teepee tipi wigwam moccasin mocassin tomahawk wampum powwow hominy succotash chinook
potlatch cultus klootchman tyee hyas saltchuck mowitch kloshe
""".split()

# Historical peoples whose names are on rivers but who are missing from the
# federal list or appear there under another name.
ETHNONYM_SUPPLEMENT = """
missouria missouri kansa kaw osage illinois illiniwek miami erie huron mohican susquehannock
natchez yazoo tunica biloxi pascagoula quapaw arkansa arkansas ottawa iowa ioway omaha ponca
pawnee wichita kiowa comanche apache navajo ute klamath modoc yakama yakima spokane umatilla
kalispel kootenai kootenay chinook tillamook clackamas santiam molalla calapooia siuslaw
coquille shasta hupa yuma mohave mojave hopi zuni pima seminole muscogee choctaw chickasaw
yamasee catawba cherokee tuscarora seneca onondaga oneida mohawk abenaki penobscot
passamaquoddy narragansett pequot mohegan wampanoag nipmuc pocumtuc massachusett powhatan
pamunkey mattaponi rappahannock nanticoke piscataway conoy shawnee kickapoo sauk mesquakie
menominee winnebago ojibwe chippewa dakota lakota sioux mandan hidatsa arikara assiniboine
shoshone bannock paiute washoe wasco cayuse palouse wenatchee okanogan methow chelan entiat
skagit snohomish stillaguamish nisqually puyallup chehalis cowlitz quinault makah elwha
duwamish snoqualmie skykomish sammamish cheyenne arapaho caddo delaware lenape munsee
mingo wyandot wyandotte potawatomi saginaw outagamie cahokia kaskaskia peoria piankashaw wea
""".split()

ETHNONYM_STOP = {"tribe", "tribes", "band", "bands", "indian", "indians", "nation", "nations",
                 "reservation", "rancheria", "community", "colony", "mission", "valley", "tribal",
                 "town", "confederated", "people", "pueblo", "village", "american", "americans",
                 "federally", "hawaiians", "northfork", "alaska", "dania", "ronde", "sault"}


def ethnonyms(english, people, foreign):
    """Distinctive words of federally recognised tribe names, plus a supplement.

    Words of the official names that are English words or plurals, personal
    names, Spanish or French words, or town-like ('Laytonville', 'Stockbridge')
    are dropped: they come from reservation and place names, not peoples.
    """
    text = json.load(open(RAW / "wikipedia" / "native_peoples.json", encoding="utf-8"))
    words = set()
    for page in text.values():
        for m in re.finditer(r"\[\[([^\]|#]+)(?:\|([^\]]+))?\]\]", page):
            for w in re.split(r"[\s\-]+", fold(m.group(2) or m.group(1))):
                if (len(w) < 4 or w in english or w[:-1] in english or w in people
                        or w in foreign or w in ETHNONYM_STOP
                        or w.endswith(("ville", "bridge", "town", "ton", "burg"))):
                    continue
                words.add(w)
    return words | set(ETHNONYM_SUPPLEMENT)


def aboriginal_wikipedia():
    """Australian place names of Aboriginal origin, and English words borrowed from
    Aboriginal languages (Coolabah, Brigalow), from two Wikipedia lists."""
    pages = json.load(open(RAW / "wikipedia" / "aboriginal_origin_lists.json", encoding="utf-8"))
    link = re.compile(r"\[\[([^\]|#]+)(?:\|([^\]]+))?\]\]")
    places = []
    for line in pages["List of Australian place names of Aboriginal origin"].splitlines():
        m = link.search(line) if line.startswith("*") else None
        if m:
            name = re.sub(r"\(.*?\)", "", m.group(2) or m.group(1)).split(",")[0]
            words = [w for w in fold(name).split() if w not in GENERIC_PLACE]
            spec, _ = split_name(" ".join(words)) if words else ("", "")
            if spec and len(spec) >= 3:
                places.append(spec)
    loans = set()
    for line in pages["List of English words of Australian Aboriginal origin"].splitlines():
        m = link.search(line) if line.startswith("*") else None
        if m:
            loans.add(fold(re.sub(r"\(.*?\)", "", m.group(2) or m.group(1))).strip())
    return sorted(set(places)), {w for w in loans if w and " " not in w}


def main():
    en, en_p = hunspell("en")
    gb, gb_p = hunspell("en-GB")
    # common words come only from lower-case dictionary entries: frequency
    # lists are full of proper nouns (thames, ohio, moines) and would leak them
    english = en | gb
    write("english", english)
    write("english_proper", en_p | gb_p)
    for lang in ("fr", "es", "nl", "de"):
        c, p = hunspell(lang)
        write(lang, c)
        write(f"{lang}_proper", p)
    write("saints", {fold(w) for w in SAINTS})
    sn, hisp, sn_common, sn_very = surnames()
    write("surnames", sn)
    write("surnames_common", sn_common)
    write("surnames_very_common", sn_very)
    write("force_english", {fold(w) for w in FORCE_ENGLISH})
    write("surnames_hispanic", hisp)
    given, given_common = given_names()
    write("given", given)
    write("given_common", given_common)
    write("elements", set(ELEMENTS))
    foreign = set().union(*(hunspell(l)[0] for l in ("es", "fr")))
    eth = ethnonyms(english, sn | given, foreign)
    write("ethnonyms", eth)
    print(f"ethnonyms {len(eth):,}")
    abo, loans = aboriginal_wikipedia()
    pd.DataFrame({"specific": abo}).to_csv(OUT / "aboriginal_wikipedia.csv", index=False)
    write("au_loanwords", loans)
    print(f"aboriginal place specifics {len(abo):,}, loanwords {len(loans):,}")
    nat = native_wikipedia()
    nat.to_csv(OUT / "native_wikipedia.csv", index=False)
    print(f"english {len(english):,}  surnames {len(sn):,}  native-wikipedia specifics {len(nat):,}")
    print(nat.sample(15, random_state=0).to_string())


def write(name, words):
    (OUT / f"{name}.txt").write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")


def load(name):
    return set((OUT / f"{name}.txt").read_text(encoding="utf-8").split("\n")) - {""}


def frequent(word, lang="en"):
    return zipf_frequency(word, lang) >= 3.0


if __name__ == "__main__":
    main()
