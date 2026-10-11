"""Fetch every raw input into data/raw (about 3 GB). Existing files are skipped.

Run once before run_all.py. Wikipedia is fetched slowly with backoff, because
its API answers bursts with HTTP 429.
"""
import json
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from common import RAW

UA = {"User-Agent": "old-water-research/0.1 (river-name study)"}
V2 = RAW / "v2"
SB = "https://www.sciencebase.gov/catalog/file/get/65cbbb98d34ef4b119cb37c9?name="
GNIS = "https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/Topical/"
WOOORM = "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/"

FILES = {
    RAW / "oprvrs_gpkg_gb.zip": "https://api.os.uk/downloads/v1/products/OpenRivers/downloads?area=GB&format=GeoPackage&redirect",
    RAW / "ne_10m_admin_0_map_units.zip": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_map_units.zip",
    RAW / "ne_10m_admin_1_states_provinces.zip": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
    RAW / "enhd_nhdplusatts.parquet": SB + "enhd_nhdplusatts.parquet",
    RAW / "enhd_nhdplus_flowlines.gpkg": SB + "enhd_nhdplus_flowlines.gpkg",
    RAW / "ekwall_1928_djvu.txt": "https://archive.org/download/englishrivername0000eile/englishrivername0000eile_djvu.txt",
    RAW / "nrfa_stations.json": "https://nrfaapps.ceh.ac.uk/nrfa/ws/station-info?station=*&format=json-object&fields=id,name,river,catchment-area,easting,northing",
    RAW / "lexicon" / "census_surnames_2010.zip": "https://www2.census.gov/topics/genealogy/2010surnames/names.zip",
    RAW / "lexicon" / "baby_names_1880_2008.csv": "https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv",
    V2 / "FeatureDescriptionHistory_National_Text.zip": GNIS + "FeatureDescriptionHistory_National_Text.zip",
    V2 / "HydroRIVERS_v10_au_shp.zip": "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_au_shp.zip",
    V2 / "HydroRIVERS_v10_eu_shp.zip": "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_eu_shp.zip",
    V2 / "SurfaceHydrologyLinesNational.gdb.zip": "https://d28rz98at9flks.cloudfront.net/83130/SurfaceHydrologyLinesNational.gdb.zip",
    V2 / "nz_gazetteer.csv": "https://gazetteer.linz.govt.nz/gaz.csv",
    V2 / "US_AtlasHCB_Counties_Gen001.zip": "https://digital.newberry.org/ahcb/downloads/gis/US_AtlasHCB_Counties_Gen001.zip",
}
for lang in ("en-GB", "en", "fr", "es", "nl", "de"):
    for ext in ("dic", "aff"):
        FILES[RAW / "lexicon" / f"hunspell_{lang}.{ext}"] = f"{WOOORM}{lang}/index.{ext}"

UNZIP = {
    RAW / "oprvrs_gpkg_gb.zip": RAW / "oprvrs",
    V2 / "SurfaceHydrologyLinesNational.gdb.zip": V2 / "ga_hydro",
    V2 / "US_AtlasHCB_Counties_Gen001.zip": V2 / "newberry",
}

NZ_OVERPASS = """[out:json][timeout:900];
area["ISO3166-1"="NZ"][admin_level=2]->.nz;
way["waterway"~"^(river|stream|canal)$"]["name"](area.nz);
out tags geom qt;"""


def fetch(url, dest: Path, data=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA, data=data)
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"  {dest.name}: {dest.stat().st_size / 1e6:,.1f} MB")


def wiki(params, tries=6):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json", "maxlag": 5})
    for k in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15 * (k + 1))
                continue
            raise
    raise RuntimeError("Wikipedia kept refusing; try again later")


def wiki_pages(titles, dest):
    if dest.exists():
        return
    out = {}
    for t in titles:
        out[t] = wiki({"action": "parse", "page": t, "prop": "wikitext", "redirects": 1})["parse"]["wikitext"]["*"]
        time.sleep(4)
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w", encoding="utf-8"))
    print(f"  {dest.name}: {len(out)} pages")


def search(query):
    r = wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": 100})
    time.sleep(3)
    return {x["title"] for x in r["query"]["search"]}


def main():
    for dest, url in FILES.items():
        if not dest.exists():
            fetch(url, dest)
    for z, out in UNZIP.items():
        if not out.exists():
            # some archives use compression GDAL cannot stream, so extract them
            zipfile.ZipFile(z).extractall(out)
    nz = V2 / "nz_osm_waterways.json"
    if not nz.exists():
        fetch("https://overpass-api.de/api/interpreter", nz,
              data=urllib.parse.urlencode({"data": NZ_OVERPASS}).encode())
    native = search('intitle:"Native American origin"') | search('intitle:placenames intitle:"Native American"')
    native = sorted(t for t in native if not any(x in t for x in ("English words", "Alaska", "Hawaii", "Canada")))
    wiki_pages(native, RAW / "wikipedia" / "native_origin_lists.json")
    wiki_pages(["List of Native American peoples in the United States"], RAW / "wikipedia" / "native_peoples.json")
    wiki_pages(["List of Australian place names of Aboriginal origin",
                "List of English words of Australian Aboriginal origin"],
               RAW / "wikipedia" / "aboriginal_origin_lists.json")
    print("all inputs present")


if __name__ == "__main__":
    main()
