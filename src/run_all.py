"""Run the whole pipeline in order. Assumes download.py has fetched data/raw.

python run_all.py            everything
python run_all.py analyze    start from a named step
"""
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    "england_network",  # OS Open Rivers -> named rivers, upstream length, basins
    "validate_size",    # upstream length vs NRFA gauged areas
    "us_rivers",        # NHDPlus -> named US streams, drainage area, HUC4
    "ekwall",           # Ekwall's 1928 labels -> OS rivers
    "england_area",     # HydroRIVERS catchment area for English rivers
    "lexicon",          # word lists, answer keys
    "classify",         # origin of every English and US name
    "gnis_gold",        # GNIS histories as an external answer key
    "australasia",      # Australia and New Zealand
    "analyze",          # retention curves, four countries
    "audit",            # accuracy-corrected US curve (needs data/labels/us_audit_labels.csv)
    "mechanisms",       # contact time, conquest date, density, renaming
    "charts",           # curve figures
    "surface",          # threshold surface maps
    "maps",             # hero maps
    "webmap",           # interactive map data
]


def main():
    here = Path(__file__).parent
    start = STEPS.index(sys.argv[1]) if len(sys.argv) > 1 else 0
    for step in STEPS[start:]:
        t = time.time()
        print(f"== {step}", flush=True)
        subprocess.run([sys.executable, "-u", str(here / f"{step}.py")], check=True, cwd=here)
        print(f"   {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
