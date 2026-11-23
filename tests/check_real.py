"""Go-live check: is everything in the deck built from the REAL data, and is it safe to present?

Run:  python -m tests.check_real          (after python run_all.py --deck --manim --pdf)
Prints PASS / FAIL / WARN for each check with the fix, and exits with code 1 if anything FAILs.
Run it after every go-live step (GO_LIVE.md) and fix what it reports.
"""
import os as _os
import subprocess as _sp
import sys as _sys
if _os.name == "nt" and _os.environ.get("PYTHONUTF8") != "1":   # Windows: read/write every file as UTF-8 (the deck uses λ, ≈, ↓)
    _os.environ["PYTHONUTF8"] = "1"
    _sys.exit(_sp.call([_sys.executable, *_sys.argv]))
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

RAW, PROC, FIG, DOCS = C.DATA_RAW, C.DATA_PROC, C.FIG, ROOT / "docs"
results = []


def check(ok, name, fix, level="FAIL"):
    results.append(("PASS" if ok else level, name, "" if ok else fix))


def fake_share(path, col):
    try:
        names = pd.read_csv(path, usecols=[col])[col].astype(str)
        return names.str.fullmatch(r"Player \d+").mean()
    except Exception:
        return None


def newest(paths):
    paths = [p for p in paths if p.exists()]
    return max((p.stat().st_mtime for p in paths), default=0)


# 1 raw data is the real Transfermarkt download (it comes as players.csv.gz; the fake build writes players.csv)
PLAYERS_RAW = next(iter(sorted(RAW.glob("players.csv*"))), RAW / "players.csv")
share = fake_share(PLAYERS_RAW, "name") if PLAYERS_RAW.exists() else None
check(share is not None, "raw data present (data/raw/players.csv)",
      'python -c "from src.data_io import download_data; download_data()"')
if share is not None:
    check(share < 0.05, "raw data is real (not the fake 'Player N' names)",
          "delete data/raw/*, then download_data() again (see GO_LIVE.md step 2)")
    cols = pd.read_csv(PLAYERS_RAW, nrows=1).columns
    check("country_of_citizenship" in cols, "full players table (nationality, foot, height for the cards)",
          "re-download: the fake table has only 4 columns", level="WARN")

# 2 processed tables rebuilt from it
share = fake_share(PROC / "player_seasons.csv", "name")
check(share is not None and share < 0.05, "data/processed rebuilt from the real data", "python run_all.py --deck")
proc_time = newest(list(PROC.glob("*.csv")))
check(newest([PLAYERS_RAW]) <= proc_time + 1, "processed tables newer than the raw data", "python run_all.py --deck")

# 3 every visual is newer than the tables it reads
stale = [p.name for p in (FIG / "interactive").glob("R*.html") if p.stat().st_mtime < proc_time]
check(not stale, "interactive visuals R0-R22 rebuilt", f"python run_all.py --deck   (stale: {', '.join(sorted(stale)[:6])})")
videos = list(FIG.glob("manim_*.mp4"))
stale_v = [p.name for p in videos if p.stat().st_mtime < proc_time]
check(not stale_v, "Manim videos rebuilt after the data (they show FAKE DATA otherwise)",
      "python -m src.manim_videos   (or delete figures/manim_* to drop the video slides)")
check(bool(videos), "Manim videos exist", "python -m src.manim_videos (optional; the deck works without them)", level="WARN")

# 4 the deck itself
idx = DOCS / "index.html"
check(idx.exists(), "deck built (docs/index.html)", "python run_all.py --deck --pdf")
if idx.exists():
    html = idx.read_text()
    check('class="fake"' not in html, "no FAKE DATA badge in the deck", "rebuild after step 2: python run_all.py --deck --pdf")
    check(idx.stat().st_mtime >= proc_time, "deck newer than the data", "python -m src.slides --pdf")
    check("YOUR-APP" not in html, "APP_URL set in src/slides.py", "paste your Streamlit link into APP_URL", level="WARN")
check((DOCS / "deck.pdf").exists(), "docs/deck.pdf exists (the file to submit)", "python -m src.slides --pdf", level="WARN")

# 5 player photos: optional, but if present they must be named right and licensed
pdir = FIG / "players"
photos = [p for p in pdir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")] if pdir.exists() else []
if photos:
    ids = set(pd.read_csv(PROC / "p3_rm_attackers.csv").player_id) if (PROC / "p3_rm_attackers.csv").exists() else set()
    bad = [p.name for p in photos if not p.stem.isdigit() or int(p.stem) not in ids]
    check(not bad, "photo files named <player_id>.jpg for Real Madrid attackers",
          f"rename or remove: {', '.join(bad)} (ids are in data/processed/p3_rm_attackers.csv)")
    lic = (pdir / "LICENSES.md").read_text() if (pdir / "LICENSES.md").exists() else ""
    unl = [p.name for p in photos if p.name not in lic]
    check(not unl, "every photo has a licence line in figures/players/LICENSES.md", f"add a line for: {', '.join(unl)}")
    gi = (ROOT / ".gitignore").read_text() if (ROOT / ".gitignore").exists() else ""
    free = all("publishing allowed" in line for line in lic.splitlines() if line.strip().startswith("-"))   # CC BY / BY-SA lines
    check(free or "figures/players/*.jpg" in gi or "figures/players/" in gi, "photos may be published (free licences) or are kept out of git",
          "add figures/players/*.jpg, *.png, *.webp to .gitignore unless your licence allows publishing", level="WARN")
else:
    results.append(("INFO", "no player photos added: the cards show silhouettes", ""))

# 6 key numbers exist
try:
    n = json.loads((PROC / "deck_numbers.json").read_text())
    missing = [k for k in ("r2", "p_title", "perm_p", "auc", "mle_a") if k not in n]
    check(not missing, "deck_numbers.json complete", f"python run_all.py --deck (missing: {missing})")
except FileNotFoundError:
    check(False, "deck_numbers.json exists", "python run_all.py --deck")

width = max(len(r[1]) for r in results if r[1])
for level, name, fix in results:
    if name:
        print(f"{level:5} {name:<{width}}  {('-> ' + fix) if fix else ''}")
fails = sum(r[0] == "FAIL" for r in results)
print(f"\n{'READY TO PRESENT' if not fails else f'{fails} check(s) failed: fix them, rebuild, run this again'}")
sys.exit(1 if fails else 0)
