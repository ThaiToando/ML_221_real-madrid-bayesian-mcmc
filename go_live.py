"""One command from the unzipped folder to the finished, checked deck:   python go_live.py

What it does, in order (each step is skipped when it is already done, so you can re-run it any time):
  1  check Python and install requirements.txt, Chromium for Playwright, reveal.js/KaTeX (if Node.js is installed)
  2  remove the FAKE test build (fake tables, charts, videos, deck)            - asks first unless --yes
  3  download the real Transfermarkt data (~200 MB) and confirm the 15 club ids
  4  run the whole analysis, the interactive visuals, the Manim videos (if Manim + Pango + LaTeX are installed)
     and the deck: index.html, deck.pptx, deck.pdf
  5  run tests/check_real.py and print READY TO PRESENT or what to fix
Everything is logged to go_live.log. Options:
  --yes          do not ask before removing the fake build
  --no-manim     skip the videos (the deck works without them)
  --no-install   skip step 1 (packages already installed)
  --fake         CI only: use tests/make_fake_data.py instead of the download (the result is NOT presentable)
What it never does: download or generate player photos, type numbers, publish anything, push to git.
"""
import os as _os
import subprocess as _sp
import sys as _sys
if _os.name == "nt" and _os.environ.get("PYTHONUTF8") != "1":   # Windows: read/write every file as UTF-8 (the deck uses λ, ≈, ↓)
    _os.environ["PYTHONUTF8"] = "1"
    _sys.exit(_sp.call([_sys.executable, *_sys.argv]))
import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "go_live.log"
PY = sys.executable


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, check=True, **kw):
    say("$ " + " ".join(map(str, cmd)))
    with LOG.open("a", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True, **kw)
    if check and r.returncode != 0:
        say(f"FAILED (exit {r.returncode}). The last lines of go_live.log show why.")
        tail = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        print("\n".join(tail))
        sys.exit(r.returncode)
    return r.returncode


def step_install(no_manim):
    say("STEP 1  install")
    if sys.version_info < (3, 10):
        sys.exit(f"Python 3.10+ needed, this is {platform.python_version()}. Install a newer Python and re-run.")
    core = [l for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip() and not l.startswith("manim")]
    tmp = ROOT / ".requirements-core.txt"
    tmp.write_text("\n".join(core) + "\n")
    run([PY, "-m", "pip", "install", "-q", "-r", str(tmp)])          # everything except Manim
    tmp.unlink()
    if not no_manim and importlib.util.find_spec("manim") is None:
        run([PY, "-m", "pip", "install", "-q", "manim"], check=False)  # needs Pango; failing here is fine
    for pkg in ("pandas", "plotly", "statsmodels", "kaleido", "pptx", "playwright"):
        if importlib.util.find_spec(pkg) is None:
            run([PY, "-m", "pip", "install", "-q", {"pptx": "python-pptx"}.get(pkg, pkg)])
    run([PY, "-m", "playwright", "install", "chromium"], check=False)
    if shutil.which("npm") and not (ROOT / "docs" / "node_modules" / "reveal.js").exists():
        run([shutil.which("npm"), "install", "--prefix", "docs", "reveal.js@5.1.0", "katex@0.16", "--silent"], check=False)
    elif not shutil.which("npm"):
        say("  Node.js not found: the deck will load reveal.js from the internet (fine with Wi-Fi).")
    if not no_manim:
        manim_ready(report=True)


def manim_ready(report=False):
    ok_py = importlib.util.find_spec("manim") is not None
    ok_tex = shutil.which("latex") is not None and shutil.which("dvisvgm") is not None
    if report and not (ok_py and ok_tex):
        sysname = platform.system()
        how = {"Darwin": "brew install pango && brew install --cask mactex-no-gui && pip install manim",
               "Linux": "sudo apt-get install -y libpango1.0-dev texlive-latex-extra dvisvgm && pip install manim"}.get(
            sysname, "install MiKTeX (miktex.org, it includes dvisvgm), then pip install manim; or run with --no-manim")
        say(f"  Manim videos will be skipped: {'manim' if not ok_py else 'LaTeX/dvisvgm'} missing. To add them: {how}")
    return ok_py and ok_tex


def is_fake_build():
    p = next(iter(sorted((ROOT / "data" / "raw").glob("players.csv*"))), ROOT / "data" / "raw" / "players.csv")  # real data is .csv.gz
    q = ROOT / "data" / "processed" / "player_seasons.csv"
    import pandas as pd
    for f, col in ((p, "name"), (q, "name")):
        if f.exists():
            try:
                if pd.read_csv(f, usecols=[col])[col].astype(str).str.fullmatch(r"Player \d+").mean() > 0.5:
                    return True
            except Exception:
                return True
    return not p.exists()


def step_clean(yes):
    say("STEP 2  remove the fake test build")
    if not is_fake_build():
        say("  real data already in place: nothing to remove")
        return
    if not yes:
        a = input("Remove the FAKE tables, charts, videos and deck (all rebuilt in step 4)? [y/N] ").strip().lower()
        if a != "y":
            sys.exit("Stopped. Re-run with --yes to skip this question.")
    globs = ["data/raw/*.csv*", "data/processed/*.csv", "data/processed/deck_numbers.json", "figures/*.png", "figures/*.mp4",
             "figures/*.gif", "figures/interactive/*.html", "docs/index.html", "docs/deck.pptx", "docs/deck.pdf"]
    n = 0
    for g in globs:
        for f in ROOT.glob(g):
            f.unlink()
            n += 1
    for d in ("docs/assets", "animations/media"):
        if (ROOT / d).exists():
            shutil.rmtree(ROOT / d)
    say(f"  removed {n} fake files (player photos in figures/players/ are kept)")


def step_data(fake):
    say("STEP 3  real data")
    sys.path.insert(0, str(ROOT))
    if fake:
        run([PY, "-m", "tests.make_fake_data"])
        say("  --fake: test data only. NOT presentable.")
        return
    from src.data_io import check_club_ids, download_data
    try:
        download_data()
    except Exception as e:
        sys.exit(f"Download failed ({e.__class__.__name__}: {e}). Check the internet connection and DATA_URL in config.py, "
                 "or download transfermarkt-datasets.zip yourself and unzip it into data/raw/.")
    ids = check_club_ids()
    say("  clubs found:\n" + ids.to_string(index=False))
    import config as C
    missing = set(C.BIG_CLUBS) - set(ids.club_id)
    if missing:
        sys.exit(f"Club ids not in the data: {missing}. Fix them in config.py (BIG_CLUBS), then re-run.")


def step_build(no_manim):
    say("STEP 4  analysis, visuals, videos, deck (about 5-10 minutes)")
    args = [PY, "run_all.py", "--deck", "--pdf"]
    if not no_manim and manim_ready():
        args.insert(3, "--manim")
    run(args)


def step_check():
    say("STEP 5  go-live check")
    code = run([PY, "-m", "tests.check_real"], check=False)
    lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(i for i, l in enumerate(lines) if "tests.check_real" in l) + 1
    print("\n".join(lines[start:]))
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--no-manim", action="store_true")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    LOG.write_text(encoding="utf-8", data=f"go_live.py started {time.ctime()} on {platform.platform()} with Python {platform.python_version()}\n")
    if not a.no_install:
        step_install(a.no_manim)
    step_clean(a.yes or a.fake)
    step_data(a.fake)
    step_build(a.no_manim)
    code = step_check()
    say("Next (optional): choose players with PLAYER_CARDS in config.py, add licensed photos to figures/players/, "
        "paste the Streamlit link into APP_URL in src/slides.py, then: python run_all.py --deck --pdf")
    sys.exit(code)


if __name__ == "__main__":
    main()
