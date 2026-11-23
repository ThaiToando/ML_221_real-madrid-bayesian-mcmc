"""Phase 1: download and load the Transfermarkt tables."""
import zipfile
import pandas as pd
import requests
from config import DATA_RAW, DATA_URL, BIG_CLUBS

TABLES = ["games", "club_games", "clubs", "players", "appearances",
          "player_valuations", "transfers", "game_events", "competitions"]


def download_data(force: bool = False) -> None:
    """Download the dataset zip (~200 MB) once and unzip it into data/raw/."""
    if not force and any(DATA_RAW.glob("games.csv*")):
        print("Data already downloaded.")
        return
    zip_path = DATA_RAW / "transfermarkt-datasets.zip"
    with requests.get(DATA_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_RAW)
    # some versions put the files in a sub-folder: move them up
    for f in DATA_RAW.rglob("*.csv*"):
        if f.parent != DATA_RAW:
            f.rename(DATA_RAW / f.name)
    zip_path.unlink()
    print("Downloaded:", sorted(p.name for p in DATA_RAW.glob("*.csv*")))


def load(name: str) -> pd.DataFrame:
    """Load one table, e.g. load('games'). Works for .csv and .csv.gz."""
    matches = sorted(DATA_RAW.glob(f"{name}.csv*"))
    if not matches:
        raise FileNotFoundError(f"{name}.csv not found in {DATA_RAW}. Run download_data().")
    date_cols = {"games": ["date"], "appearances": ["date"], "player_valuations": ["date"],
                 "transfers": ["transfer_date"], "game_events": ["date"],
                 "players": ["date_of_birth"]}.get(name, [])
    df = pd.read_csv(matches[0], low_memory=False)
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def check_club_ids() -> pd.DataFrame:
    """Print the names Transfermarkt stores for our club ids, so you can confirm them."""
    clubs = load("clubs")
    out = clubs[clubs.club_id.isin(BIG_CLUBS)][["club_id", "name", "domestic_competition_id"]]
    missing = set(BIG_CLUBS) - set(out.club_id)
    if missing:
        print("Not found (fix these ids in config.py):", missing)
    return out.sort_values("club_id")


def season_of(dates: pd.Series) -> pd.Series:
    """Season start year from a date: matches from July onward belong to that year's season."""
    return (dates.dt.year - (dates.dt.month < 7)).astype("Int64")
