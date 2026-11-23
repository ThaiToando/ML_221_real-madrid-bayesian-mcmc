"""Project-wide settings. Every other file imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"          # downloaded Transfermarkt CSVs
DATA_PROC = ROOT / "data" / "processed"   # cleaned tables we build
FIG = ROOT / "figures"                    # every chart we save
for p in (DATA_RAW, DATA_PROC, FIG):
    p.mkdir(parents=True, exist_ok=True)

# Transfermarkt dataset (snapshot up to 6 July 2026 = end of the 2025/26 season)
DATA_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.zip"

# Competition ids used by Transfermarkt
LALIGA = "ES1"
UCL = "CL"
TOP5 = ["ES1", "GB1", "IT1", "L1", "FR1"]   # Spain, England, Italy, Germany, France

# Seasons are labelled by their START year: 2025 means 2025/26
SEASONS = list(range(2014, 2026))
SIGNING_SEASONS = list(range(2014, 2025))   # need 2 seasons after a signing to judge it

# Transfermarkt club ids (verify in Phase 1 with check_club_ids())
REAL_MADRID = 418
BIG_CLUBS = {
    418: "Real Madrid", 131: "Barcelona", 13: "Atletico Madrid",
    281: "Manchester City", 31: "Liverpool", 985: "Manchester United",
    631: "Chelsea", 11: "Arsenal", 148: "Tottenham",
    27: "Bayern Munich", 16: "Borussia Dortmund",
    583: "Paris Saint-Germain", 506: "Juventus", 46: "Inter", 5: "AC Milan",
}

SEED = 221   # random seed so every simulation is reproducible

# Player profile cards (R22). Empty list = the six Real Madrid attackers with the most minutes.
# To choose the players yourself, list their Transfermarkt player ids in the order you want them shown,
# e.g. PLAYER_CARDS = [123456, 234567]. Find the ids in data/processed/p3_rm_attackers.csv (column player_id).
PLAYER_CARDS: list[int] = []
