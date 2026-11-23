"""Phase 1 (cleaning): turn the raw tables into the 4 analysis tables.

Run once:  python -m src.build_tables
Creates in data/processed/:
  club_seasons.csv    one row per club per league season (points, goals, squad value)
  player_seasons.csv  one row per player per club per league season
  signings.csv        one row per paid transfer INTO a big club, with a success label
  league_games.csv    La Liga matches (for the Poisson model)
"""
import numpy as np
import pandas as pd
from config import DATA_PROC, TOP5, LALIGA, SEASONS, SIGNING_SEASONS, BIG_CLUBS
from src.data_io import load, season_of


def club_seasons(games, club_games, clubs):
    g = games[games.competition_id.isin(TOP5) & games.season.isin(SEASONS)]
    cg = club_games.merge(g[["game_id", "season", "competition_id"]], on="game_id")
    cg["draw"] = cg.own_goals == cg.opponent_goals
    cg["points"] = np.where(cg.is_win == 1, 3, np.where(cg.draw, 1, 0))
    out = (cg.groupby(["club_id", "season", "competition_id"])
             .agg(games=("game_id", "nunique"), points=("points", "sum"),
                  gf=("own_goals", "sum"), ga=("opponent_goals", "sum"))
             .reset_index())
    out["ppg"] = out.points / out.games      # points per game: fair across 34- and 38-game leagues
    return out.merge(clubs[["club_id", "name"]], on="club_id", how="left")


def player_seasons(games, appearances, players, valuations):
    g = games[games.competition_id.isin(TOP5) & games.season.isin(SEASONS + [2026])]
    a = appearances.merge(g[["game_id", "season"]], on="game_id")
    ps = (a.groupby(["player_id", "player_club_id", "season"])
            .agg(apps=("game_id", "nunique"), minutes=("minutes_played", "sum"),
                 goals=("goals", "sum"), assists=("assists", "sum"))
            .reset_index().rename(columns={"player_club_id": "club_id"}))
    ps = ps.merge(players[["player_id", "name", "position", "date_of_birth"]], on="player_id", how="left")
    ps["season_start"] = pd.to_datetime(ps.season.astype(str) + "-09-01")
    ps["age"] = (ps.season_start - ps.date_of_birth).dt.days / 365.25
    # market value on 1 September of that season (latest valuation before it, within 1 year)
    v = valuations[["player_id", "date", "market_value_in_eur"]].dropna().sort_values("date")
    ps = ps.sort_values("season_start")
    ps = pd.merge_asof(ps, v, left_on="season_start", right_on="date", by="player_id",
                       direction="backward", tolerance=pd.Timedelta(days=365))
    ps = ps.rename(columns={"market_value_in_eur": "value_eur"}).drop(columns="date")
    ps["ga"] = ps.goals + ps.assists
    ps["ga_per90"] = np.where(ps.minutes > 0, ps.ga / (ps.minutes / 90), np.nan)
    return ps


def add_squad_value(cs, ps):
    """Squad value = sum of 1-Sept market values of everyone who played a league game."""
    sv = (ps.dropna(subset=["value_eur"]).groupby(["club_id", "season"]).value_eur.sum()
            .rename("squad_value").reset_index())
    cs = cs.merge(sv, on=["club_id", "season"], how="left")
    # relative value = squad value / average squad value in that league-season (removes inflation)
    cs["rel_value"] = cs.squad_value / cs.groupby(["competition_id", "season"]).squad_value.transform("mean")
    return cs


def signings(transfers, ps, cs, clubs):
    t = transfers[transfers.to_club_id.isin(BIG_CLUBS) & (transfers.transfer_fee > 0)].copy()
    t["season"] = season_of(t.transfer_date)
    t = t[t.season.isin(SIGNING_SEASONS)]
    t = t[t.from_club_id != t.to_club_id]
    # minutes for the new club in the 2 league seasons after the signing
    team_minutes = (cs.set_index(["club_id", "season"]).games * 90).to_dict()
    played = ps.groupby(["player_id", "club_id", "season"]).minutes.sum().to_dict()
    shares = []
    for p, c, s in zip(t.player_id, t.to_club_id, t.season):
        mins = sum(played.get((p, c, s + k), 0) for k in (0, 1))
        poss = sum(team_minutes.get((c, s + k), 38 * 90) for k in (0, 1))
        shares.append(mins / poss)
    t["minutes_share"] = shares
    t["success"] = (t.minutes_share >= 0.40).astype(int)   # main definition of a "hit"
    info = ps.sort_values("season").drop_duplicates("player_id")[["player_id", "position", "date_of_birth"]]
    t = t.merge(info, on="player_id", how="left")
    t["age"] = (t.transfer_date - t.date_of_birth).dt.days / 365.25
    t["fee_m"] = t.transfer_fee / 1e6
    t["log_fee"] = np.log(t.fee_m)
    league = clubs.set_index("club_id").domestic_competition_id
    t["from_top5"] = t.from_club_id.map(league).isin(TOP5).astype(int)
    t["club"] = t.to_club_id.map(BIG_CLUBS)
    t["is_real_madrid"] = (t.to_club_id == 418).astype(int)
    # second definition: market value 2 years later >= fee paid
    v2 = ps[["player_id", "season", "value_eur"]].rename(columns={"season": "s2", "value_eur": "value_2y"})
    t["s2"] = t.season + 2
    t = t.merge(v2.drop_duplicates(["player_id", "s2"]), on=["player_id", "s2"], how="left")
    t["value_up"] = (t.value_2y >= t.transfer_fee).astype(float).where(t.value_2y.notna())
    return t.dropna(subset=["age", "position"])


def main():
    games, club_games, clubs = load("games"), load("club_games"), load("clubs")
    players, appearances, vals, transfers = load("players"), load("appearances"), load("player_valuations"), load("transfers")
    cs = club_seasons(games, club_games, clubs)
    ps = player_seasons(games, appearances, players, vals)
    cs = add_squad_value(cs, ps)
    sg = signings(transfers, ps, cs, clubs)
    lg = games[(games.competition_id == LALIGA) & games.season.isin(SEASONS)]
    lg = lg[["game_id", "season", "date", "round", "home_club_id", "away_club_id",
             "home_club_name", "away_club_name", "home_club_goals", "away_club_goals"]]
    for name, df in [("club_seasons", cs), ("player_seasons", ps), ("signings", sg), ("league_games", lg)]:
        df.to_csv(DATA_PROC / f"{name}.csv", index=False)
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
