"""Builds FAKE tables with the exact Transfermarkt column names, to test the code
before (or without) downloading the real data.  python -m tests.make_fake_data"""
import numpy as np
import pandas as pd
from config import DATA_RAW, BIG_CLUBS

rng = np.random.default_rng(0)
LEAGUES = {"ES1": 20, "GB1": 10, "L1": 10, "FR1": 10, "IT1": 10, "PO1": 8}
BIG_BY_LEAGUE = {"ES1": [418, 131, 13], "GB1": [281, 31, 985, 631, 11, 148], "L1": [27, 16],
                 "FR1": [583], "IT1": [506, 46, 5], "PO1": []}
POS = ["Goalkeeper"] * 3 + ["Defender"] * 8 + ["Midfield"] * 8 + ["Attack"] * 6
SEASONS = range(2013, 2026)

clubs, cid = [], 10000
for lg, n in LEAGUES.items():
    ids = BIG_BY_LEAGUE[lg] + list(range(cid, cid + n - len(BIG_BY_LEAGUE[lg])))
    cid += n
    for c in ids:
        clubs.append(dict(club_id=c, name=BIG_CLUBS.get(c, f"Club {c}"), domestic_competition_id=lg,
                          strength=(0.9 if c == 418 else 0.6 if c in BIG_CLUBS else rng.normal(0, 0.25))))
clubs = pd.DataFrame(clubs)
strength = clubs.set_index("club_id").strength.to_dict()
league_of = clubs.set_index("club_id").domestic_competition_id.to_dict()

players, pid = [], 1
roster = {}
for c in clubs.club_id:
    roster[c] = []
    for pos in POS:
        talent = strength[c] + rng.normal(0, 0.3)
        dob = pd.Timestamp("1985-01-01") + pd.Timedelta(days=int(rng.integers(0, 365 * 15)))
        players.append(dict(player_id=pid, name=f"Player {pid}", position=pos, date_of_birth=dob, talent=talent))
        roster[c].append(pid)
        pid += 1
players = pd.DataFrame(players).set_index("player_id")

games, club_games, apps, events, vals, transfers = [], [], [], [], [], []
gid = 1
for s in SEASONS:
    # ---- transfer window: big clubs buy talented players, others swap randomly
    for c in clubs.club_id:
        for _ in range(int(rng.integers(1, 4))):
            seller = int(rng.choice(clubs.club_id))
            if seller == c or len(roster[seller]) < 20:
                continue
            p = int(rng.choice(roster[seller]))
            roster[seller].remove(p); roster[c].append(p)
            age = (pd.Timestamp(f"{s}-08-01") - players.at[p, "date_of_birth"]).days / 365.25
            value = np.exp(15.5 + 1.6 * players.at[p, "talent"] - 0.02 * (age - 26) ** 2)
            fee = 0 if rng.random() < 0.15 else round(value * rng.uniform(0.7, 1.6), -5)
            transfers.append(dict(player_id=p, player_name=players.at[p, "name"],
                                  transfer_date=pd.Timestamp(f"{s}-07-15") + pd.Timedelta(days=int(rng.integers(0, 40))),
                                  transfer_season=f"{s % 100:02d}/{(s + 1) % 100:02d}", from_club_id=seller, to_club_id=c,
                                  from_club_name=f"Club {seller}", to_club_name=f"Club {c}", transfer_fee=fee,
                                  market_value_in_eur=round(value, -5)))
    # ---- valuations on 1 Aug and 1 Feb
    for p, row in players.iterrows():
        for d in (f"{s}-08-01", f"{s + 1}-02-01"):
            age = (pd.Timestamp(d) - row.date_of_birth).days / 365.25
            v = np.exp(15.5 + 1.6 * row.talent - 0.02 * (age - 26) ** 2 + rng.normal(0, 0.2))
            vals.append(dict(date=d, player_id=p, current_club_name="", current_club_id=0,
                             market_value_in_eur=round(v, -4), player_club_domestic_competition_id=""))
    # ---- league games (double round robin)
    for lg in LEAGUES:
        ids = list(clubs[clubs.domestic_competition_id == lg].club_id)
        for h in ids:
            for a in ids:
                if h == a:
                    continue
                date = pd.Timestamp(f"{s}-08-15") + pd.Timedelta(days=int(rng.integers(0, 270)))
                lh = np.exp(0.25 + 0.3 + strength[h] - strength[a] * 0.8)
                la = np.exp(0.25 + strength[a] - strength[h] * 0.8)
                hg, ag = rng.poisson(lh), rng.poisson(la)
                games.append(dict(game_id=gid, competition_id=lg, competition_type="domestic_league", season=s,
                                  round="1. Matchday", date=date, home_club_id=h, away_club_id=a,
                                  home_club_goals=hg, away_club_goals=ag, home_club_name=f"Club {h}",
                                  away_club_name=f"Club {a}"))
                for own, opp, og, opg, host in ((h, a, hg, ag, "Home"), (a, h, ag, hg, "Away")):
                    club_games.append(dict(club_id=own, game_id=gid, own_goals=og, opponent_id=opp,
                                           opponent_goals=opg, hosting=host, is_win=int(og > opg)))
                    squad = roster[own]
                    tal = players.loc[squad, "talent"].to_numpy()
                    w = np.exp(tal); w /= w.sum()
                    picked = rng.choice(squad, size=14, replace=False, p=w)
                    mins = [90] * 11 + [20] * 3
                    att_w = np.array([3.0 if players.at[x, "position"] == "Attack" else
                                      1.0 if players.at[x, "position"] == "Midfield" else 0.2 for x in picked])
                    att_w = att_w * np.exp(players.loc[picked, "talent"].to_numpy()); att_w /= att_w.sum()
                    scorers = rng.choice(picked, size=og, p=att_w) if og else []
                    assisters = rng.choice(picked, size=og, p=att_w) if og else []
                    for x, m in zip(picked, mins):
                        apps.append(dict(appearance_id=f"{gid}_{x}", game_id=gid, player_id=x, player_club_id=own,
                                         player_current_club_id=own, date=date, player_name=f"Player {x}",
                                         competition_id=lg, yellow_cards=0, red_cards=0,
                                         goals=int(np.sum(scorers == x)) if og else 0,
                                         assists=int(np.sum(assisters == x)) if og else 0, minutes_played=m))
                    for x in scorers:
                        late = own == 418 and rng.random() < 0.08
                        minute = int(rng.integers(76, 91)) if late else int(rng.integers(1, 91))
                        events.append(dict(game_event_id=f"e{len(events)}", date=date, game_id=gid, player_id=x,
                                           club_id=own, club_name="", type="Goals", minute=minute,
                                           description="", player_in_id=None, player_assist_id=None))
                gid += 1

players = players.reset_index().drop(columns="talent")
clubs = clubs.drop(columns="strength")
comps = pd.DataFrame([dict(competition_id=k, name=k) for k in LEAGUES])
for name, df in [("clubs", clubs), ("players", players), ("games", pd.DataFrame(games)),
                 ("club_games", pd.DataFrame(club_games)), ("appearances", pd.DataFrame(apps)),
                 ("player_valuations", pd.DataFrame(vals)), ("transfers", pd.DataFrame(transfers)),
                 ("game_events", pd.DataFrame(events)), ("competitions", comps)]:
    df.to_csv(DATA_RAW / f"{name}.csv", index=False)
    print(name, len(df))
