"""Phase 2: first look at the data.   python -m src.eda"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DATA_PROC, FIG, LALIGA, REAL_MADRID, SEASONS
from src.style import GOLD, GREY, RIVALS, source  # noqa: E402  (theme)
from src.data_io import load, season_of

RIVALS_C = {418: ("Real Madrid", GOLD), 131: ("Barcelona", RIVALS[131]), 13: ("Atletico Madrid", RIVALS[13])}


def run():
    cs = pd.read_csv(DATA_PROC / "club_seasons.csv")
    tr = load("transfers")
    tr["season"] = season_of(tr.transfer_date)
    tr = tr[tr.season.isin(SEASONS)]

    # 1) Spending: money in vs money out, per season
    spent = tr[tr.to_club_id == REAL_MADRID].groupby("season").transfer_fee.sum() / 1e6
    sold = tr[tr.from_club_id == REAL_MADRID].groupby("season").transfer_fee.sum() / 1e6
    money = pd.DataFrame(dict(spent=spent, sold=sold)).reindex(SEASONS).fillna(0)
    money["net"] = money.spent - money.sold
    print(money.round(1))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    x = np.arange(len(money))
    ax[0].bar(x - .2, money.spent, .4, color=GOLD, label="bought")
    ax[0].bar(x + .2, money.sold, .4, color=GREY, label="sold")
    ax[0].set_xticks(x, [f"{s % 100:02d}/{(s + 1) % 100:02d}" for s in money.index], rotation=45)
    ax[0].set(ylabel="EUR millions", title="Real Madrid transfer spending"); ax[0].legend()

    # 2) Squad value and points over time: Real Madrid vs rivals
    la = cs[cs.competition_id == LALIGA]
    for cid, (name, col) in RIVALS_C.items():
        d = la[la.club_id == cid].sort_values("season")
        ax[1].plot(d.season, d.squad_value / 1e6, marker="o", color=col, label=name)
        ax[2].plot(d.season, d.points, marker="o", color=col, label=name)
    ax[1].set(title="Squad market value", ylabel="EUR millions"); ax[1].legend()
    ax[2].set(title="La Liga points", ylabel="points"); ax[2].legend()
    fig.tight_layout(); source(fig); fig.savefig(FIG / "eda_overview.png")

    # 3) Biggest signings table (for a slide)
    top = (tr[tr.to_club_id == REAL_MADRID].nlargest(10, "transfer_fee")
             [["season", "player_name", "from_club_name", "transfer_fee"]])
    top["fee_m"] = top.transfer_fee / 1e6
    print(top.drop(columns="transfer_fee").to_string(index=False))

    # 4) Is a Poisson model reasonable for goals?  mean vs variance of goals per team per match
    g = load("games")
    g = g[(g.competition_id == LALIGA) & g.season.isin(SEASONS)]
    goals = pd.concat([g.home_club_goals, g.away_club_goals])
    print(f"Goals per team per match: mean {goals.mean():.2f}, variance {goals.var():.2f}  (Poisson: equal)")
    print(f"Corr(home goals, away goals) = {np.corrcoef(g.home_club_goals, g.away_club_goals)[0, 1]:.3f}  (independence: ~0)")
    return money, top


if __name__ == "__main__":
    run()
