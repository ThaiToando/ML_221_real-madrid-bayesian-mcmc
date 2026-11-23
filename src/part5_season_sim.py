"""Part 5: Poisson match model (MLE) + Monte Carlo season simulation + 'what if' a star leaves.

Model for each match:  home goals ~ Poisson(exp(mu + home + attack_h - defense_a))
                       away goals ~ Poisson(exp(mu + attack_a - defense_h)),  independent given the rates.
Fitted as a Poisson GLM, which maximises exactly this likelihood.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from config import DATA_PROC, FIG, REAL_MADRID, SEED
from src.style import BG, GOLD, GREY, INK, source  # noqa: E402  (theme)


def long_format(games):
    h = pd.DataFrame(dict(team=games.home_club_id, opp=games.away_club_id, goals=games.home_club_goals, home=1))
    a = pd.DataFrame(dict(team=games.away_club_id, opp=games.home_club_id, goals=games.away_club_goals, home=0))
    return pd.concat([h, a], ignore_index=True).astype({"team": str, "opp": str})


def fit_poisson(games):
    d = long_format(games)
    m = smf.glm("goals ~ home + C(team) + C(opp)", data=d, family=sm.families.Poisson()).fit()
    disp = m.pearson_chi2 / m.df_resid
    print(f"Home advantage: x{np.exp(m.params['home']):.3f} goals   Dispersion (want ~1): {disp:.2f}")
    return m


def rates(m, fixtures):
    """Expected goals for each fixture from the fitted model."""
    fx = fixtures.astype({"home_club_id": str, "away_club_id": str})
    lam_h = m.predict(pd.DataFrame(dict(team=fx.home_club_id, opp=fx.away_club_id, home=1)))
    lam_a = m.predict(pd.DataFrame(dict(team=fx.away_club_id, opp=fx.home_club_id, home=0)))
    return lam_h.to_numpy(), lam_a.to_numpy()


def match_probs(lam_h, lam_a, max_goals=10):
    """P(home win, draw, away win). Goal difference of two independent Poissons is Skellam."""
    p_home = stats.skellam.sf(0, lam_h, lam_a)          # P(H - A >= 1)
    p_draw = stats.skellam.pmf(0, lam_h, lam_a)
    return p_home, p_draw, 1 - p_home - p_draw


def simulate_season(fixtures, lam_h, lam_a, n_sims=10_000, base_points=None, seed=SEED):
    """Play every fixture n_sims times. Returns points matrix (n_sims x teams) and team list."""
    rng = np.random.default_rng(seed)
    teams = np.unique(np.r_[fixtures.home_club_id, fixtures.away_club_id])
    idx = {t: i for i, t in enumerate(teams)}
    hi = fixtures.home_club_id.map(idx).to_numpy(); ai = fixtures.away_club_id.map(idx).to_numpy()
    hg = rng.poisson(lam_h, size=(n_sims, len(fixtures)))
    ag = rng.poisson(lam_a, size=(n_sims, len(fixtures)))
    hp = np.where(hg > ag, 3, np.where(hg == ag, 1, 0)); ap = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
    pts = np.zeros((n_sims, len(teams))); gd = np.zeros_like(pts)
    for j in range(len(fixtures)):                      # add each match's points to both teams
        pts[:, hi[j]] += hp[:, j]; pts[:, ai[j]] += ap[:, j]
        gd[:, hi[j]] += hg[:, j] - ag[:, j]; gd[:, ai[j]] += ag[:, j] - hg[:, j]
    if base_points is not None:
        pts += np.array([base_points.get(t, 0) for t in teams])
    champion = np.argmax(pts + gd / 1000 + rng.random(pts.shape) / 1e6, axis=1)   # tie-break on goal difference
    return pts, teams, champion


def title_odds(teams, champion, names):
    p = pd.Series(np.bincount(champion, minlength=len(teams)) / len(champion), index=teams)
    return p.rename(index=names).sort_values(ascending=False)


def what_if(m, fixtures, team, share, replacement=0.5, n_sims=10_000):
    """Lower a team's scoring rate by the share of its goals a player scored,
    assuming his replacement produces `replacement` of his output."""
    lam_h, lam_a = rates(m, fixtures)
    factor = 1 - share * (1 - replacement)
    lam_h = np.where(fixtures.home_club_id == team, lam_h * factor, lam_h)
    lam_a = np.where(fixtures.away_club_id == team, lam_a * factor, lam_a)
    return simulate_season(fixtures, lam_h, lam_a, n_sims=n_sims)


def run(season=None, n_sims=10_000):
    g = pd.read_csv(DATA_PROC / "league_games.csv", parse_dates=["date"])
    season = season or int(g.season.max())
    gs = g[g.season == season].sort_values("date")
    names = dict(zip(gs.home_club_id, gs.home_club_name))

    # (a) Fit on the first half, simulate the second half, compare to what really happened
    half = len(gs) // 2
    first, second = gs.iloc[:half], gs.iloc[half:]
    m_half = fit_poisson(first)
    base_pts = {}
    for r in first.itertuples():
        hp = 3 if r.home_club_goals > r.away_club_goals else 1 if r.home_club_goals == r.away_club_goals else 0
        ap = 3 if r.away_club_goals > r.home_club_goals else 1 if r.home_club_goals == r.away_club_goals else 0
        base_pts[r.home_club_id] = base_pts.get(r.home_club_id, 0) + hp
        base_pts[r.away_club_id] = base_pts.get(r.away_club_id, 0) + ap
    lh, la = rates(m_half, second)
    pts, teams, champ = simulate_season(second, lh, la, n_sims=n_sims, base_points=base_pts)
    odds = title_odds(teams, champ, names)
    print("Title probabilities at the halfway point:\n", odds.head(5).round(3))

    # (b) Full-season model, El Clasico probabilities, and the 'what if' experiment
    m = fit_poisson(gs)
    clasico = gs[(gs.home_club_id == REAL_MADRID) & (gs.away_club_id == 131)]
    if len(clasico):
        lh_c, la_c = rates(m, clasico.head(1))
        ph, pd_, pa = match_probs(lh_c[0], la_c[0])
        print(f"Real Madrid vs Barcelona at home: W {ph:.3f}  D {pd_:.3f}  L {pa:.3f}")
    lh_f, la_f = rates(m, gs)
    pts0, teams0, ch0 = simulate_season(gs, lh_f, la_f, n_sims=n_sims)
    p0 = title_odds(teams0, ch0, names).get(names.get(REAL_MADRID, REAL_MADRID), 0)
    ps = pd.read_csv(DATA_PROC / "player_seasons.csv")
    rm = ps[(ps.club_id == REAL_MADRID) & (ps.season == season)]
    star = rm.sort_values("goals", ascending=False).iloc[0]
    share = star.goals / max(rm.goals.sum(), 1)
    pts1, teams1, ch1 = what_if(m, gs, REAL_MADRID, share, n_sims=n_sims)
    p1 = title_odds(teams1, ch1, names).get(names.get(REAL_MADRID, REAL_MADRID), 0)
    print(f"Without {star['name']} ({share:.0%} of goals): title prob {p0:.3f} -> {p1:.3f}  (drop {p0 - p1:.3f})")

    # Charts: points distribution (CLT shape) and running title probability (law of large numbers)
    rm_i = list(teams0).index(REAL_MADRID)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].hist(pts0[:, rm_i], bins=range(int(pts0[:, rm_i].min()), int(pts0[:, rm_i].max()) + 2), alpha=.6,
               label="with star", color=GOLD, edgecolor=BG)
    ax[0].hist(pts1[:, rm_i], bins=range(int(pts1[:, rm_i].min()), int(pts1[:, rm_i].max()) + 2), alpha=.5,
               label="without star", color=GREY)
    ax[0].set(xlabel="Real Madrid season points", ylabel="simulations", title=f"{n_sims:,} simulated seasons"); ax[0].legend()
    running = np.cumsum(ch0 == rm_i) / np.arange(1, n_sims + 1)
    ax[1].plot(running, color=GOLD); ax[1].axhline(p0, ls="--", color=INK)
    ax[1].set(xscale="log", xlabel="number of simulations", ylabel="P(Real Madrid champion)",
              title="Law of large numbers: the estimate settles down")
    fig.tight_layout(); source(fig); fig.savefig(FIG / "p5_simulation.png")
    return dict(model=m, half_odds=odds, p_title=p0, p_title_without_star=p1, star=star["name"])


def odds_comparison(path_csv):
    """Optional: compare to Bet365 using a football-data.co.uk file (e.g. SP1.csv for La Liga).
    Download: https://www.football-data.co.uk/mmz4281/2526/SP1.csv  (2526 = 2025/26)."""
    fd = pd.read_csv(path_csv, parse_dates=["Date"], dayfirst=True).sort_values("Date")
    fd = fd.rename(columns={"HomeTeam": "home_club_id", "AwayTeam": "away_club_id",
                            "FTHG": "home_club_goals", "FTAG": "away_club_goals"})
    half = len(fd) // 2
    m = fit_poisson(fd.iloc[:half]); test = fd.iloc[half:].copy()
    lh, la = rates(m, test)
    ph, pdr, pa = match_probs(lh, la)
    inv = 1 / test[["B365H", "B365D", "B365A"]]
    book = inv.div(inv.sum(axis=1), axis=0).to_numpy()       # remove the bookmaker's margin
    outcome = np.select([test.home_club_goals > test.away_club_goals, test.home_club_goals == test.away_club_goals], [0, 1], 2)
    onehot = np.eye(3)[outcome]
    brier_model = np.mean(np.sum((np.c_[ph, pdr, pa] - onehot) ** 2, axis=1))
    brier_book = np.mean(np.sum((book - onehot) ** 2, axis=1))
    print(f"Brier score (lower is better): our model {brier_model:.4f}  vs  Bet365 {brier_book:.4f}")
    return brier_model, brier_book


if __name__ == "__main__":
    run()
