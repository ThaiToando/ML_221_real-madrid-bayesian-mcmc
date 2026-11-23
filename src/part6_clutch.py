"""Part 6: 'Madrid magic' - do they score more late in games?

If goals arrive as a Poisson process with constant rate, then GIVEN the number of goals,
the goal minutes are independent Uniform(0, 90). So the share of goals in minutes 76-90 should be 15/90.
Stoppage time inflates the last bin for EVERY team, so we compare Real Madrid with other big clubs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from config import FIG, DATA_PROC, REAL_MADRID, BIG_CLUBS, SEED
from src.style import BG, GOLD, GREY, INK, source  # noqa: E402  (theme)
from src.data_io import load

rng = np.random.default_rng(SEED)
BINS = [0, 15, 30, 45, 60, 75, 120]
LABELS = ["1-15", "16-30", "31-45+", "46-60", "61-75", "76-90+"]


def run():
    ev = load("game_events")
    goals = ev[ev.type.astype(str).str.lower().str.contains("goal") & ev.club_id.isin(BIG_CLUBS) & ev.minute.between(1, 90)]
    goals = goals.assign(bin=pd.cut(goals.minute, BINS, labels=LABELS), late=goals.minute > 75)
    rm, others = goals[goals.club_id == REAL_MADRID], goals[goals.club_id != REAL_MADRID]

    # 1) Chi-squared goodness of fit: Real Madrid vs the uniform (constant-rate Poisson) model
    obs = rm.bin.value_counts().reindex(LABELS).to_numpy()
    exp = len(rm) * np.diff(np.minimum(BINS, 90)) / 90
    chi = stats.chisquare(obs, exp)
    print(f"Real Madrid goals by 15 min: {dict(zip(LABELS, obs.astype(int).tolist()))}\nChi-square vs uniform: stat {chi.statistic:.1f}, p = {chi.pvalue:.4f}")

    # 2) Bayesian comparison of late-goal share: Beta(1,1) prior -> Beta(1 + late, 1 + not late)
    post = {}
    for cid, g in goals.groupby("club_id"):
        post[BIG_CLUBS[cid]] = (1 + g.late.sum(), 1 + (~g.late).sum())
    draws_rm = rng.beta(*post["Real Madrid"], 100_000)
    draws_oth = rng.beta(1 + others.late.sum(), 1 + (~others.late).sum(), 100_000)
    print(f"Late share: Real Madrid {rm.late.mean():.3f}  vs other big clubs {others.late.mean():.3f}")
    print(f"P(Real Madrid's true late share > other big clubs') = {np.mean(draws_rm > draws_oth):.3f}")
    lo, hi = np.percentile(draws_rm - draws_oth, [2.5, 97.5])
    print(f"95% credible interval for the difference: [{lo:.3f}, {hi:.3f}]")

    # 3) Chart: goal rate per minute in each bin, Real Madrid vs other big clubs
    fig, ax = plt.subplots(figsize=(9, 4.5))
    share_rm = rm.bin.value_counts(normalize=True).reindex(LABELS)
    share_ot = others.bin.value_counts(normalize=True).reindex(LABELS)
    x = np.arange(len(LABELS))
    ax.bar(x - .2, share_rm, .4, color=GOLD, edgecolor=BG, label="Real Madrid")
    ax.bar(x + .2, share_ot, .4, color=GREY, label="Other big clubs")
    ax.axhline(1 / 6, ls="--", color=INK, label="constant-rate Poisson")
    ax.set_xticks(x, LABELS); ax.set(ylabel="share of goals", title="When are the goals scored?"); ax.legend()
    fig.tight_layout(); source(fig); fig.savefig(FIG / "p6_clutch.png")
    n_ot = others.bin.value_counts().reindex(LABELS).to_numpy()
    pd.DataFrame(dict(real_madrid=share_rm, others=share_ot, n_rm=obs, exp_rm=exp, n_others=n_ot),
                 index=pd.Index(LABELS, name="bin")).to_csv(DATA_PROC / "p6_goal_timing.csv")   # counts feed the chi-square chart
    return chi, np.mean(draws_rm > draws_oth)


if __name__ == "__main__":
    run()
