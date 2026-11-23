"""Part 7 figures: grid vs MCMC vs exact posterior, an MCMC trace plot, and Weibull hazards by club.
python -m src.part7_figures"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logit
from config import DATA_PROC, FIG, REAL_MADRID, BIG_CLUBS
from src.style import GOLD, GREY, INK, NAVY, source  # noqa: E402  (theme)
from src.bayes_compute import grid_posterior, metropolis, first_goal_times, fit_weibull



def run():
    sg = pd.read_csv(DATA_PROC / "signings.csv")
    rm = sg[sg.to_club_id == REAL_MADRID]
    h, n = int(rm.success.sum()), len(rm)

    # 1) Three ways to the same posterior (conjugate prior) + the non-conjugate grid
    beta_lp = lambda p: stats.beta.logpdf(p, 5, 6)
    p, w, _, _ = grid_posterior(h, n, beta_lp)
    draws, acc = metropolis(lambda q: beta_lp(q) + stats.binom.logpmf(h, n, q), start=0.4)
    nc_lp = lambda q: stats.norm.logpdf(logit(q), -0.2, 0.5) - np.log(q * (1 - q))
    _, w_nc, _, _ = grid_posterior(h, n, nc_lp)
    dp = p[1] - p[0]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].hist(draws, bins=60, density=True, color=GREY, alpha=.6, label=f"MCMC draws (acceptance {acc:.0%})")
    ax[0].plot(p, stats.beta.pdf(p, 5 + h, 6 + n - h), "-", color=INK, lw=2, label="exact Beta posterior")
    ax[0].plot(p, w / dp, "--", color=GOLD, lw=2, label="grid approximation")
    ax[0].plot(p, w_nc / dp, ":", color=NAVY, lw=2, label="grid, logit-normal prior")
    ax[0].set(xlim=(0, 0.8), xlabel="Real Madrid's true hit rate p", ylabel="density",
              title="Grid, MCMC and the exact formula agree")
    ax[0].legend()
    # 2) Trace plot: is the chain mixing?
    ax[1].plot(draws[:3000], lw=.6, color=GREY)
    ax[1].plot(np.cumsum(draws[:3000]) / np.arange(1, 3001), color=GOLD, lw=2, label="running mean")
    ax[1].set(xlabel="iteration (after burn-in)", ylabel="p", title="MCMC trace: a 'fuzzy caterpillar' = good mixing")
    ax[1].legend()
    fig.tight_layout(); source(fig); fig.savefig(FIG / "p7_grid_mcmc.png")

    # 3) Weibull hazard for every big club: does the scoring rate rise during matches?
    rows = []
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tt = np.linspace(1, 90, 200)
    for cid, name in BIG_CLUBS.items():
        t, obs = first_goal_times(cid)
        if obs.sum() < 20:                      # club not in La Liga data: skip
            continue
        k, lam, lr, pval = fit_weibull(t, obs)
        rows.append(dict(club=name, k=k, scale=lam, LR=lr, p=pval, matches=len(t)))
        hz = (k / lam) * (tt / lam) ** (k - 1) * 90           # hazard per 90 minutes
        ax.plot(tt, hz, color=GOLD if cid == REAL_MADRID else GREY, lw=3 if cid == REAL_MADRID else 1.5, label=name)
    ax.set(xlabel="minute", ylabel="first-goal hazard (per 90 min)", title="Does the scoring hazard rise during matches?")
    ax.legend(fontsize=8)
    fig.tight_layout(); source(fig); fig.savefig(FIG / "p7_weibull_hazard.png")
    out = pd.DataFrame(rows).sort_values("k", ascending=False)
    print(out.round(3).to_string(index=False))
    out.to_csv(DATA_PROC / "p7_weibull_by_club.csv", index=False)
    return out


if __name__ == "__main__":
    run()
