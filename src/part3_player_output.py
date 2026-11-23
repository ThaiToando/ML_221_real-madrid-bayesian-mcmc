"""Part 3: Real output or luck?  Gamma-Poisson model for goals + assists per 90 (like the shortstop problem).

Model:  goal contributions k ~ Poisson(lambda * t),  t = minutes / 90,  lambda ~ Gamma(alpha, beta).
Integrating lambda out gives a Negative Binomial (HW7 #3). We fit alpha, beta by maximum likelihood.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import gammaln
from config import DATA_PROC, FIG, REAL_MADRID
from src.style import BG, GOLD, GREY, INK, source  # noqa: E402  (theme)


def neg_loglik(log_params, k, t):
    a, b = np.exp(log_params)                    # work on the log scale so a, b stay positive
    ll = gammaln(k + a) - gammaln(a) - gammaln(k + 1) + a * np.log(b / (b + t)) + k * np.log(t / (b + t))
    return -ll.sum()


def fit_prior(ps):
    att = ps[(ps.position == "Attack") & (ps.minutes >= 900)]
    k, t = att.ga.to_numpy(), (att.minutes / 90).to_numpy()
    res = optimize.minimize(neg_loglik, x0=np.log([2.0, 4.0]), args=(k, t), method="Nelder-Mead")
    a, b = np.exp(res.x)
    print(f"Prior by MLE: lambda ~ Gamma(alpha={a:.2f}, rate={b:.2f}); mean {a / b:.3f} G+A per 90")
    # Does letting lambda vary between players help? Compare with one common Poisson rate for everyone.
    lam_hat = k.sum() / t.sum()                                   # MLE of a single common rate
    ll_pois = stats.poisson.logpmf(k, lam_hat * t).sum()
    ll_nb = -res.fun
    print(f"Log-likelihood: one common rate {ll_pois:.1f}  vs  Gamma-Poisson {ll_nb:.1f}  "
          f"(LR statistic {2 * (ll_nb - ll_pois):.1f}; big = players really differ)")
    return a, b, att


def posterior_table(ps, sg, a, b):
    rm = ps[(ps.club_id == REAL_MADRID) & (ps.position == "Attack") & (ps.minutes > 0)]
    g = rm.groupby(["player_id", "name"]).agg(k=("ga", "sum"), minutes=("minutes", "sum")).reset_index()
    g["t"] = g.minutes / 90
    g["raw_per90"] = g.k / g.t
    g["post_a"], g["post_b"] = a + g.k, b + g.t
    g["post_mean"] = g.post_a / g.post_b
    g["lo"] = stats.gamma.ppf(0.025, g.post_a, scale=1 / g.post_b)
    g["hi"] = stats.gamma.ppf(0.975, g.post_a, scale=1 / g.post_b)
    g["P_over_0.6"] = stats.gamma.sf(0.6, g.post_a, scale=1 / g.post_b)
    fee = sg[sg.to_club_id == REAL_MADRID].groupby("player_id").fee_m.sum()
    g["fee_m"] = g.player_id.map(fee)
    g["fee_per_ga90_m"] = g.fee_m / g.post_mean      # EUR millions per (goal contribution per 90)
    g = g.sort_values("post_mean", ascending=False)
    print(g[["name", "k", "minutes", "raw_per90", "post_mean", "lo", "hi", "P_over_0.6", "fee_m"]].head(15).round(3))
    g.to_csv(DATA_PROC / "p3_rm_attackers.csv", index=False)
    return g


def plot(g, a, b):
    top = g[g.minutes >= 450].head(12).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    y = np.arange(len(top))
    ax[0].hlines(y, top.lo, top.hi, color=GREY, lw=3)
    ax[0].scatter(top.post_mean, y, color=GOLD, edgecolor=BG, s=70, zorder=3, label="posterior mean")
    ax[0].scatter(top.raw_per90, y, marker="x", color=INK, label="raw per 90")
    ax[0].set_yticks(y, top.name); ax[0].set_xlabel("Goals + assists per 90")
    ax[0].set_title("Real Madrid attackers: shrinkage toward the prior"); ax[0].legend()
    lam = np.linspace(0, max(1.6, top.hi.max() * 1.15), 400)
    ax[1].plot(lam, stats.gamma.pdf(lam, a, scale=1 / b), "--", color=INK, label="prior (all attackers)")
    for r in top.iloc[::-1].head(3).itertuples():
        ax[1].plot(lam, stats.gamma.pdf(lam, r.post_a, scale=1 / r.post_b), label=r.name)
    ax[1].set(xlabel="lambda (true G+A per 90)", ylabel="density", title="Prior vs posteriors")
    ax[1].legend(); fig.tight_layout(); source(fig); fig.savefig(FIG / "p3_shrinkage.png")


def run():
    ps = pd.read_csv(DATA_PROC / "player_seasons.csv")
    sg = pd.read_csv(DATA_PROC / "signings.csv")
    a, b, _ = fit_prior(ps)
    g = posterior_table(ps, sg, a, b)
    plot(g, a, b)
    return g, (a, b)


if __name__ == "__main__":
    run()
