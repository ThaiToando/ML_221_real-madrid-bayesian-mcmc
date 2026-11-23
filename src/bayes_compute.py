"""Section 11A: Bayesian computation (grid + MCMC) and Weibull goal timing.
python -m src.bayes_compute"""
import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import expit, logit
from config import DATA_PROC, REAL_MADRID, SEED
from src.data_io import load

rng = np.random.default_rng(SEED)


# ---------- 1. Grid approximation: posterior for a hit rate ----------
def grid_posterior(h, n, log_prior, grid_size=2001):
    """Posterior on a grid of p values: prior x likelihood, then normalise."""
    p = np.linspace(0.0005, 0.9995, grid_size)
    log_post = log_prior(p) + stats.binom.logpmf(h, n, p)   # log scale avoids underflow
    w = np.exp(log_post - log_post.max())
    w /= w.sum()                                               # normalise: weights sum to 1
    mean = np.sum(p * w)
    cdf = np.cumsum(w)
    lo, hi = p[np.searchsorted(cdf, 0.025)], p[np.searchsorted(cdf, 0.975)]
    return p, w, mean, (lo, hi)


# ---------- 2. Metropolis MCMC: the same posterior by random walk ----------
def metropolis(log_post, start, n_iter=20_000, step=1.0, burn=2_000):
    """Random-walk Metropolis on the logit scale (so p stays in (0, 1))."""
    x = logit(start)
    def target(z):                                   # log posterior of p plus the change-of-variable term
        p = expit(z)
        return log_post(p) + np.log(p) + np.log(1 - p)
    cur = target(x)
    draws, accepted = np.empty(n_iter), 0
    for i in range(n_iter):
        prop = x + step * rng.normal()                       # 1. propose a nearby value
        new = target(prop)
        if np.log(rng.random()) < new - cur:                 # 2. accept with prob min(1, ratio)
            x, cur = prop, new
            accepted += 1
        draws[i] = expit(x)                                  # 3. record the current state
    return draws[burn:], accepted / n_iter


# ---------- 3. Weibull model for the time to Real Madrid's first goal (censored) ----------
def first_goal_times(club_id=REAL_MADRID):
    ev, cg = load("game_events"), load("club_games")
    g = load("games")
    league = g[g.competition_id == "ES1"].game_id
    rm_games = cg[(cg.club_id == club_id) & cg.game_id.isin(league)].game_id.unique()
    goals = ev[ev.type.astype(str).str.lower().str.contains("goal") & (ev.club_id == club_id)]
    first = goals.groupby("game_id").minute.min().reindex(rm_games)
    observed = first.notna().to_numpy()                      # False = no goal: censored at 90
    t = first.fillna(90).clip(0.5, 90).to_numpy()
    return t, observed


def weibull_negloglik(params, t, observed):
    k, lam = np.exp(params)                                  # shape k, scale lambda (minutes)
    log_f = np.log(k / lam) + (k - 1) * np.log(t / lam) - (t / lam) ** k
    log_S = -(t / lam) ** k                                  # survival: no goal yet by time t
    return -(np.sum(log_f[observed]) + np.sum(log_S[~observed]))


def fit_weibull(t, observed):
    res = optimize.minimize(weibull_negloglik, x0=np.log([1.0, 40.0]), args=(t, observed), method="Nelder-Mead")
    k, lam = np.exp(res.x)
    # Exponential = Weibull with k = 1: fit only the scale, then likelihood-ratio test
    res0 = optimize.minimize_scalar(lambda ll: weibull_negloglik(np.array([0.0, ll]), t, observed),
                                    bounds=(0, 8), method="bounded")
    lr = 2 * (res0.fun - res.fun)
    return k, lam, lr, stats.chi2.sf(lr, 1)


def run():
    sg = pd.read_csv(DATA_PROC / "signings.csv")
    rm = sg[sg.to_club_id == REAL_MADRID]
    h, n = int(rm.success.sum()), len(rm)
    # (a) grid with the conjugate Beta(5, 6) prior: must match the exact Beta posterior
    p, w, mean, ci = grid_posterior(h, n, lambda p: stats.beta.logpdf(p, 5, 6))
    print(f"Grid, Beta(5,6) prior: mean {mean:.4f}, 95% {np.round(ci, 3)}  | exact mean {(5 + h) / (11 + n):.4f}")
    # (b) grid with a NON-conjugate prior: logit(p) ~ Normal(-0.2, 0.5^2)
    lp = lambda p: stats.norm.logpdf(logit(p), -0.2, 0.5) - np.log(p * (1 - p))
    _, _, mean2, ci2 = grid_posterior(h, n, lp)
    print(f"Grid, logit-normal prior: mean {mean2:.4f}, 95% {np.round(ci2, 3)}")
    # (c) MCMC for the same non-conjugate posterior: should agree with the grid
    draws, acc = metropolis(lambda p: lp(p) + stats.binom.logpmf(h, n, p), start=0.4)
    print(f"MCMC: mean {draws.mean():.4f}, 95% {np.round(np.percentile(draws, [2.5, 97.5]), 3)}, acceptance {acc:.2f}")
    # (d) Weibull vs Exponential for the time to Real Madrid's first goal
    t, obs = first_goal_times()
    k, lam, lr, pval = fit_weibull(t, obs)
    print(f"Weibull: shape k = {k:.3f}, scale = {lam:.1f} min; LR vs Exponential = {lr:.2f}, p = {pval:.4f}")
    # (e) Gamma MLE for attackers' per-90 rates (compare with Part 3's Negative Binomial fit)
    ps = pd.read_csv(DATA_PROC / "player_seasons.csv")
    rates = ps[(ps.position == "Attack") & (ps.minutes >= 900)].ga_per90
    a, loc, scale = stats.gamma.fit(rates[rates > 0], floc=0)
    print(f"Gamma fit to observed rates: shape {a:.2f}, rate {1 / scale:.2f}  (wider than the true-rate prior: includes Poisson noise)")
    return dict(grid=(mean, ci), grid_nc=(mean2, ci2), mcmc=(draws.mean(), acc), weibull=(k, lam, lr, pval))


if __name__ == "__main__":
    run()
