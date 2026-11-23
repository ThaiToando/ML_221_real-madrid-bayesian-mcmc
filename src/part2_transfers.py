"""Part 2: Transfer hit rate.  Bayesian Beta-Binomial (like the NBA problem) + logistic regression."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score
from config import DATA_PROC, FIG, SEED
from src.style import GOLD, GREY, INK, source  # noqa: E402  (theme)

rng = np.random.default_rng(SEED)


def beta_prior_from_clubs(by_club):
    """Empirical Bayes: choose Beta(a, b) whose mean and variance match the clubs' hit rates
    (method of moments). The binomial noise in each club's rate is removed from the variance."""
    rates, n = by_club.hits / by_club.n, by_club.n
    m = rates.mean()
    v = max(rates.var(ddof=1) - (m * (1 - m) / n).mean(), 1e-4)   # between-club variance only
    strength = np.clip(m * (1 - m) / v - 1, 2, 200)                 # a + b = "prior sample size"
    return m * strength, (1 - m) * strength


def bayes_hit_rates(sg):
    by_club = sg.groupby("club").success.agg(hits="sum", n="count")
    a, b = beta_prior_from_clubs(by_club)
    print(f"Prior: Beta({a:.2f}, {b:.2f})  mean {a / (a + b):.3f}  (worth {a + b:.1f} signings)")
    out = by_club.copy()
    out["post_a"], out["post_b"] = a + out.hits, b + out.n - out.hits
    out["raw_rate"] = out.hits / out.n
    out["post_mean"] = out.post_a / (out.post_a + out.post_b)
    out["cred_lo"] = stats.beta.ppf(0.025, out.post_a, out.post_b)
    out["cred_hi"] = stats.beta.ppf(0.975, out.post_a, out.post_b)
    se = np.sqrt(out.raw_rate * (1 - out.raw_rate) / out.n)              # classical (Wald) interval
    out["wald_lo"], out["wald_hi"] = out.raw_rate - 1.96 * se, out.raw_rate + 1.96 * se
    # P(Real Madrid's true hit rate > rival's) by simulation from both posteriors
    draws = {c: rng.beta(r.post_a, r.post_b, 100_000) for c, r in out.iterrows()}
    out["P_RM_better"] = [np.mean(draws["Real Madrid"] > draws[c]) for c in out.index]
    out = out.sort_values("post_mean")
    print(out[["n", "hits", "raw_rate", "post_mean", "cred_lo", "cred_hi", "P_RM_better"]].round(3))
    # Forest plot: credible intervals with raw rates
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(out))
    colors = [GOLD if c == "Real Madrid" else GREY for c in out.index]
    ax.hlines(y, out.cred_lo, out.cred_hi, color=colors, lw=3)
    ax.scatter(out.post_mean, y, color=colors, s=60, zorder=3, label="posterior mean")
    ax.scatter(out.raw_rate, y, marker="x", color=INK, label="raw hit rate")
    ax.axvline(a / (a + b), ls="--", color="gray", label="prior mean")
    ax.set_yticks(y, out.index); ax.set_xlabel("P(signing becomes a regular)")
    ax.set_title("Transfer hit rate: 95% credible intervals"); ax.legend(loc="lower right")
    fig.tight_layout(); source(fig); fig.savefig(FIG / "p2_hit_rates.png")
    out.to_csv(DATA_PROC / "p2_hit_rates.csv")
    return out, (a, b)


def logistic(sg):
    d = sg.dropna(subset=["age", "log_fee"]).copy()
    d["age_c"] = d.age - 24                        # centre age so age^2 is less correlated with age
    m = smf.logit("success ~ age_c + I(age_c**2) + log_fee + C(position) + from_top5 + is_real_madrid",
                  data=d).fit(disp=0)
    print(m.summary().tables[1])
    print("Odds ratios:\n", np.exp(m.params).round(3))
    print(f"McFadden pseudo-R^2 = {m.prsquared:.3f}   AUC = {roc_auc_score(d.success, m.predict(d)):.3f}")
    # Calibration check: predicted vs observed in 5 bins (is the model honest?)
    d["p_hat"] = m.predict(d)
    d["bin"] = pd.qcut(d.p_hat, 5, duplicates="drop")
    cal = d.groupby("bin", observed=True).agg(pred=("p_hat", "mean"), obs=("success", "mean"), n=("success", "size"))
    print(cal.round(3))
    # Prediction for a hypothetical signing (like 'radius 2.5' in the homework)
    new = pd.DataFrame(dict(age_c=[21 - 24], log_fee=[np.log(40)], position=["Attack"], from_top5=[0], is_real_madrid=[1]))
    pred = m.get_prediction(new).summary_frame(alpha=0.05)
    print("21-year-old attacker, EUR 40M, from outside the top 5 leagues, to Real Madrid:")
    print(pred.round(3))
    return m


def run():
    sg = pd.read_csv(DATA_PROC / "signings.csv", parse_dates=["transfer_date"])
    out, prior = bayes_hit_rates(sg)
    m = logistic(sg)
    return out, prior, m


if __name__ == "__main__":
    run()
