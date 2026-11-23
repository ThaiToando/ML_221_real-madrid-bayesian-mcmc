"""Part 4: When does market value peak?  Quadratic + interaction regression (like HW7 #11),
peak age = vertex of the parabola, with a cluster bootstrap confidence interval."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from config import DATA_PROC, FIG, SEED
from src.style import source  # noqa: E402  (theme)

rng = np.random.default_rng(SEED)
POSITIONS = ["Attack", "Midfield", "Defender", "Goalkeeper"]


def prep():
    ps = pd.read_csv(DATA_PROC / "player_seasons.csv")
    d = ps[(ps.minutes >= 450) & ps.value_eur.gt(0) & ps.age.between(17, 38) & ps.position.isin(POSITIONS)].copy()
    d["log_value"] = np.log(d.value_eur)
    d["age_c"] = d.age - 26
    d["min_share"] = d.minutes / (38 * 90)
    return d


def backward_stepwise(formula_terms, data, response="log_value"):
    """Drop terms one at a time while AIC improves (the 'stepwise' idea from class)."""
    terms = list(formula_terms)
    best = smf.ols(f"{response} ~ " + " + ".join(terms), data=data).fit()
    improved = True
    while improved and len(terms) > 1:
        improved = False
        for t in terms:
            trial = [x for x in terms if x != t]
            m = smf.ols(f"{response} ~ " + " + ".join(trial), data=data).fit()
            if m.aic < best.aic - 2:
                best, terms, improved = m, trial, True
                print(f"  dropped {t}  (AIC {m.aic:.1f})")
                break
    return best


def peak_age(d):
    """Peak age = 26 - b1 / (2 b2) from log_value ~ b0 + b1*age_c + b2*age_c^2 + ..."""
    m = smf.ols("log_value ~ age_c + I(age_c**2) + ga_per90 + min_share", data=d).fit()
    b1, b2 = m.params["age_c"], m.params["I(age_c ** 2)"]
    return 26 - b1 / (2 * b2), m


def cluster_bootstrap_peak(d, B=1000):
    """Resample PLAYERS (not rows) because seasons of the same player are not independent."""
    ids = d.player_id.unique()
    groups = {p: g for p, g in d.groupby("player_id")}
    peaks = []
    for _ in range(B):
        sample = pd.concat([groups[p] for p in rng.choice(ids, size=len(ids), replace=True)])
        try:
            peaks.append(peak_age(sample)[0])
        except Exception:
            continue
    return np.array(peaks)


def run(B=500):
    d = prep()
    full = backward_stepwise(["age_c", "I(age_c**2)", "ga_per90", "min_share", "C(position)",
                              "age_c:C(position)", "I(age_c**2):C(position)"], d)
    print(full.summary().tables[1]); print(f"R^2 = {full.rsquared:.3f}")
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for pos in POSITIONS:
        dp = d[d.position == pos]
        pk, m = peak_age(dp)
        boots = cluster_bootstrap_peak(dp, B=B)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(position=pos, peak_age=pk, ci_lo=lo, ci_hi=hi, n=len(dp)))
        ages = np.linspace(18, 36, 100)
        base = m.params["Intercept"] + m.params["ga_per90"] * dp.ga_per90.mean() + m.params["min_share"] * dp.min_share.mean()
        curve = base + m.params["age_c"] * (ages - 26) + m.params["I(age_c ** 2)"] * (ages - 26) ** 2
        ax[0].plot(ages, np.exp(curve) / 1e6, label=f"{pos} (peak {pk:.1f})")
        ax[1].hist(boots, bins=30, alpha=.5, label=pos)
    ax[0].set(xlabel="Age", ylabel="Predicted value (EUR M)", title="Market value vs age by position"); ax[0].legend()
    ax[1].set(xlabel="Bootstrapped peak age", title=f"Cluster bootstrap, B = {B}"); ax[1].legend()
    fig.tight_layout(); source(fig); fig.savefig(FIG / "p4_peak_age.png")
    out = pd.DataFrame(rows); print(out.round(2)); out.to_csv(DATA_PROC / "p4_peak_age.csv", index=False)
    return out, full


if __name__ == "__main__":
    run()
