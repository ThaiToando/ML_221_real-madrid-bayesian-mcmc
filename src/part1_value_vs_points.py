"""Part 1: Does squad value predict points?  (log-linear regression, like HW7 #12)"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from config import DATA_PROC, FIG, LALIGA, REAL_MADRID
from src.style import BG, DOWN, GOLD, INK, UP, source  # noqa: E402  (theme)


def run():
    cs = pd.read_csv(DATA_PROC / "club_seasons.csv")
    d = cs[(cs.competition_id == LALIGA) & cs.rel_value.notna()].copy()
    d["log_rel_value"] = np.log(d.rel_value)

    # Model A (not logged) vs Model B (logged): compare R^2 like HW7 #12
    m_raw = smf.ols("points ~ rel_value", data=d).fit()
    m_log = smf.ols("points ~ log_rel_value", data=d).fit()
    print(m_log.summary().tables[1])
    print(f"R^2 raw = {m_raw.rsquared:.3f}   R^2 logged = {m_log.rsquared:.3f}")
    b = m_log.params["log_rel_value"]
    print(f"A 20% bigger squad value -> {b * np.log(1.2):+.2f} points per season")
    print("95% CI for slope:", m_log.conf_int().loc["log_rel_value"].round(2).tolist())

    # Assumption checks: residuals vs fitted, QQ plot, normality test
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].scatter(d.log_rel_value, d.points, alpha=.4, label="all clubs")
    rm = d[d.club_id == REAL_MADRID]
    ax[0].scatter(rm.log_rel_value, rm.points, color=GOLD, edgecolor=BG, s=80, label="Real Madrid")
    xs = np.linspace(d.log_rel_value.min(), d.log_rel_value.max(), 50)
    ax[0].plot(xs, m_log.params.iloc[0] + b * xs, "--", color=INK)
    ax[0].set(xlabel="log(squad value / league average)", ylabel="Points", title="Money vs points, La Liga")
    ax[0].legend()
    ax[1].scatter(m_log.fittedvalues, m_log.resid, alpha=.4); ax[1].axhline(0, color=INK)
    ax[1].set(xlabel="Fitted points", ylabel="Residual", title="Residuals vs fitted")
    sm.qqplot(m_log.resid, line="s", ax=ax[2]); ax[2].set_title("Normal QQ plot of residuals")
    fig.tight_layout(); source(fig); fig.savefig(FIG / "p1_value_vs_points.png")
    print("Shapiro-Wilk p-value (normal residuals?):", round(stats.shapiro(m_log.resid).pvalue, 3))

    # Real Madrid over/under-performance by season (residual = actual - predicted points)
    rm = rm.assign(residual=m_log.resid.loc[rm.index]).sort_values("season")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(rm.season.astype(str), rm.residual, color=np.where(rm.residual > 0, UP, DOWN))
    ax.axhline(0, color=INK); ax.set(ylabel="Points above/below what money predicts",
                                     title="Real Madrid: over- or under-performing the budget?")
    fig.tight_layout(); source(fig); fig.savefig(FIG / "p1_rm_residuals.png")
    rm[["season", "points", "squad_value", "rel_value", "residual"]].to_csv(DATA_PROC / "p1_rm.csv", index=False)
    return m_log


if __name__ == "__main__":
    run()
