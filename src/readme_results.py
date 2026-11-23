"""Fill the results table in README.md from data/processed (numbers are never typed by hand).   python -m src.readme_results
Rewrites only the lines between <!-- results:start --> and <!-- results:end -->."""
import re
from pathlib import Path

from src.slides import numbers

README = Path(__file__).resolve().parents[1] / "README.md"


def table(n: dict) -> str:
    p = lambda x: f"{x:.0%}"  # noqa: E731
    rows = [
        ("1 · Regression", "Does squad value predict league points?",
         f"Squad value explains **{p(n['r2'])}** of the variation in points; +20% value ≈ +{n['pts_20']:.1f} points a season"),
        ("1 · Residuals", "Does Madrid beat its budget?", f"Above the budget line in **{n['above']} of {n['seasons']}** seasons"),
        ("2 · Beta-Binomial", "How often do signings work out?",
         f"**{p(n['post'])}** become regulars (raw {n['hits']}/{n['n']}, 95% credible interval {p(n['lo'])}–{p(n['hi'])})"),
        ("2 · Logistic", "Can age and fee pick the better signing?", f"AUC **{p(n['auc'])}**"),
        ("3 · Gamma-Poisson", "Real output or a hot streak?",
         f"{n['top_name']}: **{n['top_post']:.2f}** goals + assists per 90 after shrinkage"),
        ("4 · Quadratic + bootstrap", "When is a player worth the most?",
         f"Attackers peak at **{n['peak']:.1f}** (95% CI {n['peak_lo']:.1f}–{n['peak_hi']:.1f})"),
        ("5 · Poisson", "El Clásico at home?", f"Madrid win **{p(n['p_win'])}**, draw {p(n['p_draw'])}"),
        ("5 · Monte Carlo", f"What is {n['star']} worth?",
         f"Title chance **{p(n['p_title'])} → {p(n['p_without'])}** without him ({n['n_sims']:,} simulated seasons)"),
        ("6 · Permutation test", "Are the late goals luck?",
         f"{p(n['late_rm'])} of Madrid's goals after 75' vs {p(n['late_oth'])} for rivals, **{n['perm_txt']}**"),
        ("7 · MCMC, Weibull", "Do the answers hold up?", f"Same posterior by MCMC; Weibull shape {n['weib_k']:.2f}"),
    ]
    out = ["| Part | Question | Answer |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out)


def main():
    text = README.read_text(encoding="utf-8")
    new = re.sub(r"(<!-- results:start -->\n).*?(\n<!-- results:end -->)", lambda m: m.group(1) + table(numbers()) + m.group(2),
                 text, flags=re.S)
    README.write_text(new, encoding="utf-8")
    print("README.md results table refreshed")


if __name__ == "__main__":
    main()
