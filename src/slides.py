"""src/slides.py - builds the final presentation from the project's own outputs.

Two outputs from ONE slide list (so they never disagree), written to docs/ (GitHub Pages serves this folder):
  docs/index.html   reveal.js deck: interactive Real Madrid visuals (play a season, pick a player, rotate a
                    3D scoreline), looping videos, formulas typeset with KaTeX, speaker notes (press S)
  docs/deck.pptx    PowerPoint copy: static PNG twins, looping GIFs, the same titles, numbers and notes
  docs/deck.pdf     (--pdf) a print of index.html with every interactive chart swapped for its PNG
Every number on a slide is read from data/processed/ (CSV files + deck_numbers.json from src.deck_visuals).
If the data are the FAKE test tables, every slide carries a red "FAKE DATA" badge.

Run: python -m src.slides [--pdf]     (after python run_all.py, python -m src.animate, python -m src.deck_visuals)
Content rule: few words; only ML 221 formulas (Poisson, Binomial, Beta/Gamma posteriors, regression, CLT/LLN,
chi-square, Skellam); every slide ties the math to Real Madrid.
"""
import html
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

C.TAB = C.DATA_PROC                       # the engine below reads tables from here
DECK = C.ROOT / "docs"
ASSETS = DECK / "assets"
REVEAL_CDN = "https://cdn.jsdelivr.net/npm/reveal.js@5.1.0"
NODE = DECK / "node_modules"          # npm install --prefix docs reveal.js@5.1.0 katex@0.16  (offline deck)
from src.style import DECK as _D  # noqa: E402
NAVY, LLY, NVO, SPX = _D["accent"], _D["c1"], _D["c2"], _D["c3"]     # accent / Real Madrid / rivals / others
APP_URL = "https://YOUR-APP.streamlit.app"          # paste the deployed Streamlit link here
SITE_URL = "https://thaitoando.github.io/ML_221_real-madrid-bayesian-mcmc/"   # GitHub Pages: link previews use docs/img/social.png
AUTHOR = "Do Thai Toan"
COURSE = "ML 221 Applied Probability"
TITLE = "Does money buy wins? Bayesian inference and MCMC at Real Madrid"
REPORT = "Real Madrid CF, La Liga 2014/15 to 2025/26"


def is_fake() -> bool:
    """The fake generator names every player 'Player <n>'; real Transfermarkt names never look like that."""
    try:
        names = pd.read_csv(C.DATA_PROC / "player_seasons.csv", usecols=["name"]).name.astype(str)
        return names.str.fullmatch(r"Player \d+").mean() > 0.9
    except Exception:
        return True


def numbers() -> dict:
    """Every number used on a slide, read from data/processed/ (never typed by hand)."""
    P = C.DATA_PROC
    j = json.loads((P / "deck_numbers.json").read_text())
    rm1 = pd.read_csv(P / "p1_rm.csv")
    hr = pd.read_csv(P / "p2_hit_rates.csv").set_index("club")
    att = pd.read_csv(P / "p3_rm_attackers.csv")
    att = att[att.minutes >= 450].sort_values("post_mean", ascending=False)
    pk = pd.read_csv(P / "p4_peak_age.csv").set_index("position")
    gt = pd.read_csv(P / "p6_goal_timing.csv").set_index("bin")
    wb = pd.read_csv(P / "p7_weibull_by_club.csv").set_index("club")
    best = rm1.loc[rm1.residual.idxmax()]
    rm = hr.loc["Real Madrid"]
    a0, b0 = rm.post_a - rm.hits, rm.post_b - (rm.n - rm.hits)
    top = att.iloc[0]
    lucky = att.assign(gap=att.raw_per90 - att.post_mean).sort_values("gap").iloc[0]
    return dict(
        seasons=len(rm1), above=int((rm1.residual > 0).sum()), best_season=int(best.season), best_res=best.residual,
        rel_value=rm1.rel_value.mean(), r2=j["r2"], pts_20=j["pts_20pct"],
        hits=int(rm.hits), n=int(rm.n), raw=rm.raw_rate, post=rm.post_mean, lo=rm.cred_lo, hi=rm.cred_hi,
        prior_mean=a0 / (a0 + b0), prior_n=a0 + b0, p_vs_bar=hr.loc["Barcelona", "P_RM_better"] if "Barcelona" in hr.index else float("nan"),
        top_name=top["name"], top_post=top.post_mean, top_lo=top.lo, top_hi=top.hi, lucky_name=lucky["name"],
        lucky_raw=lucky.raw_per90, lucky_post=lucky.post_mean,
        peak=pk.loc["Attack", "peak_age"], peak_lo=pk.loc["Attack", "ci_lo"], peak_hi=pk.loc["Attack", "ci_hi"],
        lam_rm=j["lam_rm"], lam_bar=j["lam_bar"], p_win=j["p_win"], p_draw=j["p_draw"],
        p_title=j["p_title"], p_without=j["p_title_without"], star=j["star"], share=j["star_share"],
        pts_with=j["pts_with"], pts_without=j["pts_without"], n_sims=int(j["n_sims"]),
        late_rm=gt.loc["76-90+", "real_madrid"], late_oth=gt.loc["76-90+", "others"],
        weib_k=wb.loc["Real Madrid", "k"], weib_p=wb.loc["Real Madrid", "p"],
        **{k: j.get(k, float("nan")) for k in ("perm_p", "auc", "mcmc_acc", "cov_wald", "cov_cred", "clt_mu", "clt_sd",
                                               "first_theory", "mle_a", "mle_b")},
        perm_B=int(j.get("perm_B", 0)), r0_score=j.get("r0_score", "–"),
        perm_txt=(f"p ≈ {j.get('perm_p', float('nan')):.4f}" if j.get("perm_p", 1) > 0     # notes 9.14: never write p = 0
                  else f"p < {1 / max(j.get('perm_B', 1), 1):.4f}"),
    )


def photo_credit() -> list[str]:
    """CC BY / CC BY-SA need visible credit: one line with the photographers, read from figures/players/LICENSES.md."""
    import re as _re
    lic = C.FIG / "players" / "LICENSES.md"
    if not lic.exists():
        return []
    who = list(dict.fromkeys(_re.findall(r"\bby (.+?), (CC[^,]*|Public domain)", lic.read_text(encoding="utf-8"))))
    return ["Photos: Wikimedia Commons, " + "; ".join(f"{a} ({l})" for a, l in who)] if who else []


HINT_WORDS = ("Press", "Drag", "Rotate", "Click", "Move", "Pick", "Scan")


def shown(s):
    """Bullets for the audience. Presenter instructions ("Press ▶ ...", "Drag ...") go to the speaker notes instead."""
    return [b for b in s.get("bullets", []) if not b.startswith(HINT_WORDS) or b.startswith("Scan")]


def hints(s):
    return [b for b in s.get("bullets", []) if b.startswith(HINT_WORDS) and not b.startswith("Scan")]


def numbered(cards):
    """Cards as (number, text): a leading '1 ' is used as the number, otherwise they are counted."""
    out = []
    for k, c in enumerate(cards):
        head, _, rest = c.partition(" ")
        out.append((f"{int(head):02d}", rest) if head.isdigit() else (f"{k + 1:02d}", c))
    return out


def slide_list(n: dict) -> list[dict]:
    """The talk (about 15 minutes on the main row). title = the finding; at most 3 short bullets; one main dynamic visual.
    extras = optional ↓ sub-slides: ("iframe" | "video" | "image", file name, one-line caption). The main row alone
    tells the story; open the ↓ extras when there is time or someone asks. inwords = the formula said out loud;
    guess=True hides the big number until the next click, so the room can guess first."""
    p = lambda x: f"{x:.0%}"  # noqa: E731
    season = lambda s: f"{s}/{(s + 1) % 100:02d}"  # noqa: E731
    return [
        dict(id=1, kicker=f"Final Project  ·  {COURSE}", title=TITLE,
             byline=AUTHOR, subtitle="Squad value, transfers and results in La Liga, 2014/15 to 2025/26",
             bullets=["Stadium photo: Badefa, CC0, via Wikimedia Commons"]
                     + ([f"Scan to play with the models: {APP_URL}"] if "YOUR-APP" not in APP_URL else []),
             speaker="A", time="0:30", layout="title",
             notes="Real Madrid spend more on players than almost anyone, and they win a lot. So is it just the money? "
                   "That's the question. We'll answer it with the tools from this course."),
        dict(id=2, kicker="Kick-off", title=f"Every match is a run of random chances. Tonight's simulated game ended {n['r0_score']}",
             stat=(f"{n['lam_rm']:.2f}", "goals a game for Madrid, on average"),
             bullets=["Each shot has some chance p of going in", "Count the goals and you get a Poisson again (thinning)"],
             formula=r"N_{90}\sim\text{Pois}(\lambda),\quad \lambda_{RM}=" + f"{n['lam_rm']:.2f}",
             inwords="The number of goals in 90 minutes is Poisson with rate λ.",
             iframe="R0_matchday", speaker="A", time="0:45",
             extras=[("video", "manim_M1PoissonProcess", "Why counting random arrivals gives a Poisson")],
             notes="Hit play: shots come in at random and some of them go in. You can drag to look around. "
                   "Everything else today is about modelling what happens on that pitch."),
        dict(id=3, kicker="Why this question", title=f"Madrid's squad is worth {n['rel_value']:.1f}× the average La Liga squad",
             stat=(f"{n['rel_value']:.1f}×", "the league-average squad value"),
             bullets=["They spend the most and win the most. Is it just the money?", "Squad value and points, 2014/15 to 2025/26"],
             iframe="R23_value_and_points", speaker="B", time="1:00",
             notes="Hit play. Left is squad value, right is league points, Madrid in red. If money were everything, "
                   "the two sides would move together. They don't, not exactly, and that gap is what we look at."),
        dict(id=4, kicker="Game plan", title="Seven questions, one per part",
             cards=["1 Does money predict points? (regression)", "2 How often do signings work out? (Beta-Binomial, logistic)",
                    "3 Real output or a hot streak? (Gamma-Poisson)", "4 When is a player worth the most? (quadratic, bootstrap)",
                    "5 What is one star worth? (Poisson, Monte Carlo)", "6 Do they really score late? (permutation test)",
                    "7 Do the answers hold up? (MCMC, Weibull)"],
             speaker="B", time="0:45",
             notes="Each part is one question and one chunk of the course. We go in this order."),
        dict(id="S1", kicker="First half", title="Money and the transfer market", big="01", layout="divider", photo="trophies",
             credit="European Cups in the Bernabéu trophy room. Photo: Daniel from Calgary, CC BY 2.0, Wikimedia Commons",
             speaker="A", time="0:10", question="Does spending more get you more points? And do the expensive signings actually work out?",
             notes="Parts 1 and 2: Madrid as a buyer."),
        dict(id=5, kicker="Part 1 · regression", title=f"Squad value explains about {p(n['r2'])} of league points",
             stat=(p(n['r2']), "of the variation in points"),
             bullets=[f"20% more squad value is worth about {n['pts_20']:.1f} extra points a season"],
             formula=r"\text{points}=\beta_0+\beta_1\ln(\text{value})+\varepsilon",
             inwords="Points rise by β₁ for each one-unit rise in log squad value.",
             iframe="R1_money_vs_points", speaker="A", time="1:15",
             extras=[("iframe", "R21_leverage", "Leverage: drag Madrid and watch the line move"),
                     ("video", "manim_M6LeastSquares", "Least squares, step by step")],
             notes="Hit play to step through the seasons. The x-axis is log of squad value relative to the league. "
                   "R squared is how much of the spread in points money accounts for. If someone asks, go down: drag "
                   "Madrid's point and the slope moves a lot, because they sit far from the middle of x."),
        dict(id=6, kicker="Part 1 · residuals", title=f"Madrid finished above their budget line in {n['above']} of {n['seasons']} seasons",
             stat=(f"{n['above']}/{n['seasons']}", "seasons above the line"),
             bullets=[f"Best season: {season(n['best_season'])}, {n['best_res']:+.0f} points"],
             formula=r"e_i=y_i-\hat y_i",
             inwords="Residual = the points they got minus the points the line predicted.",
             iframe="R24_residuals", speaker="A", time="0:45",
             extras=[("iframe", "R27_residual_checks", "Is a straight line fair here? Residual checks")],
             notes="Red bars are seasons where Madrid got more than a club with that budget normally would. "
                   "Hover a bar to see actual against predicted."),
        dict(id=7, kicker="Part 2 · Bayes", title=f"About {p(n['post'])} of Madrid's signings become regulars",
             stat=(p(n['post']), "of paid signings become regulars"),
             bullets=[f"Raw count: {n['hits']} of {n['n']}, so {p(n['raw'])}. Bayes pulls it toward the big-club average of {p(n['prior_mean'])}",
                      f"95% credible interval: {p(n['lo'])} to {p(n['hi'])}"],
             formula=r"p\mid\text{data}\sim\text{Beta}(a+h,\;b+n-h)",
             inwords="Start from the prior, then add the hits and the misses.",
             iframe="R25_hit_rates", speaker="B", time="1:15",
             extras=[("iframe", "R20_bayes_tree", "Bayes' rule for one player: drag the prior"),
                     ("iframe", "R19_coverage", f"100 repeats: Wald covers {p(n['cov_wald'])}, credible {p(n['cov_cred'])}")],
             notes="A 'regular' means he played at least 40 percent of league minutes within two seasons. Hit play and "
                   "every club slides from its raw rate toward the average. Small samples move the most."),
        dict(id=8, kicker="Part 2 · learning", title="The estimate gets sharper with every signing",
             bullets=["Each row adds one more signing", "Blue line: the posterior mean"],
             formula=r"E[p\mid h]=\tfrac{a+h}{a+b+n}",
             inwords="Prior hits plus real hits, over prior total plus signings.",
             iframe="R5_bayes_learning_3d", speaker="B", time="0:45",
             extras=[("video", "manim_M2BetaUpdate", "Each signing moves the Beta curve"),
                     ("video", "anim_posterior", "The posterior, one signing at a time")],
             notes="Rotate it. The curve starts wide, that's the prior, and gets narrower as signings come in. "
                   "A hit nudges it right, a miss nudges it left."),
        dict(id=9, kicker="Part 2 · logistic regression", title=f"Given a hit and a flop, age and fee pick the hit {p(n['auc'])} of the time",
             stat=(p(n['auc']), "AUC"),
             bullets=["Put in an age and a fee, get the chance he becomes a regular", "a = age − 24"],
             formula=r"\text{logit}\,p=\beta_0+\beta_1a+\beta_2a^2+\beta_3\ln\text{fee}+\dots",
             inwords="The log-odds of becoming a regular, built from age, age squared and log fee.",
             iframe="R14_sporting_director", speaker="B", time="1:00",
             notes="Ask someone in the room for an age and a fee and type it in. The band is the 95% interval. "
                   "The ROC curve on the right shows how well the model tells hits from flops."),
        dict(id="S2", kicker="Second half", title="Players and matches", big="02", layout="divider", photo="distefano",
             credit="Alfredo Di Stéfano and José Santamaría, 1962. Photo: Joop van Bilsen / Anefo, Nationaal Archief, CC0",
             speaker="B", time="0:10", question="Who's genuinely good, when do players peak, and how much is one star worth?",
             notes="Parts 3 to 5: from the squad down to single matches."),
        dict(id=10, kicker="Part 3 · Gamma-Poisson", title="Who's actually good, and who just had a hot streak?",
             stat=(f"{n['top_post']:.2f}", f"{n['top_name']}, goals + assists per 90"),
             bullets=[f"His 95% credible interval: {n['top_lo']:.2f} to {n['top_hi']:.2f}",
                      f"{n['lucky_name']} drops from {n['lucky_raw']:.2f} to {n['lucky_post']:.2f} once we allow for his few minutes"],
             formula=r"\lambda\mid k\sim\text{Gamma}(\alpha+k,\;\beta+t)",
             inwords="Add his goals + assists to α and his minutes (in 90s) to β.",
             iframe="R3_shrinkage_bubbles", speaker="B", time="1:15",
             extras=[("iframe", "R2_player_posteriors", "Pick any attacker"),
                     ("iframe", "R11_mle_surface", f"Where the prior comes from: MLE Gamma({n['mle_a']:.2f}, {n['mle_b']:.2f})"),
                     ("video", "manim_M3Shrinkage", "Why small samples get pulled toward the average")],
             notes="Each arrow goes from a player's raw rate to our estimate. Players with only a few games get pulled "
                   "hard toward the average attacker. Players with years of minutes barely move."),
        dict(id="10b", kicker="Part 3 · the squad", title="Meet the attackers", layout="wide",
             bullets=["All stats from Transfermarkt"] + photo_credit(),
             iframe="R22_player_cards", speaker="B", time="0:45",
             notes="Click any card to open his radar and his posterior. Everything on the cards comes from the data."),
        dict(id=11, kicker="Part 4 · quadratic regression", title=f"Players are worth the most at about {n['peak']:.0f}",
             stat=(f"{n['peak']:.1f}", "peak age for attackers"),
             bullets=[f"95% interval: {n['peak_lo']:.1f} to {n['peak_hi']:.1f} (we resampled players, not seasons)"],
             formula=r"\text{age}^*=26-\frac{\beta_1}{2\beta_2}",
             inwords="The top of the parabola.",
             iframe="R8_value_surface", speaker="A", time="1:00",
             extras=[("iframe", "R18_bootstrap", "The bootstrap, one resample at a time")],
             notes="Rotate it. Log value against age is a parabola, so the peak is its vertex. The ridge you see is "
                   "that peak at every level of output."),
        dict(id=12, kicker="Part 5 · Poisson", title="Clásico at the Bernabéu. What are Madrid's chances?",
             stat=(p(n['p_win']), "chance Madrid win"), guess=True,
             bullets=[f"Win {p(n['p_win'])}, draw {p(n['p_draw'])}, lose {p(1 - n['p_win'] - n['p_draw'])}",
                      f"Madrid's goals ~ Pois({n['lam_rm']:.2f}), Barça's ~ Pois({n['lam_bar']:.2f})",
                      "Red bars are Madrid wins, grey draws, blue Barça wins"],
             formula=r"P(H=i,A=j)=\frac{e^{-\lambda_h}\lambda_h^{i}}{i!}\cdot\frac{e^{-\lambda_a}\lambda_a^{j}}{j!}",
             inwords="Two independent Poissons, so we multiply the two chances.",
             iframe="R4_clasico_3d", speaker="A", time="1:00",
             extras=[("iframe", "R16_poisson_splitting", "Given n goals, how many are Madrid's? Binomial"),
                     ("iframe", "R15_competing_exponentials", f"Who scores first? λh/(λh+λa) = {p(n['first_theory'])}"),
                     ("video", "manim_M4Skellam", "Two Poissons, one result")],
             notes="Ask the room first: what's Madrid's chance of beating Barça at home? Then click to reveal. Each bar "
                   "is one scoreline. Add up the red ones and you get the win chance."),
        dict(id=13, kicker="Part 5 · what-if", title=f"What is {n['star']} worth to the title race?",
             stat=(f"{(n['p_without'] - n['p_title']) * 100:+.0f} pts".replace('-', '−'), "title chance without him"), guess=True,
             bullets=[f"Title chance: {p(n['p_title'])} with him, {p(n['p_without'])} without",
                      f"He was involved in {p(n['share'])} of the goals", f"Expected points: {n['pts_with']:.0f} with him, {n['pts_without']:.0f} without"],
             formula=r"\lambda'=\lambda\,[1-s(1-r)]",
             inwords="Remove his share s of the goals, give back r of it for whoever replaces him.",
             iframe="R9_star_points", speaker="A", time="1:15",
             extras=[("video", "manim_M8Thinning", "Thinning: take the star's goals away at random")],
             notes="Ask first: how much is one player worth? Then reveal. We take his share out of Madrid's scoring "
                   "rate, assume his replacement is half as good, and replay the season again and again."),
        dict(id=14, kicker="Part 5 · Monte Carlo", title=f"Why we simulated {n['n_sims']:,} seasons",
             stat=(f"{n['clt_mu']:.0f}", f"expected points a season (SD {n['clt_sd']:.1f})"),
             bullets=["More runs, steadier answer (law of large numbers)", "The error shrinks like 1/√n",
                      f"Season points look like N({n['clt_mu']:.0f}, {n['clt_sd']:.1f}²)"],
             formula=r"\hat p\pm1.96\sqrt{\hat p(1-\hat p)/n}",
             inwords="The 95% margin of error for a probability we got by simulation.",
             iframe="R6_title_race_lln", speaker="A", time="0:45",
             extras=[("iframe", "R13_galton_board", "Galton board: 38 matches add up to a bell"),
                     ("video", "manim_M5CLT", "The central limit theorem, one match at a time"),
                     ("video", "anim_lln", "The running estimate settling down")],
             notes="Hit play. With ten seasons the line jumps all over the place. By ten thousand it's steady."),
        dict(id="S3", kicker="Extra time", title="Is 'Madrid magic' real?", big="03", layout="divider", photo="undecima",
             credit="Sergio Ramos with the Champions League trophy, Puerta del Sol, May 2016. Photo: Cristina Cifuentes, CC BY 2.0",
             speaker="B", time="0:10", question="Do they really score more late in games, or is that just luck? And do our answers hold up?",
             notes="Parts 6 and 7: testing the story."),
        dict(id=15, kicker="Part 6 · permutation test", title=(f"Madrid's late goals aren't just luck ({n['perm_txt']})" if n["perm_p"] < 0.05
                    else f"Madrid's late goals could be luck (p ≈ {n['perm_p']:.2f})"),
             stat=(n['perm_txt'].replace("p ≈ ", "").replace("p ", ""), "p-value"),
             bullets=[f"Goals after 75': {p(n['late_rm'])} for Madrid, {p(n['late_oth'])} for the other big clubs",
                      f"We shuffled the club labels {n['perm_B']:,} times", "Red line: what actually happened"],
             formula=r"p=\frac{\text{shuffles with }|d^*|\ge|d|}{B}",
             inwords="The share of shuffles with a gap at least as big as the real one.",
             iframe="R12_permutation_test", speaker="B", time="1:00",
             extras=[("iframe", "R26_goal_timing", "Goals by 15-minute block, and the χ² test")],
             notes="If the club didn't matter, shuffling the labels would give gaps like ours all the time. Hit play: "
                   "the grey pile builds up and almost never reaches the red line. Stoppage time pads the last block "
                   "for everyone, which is why we compare with other big clubs and not with zero."),
        dict(id=16, kicker="Part 7 · robustness", title="We tried harder methods. Same answers",
             stat=(p(n['mcmc_acc']), "of MCMC proposals accepted"),
             bullets=["A random-walk sampler (MCMC) gives the same posterior",
                      f"Weibull shape {n['weib_k']:.2f}: the chance of a goal creeps up as the game goes on"],
             formula=r"r=\frac{\text{posterior}(\theta^*)}{\text{posterior}(\theta)},\quad P(\text{accept})=\min(1,r)",
             inwords="Always move to a better spot; move to a worse one with probability r.",
             iframe="R10_mcmc_walk", speaker="B", time="1:00",
             extras=[("iframe", "R10b_mcmc_step_size", "Step size: too small crawls, too big gets stuck"),
                     ("video", "manim_M7MCMC", "Metropolis: propose, accept or reject, repeat"),
                     ("iframe", "R17_weibull_hazard", "Weibull hazard: drag the shape k")],
             notes="Hit play. The walker's histogram fills in the exact curve. The Weibull lets the scoring rate "
                   "change during the match instead of staying flat."),
        dict(id=17, kicker="Live", title="Clásico, minute by minute", layout="wide",
             bullets=["Goals arrive as a Poisson process", "Win chance comes from a Skellam over the minutes left"],
             formula=r"P(\text{win})=P(K>-d),\;K\sim\text{Skellam}\!\left(\lambda_h\tfrac{m}{90},\lambda_a\tfrac{m}{90}\right)",
             iframe="R7_live_clasico", speaker="A + B", time="1:30",
             notes="Hit play and talk over it: the bands are win, draw and loss given the score so far. After a goal "
                   "the chances jump; between goals they drift toward whoever is ahead."),
        dict(id=18, kicker="Full time", title="Money helps a lot. It isn't the whole story", bg="decima",
             credit="La Décima celebrations, Madrid, May 2014. Photo: Presidencia de la Comunidad de Madrid, CC BY 3.0",
             cards=[f"Money explains about {p(n['r2'])} of points", f"Madrid beat their budget line in {n['above']} of {n['seasons']} seasons",
                    f"One star is worth about {abs(n['p_title'] - n['p_without']) * 100:.0f} points of title chance",
                    f"The late goals aren't luck: {n['perm_txt']}",
                    "All of this is association, not causation"],
             speaker="B", time="1:00",
             notes="Wrap up: what we can say, and what we can't. We never claim money causes the wins."),
        dict(id="Q", kicker="Thanks for listening", title="Questions?", layout="title", byline=AUTHOR,
             subtitle="Every chart in this deck is live. Ask about any of them and we'll pull it up.",
             bullets=["Stadium photo: Badefa, CC0, via Wikimedia Commons"], speaker="A + B", time="",
             notes="Open whichever slide the question is about. The extras below each slide are there for this."),
        dict(id="B1", kicker="Backup", title="Peak market value by position", iframe="R28_peak_by_position",
             bullets=["Same quadratic model for every position", "Intervals from resampling players"]),
    ]


# ------------------------------------------------------------------ HTML (reveal.js)
from src.style import DECK as D  # noqa: E402

def _pitch_svg():
    """Faint pitch markings behind every slide: the deck lives inside the stadium. No logos, no crests."""
    from urllib.parse import quote
    c, o = "#e8e8e8", 0.045 if D["theme"] == "terminal" else 0.07
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" preserveAspectRatio="none">'
           f'<g fill="none" stroke="{c}" stroke-opacity="{o}" stroke-width="2">'
           '<rect x="40" y="60" width="1520" height="800"/><line x1="800" y1="60" x2="800" y2="860"/>'
           '<circle cx="800" cy="460" r="120"/><circle cx="800" cy="460" r="4" fill="#e8e8e8"/>'
           '<rect x="40" y="250" width="230" height="420"/><rect x="1330" y="250" width="230" height="420"/>'
           '<rect x="40" y="370" width="80" height="180"/><rect x="1480" y="370" width="80" height="180"/>'
           '<path d="M270 400 A110 110 0 0 1 270 520"/><path d="M1330 400 A110 110 0 0 0 1330 520"/></g>'
           '<g fill="#e8e8e8" fill-opacity="0.012">' + "".join(
               f'<rect x="{40 + 152 * i}" y="60" width="76" height="800"/>' for i in range(10)) + '</g></svg>')
    return "data:image/svg+xml;utf8," + quote(svg)


PITCH = _pitch_svg()
FONT_SANS = ", ".join(f"'{f}'" if " " in f else f for f in D["sans"])
FONT_MONO = ", ".join(f"'{f}'" if " " in f else f for f in D["mono"])
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700'
         '&family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet">')   # offline: Georgia / Segoe UI
FONT_SERIF = ", ".join(f"'{f}'" if " " in f else f for f in D["serif"])
CSS = f"""
:root {{ --r-background-color: {D['bg']}; --r-main-font: {FONT_SANS}; --r-heading-font: {FONT_SERIF};
        --r-main-color: {D['ink']}; --r-heading-color: {D['ink']}; --r-link-color: {D['accent']}; --r-main-font-size: 30px; }}
.reveal-viewport {{ background: {D['bg']} url("assets/bernabeu_soft.jpg") center bottom / cover no-repeat; }}
.reveal .slides section {{ text-align: left; padding-top: 40px; box-sizing: border-box; }}
.reveal h1, .reveal h2 {{ font-family: {FONT_SERIF}; text-transform: none; letter-spacing: -0.012em; font-weight: 650; color: {D['ink']}; }}
.reveal h2 {{ font-size: 1.55em; line-height: 1.12; margin: 0.2em 0 0.55em; max-width: 88%; }}
.reveal .kicker {{ font-family: {FONT_SANS}; font-weight: 700; font-size: 0.6em; letter-spacing: 0.08em; text-transform: uppercase;
                  color: {D['accent']}; margin: 0; }}
.reveal .kicker .sep {{ color: {D['muted']}; font-weight: 400; margin: 0 0.5em; }}
.reveal .row {{ display: flex; gap: 40px; align-items: flex-start; }}
.reveal .col-text {{ flex: 0 0 32%; text-align: left; }}
.reveal .col-vis {{ flex: 1; }}
.reveal ul {{ list-style: none; margin: 0.2em 0 0 0; padding: 0; font-size: 0.86em; color: {D['ink2']}; }}
.reveal li {{ margin: 0; padding: 0.55em 0; border-top: 1px solid {D['grid']}; line-height: 1.35; font-variant-numeric: tabular-nums; }}
.reveal li:last-child {{ border-bottom: 1px solid {D['grid']}; }}
.reveal .formula {{ margin-top: 1.1em; padding: 14px 0 0; border-top: 2px solid {D['ink']}; font-size: 0.74em; color: {D['ink']}; }}
.reveal iframe.vis {{ width: 100%; height: 640px; border: 0; background: {D['panel']}; box-shadow: 0 1px 0 {D['grid']}, 0 0 0 1px {D['grid']}; }}
.reveal img.vis {{ width: 100%; box-shadow: 0 0 0 1px {D['grid']}; margin: 0 0 10px; }}
.reveal video.vis {{ width: 100%; }}
.reveal .vis-stack img.vis, .reveal .vis-stack video.vis {{ max-height: 660px; width: auto; max-width: 100%; display: block; margin: 0 auto 10px; }}
.reveal .vis-stack.n2 img.vis, .reveal .vis-stack.n2 video.vis {{ max-height: 320px; }}
.reveal .cards {{ display: grid; gap: 0; margin-top: 40px; border-top: 2px solid {D['ink']}; }}
.reveal .card {{ display: flex; flex-direction: column; gap: 12px; min-height: 180px; padding: 22px 26px 22px 0; margin-right: 26px;
                border-bottom: 1px solid {D['grid']}; font-family: {FONT_SANS}; font-size: 0.9em; color: {D['ink']}; text-align: left; }}
.reveal .card .no {{ font-family: {FONT_SERIF}; font-size: 2.1em; font-weight: 600; color: {D['accent']}; line-height: 1; }}
.reveal .card .txt {{ font-weight: 500; line-height: 1.3; }}
.reveal .stat {{ display: flex; flex-direction: column; margin: 0 0 0.8em; }}
.reveal .stat .num {{ font-family: {FONT_SERIF}; font-weight: 600; font-size: 3.2em; line-height: 0.95; color: {D['ink']};
                     font-variant-numeric: lining-nums tabular-nums; letter-spacing: -0.02em; }}
.reveal .stat .num::after {{ content: ""; display: block; width: 56px; height: 4px; background: {D['c1']}; margin-top: 12px; }}
.reveal .stat .lab {{ font-family: {FONT_SANS}; font-size: 0.68em; font-weight: 600; color: {D['ink2']}; margin-top: 10px; max-width: 420px; }}
.reveal .more {{ font-family: {FONT_SANS}; font-size: 0.6em; font-weight: 600; color: {D['ink2']}; margin-top: 1.4em; }}
.reveal .strip {{ display: flex; flex-wrap: wrap; gap: 6px 26px; align-items: baseline; font-size: 0.72em; color: {D['ink2']}; margin: -0.2em 0 0.6em; }}
.reveal .strip span + span::before {{ content: "/"; color: {D['muted']}; margin-right: 26px; }}
.reveal .strip .more {{ margin: 0 0 0 auto; }}
.reveal .strip .more::before {{ content: none; }}
.reveal .wide-vis iframe.vis {{ height: 620px; }}
.reveal .strip span.small {{ font-size: 0.68em; color: {D['muted']}; }}
.reveal .title-slide .title-wrap {{ position: absolute; left: 0; top: 120px; width: 760px; }}
.reveal .title-slide .photo {{ position: absolute; right: -90px; top: -60px; width: 880px; height: 1000px;
                              background: url("assets/bernabeu.jpg") 38% center / cover no-repeat; }}
.reveal .title-slide .photo::after {{ content: ""; position: absolute; inset: 0;
                              background: linear-gradient(90deg, {D['bg']} 0%, rgba(43,33,27,0.6) 22%, rgba(43,33,27,0) 48%); }}
.reveal .title-slide .kicker {{ font-size: 0.66em; }}
.reveal .title-slide h1 {{ font-size: 2.75em; line-height: 1.02; margin: 0.4em 0 0.35em; font-weight: 650; }}
.reveal .title-slide .subtitle {{ font-size: 0.88em; color: {D['ink2']}; max-width: 700px; margin: 0 0 2.2em; font-family: {FONT_SANS}; }}
.reveal .title-slide .byline {{ display: flex; flex-direction: column; gap: 4px; padding-top: 16px; border-top: 2px solid {D['ink']}; width: 640px; }}
.reveal .title-slide .byline .name {{ font-family: {FONT_SERIF}; font-size: 1.05em; font-weight: 600; color: {D['ink']}; }}
.reveal .title-slide .byline .meta {{ font-family: {FONT_SANS}; font-size: 0.68em; color: {D['ink2']}; }}
.reveal .title-slide .title-note {{ font-size: 0.5em; color: {D['ink2']}; margin-top: 2.4em; }}
.reveal .title-slide .goldbar {{ position: absolute; left: 0; top: 60px; width: 120px; height: 8px; background: {D['c1']}; }}
.reveal .divider .big {{ position: absolute; left: 0; top: 150px; font-family: {FONT_SERIF}; font-size: 7.5em; font-weight: 600; line-height: 1;
                        color: {D['c1']}; letter-spacing: -0.03em; }}
.reveal .divider .label {{ position: absolute; left: 0; top: 400px; font-family: {FONT_SANS}; font-weight: 700; font-size: 0.62em;
                          letter-spacing: 0.09em; text-transform: uppercase; color: {D['c1']}; }}
.reveal .divider h2 {{ position: absolute; left: 0; top: 440px; font-size: 1.9em; max-width: 760px; color: #ffffff; }}
.reveal .divider .q {{ position: absolute; left: 0; top: 590px; font-size: 0.92em; color: #f1ede4; max-width: 720px; }}
.reveal .katex {{ color: {D['ink']}; }}
.reveal .inwords {{ font-family: {FONT_SANS}; font-size: 0.82em; color: {D['ink2']}; margin: 10px 0 0; font-style: italic; }}
.reveal .stat.guess .q {{ font-family: {FONT_SERIF}; font-size: 2.1em; font-style: italic; color: {D['muted']}; line-height: 1.1; }}
.reveal .stat.guess:has(.num.visible) .q {{ display: none; }}
.reveal .stat.guess .num.fragment:not(.visible) {{ display: none; }}
.reveal .slides section ul li.fragment.visible:not(.current-fragment) {{ opacity: 0.72; }}
.reveal .slides section h2::after {{ content: ""; display: block; width: 0; height: 3px; background: {D['c1']}; margin-top: 12px; }}
.reveal .slides section.present > h2::after {{ animation: underline 0.9s 0.2s ease-out forwards; }}
@keyframes underline {{ to {{ width: 90px; }} }}
.reveal .divider h2::after, .reveal .title-slide h1::after {{ display: none; }}
.reveal video.vis, .reveal video.vis-extra {{ background: #000; box-shadow: 0 0 0 1px {D['grid']}; }}
.reveal .divider .dphoto {{ position: absolute; right: -90px; top: -60px; width: 800px; height: 1010px; background-size: cover;
                           background-position: center; animation: kenburns 26s ease-in-out infinite alternate; }}
.reveal .divider .dphoto::after {{ content: ""; position: absolute; inset: 0;
                           background: linear-gradient(90deg, #1c1511 0%, rgba(28,21,17,0.55) 18%, rgba(28,21,17,0) 45%); }}
.reveal .credit {{ position: absolute; right: 0; top: 815px; font-size: 0.44em; color: #c9c3b5; max-width: 700px; text-align: right; }}
.reveal section.has-bg .cards {{ background: rgba(43,33,27,0.9); padding: 0 0 6px 26px; }}
.reveal .credit.dark {{ color: {D['ink2']}; }}
.reveal .title-slide .photo {{ animation: kenburns 30s ease-in-out infinite alternate; }}
@keyframes kenburns {{ from {{ transform: scale(1) translate(0, 0); }} to {{ transform: scale(1.07) translate(-1.5%, -1%); }} }}
.reveal .slides section .fragment.fade-up {{ transform: translate(0, 24px); }}
.reveal .slides section .fragment.fade-up.visible {{ transform: none; }}
.topbar {{ position: fixed; top: 0; left: 0; right: 0; height: 36px; display: flex; align-items: center; gap: 16px; padding: 0 30px;
          font: 600 17px {FONT_SANS}; letter-spacing: .04em; color: {D['ink2']}; z-index: 50; background: {D['bg']};
          border-bottom: 1px solid {D['grid']}; }}
.topbar .part {{ color: {D['ink']}; text-transform: uppercase; letter-spacing: .09em; }}
.topbar .chap {{ display: flex; gap: 4px; }}
.topbar .chap i {{ display: block; width: 22px; height: 4px; background: {D['grid']}; }}
.topbar .chap i.on {{ background: {D['c1']}; }}
.topbar .right {{ margin-left: auto; }}
.footbar {{ position: fixed; bottom: 0; left: 0; right: 0; height: 32px; display: flex; align-items: center; gap: 14px; padding: 0 30px;
           font: 500 17px {FONT_SANS}; color: {D['ink2']}; z-index: 50; }}
.footbar .who {{ color: {D['ink']}; font-weight: 600; }}
.footbar .clock {{ margin-left: auto; font-variant-numeric: tabular-nums; }}
.footbar .page {{ color: {D['ink']}; font-weight: 600; min-width: 56px; text-align: right; font-variant-numeric: tabular-nums; }}
.reveal h2.sub {{ font-size: 1.05em; color: {D['ink2']}; }}
.reveal .sub-tag {{ color: {D['muted']}; }}
.fake {{ position: fixed; top: 44px; right: 30px; background: #b3362b; color: white; font: 700 13px {FONT_SANS};
        padding: 3px 9px; z-index: 99; letter-spacing: .08em; }}
.reveal .vis-extra {{ width: 100%; height: 700px; border: 0; background: {D['panel']}; box-shadow: 0 0 0 1px {D['grid']}; }}
.reveal video.vis-extra, .reveal img.vis-extra {{ height: auto; max-height: 700px; width: auto; max-width: 100%; display: block; margin: 0 auto; }}
.reveal .progress {{ color: {D['c1']}; height: 3px; }}
.reveal .controls {{ color: {D['ink2']}; }}
"""


def _vis_html(s):
    parts = []
    if s.get("iframe"):
        parts.append(f'<iframe class="vis" data-src="assets/{s["iframe"]}.html" data-png="assets/{s["iframe"]}.png"></iframe>')
    if not s.get("iframe"):                     # with an iframe, static images move to the ↓ sub-slides
        for im in s.get("images", []):
            parts.append(f'<img class="vis" src="assets/{im}.png" alt="{im}">')
    if s.get("video") and s.get("layout") != "title":
        parts.append(f'<video class="vis" controls preload="metadata" muted playsinline src="assets/{s["video"]}.mp4"></video>')
    n = len(parts)
    return f'<div class="vis-stack n{n}">' + "\n".join(parts) + "</div>"


def kick(k):
    """Kicker text with a quiet slash between its parts instead of a middle dot."""
    return '<span class="sep">/</span>'.join(html.escape(x.strip()) for x in k.split("·"))


def _part(s):
    k = s.get("kicker", "")
    return k.split("·")[0].strip().upper() if "·" in k else k.upper()


def _extra_html(kind, name, caption, s, banner, k, total):
    head = (f'<section data-part="{html.escape(_part(s))}">{banner}<p class="kicker">{html.escape(s["kicker"])}'
            f' <span class="sub-tag">· ↓ {k}/{total}</span></p><h2 class="sub">{html.escape(caption)}</h2>')
    if kind == "iframe":
        body = f'<iframe class="vis-extra" data-src="assets/{name}.html" data-png="assets/{name}.png"></iframe>'
    elif kind == "video" and not (C.FIG / f"{name}.mp4").exists():   # no ffmpeg -> GIF only: show it animated
        body = f'<img class="vis-extra" src="assets/{name}.gif" alt="{html.escape(caption)}">'
    elif kind == "video":
        poster = f"assets/{name}.png" if (C.FIG / f"{name}.png").exists() else f"assets/{name}.gif"
        body = (f'<video class="vis-extra" controls preload="metadata" muted playsinline poster="{poster}" data-png="{poster}" '
                f'src="assets/{name}.mp4"></video>')
    else:
        body = f'<img class="vis-extra" src="assets/{name}.png" alt="{html.escape(caption)}">'
    return head + body + "</section>"


def extras_of(s):
    """Optional ↓ sub-slides. Old keys (extra_iframe, extra_video, images next to an iframe) still work."""
    ex = list(s.get("extras", []))
    if s.get("extra_iframe"):
        ex.append(("iframe", s["extra_iframe"], s["title"]))
    if s.get("extra_video"):
        ex.append(("video", s["extra_video"], s["title"]))
    if s.get("iframe") and s.get("images"):
        ex += [("image", im, s["title"]) for im in s["images"]]
    have = {"iframe": lambda m: (C.FIG / "interactive" / f"{m}.html").exists(),
            "video": lambda m: (C.FIG / f"{m}.mp4").exists() or (C.FIG / f"{m}.gif").exists(),
            "image": lambda m: (C.FIG / f"{m}.png").exists()}
    return [e for e in ex if have[e[0]](e[1])]            # e.g. no Manim installed: those ↓ slides are skipped


def slide_html(s, fake):
    banner = '<div class="fake">FAKE DATA</div>' if fake else ""
    extra_notes = " ".join(f"[{h}]" for h in hints(s))
    notes = (f'<aside class="notes">{html.escape(s.get("notes", ""))} {html.escape(extra_notes)} '
             f'({s.get("speaker", "")}, {s.get("time", "")})</aside>')
    kicker = f'<p class="kicker">{kick(s["kicker"])}</p>'
    part = f' data-part="{html.escape(_part(s))}"'
    if s.get("layout") == "title":
        bg = "" if not s.get("video") else (f'data-background-video="assets/{s["video"]}.mp4" data-background-video-loop data-background-video-muted'
              if (C.FIG / f'{s["video"]}.mp4').exists()            # no ffmpeg -> no mp4: the GIF plays instead
              else f'data-background-image="assets/{s["video"]}.gif" data-background-size="contain"')
        return (f'<section class="title-slide"{part} {bg} data-background-opacity="0.22" data-background-color="{D["bg"]}">'
                f'{banner}<div class="photo"></div><div class="goldbar"></div><div class="title-wrap"><p class="kicker">{kick(s["kicker"])}</p>'
                f'<h1>{html.escape(s["title"])}</h1><p class="subtitle">{html.escape(s.get("subtitle", ""))}</p>'
                f'<div class="byline"><span class="name">{html.escape(s.get("byline", ""))}</span>'
                f'<span class="meta">{html.escape(COURSE)}, Final Project</span></div>'
                + "".join(f'<p class="title-note">{html.escape(b)}</p>' for b in s.get("bullets", []))
                + "</div>" + notes + "</section>")
    if s.get("layout") == "divider":
        return (f'<section class="divider"{part} data-background-color="#1c1511">{banner}'
                f'<div class="dphoto" style="background-image: url(\'assets/{s.get("photo", "bernabeu")}.jpg\')"></div>'
                f'<p class="credit">{html.escape(s.get("credit", ""))}</p><div class="big">{html.escape(s["big"])}</div>'
                f'<p class="label">{html.escape(s["kicker"])}</p><h2>{html.escape(s["title"])}</h2>'
                f'<p class="q">{html.escape(s.get("question", ""))}</p>{notes}</section>')
    bullets = "".join(f'<li class="fragment fade-up">{html.escape(b)}</li>' for b in shown(s))
    stat = ""
    if s.get("stat"):
        val, lab = s["stat"]
        if s.get("guess"):
            stat = (f'<div class="stat guess"><span class="q">Your guess?</span><span class="num fragment" data-final="{html.escape(val)}">'
                    f'{html.escape(val)}</span><span class="lab">{html.escape(lab)}</span></div>')
        else:
            stat = f'<div class="stat"><span class="num" data-final="{html.escape(val)}">{html.escape(val)}</span><span class="lab">{html.escape(lab)}</span></div>'
    words = f'<p class="inwords">{html.escape(s["inwords"])}</p>' if s.get("inwords") else ""
    formula = f'<div class="formula fragment fade-up">\\[{s["formula"]}\\]{words}</div>' if s.get("formula") else ""
    ex = extras_of(s)
    hint = f'<p class="more">↓ {len(ex)} more {"view" if len(ex) == 1 else "views"}</p>' if ex else ""
    if s.get("cards"):
        cards = "".join(f'<div class="card fragment fade-up"><span class="no">{no}</span><span class="txt">{html.escape(c)}</span></div>'
                        for no, c in numbered(s["cards"]))
        cols = 4 if len(s['cards']) in (4, 7, 8) else 3
        body = f'<div class="cards" style="grid-template-columns: repeat({cols}, 1fr)">{cards}</div>'
    elif s.get("layout") == "wide":
        strip = "".join(f'<span class="small">{html.escape(b)}</span>' if b.startswith("Photos:") else f'<span>{html.escape(b)}</span>'
                        for b in shown(s))
        body = f'<div class="strip">{strip}{hint}</div><div class="wide-vis">{_vis_html(s)}</div>'
    else:
        body = (f'<div class="row"><div class="col-text">{stat}<ul>{bullets}</ul>{formula}{hint}</div>'
                f'<div class="col-vis">{_vis_html(s)}</div></div>')
    bgattr, credit = "", ""
    if s.get("bg"):
        bgattr = (f' data-background-image="assets/{s["bg"]}.jpg" data-background-size="cover" data-background-opacity="0.08"'
                  f' data-background-color="{D["bg"]}"')
        credit = f'<p class="credit dark">{html.escape(s.get("credit", ""))}</p>'
    has_bg = ' class="has-bg"' if bgattr else ""
    main = (f'<section data-auto-animate{part}{bgattr}{has_bg}>{banner}{kicker}<h2>{html.escape(s["title"])}</h2>{body}{credit}{notes}'
            f'</section>')
    if ex:
        return "<section>" + main + "".join(_extra_html(*e, s, banner, k + 1, len(ex)) for k, e in enumerate(ex)) + "</section>"
    return main


def vendor() -> tuple[str, str]:
    """Use a local copy of reveal.js + KaTeX when installed (works with no internet); else the CDN."""
    if (NODE / "reveal.js" / "dist").exists():
        for pkg in ("reveal.js", "katex"):
            dst = DECK / "vendor" / pkg
            if not dst.exists():
                shutil.copytree(NODE / pkg / ("dist" if pkg == "katex" else ""), dst / ("dist" if pkg == "katex" else ""),
                                ignore=shutil.ignore_patterns("node_modules", "test", "examples", "*.map"))
        return "vendor/reveal.js", "local: 'vendor/katex',"
    if (DECK / "vendor" / "reveal.js" / "dist" / "reveal.js").exists() and (DECK / "vendor" / "katex" / "dist" / "katex.min.js").exists():
        return "vendor/reveal.js", "local: 'vendor/katex',"          # copies already in docs/vendor: works with no internet
    return REVEAL_CDN, ""


SHARE_TEXT = ("Seven probability models on twelve seasons of Transfermarkt data, presented as a live, interactive deck. "
              f"{COURSE} final project by {AUTHOR}.")
FAVICON = ("data:image/svg+xml," + __import__("urllib.parse").parse.quote(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="12" fill="#2b211b"/>'
    '<circle cx="32" cy="32" r="18" fill="none" stroke="#e53935" stroke-width="6"/><circle cx="32" cy="32" r="5" fill="#f5ede2"/></svg>'))


def build_html(slides, fake):
    ASSETS.mkdir(parents=True, exist_ok=True)
    REVEAL, katex_opt = vendor()
    if katex_opt:     # local copy: load KaTeX directly (the reveal math plugin does not render reliably from local files)
        MATH_JS = ('<link rel="stylesheet" href="vendor/katex/dist/katex.min.css"><script src="vendor/katex/dist/katex.min.js"></script>'
                   '<script src="vendor/katex/dist/contrib/auto-render.min.js"></script>')
        MATH_PLUGIN = ""
    else:
        MATH_JS, MATH_PLUGIN = f'<script src="{REVEAL}/plugin/math/math.js"></script>', "RevealMath.KaTeX, "
    plotly_js = C.FIG / "interactive" / "plotly.min.js"
    if plotly_js.exists():
        shutil.copy(plotly_js, ASSETS)
    for bg in (C.FIG / "backgrounds").glob("*.jpg"):              # stadium photo (CC0), see figures/backgrounds/
        shutil.copy(bg, ASSETS)
    photos = C.FIG / "interactive" / "players"                # licensed player photos used by R22 (see figures/players/)
    if photos.exists():
        shutil.copytree(photos, ASSETS / "players", dirs_exist_ok=True)
    def cp(name, exts):
        for ext in exts:
            src = (C.FIG / "interactive" / f"{name}.{ext}") if ext == "html" else (C.FIG / f"{name}.{ext}")
            if src.exists():
                shutil.copy(src, ASSETS)
    for s in slides:                                         # copy every file the deck points to
        if s.get("iframe"):
            cp(s["iframe"], ("html", "png"))
        for im in s.get("images", []) + [s.get("png")]:
            if im:
                cp(im, ("png",))
        if s.get("video"):
            cp(s["video"], ("mp4", "gif", "png"))
        for kind, name, _ in extras_of(s):
            cp(name, {"iframe": ("html", "png"), "video": ("mp4", "gif", "png"), "image": ("png",)}[kind])
    sections = "\n".join(slide_html(s, fake) for s in slides)
    STATUS = (f'<div class="topbar"><span class="chap" id="chap"></span><span class="part" id="part"></span>'
              f'<span class="right">{html.escape(REPORT)}</span></div>'
              f'<div class="footbar"><span class="who">{html.escape(AUTHOR)}</span><span>{html.escape(COURSE)}, Final Project</span>'
              f'<span class="clock" id="clock">00:00</span><span class="page" id="page"></span></div>')
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>{TITLE} · {AUTHOR} · {COURSE}</title>
<meta name="description" content="{SHARE_TEXT}">
<meta name="author" content="{AUTHOR}">
<meta property="og:type" content="website">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{SHARE_TEXT}">
<meta property="og:url" content="{SITE_URL}">
<meta property="og:image" content="{SITE_URL}img/social.png">
<meta property="og:image:width" content="1280"><meta property="og:image:height" content="640">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{SITE_URL}img/social.png">
<link rel="icon" href="{FAVICON}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="{REVEAL}/dist/reveal.css"><link rel="stylesheet" href="{REVEAL}/dist/theme/{'white' if D['theme'] == 'paper' else 'black'}.css">
<style>{CSS}</style></head><body>{STATUS}<div class="reveal"><div class="slides">
{sections}
</div></div>
<script src="{REVEAL}/dist/reveal.js"></script><script src="{REVEAL}/plugin/notes/notes.js"></script>
{MATH_JS}<script src="{REVEAL}/plugin/zoom/zoom.js"></script>
<script>Reveal.initialize({{ hash: true, width: 1600, height: 900, margin: 0.05, center: false, transition: 'fade',
  backgroundTransition: 'fade', autoAnimate: true, slideNumber: false, controlsTutorial: false, pdfSeparateFragments: false,
  katex: {{ {katex_opt} delimiters: [ {{left: '\\\\[', right: '\\\\]', display: true}},
                               {{left: '\\\\(', right: '\\\\)', display: false}} ] }},
  plugins: [ RevealNotes, {MATH_PLUGIN}RevealZoom ] }});
// formulas: with the local KaTeX copy we render them ourselves once the deck is ready (no internet needed)
if (window.renderMathInElement) Reveal.on('ready', () => renderMathInElement(Reveal.getSlidesElement(), {{
  delimiters: [ {{left: '\\\\[', right: '\\\\]', display: true}}, {{left: '\\\\(', right: '\\\\)', display: false}} ] }}));
// footer: section name, presenter clock (press K to restart it), page number
(function () {{
  let t0 = Date.now();
  const clock = document.getElementById('clock'), page = document.getElementById('page'), part = document.getElementById('part');
  if (!clock) return;
  setInterval(() => {{ const s = Math.floor((Date.now() - t0) / 1000);
    clock.textContent = String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0'); }}, 500);
  function upd() {{ const sl = Reveal.getCurrentSlide(), total = Reveal.getHorizontalSlides().length;
    part.textContent = (sl && sl.dataset.part) || '';
    page.textContent = String(Reveal.getIndices().h + 1).padStart(2, '0') + ' / ' + String(total).padStart(2, '0');
    const m = ((sl && sl.dataset.part) || '').match(/PART (\\d)/), chap = document.getElementById('chap');
    chap.innerHTML = m ? [1, 2, 3, 4, 5, 6, 7].map(k => '<i class="' + (k <= +m[1] ? 'on' : '') + '"></i>').join('') : '';
    const t = sl && (sl.classList.contains('title-slide') || sl.classList.contains('divider'));
    document.querySelector('.topbar').style.visibility = t ? 'hidden' : 'visible'; }}
  function countUp() {{
    if (/print-pdf/.test(location.search)) return;                  // the PDF keeps the final numbers
    const sl = Reveal.getCurrentSlide(); if (!sl) return;
    sl.querySelectorAll('.stat .num:not(.fragment)').forEach(countEl);
  }}
  function countEl(el) {{
      const fin = el.dataset.final, m = fin.match(/^([^0-9+\\-]*)([+\\-]?)([0-9]*\\.?[0-9]+)(.*)$/);
      if (!m) return;
      const target = parseFloat(m[3]), dec = (m[3].split('.')[1] || '').length, t0 = performance.now();
      function step(t) {{ const k = Math.min(1, (t - t0) / 900), e = 1 - Math.pow(1 - k, 3);
        el.textContent = m[1] + m[2] + (target * e).toFixed(dec) + m[4];
        if (k < 1) requestAnimationFrame(step); else el.textContent = fin; }}
      requestAnimationFrame(step);
  }}
  Reveal.on('fragmentshown', e => {{ if (!/print-pdf/.test(location.search))
    e.fragments.forEach(f => {{ if (f.classList.contains('num')) countEl(f); }}); }});
  // videos: V play/pause, [ back 5 s, ] forward 5 s (or use the controls under the video)
  function vid() {{ const sl = Reveal.getCurrentSlide(); return sl && sl.querySelector('video'); }}
  document.addEventListener('keydown', e => {{
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    const v = vid(); if (!v) return;
    if (e.key === 'v' || e.key === 'V' || e.code === 'KeyV') {{ v.paused ? v.play() : v.pause(); }}
    else if (e.key === ']' || e.code === 'BracketRight') {{ v.currentTime = Math.min(v.duration || 0, v.currentTime + 5); }}
    else if (e.key === '[' || e.code === 'BracketLeft') {{ v.currentTime = Math.max(0, v.currentTime - 5); }}
    else return;
    e.preventDefault(); e.stopImmediatePropagation();
  }}, true);
  Reveal.on('slidechanged', e => {{ document.querySelectorAll('video').forEach(v => {{ if (!e.currentSlide.contains(v)) v.pause(); }}); }});
  Reveal.on('ready', () => {{ upd(); countUp(); }}); Reveal.on('slidechanged', () => {{ upd(); countUp(); }});
  Reveal.addKeyBinding({{keyCode: 75, key: 'K', description: 'Restart the presenter clock'}}, () => {{ t0 = Date.now(); }});
}})();</script></body></html>"""
    out = DECK / "index.html"
    out.write_text(doc)
    return out


# ------------------------------------------------------------------ PowerPoint (python-pptx)
def build_pptx(slides, fake):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    def rgb(h):
        return RGBColor.from_string(h.lstrip("#"))

    prs = Presentation()
    prs.core_properties.author = AUTHOR
    prs.core_properties.last_modified_by = AUTHOR
    prs.core_properties.title = TITLE
    prs.core_properties.subject = COURSE + ", Final Project"
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    def text(slide, x, y, w, h, s, size, bold=False, color=None):
        color = color or D["ink"]
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        lines = s if isinstance(s, list) else [s]
        for k, line in enumerate(lines):
            para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            para.text = line
            para.font.size, para.font.bold, para.font.color.rgb = Pt(size), bold, rgb(color)
            para.font.name = "Times New Roman"
            para.space_after = Pt(8)
        return tb

    def picture(slide, name, x, y, w=None, h=None, ext="png"):
        p = C.FIG / f"{name}.{ext}"
        if p.exists():
            slide.shapes.add_picture(str(p), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)

    for s in slides:
        sl = prs.slides.add_slide(blank)
        sl.background.fill.solid()
        sl.background.fill.fore_color.rgb = rgb(D["bg"])
        if s.get("layout") == "divider":
            text(sl, 0.7, 1.2, 6.0, 1.8, s["big"], 96, True, D["c1"])
            text(sl, 0.7, 3.4, 9.0, 0.4, s["kicker"].upper(), 14, True, D["accent"])
            text(sl, 0.7, 3.9, 11.5, 1.2, s["title"], 40, True, D["ink"])
            text(sl, 0.7, 5.2, 11.0, 1.0, s.get("question", ""), 20, False, D["ink2"])
            sl.notes_slide.notes_text_frame.text = s.get("notes", "")
            continue
        if s.get("layout") == "title":
            if s.get("video"):
                picture(sl, s["video"], 6.9, 1.4, w=6.0, ext="gif")         # GIFs loop in PowerPoint
            stadium = C.FIG / "backgrounds" / "bernabeu.jpg"
            if stadium.exists():
                sl.shapes.add_picture(str(stadium), Inches(7.1), Inches(1.6), width=Inches(6.0))
            text(sl, 0.7, 1.6, 6.0, 0.4, s["kicker"].upper(), 13, False, D["accent"])
            text(sl, 0.7, 2.2, 6.2, 2.0, s["title"], 44, True, D["ink"])
            text(sl, 0.7, 4.4, 9.0, 1.5, [s.get("subtitle", ""), "", s.get("byline", ""), COURSE + ", Final Project"] + s.get("bullets", []), 18, False, D["ink2"])
        else:
            text(sl, 0.6, 0.35, 12, 0.4, s["kicker"].upper(), 12, False, D["accent"])
            text(sl, 0.6, 0.65, 12.1, 1.2, s["title"], 30, True, D["ink"])
            if s.get("cards"):
                for k, c in enumerate(s["cards"]):
                    col, row = k % 4, k // 4
                    box = sl.shapes.add_shape(1, Inches(0.6 + col * 3.1), Inches(2.2 + row * 1.9), Inches(2.9), Inches(1.6))
                    box.fill.solid(); box.fill.fore_color.rgb = rgb(D["panel"]); box.line.color.rgb = rgb([LLY, NVO, SPX][k % 3])
                    box.text_frame.text = c
                    box.text_frame.word_wrap = True
                    for para in box.text_frame.paragraphs:
                        para.font.size, para.font.bold, para.font.color.rgb = Pt(17), True, rgb(D["ink"])
            else:
                if s.get("stat"):
                    text(sl, 0.6, 1.75, 4.1, 0.9, s["stat"][0], 40, True, D["accent"])
                    text(sl, 0.6, 2.55, 4.1, 0.5, s["stat"][1].upper(), 11, False, D["muted"])
                text(sl, 0.6, 3.1 if s.get("stat") else 1.9, 4.1, 2.2, shown(s), 18, False, D["ink2"])
                if s.get("formula"):
                    text(sl, 0.6, 5.2, 4.1, 1.4, [latex_to_text(s["formula"]), s.get("inwords", "")], 15, False, D["accent"])
                main_png = s.get("png") or s.get("iframe")
                vis = ([main_png] if main_png else []) + ([] if s.get("iframe") else s.get("images", []))
                vis = vis[:2] + ([s["video"]] if s.get("video") else [])
                if len(vis) == 1:
                    picture(sl, vis[0], 4.9, 1.85, w=8.1, ext="gif" if vis[0] == s.get("video") else "png")
                elif vis:
                    for k, v in enumerate(vis[:2]):
                        picture(sl, v, 4.9, 1.85 + k * 2.75, h=2.6, ext="gif" if v == s.get("video") else "png")
        if fake:
            text(sl, 10.9, 0.15, 2.3, 0.4, "FAKE DATA", 14, True, "#c62828")
        sl.notes_slide.notes_text_frame.text = f"{s.get('notes', '')}  [{s.get('speaker', '')}, {s.get('time', '')}]"
        ex = extras_of(s)
        for k, (kind, name, caption) in enumerate(ex):            # the ↓ sub-slides become ordinary slides
            xs = prs.slides.add_slide(blank)
            xs.background.fill.solid()
            xs.background.fill.fore_color.rgb = rgb(D["bg"])
            text(xs, 0.6, 0.35, 12, 0.4, f"{s['kicker'].upper()}  ·  EXTRA {k + 1}/{len(ex)}", 12, False, D["accent"])
            text(xs, 0.6, 0.65, 12.1, 0.8, caption, 24, True, D["ink2"])
            mp4, poster = C.FIG / f"{name}.mp4", C.FIG / f"{name}.png"
            if kind == "video" and mp4.exists() and poster.exists():
                xs.shapes.add_movie(str(mp4), Inches(1.4), Inches(1.5), Inches(10.5), Inches(5.9),
                                    poster_frame_image=str(poster), mime_type="video/mp4")
            elif kind == "video":
                picture(xs, name, 1.4, 1.5, h=5.8, ext="gif")
            else:
                picture(xs, name, 1.4, 1.5, h=5.8)
            if fake:
                text(xs, 10.9, 0.15, 2.3, 0.4, "FAKE DATA", 14, True, "#c62828")
    out = DECK / "deck.pptx"
    prs.save(out)
    return out


def latex_to_text(tex: str) -> str:
    """PowerPoint has no LaTeX: turn a course formula into readable plain text (Unicode symbols)."""
    import re as _re
    sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
    sym = {r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\lambda": "λ", r"\mu": "μ", r"\sigma": "σ", r"\rho": "ρ",
           r"\varepsilon": "ε", r"\epsilon": "ε", r"\chi": "χ", r"\sum": "Σ", r"\cdot": "·", r"\pm": "±", r"\mid": "|",
           r"\sim": "~", r"\times": "×", r"\infty": "∞", r"\le": "≤", r"\ge": "≥", r"\qquad": "     ", r"\quad": "   ",
           r"\,": " ", r"\;": " ", r"\!": "", r"\left": "", r"\right": "", r"\bar R": "R̄", r"\hat p": "p̂", r"\hat y": "ŷ",
           r"\theta": "θ", r"^*": "*"}
    s = tex
    for _ in range(6):                                        # innermost braces first, repeated for nesting
        s = _re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", s)
        s = _re.sub(r"\^\{([^{}]*)\}", lambda m: m.group(1).translate(sup) if _re.fullmatch(r"[0-9+\-]+", m.group(1))
                    else "^(" + m.group(1) + ")", s)
        s = _re.sub(r"_\{([^{}]*)\}", r"_\1", s)
        s = _re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
        s = _re.sub(r"\\[td]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    for k in sorted(sym, key=len, reverse=True):
        s = s.replace(k, sym[k])
    s = _re.sub(r"\^([0-9])", lambda m: m.group(1).translate(sup), s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return _re.sub(r"\(([A-Za-z0-9_.]+)\)/\(([A-Za-z0-9_.]+)\)", r"(\1/\2)", s)      # (a)/(b) -> (a/b) for simple terms


def export_pdf(html_path):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1600, "height": 900})
        pg.goto(html_path.as_uri() + "?print-pdf", wait_until="networkidle")
        pg.wait_for_timeout(3000)
        # iframes do not print: swap each interactive chart for its static PNG twin
        pg.evaluate("""() => document.querySelectorAll('iframe[data-png], video[data-png]').forEach(f => {
            const img = document.createElement('img'); img.className = 'vis'; img.src = f.dataset.png;
            img.style.maxHeight = '620px'; img.style.width = 'auto'; f.replaceWith(img); })""")
        pg.wait_for_timeout(2000)
        out = DECK / "deck.pdf"
        pg.pdf(path=str(out), width="1600px", height="900px", print_background=True)
        b.close()
    return out


def run(argv=()):
    DECK.mkdir(exist_ok=True)
    fake = is_fake()
    slides = slide_list(numbers())
    made = [build_html(slides, fake), build_pptx(slides, fake)]
    if "--pdf" in argv:
        made.append(export_pdf(made[0]))
    for m in made:
        print("  wrote", m.relative_to(C.ROOT), "(FAKE DATA banner on)" if fake else "")
    return made


if __name__ == "__main__":
    run(sys.argv[1:])
