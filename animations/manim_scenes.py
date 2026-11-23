"""Manim scenes for the deck: one short 3Blue1Brown-style explainer per big idea, in the deck's terminal colours.

Every number on screen is read from data/processed (written by run_all.py and src.deck_sims), never typed in.
Render all of them with:   python -m src.manim_videos            (720p30, about 4 minutes)
Render one while editing:  manim -ql animations/manim_scenes.py M2BetaUpdate
Needs: pip install manim, plus Pango (Mac: brew install pango; Ubuntu: sudo apt-get install libpango1.0-dev) and LaTeX.

M1 PoissonProcess  goals arrive at random -> counts are Poisson                notes 2.4, 5.5
M2 BetaUpdate      each signing updates Real Madrid's hit-rate belief          notes 7.2, 7.4
M3 Shrinkage       raw per-90 rates slide toward the prior mean                notes 7.3
M4 Skellam         two Poisson scorelines -> goal difference -> win/draw/loss  notes 5.4
M5 CLT             0/1/3 points per match add up to a bell                     notes 8.3
M6 LeastSquares    the line that makes the squares smallest                    notes 10.2
M7 MCMC            a walker that proposes, accepts, rejects                    notes 11A.3
M8 Thinning        remove the star: thin his goals, lower lambda               notes 5.5
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Axes, Create, DashedLine, Dot, FadeIn, FadeOut, GrowFromEdge, Indicate,
                   Line, MathTex, Rectangle, ReplacementTransform, Scene, Square, Text, Transform, ValueTracker,
                   VGroup, Write, always_redraw, config, linear, rate_functions)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- look (matches src/style.py, editorial theme)
BG, PANEL, GRID = "#2b211b", "#352a22", "#4d3f33"
INK, INK2, MUTED = "#f5ede2", "#e0d3c1", "#b3a28c"
GOLD, NAVY, GREY, UPC, DOWNC = "#e53935", "#8db4dc", "#b2a99c", "#66bb6a", "#7fa7d6"
config.background_color = BG


def _font():
    try:
        import manimpango
        have = set(manimpango.list_fonts())
        for f in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
            if f in have:
                return f
    except Exception:
        pass
    return "serif"


FONT = _font()


def T(s, size=26, color=INK, **kw):
    return Text(s, font=FONT, font_size=size, color=color, **kw)


def NUM():
    return json.loads((PROC / "deck_numbers.json").read_text())


def is_fake():
    try:
        names = pd.read_csv(PROC / "p3_rm_attackers.csv").name.astype(str)
        return names.str.fullmatch(r"Player \d+").mean() > 0.5
    except Exception:
        return False


class Base(Scene):
    kicker = ""
    title = ""

    def header(self):
        k = T(self.kicker, 18, GOLD).to_corner(UP + LEFT, buff=0.35)
        t = T(self.title, 30, INK).next_to(k, DOWN, aligned_edge=LEFT, buff=0.12)
        src = T("Data: Transfermarkt via transfermarkt-datasets; authors' calculations", 12, MUTED).to_corner(DOWN + RIGHT, buff=0.2)
        bar = Line(LEFT * 7.1, RIGHT * 7.1, color=GRID, stroke_width=1).next_to(t, DOWN, buff=0.15)
        self.add(src)
        if is_fake():
            self.add(T("FAKE DATA · DO NOT PRESENT", 18, DOWNC).to_corner(UP + RIGHT, buff=0.35))
        self.play(FadeIn(k, shift=RIGHT * 0.2), Write(t), Create(bar), run_time=1.2)
        return VGroup(k, t, bar)

    def axes(self, x_range, y_range, x_len=10, y_len=4.6, x_label="", y_label="", x_dp=0, y_dp=1, pad=0.0,
             y_numbers=True, x_ticks=None):
        """Axes whose y axis always sits at the left edge (Manim puts it at x = 0 otherwise).
        Use ax.p(x, y) for points and ax.fplot(f, lo, hi) for curves, in data units. Tick labels are monospace."""
        lo, hi, st = x_range
        L = lo - pad
        ax = Axes(x_range=[0, hi - L + pad, st], y_range=y_range, x_length=x_len, y_length=y_len, tips=False,
                  axis_config=dict(color=GRID, stroke_width=2), x_axis_config=dict(include_ticks=False),
                  y_axis_config=dict(include_ticks=True, tick_size=0.05))
        ticks = x_ticks if x_ticks is not None else np.arange(lo, hi + 1e-9, st)
        ax.x_axis.add_labels({float(t - L): T(f"{t:.{x_dp}f}", 16, INK2) for t in ticks}, font_size=16)
        if y_numbers:
            y0, y1, ys = y_range
            ax.y_axis.add_labels({float(v): T(f"{v:.{y_dp}f}", 16, INK2) for v in np.arange(y0 + ys, y1 + 1e-9, ys)}, font_size=16)
        ax.L = L
        ax.p = lambda x, y: ax.c2p(x - L, y)
        ax.fplot = lambda f, a, b, step=None, **kw: ax.plot(lambda u: f(u + L), x_range=[a - L, b - L, step or (b - a) / 200], **kw)
        labs = VGroup()
        if x_label:
            labs.add(T(x_label, 18, INK2).next_to(ax.x_axis, DOWN, buff=0.45))
        if y_label:
            labs.add(T(y_label, 18, INK2).rotate(math.pi / 2).next_to(ax.y_axis, LEFT, buff=0.55))
        return ax, labs

    def bars(self, ax, xs, ys, color, width=0.8):
        g = VGroup()
        for x, y in zip(xs, ys):
            w = ax.p(x + width / 2, 0)[0] - ax.p(x - width / 2, 0)[0]
            h = max(ax.p(0, y)[1] - ax.p(0, 0)[1], 1e-3)
            r = Rectangle(width=w, height=h, stroke_width=0, fill_color=color if isinstance(color, str) else color(x),
                          fill_opacity=0.9)
            r.move_to(ax.p(x, 0), aligned_edge=DOWN)
            g.add(r)
        return g


# ---------------------------------------------------------------- M1
class M1PoissonProcess(Base):
    kicker = "M1 · POISSON PROCESS · NOTES 2.4, 5.5"
    title = "Goals arrive at random. Count them: Poisson."

    def construct(self):
        n = NUM(); lam = n["lam_rm"]
        self.header()
        rng = np.random.default_rng(221)
        line = Line(LEFT * 6 + UP * 1.3, RIGHT * 6 + UP * 1.3, color=GREY)
        ticks = VGroup(*[VGroup(Line(UP * 0.1, DOWN * 0.1, color=GREY).move_to(line.point_from_proportion(m / 90)),
                                T(f"{m}'", 16, MUTED).next_to(line.point_from_proportion(m / 90), DOWN, buff=0.15))
                         for m in (0, 15, 30, 45, 60, 75, 90)])
        self.play(Create(line), FadeIn(ticks))
        times = np.sort(rng.uniform(0, 90, max(1, rng.poisson(lam))))
        dots = VGroup(*[Dot(line.point_from_proportion(t / 90), color=GOLD, radius=0.1) for t in times])
        lab = T(f"match 1: {len(times)} goals", 20, INK).next_to(line, UP, buff=0.3)
        self.play(*[FadeIn(d, scale=2.5) for d in dots], lag_ratio=0.4, run_time=1.5)
        self.play(FadeIn(lab))
        self.wait(0.4)

        def mini(i, ts):
            y = 0.9 - 0.5 * i
            ln = Line(LEFT * 6.6 + UP * y, LEFT * 1.6 + UP * y, color=GRID, stroke_width=2)
            ds = VGroup(*[Dot(ln.point_from_proportion(t / 90), color=GOLD, radius=0.06) for t in ts])
            tx = T(f"{len(ts)}", 18, GOLD).next_to(ln, RIGHT, buff=0.2)
            return VGroup(ln, ds, tx)
        first = mini(0, times)
        self.play(ReplacementTransform(VGroup(line, dots), VGroup(first[0], first[1])), FadeOut(ticks),
                  ReplacementTransform(lab, first[2]), run_time=1.2)
        head = T("goals per match", 16, INK2).next_to(first, UP, buff=0.2).align_to(first, LEFT)
        self.play(FadeIn(head))
        for i in range(1, 8):
            ts = np.sort(rng.uniform(0, 90, rng.poisson(lam)))
            self.play(FadeIn(mini(i, ts), shift=UP * 0.15), run_time=0.35)
        sims = rng.poisson(lam, 4000)
        ax, labs = self.axes([0, 8, 1], [0, 0.3, 0.1], x_len=5.6, y_len=2.8, x_label="goals in a match (4,000 simulated)", pad=0.7)
        VGroup(ax, labs).move_to(RIGHT * 3.6 + DOWN * 1.5)
        self.play(Create(ax), FadeIn(labs))
        freq = np.bincount(sims, minlength=9)[:9] / len(sims)
        emp = self.bars(ax, range(9), freq, GREY, width=0.7)
        self.play(*[GrowFromEdge(b, DOWN) for b in emp], lag_ratio=0.08, run_time=1.4)
        pmf = [math.exp(-lam) * lam ** k / math.factorial(k) for k in range(9)]
        pts = VGroup(*[Dot(ax.p(k, p), color=GOLD, radius=0.08) for k, p in enumerate(pmf)])
        f = MathTex(r"P(N=k)=\frac{e^{-\lambda}\lambda^k}{k!}", color=INK, font_size=42).move_to(RIGHT * 3.6 + UP * 1.5)
        lt = MathTex(rf"\lambda = {lam:.2f}\ \text{{goals per match}}", color=GOLD, font_size=32).next_to(f, DOWN, buff=0.2)
        self.play(Write(f), FadeIn(lt), *[FadeIn(d, scale=2) for d in pts], run_time=1.6)
        self.play(Indicate(lt, color=GOLD))
        self.wait(1.5)


# ---------------------------------------------------------------- M2
class M2BetaUpdate(Base):
    kicker = "M2 · BAYES · BETA-BINOMIAL · NOTES 7.2"
    title = "Every signing moves the belief a little."

    def construct(self):
        from scipy import stats
        hr = pd.read_csv(PROC / "p2_hit_rates.csv").set_index("club").loc["Real Madrid"]
        h, n = int(hr.hits), int(hr.n)
        a0, b0 = hr.post_a - h, hr.post_b - (n - h)
        self.header()
        top = 6
        ax, labs = self.axes([0, 1, 0.2], [0, top, 2], x_len=8, y_len=3.9, x_label="Real Madrid's hit rate p", x_dp=1, y_dp=0)
        VGroup(ax, labs).move_to(LEFT * 2.6 + DOWN * 0.9)
        rule = MathTex(r"\text{Beta}(a,b)\ \xrightarrow{\ h \text{ hits in } n\ }\ \text{Beta}(a+h,\ b+n-h)",   # notes 7.2
                       color=INK, font_size=32).move_to(UP * 1.95)
        self.play(Create(ax), FadeIn(labs), Write(rule))
        A, B = ValueTracker(a0), ValueTracker(b0)
        pdf = lambda p, a, b: min(stats.beta.pdf(p, a, b), top)  # noqa: E731
        curve = always_redraw(lambda: ax.fplot(lambda p: pdf(p, A.get_value(), B.get_value()), 0.002, 0.998, 0.004,
                                               color=GOLD, stroke_width=5))
        prior = ax.fplot(lambda p: pdf(p, a0, b0), 0.002, 0.998, 0.004, color=GREY, stroke_width=2)
        lab = always_redraw(lambda: MathTex(rf"\text{{Beta}}({A.get_value():.1f},\ {B.get_value():.1f})",
                                            color=GOLD, font_size=36).move_to(RIGHT * 4.6 + UP * 1.05))
        prl = T("grey: prior from the other clubs", 15, GREY).move_to(RIGHT * 4.6 + UP * 0.55)
        self.play(Create(prior), Create(curve), FadeIn(lab), FadeIn(prl), run_time=1.5)
        seq = np.array([1] * h + [0] * (n - h))
        np.random.default_rng(221).shuffle(seq)
        cols = 6
        origin = RIGHT * 3.3 + DOWN * 0.1
        for i, s_ in enumerate(seq):
            sq = Square(0.28, fill_color=UPC if s_ else DOWNC, fill_opacity=0.9, stroke_width=0)
            sq.move_to(origin + RIGHT * 0.36 * (i % cols) + DOWN * 0.36 * (i // cols))
            self.play(FadeIn(sq, scale=1.6), A.animate.set_value(A.get_value() + s_), B.animate.set_value(B.get_value() + 1 - s_),
                      run_time=0.55 if i < 6 else 0.25)
        rows = (n - 1) // cols + 1
        legend = VGroup(Square(0.18, fill_color=UPC, fill_opacity=1, stroke_width=0), T("regular", 15, INK2),
                        Square(0.18, fill_color=DOWNC, fill_opacity=1, stroke_width=0), T("not", 15, INK2)).arrange(RIGHT, buff=0.12)
        legend.move_to(origin + DOWN * 0.36 * rows + RIGHT * 0.9 + DOWN * 0.1, aligned_edge=LEFT).shift(LEFT * 0.9)
        mean = (a0 + h) / (a0 + b0 + n)
        res = VGroup(T(f"{h} hits in {n} signings", 18, INK), T(f"posterior mean {mean:.2f}", 22, GOLD))
        res.arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(legend, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(legend), Write(res))
        self.wait(2)


# ---------------------------------------------------------------- M3
class M3Shrinkage(Base):
    kicker = "M3 · SHRINKAGE · NOTES 7.3"
    title = "Small samples get pulled toward the league."

    def construct(self):
        n = NUM(); a, b = n["mle_a"], n["mle_b"]
        d = pd.read_csv(PROC / "p3_rm_attackers.csv").sort_values("t").head(7).iloc[::-1]
        self.header()
        hi = max(2.5, math.ceil(max(d.raw_per90.max(), d.post_mean.max()) * 2) / 2)
        ax, labs = self.axes([0, hi, 0.5], [0, len(d) + 1, 1], x_len=8.6, y_len=3.7, x_label="goal involvements per 90 minutes",
                             x_dp=1, y_numbers=False)
        ax.y_axis.set_opacity(0)
        VGroup(ax, labs).move_to(RIGHT * 0.9 + DOWN * 0.8)
        f = MathTex(r"\hat\theta=\frac{\alpha+k}{\beta+t}=w\,\frac{k}{t}+(1-w)\,\frac{\alpha}{\beta},\qquad w=\frac{t}{\beta+t}",
                    color=INK, font_size=30).move_to(UP * 2.1)
        self.play(Create(ax), FadeIn(labs), Write(f))
        pm = a / b
        pline = DashedLine(ax.p(pm, 0), ax.p(pm, len(d) + 0.5), color=GREY, stroke_width=2)
        plab = T(f"prior mean α/β = {pm:.2f}", 15, GREY).next_to(pline, UP, buff=0.08)
        self.play(Create(pline), FadeIn(plab))
        rows = []
        for i, r in enumerate(d.itertuples()):
            y = i + 1
            name = T(f"{r.name:<12} {r.t:4.0f}×90'", 15, INK2).next_to(ax.p(0, y), LEFT, buff=0.2)
            dot = Dot(ax.p(r.raw_per90, y), color=NAVY, radius=0.1)
            rows.append((r, y, dot, name))
        self.play(*[FadeIn(nm) for *_, nm in rows], *[FadeIn(dt, scale=2) for *_, dt, _ in rows], lag_ratio=0.1, run_time=1.5)
        key = VGroup(VGroup(Dot(color=NAVY, radius=0.08), T("raw k / t", 16, NAVY)).arrange(RIGHT, buff=0.12),
                     VGroup(Dot(color=GOLD, radius=0.08), T("posterior mean", 16, GOLD)).arrange(RIGHT, buff=0.12))
        key.arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_edge(RIGHT, buff=0.25).shift(UP * 0.6)
        self.play(FadeIn(key[0]))
        anims = []
        for r, y, dot, _ in rows:
            self.add(dot.copy().set_opacity(0.25))
            anims.append(dot.animate.move_to(ax.p(r.post_mean, y)).set_color(GOLD))
        self.play(*anims, FadeIn(key[1]), run_time=2.5, rate_func=rate_functions.ease_in_out_cubic)
        note = T("Few minutes: big pull.\nMany minutes: the data win.", 16, INK).to_corner(DOWN + LEFT, buff=0.3)
        self.play(Write(note))
        self.wait(2)


# ---------------------------------------------------------------- M4
class M4Skellam(Base):
    kicker = "M4 · TWO POISSONS · NOTES 5.4"
    title = "Two scorelines become one result."

    def construct(self):
        n = NUM(); lh, la = n["lam_rm"], n["lam_bar"]
        pois = lambda lam, k: math.exp(-lam) * lam ** k / math.factorial(k)  # noqa: E731
        self.header()
        kw = dict(x_len=4.6, y_len=1.7, pad=0.6, y_dp=1)
        axL, lL = self.axes([0, 7, 1], [0, 0.4, 0.2], x_label="Real Madrid goals", **kw)
        axR, lR = self.axes([0, 7, 1], [0, 0.4, 0.2], x_label="Barcelona goals", **kw)
        VGroup(axL, lL).move_to(LEFT * 3.3 + UP * 1.0)
        VGroup(axR, lR).move_to(RIGHT * 3.3 + UP * 1.0)
        bl = self.bars(axL, range(8), [pois(lh, k) for k in range(8)], GOLD, 0.7)
        br = self.bars(axR, range(8), [pois(la, k) for k in range(8)], NAVY, 0.7)
        tl = MathTex(rf"X\sim\text{{Pois}}({lh:.2f})", color=GOLD, font_size=30).next_to(axL, UP, buff=0.05).align_to(axL, RIGHT)
        tr = MathTex(rf"Y\sim\text{{Pois}}({la:.2f})", color=NAVY, font_size=30).next_to(axR, UP, buff=0.05).align_to(axR, RIGHT)
        self.play(Create(axL), Create(axR), FadeIn(lL), FadeIn(lR))
        self.play(*[GrowFromEdge(b_, DOWN) for b_ in [*bl, *br]], Write(tl), Write(tr), run_time=1.5)
        self.wait(0.5)
        diffs = np.arange(-4, 7)
        pd_ = [sum(pois(lh, k + j) * pois(la, k) for k in range(0, 15) if k + j >= 0) for j in diffs]
        axD, lD = self.axes([-4, 6, 1], [0, 0.3, 0.1], x_len=8.4, y_len=1.9, x_label="goal difference X − Y", pad=0.6)
        VGroup(axD, lD).move_to(DOWN * 1.95 + LEFT * 1.0)
        col = lambda x: UPC if x > 0 else (GREY if x == 0 else DOWNC)  # noqa: E731
        bd = self.bars(axD, diffs, pd_, col, 0.75)
        self.play(Create(axD), FadeIn(lD))
        self.play(*[ReplacementTransform(VGroup(bl[i % 8].copy(), br[(i * 3) % 8].copy()), bd[i]) for i in range(len(diffs))],
                  run_time=2)
        pw = sum(p for j, p in zip(diffs, pd_) if j > 0)
        pdr = sum(p for j, p in zip(diffs, pd_) if j == 0)
        res = VGroup(T(f"WIN  {pw:.0%}", 24, UPC), T(f"DRAW {pdr:.0%}", 24, GREY), T(f"LOSS {1 - pw - pdr:.0%}", 24, DOWNC))
        res.arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(axD, RIGHT, buff=0.6)
        self.play(FadeIn(res, shift=LEFT * 0.3))
        self.wait(2)


# ---------------------------------------------------------------- M5
class M5CLT(Base):
    kicker = "M5 · CENTRAL LIMIT THEOREM · NOTES 8.3"
    title = "Add up 0, 1, 3 point matches. A bell appears."

    def construct(self):
        n = NUM(); pw, pdr, N = n["clt_pw"], n["clt_pd"], int(n["clt_n"])
        one = np.zeros(4); one[0], one[1], one[3] = 1 - pw - pdr, pdr, pw
        mu1, var1 = 3 * pw + pdr, 9 * pw + pdr - (3 * pw + pdr) ** 2
        self.header()
        f = MathTex(r"S_n=X_1+\dots+X_n\ \approx\ N(n\mu,\ n\sigma^2),\qquad X_i\in\{0,1,3\}", color=INK,
                    font_size=34).move_to(UP * 1.95)
        self.play(Write(f))
        pmf, stages = one.copy(), {1: one}
        for k in range(2, N + 1):
            pmf = np.convolve(pmf, one)
            if k in (2, 4, 10, N):
                stages[k] = pmf.copy()
        prev = None
        for k, p in stages.items():
            mu, sd = k * mu1, math.sqrt(k * var1)
            lo, hi = (0, 3 * k) if k < 10 else (int(max(0, mu - 4 * sd)), int(min(3 * k, mu + 4 * sd)))
            step = max(1, (hi - lo) // 6)
            xs = np.arange(len(p))
            keep = (xs >= lo) & (xs <= hi)
            ax, labs = self.axes([lo, hi, step], [0, float(p.max()) * 1.15, 1], x_len=10.5, y_len=3.6, pad=0.6,
                                 x_label=f"points after {k} match{'es' if k > 1 else ''}", y_numbers=False)
            VGroup(ax, labs).move_to(DOWN * 1.15)
            bars = self.bars(ax, xs[keep], p[keep], GOLD, 0.8 if k < 10 else 0.9)
            cap = MathTex(rf"n={k}", color=GOLD, font_size=42).move_to(RIGHT * 5.6 + UP * 1.0)
            grp = VGroup(ax, labs, bars, cap)
            if prev is None:
                self.play(Create(ax), FadeIn(labs), *[GrowFromEdge(b_, DOWN) for b_ in bars], FadeIn(cap))
            else:
                self.play(ReplacementTransform(prev, grp), run_time=1.3)
            prev = grp
            self.wait(0.6)
        ax = prev[0]
        mu, sd = N * mu1, math.sqrt(N * var1)
        normal = ax.fplot(lambda x: math.exp(-(x - mu) ** 2 / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi)),
                          max(ax.L + 0.6, mu - 4 * sd), mu + 4 * sd, 0.25, color=INK, stroke_width=4)
        lab = MathTex(rf"N({mu:.1f},\ {sd:.1f}^2)", color=INK, font_size=34).next_to(prev[3], DOWN, buff=0.2)
        self.play(Create(normal), Write(lab), run_time=1.5)
        self.wait(2)


# ---------------------------------------------------------------- M6
class M6LeastSquares(Base):
    kicker = "M6 · LEAST SQUARES · NOTES 10.2"
    title = "The best line makes the squares smallest."

    def construct(self):
        cs = pd.read_csv(PROC / "club_seasons.csv")
        d = cs[(cs.competition_id == "ES1") & cs.rel_value.notna()].copy()
        d["x"] = np.log(d.rel_value)
        rm = d[d.club_id == 418]
        oth = d[d.club_id != 418].sample(min(24, int((d.club_id != 418).sum())), random_state=221)
        pts = pd.concat([oth, rm.sample(min(4, len(rm)), random_state=221)])
        b1, b0 = np.polyfit(d.x, d.points, 1)
        self.header()
        x0, x1 = math.floor(d.x.min() * 2) / 2, math.ceil(d.x.max() * 2) / 2
        ax, labs = self.axes([x0, x1, 0.5], [0, 120, 20], x_len=7.8, y_len=4.5, x_label="log(squad value ÷ league average)",
                             y_label="points", x_dp=1, y_dp=0)
        VGroup(ax, labs).move_to(LEFT * 2.4 + DOWN * 0.75)
        self.play(Create(ax), FadeIn(labs))
        dots = VGroup(*[Dot(ax.p(r.x, r.points), radius=0.07, color=GOLD if r.club_id == 418 else GREY) for r in pts.itertuples()])
        self.play(FadeIn(dots, lag_ratio=0.05), run_time=1.2)
        slope, icpt = ValueTracker(0.0), ValueTracker(float(d.points.mean()))
        clamp = lambda v: min(max(v, 0), 120)  # noqa: E731

        def line():
            ys = [icpt.get_value() + slope.get_value() * x for x in (x0, x1)]
            return Line(ax.p(x0, clamp(ys[0])), ax.p(x1, clamp(ys[1])), color=NAVY, stroke_width=4)
        uy = ax.p(0, 1)[1] - ax.p(0, 0)[1]

        def squares():
            g = VGroup()
            for r in pts.itertuples():
                fit = icpt.get_value() + slope.get_value() * r.x
                e = r.points - fit
                side = max(abs(e) * uy, 1e-3)
                sq = Square(side_length=side, stroke_width=1, stroke_color=DOWNC, fill_color=DOWNC, fill_opacity=0.14)
                sq.move_to(ax.p(r.x, r.points - e / 2) + RIGHT * side / 2)
                g.add(sq)
            return g
        sse = lambda: float(np.sum((pts.points - icpt.get_value() - slope.get_value() * pts.x) ** 2))  # noqa: E731
        right = RIGHT * 4.75
        L_, S_ = always_redraw(line), always_redraw(squares)
        box = always_redraw(lambda: VGroup(T("sum of squares", 16, INK2), T(f"{sse():,.0f}", 34, DOWNC),
                                           MathTex(rf"\hat y = {icpt.get_value():.1f} + {slope.get_value():.1f}\,x", color=NAVY,
                                                   font_size=32)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(right + DOWN * 0.3))
        self.play(Create(L_), FadeIn(S_), FadeIn(box))
        for s_, c in ((70, d.points.mean() - 10), (12, d.points.mean() + 5)):
            self.play(slope.animate.set_value(s_), icpt.animate.set_value(c), run_time=1.6)
        self.play(slope.animate.set_value(b1), icpt.animate.set_value(b0), run_time=2.2)
        f = MathTex(r"\min_{\beta_0,\beta_1}\ \sum_i (y_i-\beta_0-\beta_1x_i)^2", color=INK, font_size=28).move_to(right + UP * 1.7)
        r2 = T(f"R² = {NUM()['r2']:.2f} · n = {len(d)}", 18, GOLD).move_to(right + DOWN * 1.9)
        self.play(Write(f), FadeIn(r2))
        self.wait(2)


# ---------------------------------------------------------------- M7
class M7MCMC(Base):
    kicker = "M7 · METROPOLIS MCMC · NOTES 11A.3"
    title = "Propose a step. Accept or reject. Repeat."

    def construct(self):
        from scipy import stats
        from scipy.special import expit, logit
        hr = pd.read_csv(PROC / "p2_hit_rates.csv").set_index("club").loc["Real Madrid"]
        h, n = int(hr.hits), int(hr.n)
        lp = lambda p: stats.norm.logpdf(logit(p), -0.2, 0.5) - np.log(p * (1 - p)) + stats.binom.logpmf(h, n, p)  # noqa: E731
        grid = np.linspace(0.005, 0.995, 400)
        Z = np.trapezoid(np.exp(lp(grid) - lp(grid).max()), grid)
        dens = lambda p: float(np.exp(lp(p) - lp(grid).max()) / Z)  # noqa: E731
        ymax = max(dens(g) for g in grid) * 1.25
        self.header()
        ax, labs = self.axes([0, 1, 0.2], [0, ymax, 1], x_len=9.6, y_len=3.5, x_label="Real Madrid's hit rate p", x_dp=1,
                             y_numbers=False)
        VGroup(ax, labs).move_to(LEFT * 1.4 + DOWN * 1.1)
        curve = ax.fplot(dens, 0.005, 0.995, 0.005, color=INK2, stroke_width=3)
        f = MathTex(r"r=\frac{\text{posterior}(\theta^*)}{\text{posterior}(\theta)},\quad \text{accept }\theta^*\text{ with probability }\min(1,r)",   # notes 11A.3
                    color=INK, font_size=32).move_to(UP * 1.85)
        self.play(Create(ax), FadeIn(labs), Create(curve), Write(f))
        rng = np.random.default_rng(221)
        target = lambda z: lp(expit(z)) + np.log(expit(z) * (1 - expit(z)))  # noqa: E731
        z = logit(0.7); cur = target(z)
        walker = Dot(ax.p(expit(z), 0), color=GOLD, radius=0.12)
        self.play(FadeIn(walker, scale=2))
        state = dict(draws=[], acc=0)
        bins = np.linspace(0, 1, 41)
        hist = VGroup()
        side = RIGHT * 5.6 + DOWN * 0.2
        counter = always_redraw(lambda: VGroup(T("accepted", 16, INK2),
                                               T(f"{state['acc']}/{max(len(state['draws']), 1)}", 30, GOLD)).arrange(DOWN, aligned_edge=LEFT).move_to(side))
        key = VGroup(VGroup(Dot(color=UPC, radius=0.07), T("accepted", 15, INK2)).arrange(RIGHT, buff=0.1),
                     VGroup(Dot(color=DOWNC, radius=0.07), T("rejected", 15, INK2)).arrange(RIGHT, buff=0.1))
        key.arrange(DOWN, aligned_edge=LEFT).next_to(counter, DOWN, buff=0.4).align_to(counter, LEFT)
        self.add(counter, key)
        N_ = 160
        for i in range(N_):
            zp = z + rng.normal()
            new = target(zp)
            ok = bool(np.log(rng.random()) < new - cur)
            slow = i < 16
            if slow:
                c = UPC if ok else DOWNC
                prop = Dot(ax.p(expit(zp), 0), color=c, radius=0.09)
                arr = Arrow(walker.get_center() + UP * 0.4, prop.get_center() + UP * 0.4, buff=0, color=c, stroke_width=3,
                            max_tip_length_to_length_ratio=0.2)
                self.play(FadeIn(prop), Create(arr), run_time=0.3)
            if ok:
                z, cur = zp, new
                state["acc"] += 1
            state["draws"].append(expit(z))
            cnt, _ = np.histogram(state["draws"], bins=bins, density=True)
            nh = self.bars(ax, (bins[:-1] + bins[1:]) / 2, np.minimum(cnt, ymax), GOLD, 1 / 40 * 0.9).set_opacity(0.55)
            if slow:
                self.play(walker.animate.move_to(ax.p(expit(z), 0)), FadeOut(prop), FadeOut(arr), Transform(hist, nh), run_time=0.3)
            elif i % 4 == 0 or i == N_ - 1:
                self.play(Transform(hist, nh), walker.animate.move_to(ax.p(expit(z), 0)), run_time=0.1, rate_func=linear)
        self.bring_to_front(walker)
        res = T(f"{N_} draws\nmean {np.mean(state['draws']):.2f}\nhistogram ≈\nposterior", 16, INK)
        res.next_to(key, DOWN, buff=0.4).align_to(key, LEFT)
        self.play(FadeIn(res))
        self.wait(2)


# ---------------------------------------------------------------- M8
class M8Thinning(Base):
    kicker = "M8 · THINNING A POISSON PROCESS · NOTES 5.5"
    title = "Take the star away: his goals disappear at random."

    def construct(self):
        n = NUM()
        s, lam = n["star_share"], n["lam_rm"]
        p_with, p_without = n["p_title"], n["p_title_without"]
        self.header()
        rng = np.random.default_rng(7)
        line = Line(LEFT * 6 + UP * 0.9, RIGHT * 6 + UP * 0.9, color=GREY)
        self.play(Create(line))
        times = np.sort(rng.uniform(0, 1, 40))
        star = rng.random(40) < s
        dots = VGroup(*[Dot(line.point_from_proportion(t), radius=0.1, color=GOLD) for t in times])
        lab = T("Real Madrid's goals over a season", 18, INK2).next_to(line, UP, buff=0.3)
        self.play(FadeIn(lab), FadeIn(dots, lag_ratio=0.04), run_time=1.6)
        self.play(*[d_.animate.set_color(NAVY) for d_, st in zip(dots, star) if st], run_time=0.8)
        sl = T(f"blue = involved the star ({s:.0%} of goals)", 18, NAVY).next_to(line, DOWN, buff=0.35)
        self.play(FadeIn(sl))
        keep = rng.random(40) < 0.5
        self.play(*[FadeOut(d_, shift=DOWN * 0.5, scale=0.3) for d_, st, k in zip(dots, star, keep) if st and not k], run_time=1.2)
        f = MathTex(r"\lambda' = \lambda\,[1-s(1-r)]", color=INK, font_size=40).move_to(DOWN * 1.0 + LEFT * 3.0)
        sub = MathTex(rf"= {lam:.2f}\,[1-{s:.2f}(1-0.5)] = {lam * (1 - s * (1 - 0.5)):.2f}", color=GOLD,
                      font_size=34).next_to(f, DOWN, buff=0.2, aligned_edge=LEFT)
        note = T("r = 0.5: the replacement scores half as often", 16, MUTED).next_to(sub, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(Write(f), Write(sub), FadeIn(note))
        bw, bh = 0.9, 2.0
        base = DOWN * 3.1 + RIGHT * 3.4
        b1 = Rectangle(width=bw, height=bh * p_with, fill_color=GOLD, fill_opacity=0.9, stroke_width=0).move_to(base, aligned_edge=DOWN)
        b2 = Rectangle(width=bw, height=bh * p_without, fill_color=GREY, fill_opacity=0.9, stroke_width=0)
        b2.move_to(base + RIGHT * 1.6, aligned_edge=DOWN)
        t1 = T(f"{p_with:.0%}", 22, GOLD).next_to(b1, UP, buff=0.1)
        t2 = T(f"{p_without:.0%}", 22, INK).next_to(b2, UP, buff=0.1)
        c1 = T("with", 16, INK2).next_to(b1, DOWN, buff=0.1)
        c2 = T("without", 16, INK2).next_to(b2, DOWN, buff=0.1)
        head = T(f"P(title), {int(n['n_sims']):,} simulated seasons", 17, INK2).next_to(VGroup(t1, t2), UP, buff=0.2)
        self.play(FadeIn(head), GrowFromEdge(b1, DOWN), FadeIn(t1), FadeIn(c1))
        self.play(GrowFromEdge(b2, DOWN), FadeIn(t2), FadeIn(c2))
        self.play(Indicate(t2, color=DOWNC))
        self.wait(2)


SCENES = ["M1PoissonProcess", "M2BetaUpdate", "M3Shrinkage", "M4Skellam", "M5CLT", "M6LeastSquares", "M7MCMC", "M8Thinning"]
