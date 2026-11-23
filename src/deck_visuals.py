"""Dynamic visuals for the final slide deck (Plotly). Every figure is saved twice:
figures/interactive/<ID>.html (play / rotate / pick a player in the browser and in the reveal.js deck)
figures/<ID>.png (static twin for PowerPoint and the PDF; needs kaleido + Chrome).

R1 Money vs points, season by season (▶ Play)          Part 1  regression line, residual = actual - predicted
R2 Player posterior explorer (pick a Real Madrid attacker) Part 3  Gamma prior -> Gamma posterior, 95% interval
R3 Raw vs posterior: who was lucky? (bubbles = minutes)  Part 3  shrinkage toward the prior mean
R4 El Clasico scoreline probabilities in 3D              Part 5  P(H = i, A = j) = Pois(i; λh) Pois(j; λa)
R5 Learning the hit rate, one signing at a time (3D)     Part 2  Beta(a + hits, b + misses)
R6 10,000 seasons: the title race settles (▶ Play)       Part 5  law of large numbers, ±1.96 √(p(1-p)/n)
R7 Live Clasico: win probability minute by minute (▶)    Part 5  Poisson process + Skellam on the minutes left
R8 Market value over age and output (3D)                 Part 4  log value = b0 + b1 (age-26) + b2 (age-26)^2 + ...
R9 With vs without the star: season points (▶ Play)      Part 5  CLT bell, what-if thinning
Run: python -m src.deck_visuals   (after python run_all.py)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from scipy import stats

from config import DATA_PROC, FIG, REAL_MADRID, SEED
from src.part4_value_curve import peak_age, prep
from src.part5_season_sim import fit_poisson, rates, simulate_season, what_if
from src.style import BG, DOWN, GOLD, GREY, GRIDC, INK, INK2, LIGHT, MUTED, NAVY, PANEL, SEQ, SOURCE, plotly_template

TEMPLATE = plotly_template()
OUT = FIG / "interactive"
OUT.mkdir(parents=True, exist_ok=True)
NUM = {}                      # key numbers for the slides, written to data/processed/deck_numbers.json


def _layout(fig, title, height=620):
    import re
    title = re.sub(r"^R\d+b?\s+", "", title)              # internal figure codes stay in file names, not on screen
    for fr in fig.frames or []:
        if fr.layout and fr.layout.title and fr.layout.title.text:
            fr.layout.title.text = re.sub(r"^R\d+b?\s+", "", fr.layout.title.text)
    is3d = any(t.type in ("surface", "scatter3d") for t in fig.data)
    fig.update_layout(template=TEMPLATE, title=dict(text=title), height=height,
                      margin=dict(l=20 if is3d else 80, r=20, t=80, b=20), legend=dict(orientation="h", y=-0.12))
    fig.add_annotation(text=SOURCE, xref="paper", yref="paper", x=1, y=-0.2, xanchor="right", showarrow=False,
                       font=dict(size=11, color=MUTED))
    for k in [k for k in fig.layout if k.startswith("scene")] if is3d else []:
        for ax in ("xaxis", "yaxis", "zaxis"):
            getattr(fig.layout[k], ax).update(backgroundcolor=PANEL, gridcolor=GRIDC, zerolinecolor=GRIDC, showspikes=False,
                                              tickfont=dict(color=INK2, size=11), title_font=dict(color=INK2))
        fig.layout[k].update(bgcolor=BG)
        if fig.layout[k].camera.eye.x is None:
            fig.layout[k].camera = dict(eye=dict(x=1.6, y=-1.5, z=0.9))
    return fig


def _play(duration=120):
    return [dict(type="buttons", x=1.0, xanchor="right", y=1.14, direction="left", showactive=False, buttons=[
        dict(label="▶ Play", method="animate",
             args=[None, dict(frame=dict(duration=duration, redraw=True), transition=dict(duration=0), fromcurrent=False)]),
        dict(label="❚❚", method="animate", args=[[None], dict(mode="immediate")])])]


def strip_codes(text):
    """Internal figure codes (R1, R10b, ...) name the files; viewers never see them in titles or captions."""
    import re
    return re.sub(r"(?<=[\"'>])R\d{1,2}b?(?:\s+·)?\s+(?=\S)", "", text)


def save(fig, name, width=1600, height=900):
    html = OUT / f"{name}.html"
    fig.write_html(html, include_plotlyjs="directory", config=dict(displaylogo=False, responsive=True))
    html.write_text(strip_codes(html.read_text(encoding="utf-8")), encoding="utf-8")
    try:
        fig.write_image(FIG / f"{name}.png", width=width, height=height)
    except Exception as e:
        print(f"  (no PNG for {name}: {e.__class__.__name__}; pip install kaleido)")
    return html


def _season(g=None):
    g = g if g is not None else pd.read_csv(DATA_PROC / "league_games.csv", parse_dates=["date"])
    gs = g[g.season == g.season.max()].sort_values("date")
    return gs, dict(zip(gs.home_club_id, gs.home_club_name))


# ---------------------------------------------------------------- R1
def r1_money_vs_points():
    cs = pd.read_csv(DATA_PROC / "club_seasons.csv")
    d = cs[(cs.competition_id == "ES1") & cs.rel_value.notna()].copy()
    d["x"] = np.log(d.rel_value)
    m = smf.ols("points ~ x", data=d).fit()
    NUM.update(r2=m.rsquared, slope=m.params.x, pts_20pct=m.params.x * np.log(1.2))
    xs = np.linspace(d.x.min(), d.x.max(), 50)
    seasons = sorted(d.season.unique())

    def traces(s):
        cur = d[d.season == s]
        rm = cur[cur.club_id == REAL_MADRID]
        oth = cur[cur.club_id != REAL_MADRID]
        return [go.Scatter(x=d.x, y=d.points, mode="markers", marker=dict(color=LIGHT, size=7, opacity=0.35),
                           hoverinfo="skip", name="all club-seasons"),
                go.Scatter(x=xs, y=m.params.Intercept + m.params.x * xs, mode="lines", line=dict(color=NAVY, width=3),
                           name=f"fitted line (R² {m.rsquared:.2f})"),
                go.Scatter(x=oth.x, y=oth.points, mode="markers", marker=dict(color=GREY, size=11), name=f"{s}/{s % 100 + 1:02d}",
                           text=oth.name, hovertemplate="%{text}<br>points %{y}<extra></extra>"),
                go.Scatter(x=rm.x, y=rm.points, mode="markers+text", text=["Real Madrid"] * len(rm), textposition="top center",
                           marker=dict(color=GOLD, size=22, line=dict(color=BG, width=2)), name="Real Madrid",
                           hovertemplate="Real Madrid<br>points %{y}<extra></extra>")]
    fig = go.Figure(data=traces(seasons[-1]),
                    frames=[go.Frame(data=traces(s), name=str(s), layout=dict(title_text=f"R1  Money vs points: La Liga {s}/{s % 100 + 1:02d}"))
                            for s in seasons])
    fig.update_layout(updatemenus=_play(700), xaxis_title="log(squad value ÷ league average)", yaxis_title="league points",
                      sliders=[dict(steps=[dict(method="animate", label=str(s), args=[[str(s)], dict(mode="immediate")])
                                           for s in seasons], currentvalue=dict(prefix="Season start: "), pad=dict(t=50))])
    fig = _layout(fig, "R1  Money vs points: every La Liga season, Real Madrid in red")
    fig.update_layout(legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"), margin=dict(t=120))   # legend clear of the slider
    return save(fig, "R1_money_vs_points")


# ---------------------------------------------------------------- R2, R3
def _attackers():
    g = pd.read_csv(DATA_PROC / "p3_rm_attackers.csv").rename(columns={"P_over_0.6": "p_over"})
    g = g[g.minutes >= 450].sort_values("post_mean", ascending=False).head(10)
    a0, b0 = float((g.post_a - g.k).iloc[0]), float((g.post_b - g.t).iloc[0])   # prior = posterior minus data
    return g, a0, b0


def r2_player_posteriors():
    g, a0, b0 = _attackers()
    lam = np.linspace(0, max(2.0, g.hi.max() * 1.1), 500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lam, y=stats.gamma.pdf(lam, a0, scale=1 / b0), mode="lines", name=f"prior Gamma({a0:.2f}, {b0:.2f})",
                             line=dict(color=GREY, dash="dash", width=3)))
    buttons = []
    for i, r in enumerate(g.itertuples()):
        dens = stats.gamma.pdf(lam, r.post_a, scale=1 / r.post_b)
        inside = (lam >= r.lo) & (lam <= r.hi)
        vis = i == 0
        fig.add_trace(go.Scatter(x=lam, y=dens, mode="lines", line=dict(color=NAVY, width=3), visible=vis,
                                 name="posterior", hovertemplate="λ %{x:.2f}<br>density %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=lam[inside], y=dens[inside], fill="tozeroy", mode="none", fillcolor="rgba(254,190,16,0.40)",
                                 visible=vis, name="95% credible interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r.raw_per90, r.raw_per90], y=[0, dens.max()], mode="lines", visible=vis,
                                 line=dict(color=DOWN, width=2, dash="dot"), name="raw per 90"))
        n = 1 + 3 * len(g)
        mask = [True] + [False] * (n - 1)
        mask[1 + 3 * i: 4 + 3 * i] = [True] * 3
        buttons.append(dict(label=r.name, method="update", args=[{"visible": mask}, {
            "title.text": f"R2  {r.name}: {r.k} G+A in {r.minutes:,} min → posterior mean {r.post_mean:.2f} per 90, "
                          f"95% [{r.lo:.2f}, {r.hi:.2f}], P(λ > 0.6) = {r.p_over:.2f}"}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=1.0, xanchor="right", y=1.12, direction="down")],
                      xaxis_title="λ = true goal contributions per 90", yaxis_title="density")
    first = g.iloc[0]
    return save(_layout(fig, f"R2  {first['name']}: posterior mean {first.post_mean:.2f} per 90 (pick a player)"),
                "R2_player_posteriors")


def r3_shrinkage_bubbles():
    g, a0, b0 = _attackers()
    paid = g.fee_m.notna()
    fig = go.Figure()
    for r in g.itertuples():
        fig.add_annotation(x=r.post_mean, y=r.name, ax=r.raw_per90, ay=r.name, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=LIGHT, text="")
    fig.add_trace(go.Scatter(x=g.raw_per90, y=g.name, mode="markers", name="raw G+A per 90",
                             marker=dict(symbol="x", size=12, color=GREY)))
    fig.add_trace(go.Scatter(x=g.post_mean, y=g.name, mode="markers", name="posterior mean (bubble = minutes)",
                             marker=dict(size=np.sqrt(g.minutes) / 3.5, color=np.where(paid, GOLD, NAVY), line=dict(color=BG, width=2)),
                             customdata=np.c_[g.k, g.minutes, g.lo, g.hi],
                             hovertemplate="%{y}<br>%{customdata[0]} G+A in %{customdata[1]:,} min<br>"
                                           "posterior %{x:.2f} [%{customdata[2]:.2f}, %{customdata[3]:.2f}]<extra></extra>"))
    fig.add_vline(x=a0 / b0, line=dict(color=GREY, dash="dash"), annotation_text=f"prior mean {a0 / b0:.2f}")
    fig.update_layout(xaxis_title="goal contributions per 90", yaxis=dict(autorange="reversed"))
    return save(_layout(fig, "R3  Few minutes → pulled hard toward the average; many minutes → barely moves", height=640),
                "R3_shrinkage_bubbles")


# ---------------------------------------------------------------- R4
def r4_clasico_3d():
    gs, names = _season()
    m = fit_poisson(gs)
    lh, la = (r[0] for r in rates(m, pd.DataFrame(dict(home_club_id=[REAL_MADRID], away_club_id=[131]))))
    k = np.arange(0, 7)
    P = np.outer(stats.poisson.pmf(k, lh), stats.poisson.pmf(k, la))           # independence given the rates
    NUM.update(lam_rm=lh, lam_bar=la, p_win=stats.skellam.sf(0, lh, la), p_draw=stats.skellam.pmf(0, lh, la))
    fig = go.Figure()
    for outcome, color, mask in (("Real Madrid win", GOLD, lambda i, j: i > j), ("draw", GREY, lambda i, j: i == j),
                                 ("Barcelona win", NAVY, lambda i, j: i < j)):
        xs, ys, zs = [], [], []
        for i in k:
            for j in k:
                if mask(i, j):
                    xs += [i, i, None]; ys += [j, j, None]; zs += [0, P[i, j], None]
        tot = sum(P[i, j] for i in k for j in k if mask(i, j))
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=16),
                                   name=f"{outcome}: {tot:.1%}", hoverinfo="skip"))
    ii, jj = np.meshgrid(k, k, indexing="ij")
    fig.add_trace(go.Scatter3d(x=ii.ravel(), y=jj.ravel(), z=P.ravel(), mode="markers", marker=dict(size=4, color=INK),
                               hovertemplate="Real Madrid %{x} – %{y} Barcelona<br>P = %{z:.3f}<extra></extra>", showlegend=False))
    fig.update_layout(scene=dict(xaxis_title="Real Madrid goals", yaxis_title="Barcelona goals", zaxis_title="probability",
                                 zaxis_tickformat=".0%"))
    return save(_layout(fig, f"R4  El Clasico at the Bernabéu: λ = {lh:.2f} vs {la:.2f}; every scoreline's probability"),
                "R4_clasico_3d")


# ---------------------------------------------------------------- R5
def r5_bayes_learning_3d():
    hr = pd.read_csv(DATA_PROC / "p2_hit_rates.csv").set_index("club")
    a0 = hr.loc["Real Madrid", "post_a"] - hr.loc["Real Madrid", "hits"]
    b0 = hr.loc["Real Madrid", "post_b"] - (hr.loc["Real Madrid", "n"] - hr.loc["Real Madrid", "hits"])
    sg = pd.read_csv(DATA_PROC / "signings.csv", parse_dates=["transfer_date"])
    rm = sg[sg.to_club_id == REAL_MADRID].sort_values("transfer_date")
    hits = np.r_[0, np.cumsum(rm.success.to_numpy())]
    n = np.arange(len(hits))
    p = np.linspace(0.001, 0.999, 300)
    Z = np.array([stats.beta.pdf(p, a0 + h, b0 + k - h) for k, h in zip(n, hits)])
    fig = go.Figure(go.Surface(x=p, y=n, z=Z, colorscale=SEQ, showscale=False, opacity=0.95,
                               hovertemplate="p %{x:.2f}<br>after %{y} signings<br>density %{z:.2f}<extra></extra>"))
    means = (a0 + hits) / (a0 + b0 + n)
    peak = [stats.beta.pdf(mu, a0 + h, b0 + k - h) for mu, k, h in zip(means, n, hits)]
    fig.add_trace(go.Scatter3d(x=means, y=n, z=peak, mode="lines+markers", line=dict(color=DOWN, width=6),
                               marker=dict(size=3), name="posterior mean",
                               text=[f"{'hit' if s else 'miss'}: {nm}" for s, nm in zip([None] + list(rm.success), [""] + list(rm.player_name))],
                               hovertemplate="%{text}<br>mean %{x:.2f}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title="hit rate p", yaxis_title="signings seen", zaxis_title="belief (density)",
                                 xaxis_tickformat=".0%"),
                      scene_camera=dict(eye=dict(x=1.7, y=1.3, z=0.8)))
    return save(_layout(fig, f"R5  Bayes, one signing at a time: Beta({a0:.1f}, {b0:.1f}) → Beta({a0 + hits[-1]:.1f}, {b0 + n[-1] - hits[-1]:.1f})"),
                "R5_bayes_learning_3d")


# ---------------------------------------------------------------- R6, R9
def _sims(n_sims=10_000):
    gs, names = _season()
    m = fit_poisson(gs)
    lh, la = rates(m, gs)
    pts, teams, champ = simulate_season(gs, lh, la, n_sims=n_sims)
    return gs, names, m, pts, teams, champ


def r6_title_race_lln():
    gs, names, m, pts, teams, champ = _sims()
    NUM.update(p_title=float(np.mean(champ == list(teams).index(REAL_MADRID))), n_sims=len(champ))
    top = pd.Series(np.bincount(champ, minlength=len(teams)), index=teams).sort_values(ascending=False).index[:3]
    n = np.arange(1, len(champ) + 1)
    stops = np.unique(np.geomspace(10, len(champ), 60).astype(int))
    col = {top[0]: GOLD, top[1]: NAVY, top[2]: GREY}

    def traces(k):
        out = []
        for t in top:
            i = list(teams).index(t)
            run = np.cumsum(champ[:k] == i) / n[:k]
            se = np.sqrt(np.clip(run * (1 - run), 1e-9, None) / n[:k])
            label = "Real Madrid" if t == REAL_MADRID else names.get(t, str(t))
            out += [go.Scatter(x=np.r_[n[:k], n[:k][::-1]], y=np.r_[run + 1.96 * se, (run - 1.96 * se)[::-1]], fill="toself",
                               fillcolor="rgba(128,140,160,0.12)", line=dict(width=0), hoverinfo="skip", showlegend=False),
                    go.Scatter(x=n[:k], y=run, mode="lines", line=dict(color=col[t], width=4), name=f"{label}: {run[-1]:.3f}")]
        return out
    fig = go.Figure(data=traces(len(champ)), frames=[go.Frame(data=traces(k), name=str(k)) for k in stops])
    fig.update_layout(updatemenus=_play(90), xaxis=dict(type="log", title="number of simulated seasons", range=[1, np.log10(len(champ))]),
                      yaxis=dict(title="P(champion)", range=[0, 1], tickformat=".0%"))
    return save(_layout(fig, "R6  10,000 simulated seasons: the estimate settles, the band shrinks like 1/√n"), "R6_title_race_lln")


def r9_star_points():
    gs, names, m, pts, teams, champ = _sims()
    i = list(teams).index(REAL_MADRID)
    ps = pd.read_csv(DATA_PROC / "player_seasons.csv")
    rm = ps[(ps.club_id == REAL_MADRID) & (ps.season == gs.season.max())]
    star = rm.sort_values("goals", ascending=False).iloc[0]
    share = star.goals / max(rm.goals.sum(), 1)
    pts1, _, ch1 = what_if(m, gs, REAL_MADRID, share)
    NUM.update(star=star["name"], star_share=share, p_title_without=float(np.mean(ch1 == i)),
               pts_with=float(pts[:, i].mean()), pts_without=float(pts1[:, i].mean()))
    a, b = pts[:, i], pts1[:, i]
    bins = np.arange(min(a.min(), b.min()), max(a.max(), b.max()) + 2)
    stops = np.unique(np.geomspace(20, len(a), 40).astype(int))

    def traces(k):
        return [go.Histogram(x=a[:k], xbins=dict(start=bins[0], end=bins[-1], size=1), marker_color=GOLD, opacity=0.75,
                             name=f"with {star['name']}: mean {a[:k].mean():.1f}"),
                go.Histogram(x=b[:k], xbins=dict(start=bins[0], end=bins[-1], size=1), marker_color=NAVY, opacity=0.6,
                             name=f"without: mean {b[:k].mean():.1f}")]
    fig = go.Figure(data=traces(len(a)), frames=[go.Frame(data=traces(k), name=str(k)) for k in stops])
    fig.update_layout(barmode="overlay", updatemenus=_play(120), xaxis_title="Real Madrid season points",
                      yaxis_title="simulated seasons")
    return save(_layout(fig, f"R9  Remove {star['name']} ({share:.0%} of goals): the whole bell shifts left"), "R9_star_points")


# ---------------------------------------------------------------- R7
def r7_live_clasico(seed=SEED):
    gs, names = _season()
    m = fit_poisson(gs)
    lh, la = (r[0] for r in rates(m, pd.DataFrame(dict(home_club_id=[REAL_MADRID], away_club_id=[131]))))
    rng = np.random.default_rng(seed)
    hg = np.r_[0, np.cumsum(rng.poisson(lh / 90, 90))]
    ag = np.r_[0, np.cumsum(rng.poisson(la / 90, 90))]
    pw, pdr = [], []
    for t in range(91):
        left, diff = (90 - t) / 90, hg[t] - ag[t]
        if left > 0:
            pw.append(stats.skellam.sf(-diff, lh * left, la * left)); pdr.append(stats.skellam.pmf(-diff, lh * left, la * left))
        else:
            pw.append(float(diff > 0)); pdr.append(float(diff == 0))
    pw, pdr = np.array(pw), np.array(pdr)
    pl = 1 - pw - pdr
    goals = [(t, "Real Madrid") for t in range(1, 91) if hg[t] > hg[t - 1]] + \
            [(t, "Barcelona") for t in range(1, 91) if ag[t] > ag[t - 1]]

    def traces(t):
        x = np.arange(t + 1)
        return [go.Scatter(x=x, y=pw[:t + 1], stackgroup="p", mode="none", fillcolor=GOLD, name="Real Madrid win"),
                go.Scatter(x=x, y=pdr[:t + 1], stackgroup="p", mode="none", fillcolor=LIGHT, name="draw"),
                go.Scatter(x=x, y=pl[:t + 1], stackgroup="p", mode="none", fillcolor=NAVY, name="Barcelona win")]
    frames = [go.Frame(data=traces(t), name=str(t),
                       layout=dict(title_text=f"R7  {t}'   Real Madrid {hg[t]} – {ag[t]} Barcelona   ·   P(win) {pw[t]:.0%}"))
              for t in range(91)]
    fig = go.Figure(data=traces(90), frames=frames)
    for t, who in goals:
        fig.add_vline(x=t, line=dict(color=DOWN if who == "Barcelona" else INK, width=1.5, dash="dot"))
        fig.add_annotation(x=t, y=1.04, yref="paper", text=f"GOAL {t}'", showarrow=False, font=dict(size=12))
    fig.update_layout(updatemenus=_play(110), xaxis=dict(range=[0, 90], title="minute"),
                      yaxis=dict(range=[0, 1], tickformat=".0%", title="probability given the score so far"))
    return save(_layout(fig, f"R7  Full time: Real Madrid {hg[90]} – {ag[90]} Barcelona (press ▶ to replay the match)"),
                "R7_live_clasico")


# ---------------------------------------------------------------- R8
def r8_value_surface():
    d = prep()
    att = d[d.position == "Attack"]
    pk, m = peak_age(att)
    ages = np.linspace(18, 36, 40)
    ga = np.linspace(0, 1.2, 30)
    A, G = np.meshgrid(ages, ga)
    logv = (m.params["Intercept"] + m.params["age_c"] * (A - 26) + m.params["I(age_c ** 2)"] * (A - 26) ** 2
            + m.params["ga_per90"] * G + m.params["min_share"] * att.min_share.mean())
    V = np.exp(logv) / 1e6                                                      # median value, EUR M
    NUM.update(peak_attack=pk)
    fig = go.Figure(go.Surface(x=A, y=G, z=V, colorscale=SEQ, showscale=False,
                               hovertemplate="age %{x:.1f}<br>G+A per 90 %{y:.2f}<br>value €%{z:.1f}M<extra></extra>"))
    ridge = np.exp(m.params["Intercept"] + m.params["ga_per90"] * ga + m.params["min_share"] * att.min_share.mean()
                   + m.params["age_c"] * (pk - 26) + m.params["I(age_c ** 2)"] * (pk - 26) ** 2) / 1e6
    fig.add_trace(go.Scatter3d(x=np.full_like(ga, pk), y=ga, z=ridge, mode="lines", line=dict(color=DOWN, width=8),
                               name=f"peak age {pk:.1f}"))
    fig.update_layout(scene=dict(xaxis_title="age", yaxis_title="G+A per 90", zaxis_title="median value (€M)"))
    return save(_layout(fig, f"R8  Attackers' market value peaks at {pk:.1f}, at every level of output"), "R8_value_surface")


# ---------------------------------------------------------------- R23  slide 3: squad value and points over time
def r23_value_and_points():
    """Real Madrid vs its two La Liga rivals, season by season (▶ Play): does the money line move with the points line?"""
    from plotly.subplots import make_subplots
    from config import LALIGA
    from src.style import RIVALS
    clubs = {REAL_MADRID: ("Real Madrid", GOLD, 5), 131: ("Barcelona", RIVALS[131], 3), 13: ("Atletico Madrid", RIVALS[13], 3)}
    la = pd.read_csv(DATA_PROC / "club_seasons.csv")
    la = la[(la.competition_id == LALIGA) & la.club_id.isin(clubs)].sort_values("season")
    seasons = sorted(la.season.unique())

    def traces(upto):
        out = []
        for cid, (name, col, w) in clubs.items():
            d = la[(la.club_id == cid) & (la.season <= upto)]
            lab = [f"{s}/{s % 100 + 1:02d}" for s in d.season]
            out += [go.Scatter(x=d.season, y=d.squad_value / 1e6, mode="lines+markers", line=dict(color=col, width=w),
                               marker=dict(size=9), name=name, legendgroup=name, text=lab,
                               hovertemplate=name + " %{text}<br>squad value EUR %{y:,.0f}M<extra></extra>"),
                    go.Scatter(x=d.season, y=d.points, mode="lines+markers", line=dict(color=col, width=w), marker=dict(size=9),
                               name=name, legendgroup=name, showlegend=False, text=lab,
                               hovertemplate=name + " %{text}<br>%{y} points<extra></extra>")]
        return out
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Squad market value (EUR millions)", "La Liga points"), horizontal_spacing=0.08)
    for k, t in enumerate(traces(seasons[-1])):
        fig.add_trace(t, 1, 1 + k % 2)
    fig.frames = [go.Frame(data=traces(s), name=str(s), traces=list(range(6))) for s in seasons]
    fig.update_xaxes(range=[seasons[0] - 0.5, seasons[-1] + 0.5], title_text="season start")
    fig.update_yaxes(range=[0, la.squad_value.max() / 1e6 * 1.08], title_text="EUR millions", row=1, col=1)
    fig.update_yaxes(range=[la.points.min() - 5, la.points.max() + 5], title_text="points", row=1, col=2)
    fig.update_layout(updatemenus=_play(500))
    rm = la[la.club_id == REAL_MADRID]
    return save(_layout(fig, f"R23  Real Madrid's squad value vs its points, {seasons[0]}/{seasons[0] % 100 + 1:02d} to "
                             f"{seasons[-1]}/{seasons[-1] % 100 + 1:02d}: corr(value, points) = {rm.squad_value.corr(rm.points):.2f}"),
                "R23_value_and_points")


# ---------------------------------------------------------------- R24  slide 6: residuals, season by season
def r24_residuals():
    """Actual minus predicted points for each Real Madrid season (Part 1 line); bars grow in one by one (▶ Play)."""
    rm = pd.read_csv(DATA_PROC / "p1_rm.csv").sort_values("season")
    rm["pred"] = rm.points - rm.residual
    lab = [f"{s}/{s % 100 + 1:02d}" for s in rm.season]
    col = [GOLD if r > 0 else DOWN for r in rm.residual]

    def bars(k):
        y = [r if i < k else 0 for i, r in enumerate(rm.residual)]
        return [go.Bar(x=lab, y=y, marker_color=col, customdata=np.c_[rm.points, rm.pred],
                       hovertemplate="%{x}<br>actual %{customdata[0]:.0f} − predicted %{customdata[1]:.1f}<br>"
                                     "= %{y:+.1f} points<extra></extra>", name="residual")]
    fig = go.Figure(data=bars(len(rm)), frames=[go.Frame(data=bars(k), name=str(k)) for k in range(1, len(rm) + 1)])
    fig.add_hline(y=0, line=dict(color=INK2, width=1))
    lim = rm.residual.abs().max() * 1.2
    fig.update_layout(updatemenus=_play(450), showlegend=False, xaxis_title="season",
                      yaxis=dict(title="points above (+) or below (−) what the budget predicts", range=[-lim, lim]))
    above = int((rm.residual > 0).sum())
    return save(_layout(fig, f"R24  Real Madrid finished above its budget line in {above} of {len(rm)} seasons (residual = actual − predicted)"),
                "R24_residuals")


# ---------------------------------------------------------------- R25  slide 7: hit rates, raw → posterior
def r25_hit_rates():
    """Every big club's transfer hit rate: ▶ slides each dot from its raw rate h/n to its Beta posterior mean (shrinkage),
    then the 95% credible interval appears."""
    hr = pd.read_csv(DATA_PROC / "p2_hit_rates.csv").sort_values("post_mean")
    a0 = (hr.post_a - hr.hits).iloc[0]
    b0 = (hr.post_b - (hr.n - hr.hits)).iloc[0]
    prior = a0 / (a0 + b0)
    is_rm = hr.club == "Real Madrid"
    col = np.where(is_rm, GOLD, GREY)
    size = np.where(is_rm, 18, 11)
    steps = np.linspace(0, 1, 16)

    def traces(t, ci):
        x = hr.raw_rate + t * (hr.post_mean - hr.raw_rate)
        err = dict(type="data", symmetric=False, array=(hr.cred_hi - hr.post_mean) if ci else np.zeros(len(hr)),
                   arrayminus=(hr.post_mean - hr.cred_lo) if ci else np.zeros(len(hr)), color=LIGHT, thickness=2, width=6)
        return [go.Scatter(x=hr.raw_rate, y=hr.club, mode="markers", marker=dict(symbol="circle-open", color=LIGHT, size=10),
                           name="raw h/n", customdata=np.c_[hr.hits, hr.n],
                           hovertemplate="%{y}: raw %{customdata[0]}/%{customdata[1]} = %{x:.0%}<extra></extra>"),
                go.Scatter(x=x, y=hr.club, mode="markers", marker=dict(color=col, size=size, line=dict(color=BG, width=1)),
                           error_x=err, name="posterior mean, 95% credible interval",
                           customdata=np.c_[hr.cred_lo, hr.cred_hi],
                           hovertemplate="%{y}: %{x:.0%}  [%{customdata[0]:.0%}, %{customdata[1]:.0%}]<extra></extra>")]
    frames = [go.Frame(data=traces(t, False), name=f"{t:.2f}") for t in steps] + [go.Frame(data=traces(1, True), name="ci")]
    fig = go.Figure(data=traces(1, True), frames=frames)
    fig.add_vline(x=prior, line=dict(color=NAVY, width=2, dash="dash"),
                  annotation_text=f"prior mean {prior:.0%}", annotation_font=dict(color=INK2, size=13))
    fig.update_layout(updatemenus=_play(90), height=640, xaxis=dict(title="share of paid signings who became regulars",
                      tickformat=".0%", range=[0, max(hr.raw_rate.max(), hr.cred_hi.max()) + 0.05]), yaxis_title="")
    rm = hr[is_rm].iloc[0]
    fig.update_yaxes(automargin=True)                    # full club names, never clipped
    fig = _layout(fig, f"R25  Real Madrid: raw {int(rm.hits)}/{int(rm.n)} = {rm.raw_rate:.0%} → posterior {rm.post_mean:.0%} "
                             f"(press ▶: every club shrinks toward {prior:.0%})", height=640)
    fig.update_layout(margin=dict(l=190))
    return save(fig, "R25_hit_rates")


# ---------------------------------------------------------------- R26  goal timing and the chi-square test (Part 6)
def r26_goal_timing():
    """Share of goals in each 15-minute bin, Real Madrid vs other big clubs; ▶ the bars fill up, the dashed line is the
    constant-rate Poisson model (every bin 1/6). Title: chi-square goodness of fit for Real Madrid (lecture notes 9.5)."""
    gt = pd.read_csv(DATA_PROC / "p6_goal_timing.csv")
    chi2 = float(((gt.n_rm - gt.exp_rm) ** 2 / gt.exp_rm).sum())
    df = len(gt) - 1
    pval = stats.chi2.sf(chi2, df)
    steps = np.linspace(0.05, 1, 14)

    def bars(t):
        return [go.Bar(x=gt.bin, y=gt.real_madrid * t, name="Real Madrid", marker_color=GOLD,
                       customdata=np.c_[gt.n_rm, gt.exp_rm],
                       hovertemplate="%{x}: %{customdata[0]:.0f} goals (expected %{customdata[1]:.0f})<extra>Real Madrid</extra>"),
                go.Bar(x=gt.bin, y=gt.others * t, name="Other big clubs", marker_color=GREY, customdata=gt.n_others,
                       hovertemplate="%{x}: %{customdata:.0f} goals<extra>Other big clubs</extra>")]
    fig = go.Figure(data=bars(1), frames=[go.Frame(data=bars(t), name=f"{t:.2f}") for t in steps])
    fig.add_hline(y=1 / 6, line=dict(color=INK, width=2, dash="dash"), annotation_text="constant-rate Poisson: 1/6 per bin",
                  annotation_position="top left", annotation_font=dict(color=INK2, size=13))
    fig.update_layout(updatemenus=_play(80), barmode="group", bargap=0.25, xaxis_title="minute of the match",
                      yaxis=dict(title="share of goals", tickformat=".0%", range=[0, max(gt.real_madrid.max(), gt.others.max()) * 1.18]))
    ptxt = f"p = {pval:.4f}" if pval >= 0.0001 else "p < 0.0001"
    return save(_layout(fig, f"R26  Real Madrid's goals by 15-minute bin: χ² = {chi2:.1f} on {df} df, {ptxt} (most of it from the last bin)"),
                "R26_goal_timing")


# ---------------------------------------------------------------- R27  residual checks for the Part 1 regression
def r27_residual_checks():
    """Residuals vs fitted and a normal Q-Q plot for points ~ log(relative value) (lecture notes 10.2); ▶ adds seasons."""
    from plotly.subplots import make_subplots
    cs = pd.read_csv(DATA_PROC / "club_seasons.csv")
    d = cs[(cs.competition_id == "ES1") & cs.rel_value.notna()].copy()
    d["x"] = np.log(d.rel_value)
    m = smf.ols("points ~ x", data=d).fit()
    d["fit"], d["res"] = m.fittedvalues, m.resid
    d = d.sort_values("season")
    order = np.argsort(d.res.to_numpy())
    theo = np.empty(len(d))
    theo[order] = stats.norm.ppf((np.arange(1, len(d) + 1) - 0.5) / len(d))
    d["theo"] = theo
    sd = d.res.std()
    seasons = sorted(d.season.unique())

    def traces(upto):
        cur = d[d.season <= upto]
        rm, oth = cur[cur.club_id == REAL_MADRID], cur[cur.club_id != REAL_MADRID]
        lab = lambda g: (g.name + " " + g.season.astype(str)).tolist()  # noqa: E731
        return [go.Scatter(x=oth.fit, y=oth.res, mode="markers", marker=dict(color=GREY, size=8, opacity=0.7), text=lab(oth),
                           name="club-season", hovertemplate="%{text}<br>fitted %{x:.0f}, residual %{y:+.1f}<extra></extra>"),
                go.Scatter(x=rm.fit, y=rm.res, mode="markers", marker=dict(color=GOLD, size=14, line=dict(color=INK, width=1)),
                           text=lab(rm), name="Real Madrid", hovertemplate="%{text}<br>fitted %{x:.0f}, residual %{y:+.1f}<extra></extra>"),
                go.Scatter(x=oth.theo, y=oth.res, mode="markers", marker=dict(color=GREY, size=8, opacity=0.7), text=lab(oth),
                           showlegend=False, hovertemplate="%{text}<br>normal quantile %{x:.2f}<extra></extra>"),
                go.Scatter(x=rm.theo, y=rm.res, mode="markers", marker=dict(color=GOLD, size=14, line=dict(color=INK, width=1)),
                           text=lab(rm), showlegend=False, hovertemplate="%{text}<br>normal quantile %{x:.2f}<extra></extra>")]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residuals vs fitted: no curve, no funnel", "Normal Q-Q: points near the line"),
                        horizontal_spacing=0.09)
    for k, t in enumerate(traces(seasons[-1])):
        fig.add_trace(t, 1, 1 + k // 2)
    fig.add_hline(y=0, line=dict(color=INK2, width=1), row=1, col=1)
    q = np.array([d.theo.min(), d.theo.max()])
    fig.add_trace(go.Scatter(x=q, y=q * sd, mode="lines", line=dict(color=NAVY, width=2), name="normal line", hoverinfo="skip"), 1, 2)
    fig.frames = [go.Frame(data=traces(s), name=str(s), traces=[0, 1, 2, 3]) for s in seasons]
    fig.update_xaxes(title_text="fitted points", row=1, col=1)
    fig.update_xaxes(title_text="normal quantile", row=1, col=2)
    fig.update_yaxes(title_text="residual (points)", row=1, col=1)
    fig.update_layout(updatemenus=_play(450))
    sw = stats.shapiro(d.res).pvalue
    return save(_layout(fig, f"R27  Part 1 regression checks: residual SD {sd:.1f} points, Shapiro-Wilk p = {sw:.2f}"),
                "R27_residual_checks")


# ---------------------------------------------------------------- R28  peak age by position (Part 4)
def r28_peak_by_position():
    """Peak-value age and its 95% cluster-bootstrap interval for each position; ▶ the intervals open up."""
    pk = pd.read_csv(DATA_PROC / "p4_peak_age.csv").sort_values("peak_age")
    col = [GOLD if p == "Attack" else NAVY for p in pk.position]

    def traces(t):
        return [go.Scatter(x=pk.peak_age, y=pk.position, mode="markers+text", text=[f"{v:.1f}" for v in pk.peak_age],
                           textposition="top center", textfont=dict(color=INK, size=15),
                           marker=dict(color=col, size=20, line=dict(color=INK, width=1)),
                           error_x=dict(type="data", symmetric=False, array=(pk.ci_hi - pk.peak_age) * t,
                                        arrayminus=(pk.peak_age - pk.ci_lo) * t, color=INK2, thickness=3, width=10),
                           customdata=np.c_[pk.ci_lo, pk.ci_hi, pk.n],
                           hovertemplate="%{y}: peak %{x:.1f} [%{customdata[0]:.1f}, %{customdata[1]:.1f}]<br>"
                                         "%{customdata[2]:,} player-seasons<extra></extra>")]
    fig = go.Figure(data=traces(1), frames=[go.Frame(data=traces(t), name=f"{t:.2f}") for t in np.linspace(0, 1, 15)])
    fig.update_layout(updatemenus=_play(70), showlegend=False, height=560,
                      xaxis=dict(title="age at peak market value (years)", range=[pk.ci_lo.min() - 0.8, pk.ci_hi.max() + 0.8]),
                      yaxis=dict(title="", automargin=True))
    lo, hi = pk.iloc[0], pk.iloc[-1]
    return save(_layout(fig, f"R28  Market value peaks between {lo.peak_age:.1f} ({lo.position.lower()}) and {hi.peak_age:.1f} "
                             f"({hi.position.lower()})", height=560), "R28_peak_by_position")


def run():
    made = [r1_money_vs_points(), r2_player_posteriors(), r3_shrinkage_bubbles(), r4_clasico_3d(), r5_bayes_learning_3d(),
            r6_title_race_lln(), r7_live_clasico(), r8_value_surface(), r9_star_points(),
            r23_value_and_points(), r24_residuals(), r25_hit_rates(), r26_goal_timing(), r27_residual_checks(),
            r28_peak_by_position()]
    for f in made:
        print("  wrote", f.relative_to(FIG.parent))
    import json
    (DATA_PROC / "deck_numbers.json").write_text(json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                            for k, v in NUM.items()}, indent=2))
    print("  wrote data/processed/deck_numbers.json")
    return made


if __name__ == "__main__":
    run()
