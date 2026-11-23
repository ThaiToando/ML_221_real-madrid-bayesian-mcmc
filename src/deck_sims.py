"""Simulation lab for the final deck: 13 more interactive visuals, one per lecture-notes idea.

Plotly figures (play / slider / rotate) are saved like R1-R9 (HTML + PNG twin). Three are small custom
widgets (several sliders at once, live recomputation in the browser): R14, R20, R21. Their PNG twin is a
browser screenshot. Everything takes its look from src/style.py.

R0  Matchday at the stadium: shots and goals arrive as a Poisson process (3D pitch, ▶)   notes 2.4, 5.5
R10 MCMC: watch the Metropolis walker explore Real Madrid's hit-rate posterior (▶)        notes 11A.3
R10b MCMC step size: too small crawls, too big never moves (slider)                       notes 11A.3
R11 MLE: the log-likelihood surface of the Gamma prior over (alpha, beta) (3D)            notes 6.2
R12 Permutation test: is "Madrid magic" chance? the null distribution builds live (▶)     notes 9.5, 9.11
R13 Galton board: 3,000 simulated seasons fall into a bell (▶)                            notes 8.3
R14 Sporting director: sliders -> P(signing becomes a regular) + ROC curve (widget)       notes 11
R15 Competing exponentials: who scores first? (▶)                                          notes 3.9
R16 Poisson splitting: given n goals, how many are Real Madrid's? (slider)                 notes 5.4
R17 Weibull hazard: drag the shape k (slider)                                              notes 11A.5
R18 Bootstrap: resample players, the peak-age interval appears (▶)                         notes 6.4
R19 Coverage: 100 samples, ~95 intervals catch the truth (▶)                               notes 7.4
R20 Bayes' rule tree for one good first season (widget)                                    notes 1.4
R21 Leverage: move Real Madrid and watch the regression line (widget)                      notes 10.10
R22 The squad: profile cards for Real Madrid's attackers, click one for its radar and posterior (widget)  notes 7.3
Run: python -m src.deck_sims    (after python run_all.py and python -m src.deck_visuals)
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from plotly.subplots import make_subplots
from scipy import optimize, stats
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, roc_curve

import config as cfg
from config import BIG_CLUBS, DATA_PROC, FIG, REAL_MADRID, SEED
from src.data_io import load
from src.style import css_family
from src.deck_visuals import OUT, _layout, _play, _season, save, strip_codes
from src.part3_player_output import neg_loglik
from src.part4_value_curve import cluster_bootstrap_peak, peak_age, prep
from src.part5_season_sim import fit_poisson, match_probs, rates
from src.style import BG, DECK, DOWN, GOLD, GREY, INK, INK2, LIGHT, MUTED, NAVY, SEQ, UP

rng = np.random.default_rng(SEED)


def _lay(fig, title, height=620):
    """deck_visuals._layout plus room under the x axis so its title never sits on the tick labels."""
    fig = _layout(fig, title, height)
    if not any(t.type in ("surface", "scatter3d") for t in fig.data):
        fig.update_xaxes(automargin=True, title_standoff=18)
        fig.update_layout(margin=dict(b=95))
    return fig


def _clasico_rates():
    gs, names = _season()
    m = fit_poisson(gs)
    lh, la = (r[0] for r in rates(m, pd.DataFrame(dict(home_club_id=[REAL_MADRID], away_club_id=[131]))))
    return gs, m, lh, la


# ---------------------------------------------------------------- R0 stadium
def r0_matchday(seed=SEED):
    """A generic stadium bowl around a pitch; shots arrive as a Poisson process and each shot is a goal with
    probability lambda/shots (Poisson thinning), so goals are Poisson(lambda) too."""
    gs, m, lh, la = _clasico_rates()
    r = np.random.default_rng(seed)
    shots_rate = 14.0                                            # shots per team per match (illustrative)
    ev = []
    for team, lam, goal_x in (("Real Madrid", lh, 52.5), ("Barcelona", la, -52.5)):
        n = r.poisson(shots_rate)
        for t in np.sort(r.uniform(0, 90, n)):
            is_goal = r.random() < lam / shots_rate
            x = goal_x - np.sign(goal_x) * r.uniform(5, 28)
            y = r.normal(0, 9 if not is_goal else 5)
            ev.append((t, team, x, y, is_goal))
    ev = pd.DataFrame(ev, columns=["t", "team", "x", "y", "goal"]).sort_values("t")
    # pitch, lines and a three-tier bowl
    th = np.linspace(0, 2 * np.pi, 120)
    bowl = []
    for k, (a, b, z0, z1) in enumerate(((70, 50, 0, 6), (82, 60, 6, 14), (95, 71, 14, 24))):
        rr = np.linspace(0, 1, 2)
        T, R = np.meshgrid(th, rr)
        X = (a + R * 10) * np.cos(T)
        Y = (b + R * 8) * np.sin(T)
        Z = z0 + R * (z1 - z0)
        bowl.append(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.9, hoverinfo="skip",
                               colorscale=[[0, "#151b24"], [1, "#2a3240" if k < 2 else "#3a4452"]]))
    pitch = go.Surface(x=np.array([[-52.5, 52.5], [-52.5, 52.5]]), y=np.array([[-34, -34], [34, 34]]), z=np.zeros((2, 2)),
                       showscale=False, colorscale=[[0, "#0e3b23"], [1, "#0e3b23"]], hoverinfo="skip")
    lines = []
    def seg(xs, ys):
        lines.append(go.Scatter3d(x=xs, y=ys, z=[0.05] * len(xs), mode="lines", line=dict(color="#c9d1c8", width=3),
                                  hoverinfo="skip", showlegend=False))
    seg([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34]); seg([0, 0], [-34, 34])
    seg(list(9.15 * np.cos(th)), list(9.15 * np.sin(th)))
    for s in (-1, 1):
        seg([s * 52.5, s * 36, s * 36, s * 52.5], [-20.2, -20.2, 20.2, 20.2])
        seg([s * 52.5, s * 52.5], [-3.66, 3.66])
    rim = go.Scatter3d(x=105 * np.cos(th), y=79 * np.sin(th), z=np.full_like(th, 24.5), mode="lines",
                       line=dict(color=GOLD, width=4), hoverinfo="skip", showlegend=False)
    static = bowl + [pitch, rim] + lines

    def dyn(t):
        cur = ev[ev.t <= t]
        out = []
        for team, col in (("Real Madrid", GOLD), ("Barcelona", NAVY)):
            sh = cur[(cur.team == team) & ~cur.goal]
            gl = cur[(cur.team == team) & cur.goal]
            out += [go.Scatter3d(x=sh.x, y=sh.y, z=np.full(len(sh), 0.6), mode="markers", name=f"{team} shot",
                                 marker=dict(size=4, color=col, opacity=0.55)),
                    go.Scatter3d(x=gl.x, y=gl.y, z=np.full(len(gl), 1.2), mode="markers", name=f"{team} goal",
                                 marker=dict(size=11, color=col, symbol="diamond", line=dict(color=INK, width=2)))]
        return out

    def score(t):
        cur = ev[(ev.t <= t) & ev.goal]
        return (cur.team == "Real Madrid").sum(), (cur.team == "Barcelona").sum()
    steps = list(range(0, 91, 2))
    frames = [go.Frame(data=dyn(t), traces=list(range(len(static), len(static) + 4)), name=str(t),
                       layout=dict(title_text=f"R0  {t:>2}'   REAL MADRID {score(t)[0]} – {score(t)[1]} BARCELONA"))
              for t in steps]
    fig = go.Figure(data=static + dyn(90), frames=frames)
    hs, as_ = score(90)
    fig.update_layout(updatemenus=_play(160), showlegend=False,
                      scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                                 aspectmode="data", camera=dict(eye=dict(x=0.0, y=-0.95, z=0.55), center=dict(x=0, y=0, z=-0.12))))
    NUM.update(r0_score=f"{hs}–{as_}", r0_lh=lh, r0_la=la)
    out = save(_lay(fig, f"R0  Matchday: REAL MADRID {hs} – {as_} BARCELONA. Goals ~ Pois({lh:.2f}) vs Pois({la:.2f})",
                       height=660), "R0_matchday")
    _orbit(out)
    return out


ORBIT_JS = """<script>
// walk-in camera: slow orbit around the bowl until the presenter touches the chart
(function () {
  const gd = document.querySelector('.plotly-graph-div'); if (!gd) return;
  let t0 = null, live = true, r = 0.95, z = 0.55;
  ['mousedown', 'wheel', 'touchstart'].forEach(e => gd.addEventListener(e, () => { live = false; }, {passive: true}));
  function step(ts) {
    if (!live) return; if (t0 === null) t0 = ts;
    const a = -Math.PI / 2 + 0.00012 * (ts - t0), zz = z + 0.25 * Math.max(0, 1 - (ts - t0) / 6000);
    Plotly.relayout(gd, {'scene.camera.eye': {x: r * Math.cos(a), y: r * Math.sin(a), z: zz}}).then(() => requestAnimationFrame(step));
  }
  gd.on ? gd.on('plotly_afterplot', () => {}) : 0; setTimeout(() => requestAnimationFrame(step), 600);
})();
</script>"""


def _orbit(html):
    txt = html.read_text()
    if "walk-in camera" not in txt:
        html.write_text(txt.replace("</body>", ORBIT_JS + "</body>"))


# ---------------------------------------------------------------- R10, R10b MCMC
def _rm_hits():
    sg = pd.read_csv(DATA_PROC / "signings.csv")
    rm = sg[sg.to_club_id == REAL_MADRID]
    return int(rm.success.sum()), len(rm)


def _log_post(h, n):
    return lambda p: stats.norm.logpdf(logit(p), -0.2, 0.5) - np.log(p * (1 - p)) + stats.binom.logpmf(h, n, p)


def _walk(lp, step, n_iter, start=0.6, seed=SEED):
    """Random-walk Metropolis on z = logit(p): same algorithm as bayes_compute.metropolis, full path kept."""
    r = np.random.default_rng(seed)
    target = lambda z: lp(expit(z)) + np.log(expit(z)) + np.log(1 - expit(z))  # noqa: E731
    z, cur = logit(start), target(logit(start))
    path, props, acc = [], [], []
    for _ in range(n_iter):
        zp = z + step * r.normal()
        new = target(zp)
        ok = np.log(r.random()) < new - cur
        props.append(expit(zp)); acc.append(ok)
        if ok:
            z, cur = zp, new
        path.append(expit(z))
    return np.array(path), np.array(props), np.array(acc)


def r10_mcmc_walk():
    h, n = _rm_hits()
    lp = _log_post(h, n)
    p = np.linspace(0.001, 0.999, 600)
    dens = np.exp(lp(p) - lp(p).max()); dens /= np.trapezoid(dens, p)
    path, props, acc = _walk(lp, 1.0, 600)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], vertical_spacing=0.08, shared_xaxes=False)
    fig.add_trace(go.Scatter(x=p, y=dens, mode="lines", line=dict(color=INK2, width=2), name="target posterior"), 1, 1)
    bins = np.linspace(0, 1, 51)

    def dyn(t):
        hist, _ = np.histogram(path[:t], bins=bins, density=True)
        col = UP if acc[t - 1] else DOWN
        return [go.Bar(x=(bins[:-1] + bins[1:]) / 2, y=hist, marker_color=GOLD, opacity=0.55, name="draws so far"),
                go.Scatter(x=[props[t - 1]], y=[0.15], mode="markers", marker=dict(size=14, color=col, symbol="x"),
                           name="proposal (green = accepted)"),
                go.Scatter(x=[path[t - 1]], y=[0.15], mode="markers", marker=dict(size=18, color=GOLD, line=dict(color=BG, width=2)),
                           name="walker"),
                go.Scatter(x=np.arange(1, t + 1), y=path[:t], mode="lines", line=dict(color=GOLD, width=1.5), name="trace")]
    stops = list(range(2, 41)) + list(range(45, 601, 15))
    frames = [go.Frame(data=dyn(t), traces=[1, 2, 3, 4], name=str(t),
                       layout=dict(title_text=f"R10  Metropolis step {t}: acceptance so far {acc[:t].mean():.0%}, "
                                              f"mean of draws {path[:t].mean():.3f}")) for t in stops]
    for i, tr in enumerate(dyn(600)):
        fig.add_trace(tr, 1 if i < 3 else 2, 1)
    fig.frames = frames
    fig.update_xaxes(title_text="Real Madrid's hit rate p", range=[0, 0.8], row=1, col=1)
    fig.update_yaxes(title_text="density", row=1, col=1)
    fig.update_xaxes(title_text="iteration", range=[0, 600], row=2, col=1)
    fig.update_yaxes(title_text="p", range=[0, 0.8], row=2, col=1)
    fig.update_layout(updatemenus=_play(60), barmode="overlay")
    NUM.update(mcmc_acc=float(acc.mean()), mcmc_mean=float(path[100:].mean()))
    return save(_lay(fig, f"R10  MCMC: a random walk that ends up spending time where the posterior is high ({h} hits in {n})",
                        height=680), "R10_mcmc_walk")


def r10b_mcmc_step_size():
    h, n = _rm_hits()
    lp = _log_post(h, n)
    p = np.linspace(0.001, 0.999, 600)
    dens = np.exp(lp(p) - lp(p).max()); dens /= np.trapezoid(dens, p)
    steps_ = [0.02, 0.2, 1.0, 3.0, 12.0]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.08)
    sliders = []
    for j, s in enumerate(steps_):
        path, _, acc = _walk(lp, s, 3000)
        vis = s == 1.0
        fig.add_trace(go.Scatter(x=np.arange(3000), y=path, mode="lines", line=dict(color=GOLD, width=1), visible=vis,
                                 name=f"trace, step {s}"), 1, 1)
        fig.add_trace(go.Histogram(x=path[300:], histnorm="probability density", nbinsx=50, marker_color=GOLD, opacity=0.6,
                                   visible=vis, name="draws"), 1, 2)
        sliders.append(dict(label=f"{s:g}", method="update",
                            args=[{"visible": [k // 2 == j for k in range(2 * len(steps_))] + [True]},
                                  {"title.text": f"R10b  step {s:g}: acceptance {acc.mean():.0%}. "
                                                 + ("Too small: it crawls" if s < 0.1 else "Too big: it rarely moves" if s > 5
                                                    else "About right: 20-50% accepted")}]))
    fig.add_trace(go.Scatter(x=p, y=dens, mode="lines", line=dict(color=INK, width=2), name="exact posterior"), 1, 2)
    fig.update_xaxes(title_text="iteration", row=1, col=1); fig.update_yaxes(title_text="p", range=[0, 0.8], row=1, col=1)
    fig.update_xaxes(title_text="p", range=[0, 0.8], row=1, col=2)
    fig.update_layout(sliders=[dict(active=2, steps=sliders, currentvalue=dict(prefix="Proposal step (logit scale): "),
                                    pad=dict(t=50))])
    return save(_lay(fig, "R10b  step 1: acceptance about right. Drag the step size", height=600), "R10b_mcmc_step_size")


# ---------------------------------------------------------------- R11 MLE surface
def r11_mle_surface():
    ps = pd.read_csv(DATA_PROC / "player_seasons.csv")
    att = ps[(ps.position == "Attack") & (ps.minutes >= 900)]
    k, t = att.ga.to_numpy(), (att.minutes / 90).to_numpy()
    res = optimize.minimize(neg_loglik, x0=np.log([2.0, 4.0]), args=(k, t), method="Nelder-Mead")
    a_hat, b_hat = np.exp(res.x)
    A = np.linspace(a_hat * 0.45, a_hat * 1.9, 45)
    B = np.linspace(b_hat * 0.45, b_hat * 1.9, 45)
    AA, BB = np.meshgrid(A, B)
    L = np.vectorize(lambda a, b: -neg_loglik(np.log([a, b]), k, t))(AA, BB)
    L = L - L.max()
    Lc = np.maximum(L, -60)
    fig = go.Figure(go.Surface(x=AA, y=BB, z=Lc, colorscale=SEQ, showscale=False, opacity=0.95,
                               contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=2)),
                               hovertemplate="α %{x:.2f}<br>β %{y:.2f}<br>log L − max %{z:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[a_hat], y=[b_hat], z=[0], mode="markers+text", text=[f"MLE α={a_hat:.2f}, β={b_hat:.2f}"],
                               textposition="top center", textfont=dict(color=INK), marker=dict(size=8, color=DOWN), name="MLE"))
    ridge_a = A
    fig.add_trace(go.Scatter3d(x=ridge_a, y=ridge_a / (a_hat / b_hat), z=[max(-60, -neg_loglik(np.log([a, a / (a_hat / b_hat)]), k, t)
                                                                           + res.fun) for a in ridge_a],
                               mode="lines", line=dict(color=GOLD, width=5), name="same prior mean α/β"))
    fig.update_layout(scene=dict(xaxis_title="α (shape)", yaxis_title="β (rate)", zaxis_title="log-likelihood − max"))
    NUM.update(mle_a=a_hat, mle_b=b_hat)
    return save(_lay(fig, f"R11  MLE: the data pick the prior where the likelihood peaks, Gamma({a_hat:.2f}, {b_hat:.2f})"),
                "R11_mle_surface")


# ---------------------------------------------------------------- R12 permutation test
def r12_permutation_test(B=10_000):
    """Lecture notes 9.11/9.13: shuffle the club labels 10,000 times; two-sided p = share of shuffles with
    |d*| >= |d|, i.e. np.mean(np.abs(perm) >= abs(obs)). If no shuffle reaches it, report p < 1/B."""
    ev = load("game_events")
    g = ev[ev.type.astype(str).str.lower().str.contains("goal") & ev.club_id.isin(BIG_CLUBS) & ev.minute.between(1, 90)]
    late = (g.minute > 75).to_numpy()
    is_rm = (g.club_id == REAL_MADRID).to_numpy()
    obs = late[is_rm].mean() - late[~is_rm].mean()
    r = np.random.default_rng(SEED)
    null = np.empty(B)
    for b in range(B):
        lab = r.permutation(is_rm)
        null[b] = late[lab].mean() - late[~lab].mean()
    edges = np.linspace(min(null.min(), -abs(obs)) * 1.1, max(null.max(), obs) * 1.15, 60)
    stops = np.unique(np.geomspace(20, B, 45).astype(int))

    def dyn(k):
        hist, _ = np.histogram(null[:k], bins=edges)
        return [go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=hist, marker_color=NAVY, opacity=0.75, name="shuffled labels")]
    frames = [go.Frame(data=dyn(k), traces=[0], name=str(k),
                       layout=dict(title_text=f"R12  {k:,} shuffles: {np.sum(np.abs(null[:k]) >= abs(obs)):,} as extreme as "
                                              f"Real Madrid's  →  p ≈ {np.mean(np.abs(null[:k]) >= abs(obs)):.4f}"))
              for k in stops]
    fig = go.Figure(data=dyn(B), frames=frames)
    fig.add_vline(x=obs, line=dict(color=GOLD, width=4), annotation_text=f"Real Madrid: {obs:+.3f}", annotation_position="top left",
                  annotation_font=dict(color=GOLD, size=15))
    fig.add_vline(x=-obs, line=dict(color=GOLD, width=1, dash="dot"))
    pval = np.mean(np.abs(null) >= abs(obs))                     # notes 9.13: share of shuffles at least as extreme
    fig.update_layout(updatemenus=_play(120), xaxis=dict(title="late share (Real Madrid) − late share (other big clubs)",
                                                         range=[edges[0], edges[-1]]),
                      yaxis_title="shuffles", bargap=0.02)
    NUM.update(perm_obs=float(obs), perm_p=float(pval), perm_B=B)
    ptxt = f"p ≈ {pval:.4f}" if pval > 0 else f"p < {1 / B:.4f}"   # notes 9.14: never write p = 0
    return save(_lay(fig, f"R12  If 'Madrid magic' were luck, shuffled labels would often look like this. {ptxt}"),
                "R12_permutation_test")


# ---------------------------------------------------------------- R13 Galton board
def r13_galton_board(n=3000):
    gs, m, _, _ = _clasico_rates()
    lh, la = rates(m, gs)
    ph, pdr, pa = match_probs(lh, la)
    home = (gs.home_club_id == REAL_MADRID).to_numpy()
    away = (gs.away_club_id == REAL_MADRID).to_numpy()
    pw = np.r_[ph[home], pa[away]]; pd_ = np.r_[pdr[home], pdr[away]]
    order = np.argsort(np.r_[gs.date[home].to_numpy(), gs.date[away].to_numpy()])
    pw, pd_ = pw[order], pd_[order]
    r = np.random.default_rng(SEED)
    u = r.random((n, len(pw)))
    pts = np.where(u < pw, 3, np.where(u < pw + pd_, 1, 0))
    cum = np.c_[np.zeros(n), np.cumsum(pts, axis=1)]
    final = cum[:, -1]
    mu = np.sum(3 * pw + pd_); var = np.sum(9 * pw + pd_ - (3 * pw + pd_) ** 2)
    xs = np.arange(final.min() - 5, final.max() + 6)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32], shared_yaxes=True, horizontal_spacing=0.02)
    stops = list(range(1, 30)) + list(np.unique(np.geomspace(30, n, 40).astype(int)))

    def dyn(k):
        show = cum[max(0, k - 25):k]
        xs_l, ys_l = [], []
        for row in show:
            xs_l += list(range(len(row))) + [None]; ys_l += list(row) + [None]
        cnt = np.bincount(final[:k].astype(int), minlength=int(final.max()) + 1)
        return [go.Scatter(x=xs_l, y=ys_l, mode="lines", line=dict(color=GOLD, width=1), opacity=0.5, name="seasons (last 25)"),
                go.Bar(y=np.arange(len(cnt)), x=cnt, orientation="h", marker_color=GOLD, opacity=0.8, name="final points")]
    fig.add_trace(dyn(n)[0], 1, 1); fig.add_trace(dyn(n)[1], 1, 2)
    norm_x = stats.norm.pdf(xs, mu, np.sqrt(var)) * n
    fig.add_trace(go.Scatter(x=norm_x, y=xs, mode="lines", line=dict(color=INK, width=2), name=f"N({mu:.1f}, {np.sqrt(var):.1f}²)"), 1, 2)
    fig.frames = [go.Frame(data=dyn(k), traces=[0, 1], name=str(k),
                           layout=dict(title_text=f"R13  {k:,} seasons dropped: mean {final[:k].mean():.1f} points")) for k in stops]
    fig.update_xaxes(title_text="match", range=[0, len(pw)], row=1, col=1)
    fig.update_yaxes(title_text="Real Madrid points", row=1, col=1)
    fig.update_xaxes(title_text="seasons", row=1, col=2)
    fig.update_layout(updatemenus=_play(90), bargap=0)
    NUM.update(clt_mu=float(mu), clt_sd=float(np.sqrt(var)), clt_pw=float(pw.mean()), clt_pd=float(pd_.mean()), clt_n=int(len(pw)))
    return save(_lay(fig, f"R13  Galton board: 38 matches of 0/1/3 points add up to a bell, N({mu:.1f}, {np.sqrt(var):.1f}²)",
                        height=640), "R13_galton_board")


# ---------------------------------------------------------------- R15, R16
def r15_competing_exponentials(n=500):
    gs, m, lh, la = _clasico_rates()
    r = np.random.default_rng(SEED)
    t_rm = r.exponential(90 / lh, n); t_b = r.exponential(90 / la, n)
    first = t_rm < t_b
    theory = lh / (lh + la)
    stops = list(range(5, n + 1, 5))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 300, 300], y=[0, 0, 300], fill="toself", fillcolor="rgba(254,190,16,0.08)",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[0, 300], y=[0, 300], mode="lines", line=dict(color=MUTED, dash="dot"), showlegend=False))

    def dyn(k):
        return [go.Scatter(x=t_b[:k][first[:k]], y=t_rm[:k][first[:k]], mode="markers", marker=dict(color=GOLD, size=7),
                           name="Real Madrid first"),
                go.Scatter(x=t_b[:k][~first[:k]], y=t_rm[:k][~first[:k]], mode="markers", marker=dict(color=NAVY, size=7),
                           name="Barcelona first")]
    for tr in dyn(n):
        fig.add_trace(tr)
    fig.frames = [go.Frame(data=dyn(k), traces=[2, 3], name=str(k),
                           layout=dict(title_text=f"R15  {k} matches: Real Madrid first {first[:k].mean():.1%}  vs  "
                                                  f"theory λh/(λh+λa) = {theory:.1%}")) for k in stops]
    fig.update_layout(updatemenus=_play(80), xaxis=dict(title="minutes until Barcelona's first goal", range=[0, 300]),
                      yaxis=dict(title="minutes until Real Madrid's first goal", range=[0, 300], scaleanchor="x", constrain="domain"))
    fig.update_xaxes(constrain="domain")
    NUM.update(first_theory=float(theory))
    return save(_lay(fig, f"R15  Two exponential clocks race: P(Real Madrid first) = {lh:.2f}/({lh:.2f}+{la:.2f}) = {theory:.1%}",
                        height=680), "R15_competing_exponentials")


def r16_poisson_splitting():
    gs, m, lh, la = _clasico_rates()
    q = lh / (lh + la)
    fig = go.Figure()
    steps_ = []
    ns = list(range(1, 7))
    for j, n in enumerate(ns):
        k = np.arange(n + 1)
        fig.add_trace(go.Bar(x=k, y=stats.binom.pmf(k, n, q), marker_color=[GOLD if kk > n - kk else LIGHT if kk == n - kk else NAVY
                                                                          for kk in k],
                             text=[f"{v:.0%}" for v in stats.binom.pmf(k, n, q)], textposition="outside",
                             visible=n == 3, name=f"n = {n}"))
        steps_.append(dict(label=str(n), method="update", args=[{"visible": [i == j for i in range(len(ns))]},
                       {"title.text": f"R16  Given {n} goals: P(all Real Madrid's) = {q ** n:.1%}, "
                                      f"P(Real Madrid score more) = {stats.binom.sf(n // 2, n, q) if n % 2 else stats.binom.sf(n / 2, n, q):.1%}"}]))
    fig.update_layout(sliders=[dict(active=2, steps=steps_, currentvalue=dict(prefix="Total goals in the match: "), pad=dict(t=50))],
                      xaxis=dict(title="goals scored by Real Madrid", dtick=1), yaxis=dict(title="probability", tickformat=".0%"))
    return save(_lay(fig, f"R16  Given 3 goals: each is Real Madrid's with p = λh/(λh+λa) = {q:.2f}  →  Binomial(3, {q:.2f})"),
                "R16_poisson_splitting")


# ---------------------------------------------------------------- R17 Weibull
def r17_weibull_hazard():
    wb = pd.read_csv(DATA_PROC / "p7_weibull_by_club.csv").set_index("club")
    k_rm, lam = wb.loc["Real Madrid", "k"], wb.loc["Real Madrid", "scale"]
    t = np.linspace(0.5, 90, 300)
    ks = np.round(np.arange(0.6, 1.61, 0.05), 2)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["hazard: chance of the first goal next minute", "P(no goal yet by minute t)"])
    steps_ = []
    for j, k in enumerate(ks):
        hz = (k / lam) * (t / lam) ** (k - 1) * 90
        S = np.exp(-(t / lam) ** k)
        vis = abs(k - 1.10) < 1e-9 or (j == int(np.argmin(np.abs(ks - k_rm))))
        vis = j == int(np.argmin(np.abs(ks - k_rm)))
        fig.add_trace(go.Scatter(x=t, y=hz, mode="lines", line=dict(color=GOLD, width=4), visible=vis, name=f"k = {k}"), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=S, mode="lines", line=dict(color=GOLD, width=4), visible=vis, showlegend=False), 1, 2)
        shape = "rising" if k > 1.001 else "falling" if k < 0.999 else "constant (Exponential)"
        steps_.append(dict(label=f"{k:.2f}", method="update",
                           args=[{"visible": [i // 2 == j for i in range(2 * len(ks))] + [True, True]},
                                 {"title.text": f"R17  shape k = {k:.2f}: hazard {shape}; median wait {lam * np.log(2) ** (1 / k):.0f} min"}]))
    fig.add_trace(go.Scatter(x=t, y=np.full_like(t, 90 / lam), mode="lines", line=dict(color=MUTED, dash="dot"),
                             name="k = 1 (constant rate)"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=np.exp(-t / lam), mode="lines", line=dict(color=MUTED, dash="dot"), showlegend=False), 1, 2)
    fig.update_xaxes(title_text="minute"); fig.update_yaxes(title_text="per 90 minutes", row=1, col=1)
    fig.update_yaxes(title_text="probability", tickformat=".0%", row=1, col=2)
    j0 = int(np.argmin(np.abs(ks - k_rm)))
    fig.update_layout(sliders=[dict(active=j0, steps=steps_, currentvalue=dict(prefix=f"Shape k (Real Madrid's fit: {k_rm:.2f}): "),
                                    pad=dict(t=50))])
    return save(_lay(fig, f"R17  Real Madrid's fitted shape k = {k_rm:.2f}: the scoring hazard rises slightly. Drag k"),
                "R17_weibull_hazard")


# ---------------------------------------------------------------- R18 bootstrap
def r18_bootstrap(B=300):
    d = prep()
    att = d[d.position == "Attack"]
    pk, _ = peak_age(att)
    boots = cluster_bootstrap_peak(att, B=B)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    edges = np.linspace(boots.min() - 0.05, boots.max() + 0.05, 40)
    stops = list(range(5, B + 1, 5))

    def dyn(k):
        hist, _ = np.histogram(boots[:k], bins=edges)
        return [go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=hist, marker_color=GOLD, opacity=0.8, name="bootstrap peaks")]
    fig = go.Figure(data=dyn(B), frames=[go.Frame(data=dyn(k), traces=[0], name=str(k),
                    layout=dict(title_text=f"R18  {k} resamples of players: 95% interval so far "
                                           f"[{np.percentile(boots[:k], 2.5):.2f}, {np.percentile(boots[:k], 97.5):.2f}]"))
                    for k in stops])
    for x, txt in ((lo, f"2.5%: {lo:.2f}"), (hi, f"97.5%: {hi:.2f}")):
        fig.add_vline(x=x, line=dict(color=INK, dash="dash"), annotation_text=txt)
    fig.add_vline(x=pk, line=dict(color=DOWN, width=3), annotation_text=f"estimate {pk:.2f}")
    fig.update_layout(updatemenus=_play(70), xaxis_title="peak age of attackers (years)", yaxis_title="resamples", bargap=0.03)
    return save(_lay(fig, f"R18  Resample players, refit, repeat: attackers peak at {pk:.1f} (95% CI {lo:.1f}–{hi:.1f})"),
                "R18_bootstrap")


# ---------------------------------------------------------------- R19 coverage
def r19_coverage(n_samples=100):
    hr = pd.read_csv(DATA_PROC / "p2_hit_rates.csv").set_index("club")
    rm = hr.loc["Real Madrid"]
    a0, b0 = rm.post_a - rm.hits, rm.post_b - (rm.n - rm.hits)
    p_true, n = float(rm.post_mean), int(rm.n)
    r = np.random.default_rng(SEED)
    h = r.binomial(n, p_true, n_samples)
    ph = h / n
    se = np.sqrt(ph * (1 - ph) / n)
    wald = np.c_[ph - 1.96 * se, ph + 1.96 * se]
    cred = np.c_[stats.beta.ppf(0.025, a0 + h, b0 + n - h), stats.beta.ppf(0.975, a0 + h, b0 + n - h)]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=["Wald interval: p̂ ± 1.96·SE", "Beta credible interval"])

    def dyn(k):
        out = []
        for col, iv in ((1, wald), (2, cred)):
            hit = (iv[:k, 0] <= p_true) & (p_true <= iv[:k, 1])
            for ok, color in ((True, GOLD), (False, DOWN)):
                xs, ys = [], []
                for i in np.where(hit == ok)[0]:
                    xs += [iv[i, 0], iv[i, 1], None]; ys += [i, i, None]
                out.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=3), showlegend=False))
        return out
    for i, tr in enumerate(dyn(n_samples)):
        fig.add_trace(tr, 1, 1 + i // 2)
    for c in (1, 2):
        fig.add_vline(x=p_true, line=dict(color=INK, width=2), row=1, col=c)

    def cov(k):
        w = np.mean((wald[:k, 0] <= p_true) & (p_true <= wald[:k, 1]))
        c = np.mean((cred[:k, 0] <= p_true) & (p_true <= cred[:k, 1]))
        return w, c
    fig.frames = [go.Frame(data=dyn(k), traces=[0, 1, 2, 3], name=str(k),
                           layout=dict(title_text=f"R19  {k} samples of {n} signings: Wald covers {cov(k)[0]:.0%}, "
                                                  f"credible covers {cov(k)[1]:.0%} (red = missed the true {p_true:.2f})"))
                  for k in range(2, n_samples + 1, 2)]
    fig.update_xaxes(title_text="hit rate", range=[-0.15, 0.85])
    fig.update_yaxes(title_text="sample", range=[-1, n_samples], row=1, col=1)
    fig.update_layout(updatemenus=_play(70))
    w, c = cov(n_samples)
    NUM.update(cov_wald=float(w), cov_cred=float(c))
    return save(_lay(fig, f"R19  100 repeats with n = {n}: Wald caught the truth {w:.0%} of the time, the credible interval {c:.0%}",
                        height=680), "R19_coverage")


# ---------------------------------------------------------------- custom widgets (R14, R20, R21)
def _widget(name, title, controls_html, script, data):
    D = DECK
    mono = css_family(D["mono"])
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<script src="plotly.min.js"></script>
<style>
 body {{ margin: 0; background: {D['bg']}; color: {D['ink']}; font-family: {mono}; }}
 .wrap {{ display: flex; gap: 18px; padding: 14px 16px; height: calc(100vh - 28px); box-sizing: border-box; }}
 .panel {{ width: 300px; flex: 0 0 300px; background: {D['panel']}; border: 1px solid {D['grid']}; padding: 14px; }}
 .main {{ flex: 1; display: flex; flex-direction: column; min-width: 0; }}
 h1 {{ font-size: 18px; color: {D['accent']}; margin: 0 0 8px; font-weight: 700; }}
 label {{ display: block; font-size: 12px; color: {D['ink2']}; margin: 14px 0 4px; text-transform: uppercase; letter-spacing: .06em; }}
 input[type=range] {{ width: 100%; accent-color: {D['accent']}; }}
 select {{ width: 100%; background: {D['bg']}; color: {D['ink']}; border: 1px solid {D['grid']}; padding: 4px; font-family: {mono}; }}
 .val {{ float: right; color: {D['accent']}; }}
 .big {{ font-size: 44px; color: {D['accent']}; margin: 16px 0 2px; }}
 .small {{ font-size: 12px; color: {D['muted']}; line-height: 1.5; }}
 .chk {{ font-size: 13px; color: {D['ink2']}; margin-top: 12px; }}
 .src {{ font-size: 10px; color: {D['muted']}; text-align: right; }}
</style></head><body><div class="wrap"><div class="panel"><h1>{title}</h1>{controls_html}</div>
<div class="main"><div id="plot" style="flex:1"></div><div class="src">Data: Transfermarkt via transfermarkt-datasets; authors' calculations</div></div></div>
<script>
const DATA = {json.dumps(data)};
const C = {json.dumps(dict(bg=D['bg'], panel=D['panel'], grid=D['grid'], ink=D['ink'], ink2=D['ink2'], muted=D['muted'],
                           gold=GOLD, navy=NAVY, grey=GREY, down=DOWN, up=UP, mono=mono))};
const LAYOUT = {{paper_bgcolor: C.bg, plot_bgcolor: C.panel, font: {{family: C.mono, color: C.ink, size: 13}},
  margin: {{l: 60, r: 20, t: 40, b: 50}}, showlegend: true, legend: {{orientation: 'h', y: -0.15, font: {{color: C.ink2}}}},
  xaxis: {{gridcolor: C.grid, zerolinecolor: C.grid}}, yaxis: {{gridcolor: C.grid, zerolinecolor: C.grid}}}};
const $ = id => document.getElementById(id);
{script}
</script></body></html>"""
    html = OUT / f"{name}.html"
    html.write_text(strip_codes(page), encoding="utf-8")
    _screenshot(html, FIG / f"{name}.png")
    return html


def _screenshot(html, png):
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1600, "height": 900})
            pg.goto(html.as_uri(), wait_until="networkidle")
            pg.wait_for_timeout(800)
            pg.screenshot(path=str(png))
            b.close()
    except Exception as e:
        print(f"  (no PNG for {html.stem}: {e.__class__.__name__}; pip install playwright && playwright install chromium)")


def r14_sporting_director():
    sg = pd.read_csv(DATA_PROC / "signings.csv")
    d = sg.dropna(subset=["age", "log_fee"]).copy()
    d["age_c"] = d.age - 24
    m = smf.logit("success ~ age_c + I(age_c**2) + log_fee + C(position) + from_top5 + is_real_madrid", data=d).fit(disp=0)
    names = list(m.params.index)
    p_hat = m.predict(d)
    fpr, tpr, _ = roc_curve(d.success, p_hat)
    auc = roc_auc_score(d.success, p_hat)
    positions = ["Attack"] + [n.split("[T.")[1].rstrip("]") for n in names if n.startswith("C(position)")]
    data = dict(names=names, beta=m.params.tolist(), cov=m.cov_params().values.tolist(), positions=positions,
                fpr=fpr.tolist(), tpr=tpr.tolist(), auc=float(auc))
    controls = """
<label>Age <span class="val" id="ageV"></span></label><input id="age" type="range" min="17" max="34" step="1" value="21">
<label>Fee (EUR M) <span class="val" id="feeV"></span></label><input id="fee" type="range" min="0" max="5.3" step="0.05" value="3.69">
<label>Position</label><select id="pos"></select>
<div class="chk"><input type="checkbox" id="top5"> from a top-5 league</div>
<div class="chk"><input type="checkbox" id="rm" checked> signed by Real Madrid</div>
<div class="big" id="prob"></div><div class="small" id="ci"></div>
<div class="small" style="margin-top:14px">logit p = β₀ + β₁(age−24) + β₂(age−24)² + β₃ ln(fee) + position + top-5 + Real Madrid.
The 95% interval is built on the logit scale, then transformed.</div>"""
    script = """
DATA.positions.forEach(p => { const o = document.createElement('option'); o.text = p; $('pos').add(o); });
function xvec(age, lf, pos, top5, rm) {
  return DATA.names.map(n => n === 'Intercept' ? 1 : n === 'age_c' ? age - 24 : n === 'I(age_c ** 2)' ? (age - 24) ** 2 :
    n === 'log_fee' ? lf : n === 'from_top5' ? top5 : n === 'is_real_madrid' ? rm :
    n.startsWith('C(position)') ? (n.includes('[T.' + pos + ']') ? 1 : 0) : 0);
}
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const sig = z => 1 / (1 + Math.exp(-z));
function pred(x) { const eta = dot(DATA.beta, x); const v = dot(x, DATA.cov.map(r => dot(r, x)));
  const se = Math.sqrt(v); return [sig(eta), sig(eta - 1.96 * se), sig(eta + 1.96 * se)]; }
function draw() {
  const age = +$('age').value, lf = +$('fee').value, pos = $('pos').value, t5 = $('top5').checked ? 1 : 0, rm = $('rm').checked ? 1 : 0;
  $('ageV').textContent = age; $('feeV').textContent = Math.exp(lf).toFixed(1);
  const [p, lo, hi] = pred(xvec(age, lf, pos, t5, rm));
  $('prob').textContent = (100 * p).toFixed(0) + '%'; $('ci').textContent = '95% CI ' + (100 * lo).toFixed(0) + '% – ' + (100 * hi).toFixed(0) + '%  ·  P(becomes a regular)';
  const fees = [...Array(80).keys()].map(i => i * 5.3 / 79);
  const curve = fees.map(f => pred(xvec(age, f, pos, t5, rm)));
  const tr = [
    {x: fees.map(Math.exp), y: curve.map(c => c[2]), mode: 'lines', line: {width: 0}, showlegend: false, hoverinfo: 'skip', xaxis: 'x', yaxis: 'y'},
    {x: fees.map(Math.exp), y: curve.map(c => c[1]), mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(254,190,16,0.15)', line: {width: 0}, name: '95% CI', xaxis: 'x', yaxis: 'y'},
    {x: fees.map(Math.exp), y: curve.map(c => c[0]), mode: 'lines', line: {color: C.gold, width: 3}, name: 'P(success) vs fee', xaxis: 'x', yaxis: 'y'},
    {x: [Math.exp(lf)], y: [p], mode: 'markers', marker: {size: 16, color: C.gold, line: {color: C.bg, width: 2}}, name: 'this signing', xaxis: 'x', yaxis: 'y'},
    {x: DATA.fpr, y: DATA.tpr, mode: 'lines', line: {color: C.navy, width: 3}, name: 'ROC (AUC ' + DATA.auc.toFixed(2) + ')', xaxis: 'x2', yaxis: 'y2'},
    {x: [0, 1], y: [0, 1], mode: 'lines', line: {color: C.muted, dash: 'dot'}, name: 'coin flip', xaxis: 'x2', yaxis: 'y2'}];
  const L = JSON.parse(JSON.stringify(LAYOUT));
  Object.assign(L, {grid: {rows: 1, columns: 2, pattern: 'independent'},
    xaxis: {type: 'log', title: 'fee (EUR M, log scale)', gridcolor: C.grid, domain: [0, 0.6]},
    yaxis: {title: 'P(becomes a regular)', range: [0, 1], tickformat: '.0%', gridcolor: C.grid},
    xaxis2: {title: 'false positive rate', range: [0, 1], gridcolor: C.grid, domain: [0.7, 1]},
    yaxis2: {title: 'true positive rate', range: [0, 1], gridcolor: C.grid, anchor: 'x2'},
    title: {text: 'Logistic model: P = ' + (100 * p).toFixed(0) + '%  ·  AUC ' + DATA.auc.toFixed(2), font: {color: C.ink, size: 15}, x: 0.01}});
  Plotly.react('plot', tr, L, {displaylogo: false, responsive: true});
}
['age', 'fee', 'pos', 'top5', 'rm'].forEach(id => $(id).addEventListener('input', draw)); draw();"""
    NUM.update(auc=float(auc))
    return _widget("R14_sporting_director", "R14 · You're the sporting director", controls, script, data)


def r20_bayes_tree():
    controls = """
<label>Prior: P(regular) <span class="val" id="pV"></span></label><input id="p" type="range" min="0.05" max="0.95" step="0.01" value="0.45">
<label>P(starts 8 of first 10 | regular) <span class="val" id="aV"></span></label><input id="a" type="range" min="0.05" max="0.95" step="0.01" value="0.60">
<label>P(starts 8 of first 10 | flop) <span class="val" id="bV"></span></label><input id="b" type="range" min="0.01" max="0.95" step="0.01" value="0.15">
<div class="big" id="post"></div><div class="small">P(regular | he started 8 of his first 10)</div>
<div class="small" style="margin-top:14px">Lecture notes 1.4. Set the prior to 0.10 and the same evidence gives only 31%: base rates matter.</div>"""
    script = """
function draw() {
  const p = +$('p').value, a = +$('a').value, b = +$('b').value;
  $('pV').textContent = p.toFixed(2); $('aV').textContent = a.toFixed(2); $('bV').textContent = b.toFixed(2);
  const j = [p * a, p * (1 - a), (1 - p) * b, (1 - p) * (1 - b)];
  const post = j[0] / (j[0] + j[2]);
  $('post').textContent = (100 * post).toFixed(1) + '%';
  const nodes = {root: [0.05, 0.5], R: [0.4, 0.75], F: [0.4, 0.25], RS: [0.78, 0.88], RN: [0.78, 0.62], FS: [0.78, 0.38], FN: [0.78, 0.12]};
  const edges = [['root', 'R', p, 'regular ' + p.toFixed(2), C.gold], ['root', 'F', 1 - p, 'flop ' + (1 - p).toFixed(2), C.navy],
    ['R', 'RS', j[0], 'starts ' + a.toFixed(2), C.gold], ['R', 'RN', j[1], 'not ' + (1 - a).toFixed(2), C.grey],
    ['F', 'FS', j[2], 'starts ' + b.toFixed(2), C.navy], ['F', 'FN', j[3], 'not ' + (1 - b).toFixed(2), C.grey]];
  const tr = edges.map(([u, v, w, lab, col]) => ({x: [nodes[u][0], nodes[v][0]], y: [nodes[u][1], nodes[v][1]], mode: 'lines',
    line: {width: 2 + 30 * w, color: col}, hoverinfo: 'text', text: lab, showlegend: false}));
  const leaves = [['RS', j[0], 'regular AND starts', C.gold], ['RN', j[1], 'regular, not', C.grey], ['FS', j[2], 'flop AND starts', C.navy], ['FN', j[3], 'flop, not', C.grey]];
  tr.push({x: leaves.map(l => nodes[l[0]][0] + 0.02), y: leaves.map(l => nodes[l[0]][1]), mode: 'text', textposition: 'middle right',
    text: leaves.map(l => l[2] + '  ' + (100 * l[1]).toFixed(1) + '%'), textfont: {color: C.ink, size: 14}, showlegend: false});
  tr.push({x: edges.map(e => (nodes[e[0]][0] + nodes[e[1]][0]) / 2), y: edges.map(e => (nodes[e[0]][1] + nodes[e[1]][1]) / 2 + 0.035),
    mode: 'text', text: edges.map(e => e[3]), textfont: {color: C.ink2, size: 12}, showlegend: false});
  const L = JSON.parse(JSON.stringify(LAYOUT));
  Object.assign(L, {xaxis: {visible: false, range: [0, 1.15]}, yaxis: {visible: false, range: [0, 1]},
    title: {text: 'Bayes: ' + (100 * j[0]).toFixed(1) + '% ÷ (' + (100 * j[0]).toFixed(1) + '% + ' + (100 * j[2]).toFixed(1) + '%) = ' + (100 * post).toFixed(1) + '%',
            font: {color: C.gold, size: 18}, x: 0.01}});
  Plotly.react('plot', tr, L, {displaylogo: false, responsive: true});
}
['p', 'a', 'b'].forEach(id => $(id).addEventListener('input', draw)); draw();"""
    return _widget("R20_bayes_tree", "R20 · Bayes' rule: one good first season", controls, script, {})


def r21_leverage():
    cs = pd.read_csv(DATA_PROC / "club_seasons.csv")
    d = cs[(cs.competition_id == "ES1") & cs.rel_value.notna()].copy()
    data = dict(x=np.log(d.rel_value).round(4).tolist(), y=d.points.astype(float).tolist(),
                rm=(d.club_id == REAL_MADRID).astype(int).tolist())
    controls = """
<label>Shift Real Madrid's points <span class="val" id="dyV"></span></label><input id="dy" type="range" min="-50" max="30" step="1" value="0">
<label>Scale Real Madrid's squad value <span class="val" id="sxV"></span></label><input id="sx" type="range" min="-1.2" max="0.6" step="0.05" value="0">
<div class="chk"><input type="checkbox" id="drop"> drop Real Madrid completely</div>
<div class="big" id="slope"></div><div class="small" id="r2"></div>
<div class="small" style="margin-top:14px">Leverage (notes 10.10): points far from the mean of x pull the line hardest.
Real Madrid sits at the far right, so moving it tilts the whole fit.</div>"""
    script = """
function ols(x, y) { const n = x.length, mx = x.reduce((a, b) => a + b) / n, my = y.reduce((a, b) => a + b) / n;
  let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  const b = sxy / sxx; return {b, a: my - b * mx, r2: sxy * sxy / (sxx * syy)}; }
const base = ols(DATA.x, DATA.y);
function draw() {
  const dy = +$('dy').value, sx = +$('sx').value, drop = $('drop').checked;
  $('dyV').textContent = (dy > 0 ? '+' : '') + dy; $('sxV').textContent = '×' + Math.exp(sx).toFixed(2);
  const X = [], Y = [], rx = [], ry = [];
  DATA.x.forEach((x, i) => { if (DATA.rm[i]) { if (!drop) { X.push(x + sx); Y.push(DATA.y[i] + dy); rx.push(x + sx); ry.push(DATA.y[i] + dy); } }
                             else { X.push(x); Y.push(DATA.y[i]); } });
  const f = ols(X, Y), xs = [Math.min(...DATA.x) - 0.1, Math.max(...DATA.x) + 0.6];
  $('slope').textContent = 'β₁ = ' + f.b.toFixed(1); $('r2').textContent = 'R² ' + f.r2.toFixed(3) + '   (original β₁ ' + base.b.toFixed(1) + ', R² ' + base.r2.toFixed(3) + ')';
  const tr = [{x: DATA.x.filter((_, i) => !DATA.rm[i]), y: DATA.y.filter((_, i) => !DATA.rm[i]), mode: 'markers', marker: {color: C.grey, size: 7, opacity: 0.6}, name: 'other clubs'},
    {x: rx, y: ry, mode: 'markers', marker: {color: C.gold, size: 13, line: {color: C.bg, width: 2}}, name: 'Real Madrid (moved)'},
    {x: xs, y: xs.map(x => base.a + base.b * x), mode: 'lines', line: {color: C.muted, dash: 'dot', width: 2}, name: 'original line'},
    {x: xs, y: xs.map(x => f.a + f.b * x), mode: 'lines', line: {color: C.navy, width: 3}, name: 'refitted line'}];
  const L = JSON.parse(JSON.stringify(LAYOUT));
  Object.assign(L, {xaxis: {title: 'log(squad value ÷ league average)', gridcolor: C.grid}, yaxis: {title: 'points', gridcolor: C.grid},
    title: {text: 'Least squares refits live: β₁ ' + base.b.toFixed(1) + ' → ' + f.b.toFixed(1), font: {color: C.ink, size: 15}, x: 0.01}});
  Plotly.react('plot', tr, L, {displaylogo: false, responsive: true});
}
['dy', 'sx', 'drop'].forEach(id => $(id).addEventListener('input', draw)); draw();"""
    return _widget("R21_leverage", "R21 · Leverage: move Real Madrid", controls, script, data)


# ---------------------------------------------------------------- R22 player profile cards
PROFILE_FIELDS = ["sub_position", "country_of_citizenship", "foot", "height_in_cm"]   # used when the real table has them


def _player_photo(pid):
    """Optional: a photo the team added themselves (licensed) at figures/players/<player_id>.jpg or .png.
    Nothing is ever downloaded; without a file the card shows a plain silhouette."""
    for ext in ("jpg", "jpeg", "png", "webp"):
        f = FIG / "players" / f"{pid}.{ext}"
        if f.exists():
            dst = OUT / "players"
            dst.mkdir(exist_ok=True)
            import shutil
            shutil.copy(f, dst / f.name)
            return f"players/{f.name}"
    return None


def r22_player_cards(n_cards=6):
    att = pd.read_csv(DATA_PROC / "p3_rm_attackers.csv")
    chosen = [i for i in getattr(cfg, "PLAYER_CARDS", []) if i in set(att.player_id)]
    missing = sorted(set(getattr(cfg, "PLAYER_CARDS", [])) - set(chosen))
    if missing:
        print(f"  R22: player ids not among Real Madrid's attackers, skipped: {missing}")
    att = (att.set_index("player_id").loc[chosen].reset_index() if chosen
           else att.sort_values("t", ascending=False).head(n_cards))
    ps = pd.read_csv(DATA_PROC / "player_seasons.csv")
    raw = load("players")
    vals = load("player_valuations")
    # the comparison group for the radar: every attacker at the big clubs with at least 900 league minutes
    pool = ps[ps.position == "Attack"].groupby("player_id").agg(minutes=("minutes", "sum"), goals=("goals", "sum"),
                                                                  assists=("assists", "sum"), seasons=("season", "nunique"),
                                                                  peak_value=("value_eur", "max"))
    pool = pool[pool.minutes >= 900]
    pool["g90"], pool["a90"] = pool.goals / pool.minutes * 90, pool.assists / pool.minutes * 90
    pool["ga90"], pool["min_season"] = (pool.goals + pool.assists) / pool.minutes * 90, pool.minutes / pool.seasons
    axes = [("g90", "Goals / 90"), ("a90", "Assists / 90"), ("ga90", "G+A / 90"), ("min_season", "Min / season"),
            ("peak_value", "Peak value")]
    pct = {k: pool[k].rank(pct=True) * 100 for k, _ in axes}
    lam = np.linspace(0.001, max(att.hi.max() * 1.3, 0.5), 240)
    cards = []
    for r in att.itertuples():
        mine = ps[(ps.player_id == r.player_id) & (ps.club_id == REAL_MADRID)]
        info = raw[raw.player_id == r.player_id]
        info = info.iloc[0] if len(info) else pd.Series(dtype=object)
        v = vals[vals.player_id == r.player_id].dropna(subset=["market_value_in_eur"]).sort_values("date")
        dob = pd.to_datetime(info.get("date_of_birth"), errors="coerce")
        v_age = ((v.date - dob).dt.days / 365.25).round(2).tolist() if pd.notna(dob) else []
        extra = {f: (None if pd.isna(info.get(f)) else (int(info.get(f)) if f == "height_in_cm" else str(info.get(f))))
                 for f in PROFILE_FIELDS if f in info.index}
        cards.append(dict(
            id=int(r.player_id), name=str(r.name), photo=_player_photo(int(r.player_id)), position=str(mine.position.iloc[0]) if len(mine) else "Attack",
            **extra,
            age=float(mine.age.max()) if len(mine) else None, last_season=int(mine.season.max()) if len(mine) else None,
            seasons=int(mine.season.nunique()), minutes=int(mine.minutes.sum()), goals=int(mine.goals.sum()),
            assists=int(mine.assists.sum()), apps=int(mine.apps.sum()),
            post=float(r.post_mean), lo=float(r.lo), hi=float(r.hi), raw=float(r.raw_per90),
            fee=None if pd.isna(r.fee_m) else float(r.fee_m),
            peak_value=float(v.market_value_in_eur.max() / 1e6) if len(v) else None,
            radar=[float(pct[k].get(r.player_id, np.nan)) if r.player_id in pool.index else None for k, _ in axes],
            v_age=v_age, v_val=(v.market_value_in_eur / 1e6).round(2).tolist(),
            dens=stats.gamma.pdf(lam, r.post_a, scale=1 / r.post_b).round(4).tolist()))
    data = dict(cards=cards, axes=[a for _, a in axes], axes_short=["G/90", "A/90", "G+A/90", "Min/szn", "Value"], lam=lam.round(4).tolist(), xmax=float(lam.max()),
                prior=stats.gamma.pdf(lam, NUM.get("mle_a", 2.0), scale=1 / NUM.get("mle_b", 2.0)).round(4).tolist(),
                fake=bool(pd.Series([c["name"] for c in cards]).str.fullmatch(r"Player \d+").mean() > 0.5),
                pool_n=int(len(pool)))
    return _cards_page("R22_player_cards", data)


SILHOUETTE = ("<svg viewBox='0 0 64 64' width='64' height='64'><circle cx='32' cy='32' r='31' fill='#0a0d12' stroke='#FEBE10' "
              "stroke-width='1.5'/><circle cx='32' cy='25' r='11' fill='#3a4452'/><path d='M12 56c2-12 10-18 20-18s18 6 20 18' "
              "fill='#3a4452'/></svg>")


def _cards_page(name, data):
    D = DECK
    mono = css_family(D["mono"])
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>R22 · The squad</title>
<script src="plotly.min.js"></script>
<style>
 * {{ box-sizing: border-box; }}
 body {{ margin: 0; background: {D['bg']}; color: {D['ink']}; font-family: {mono}; overflow: hidden; }}
 header {{ display: flex; justify-content: space-between; align-items: baseline; padding: 12px 18px 6px; }}
 h1 {{ font-size: 17px; color: {GOLD}; margin: 0; font-weight: 700; letter-spacing: .02em; }}
 .hint {{ font-size: 12px; color: {D['muted']}; }}
 .fake {{ background: #c62828; color: #fff; font-weight: 700; font-size: 12px; padding: 2px 8px; letter-spacing: .08em; }}
 .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); grid-template-rows: repeat(2, minmax(0, 1fr)); gap: 12px;
         padding: 6px 18px 0; height: calc(100vh - 70px); }}
 .card {{ background: {D['panel']}; border: 1px solid {D['grid']}; border-top: 3px solid {GOLD}; padding: 12px 14px; cursor: pointer;
         display: grid; grid-template-rows: auto auto auto minmax(0, 1fr); gap: 8px; min-height: 0; overflow: hidden;
         transition: border-color .15s, transform .15s; }}
 .card:hover {{ border-color: {GOLD}; transform: translateY(-2px); }}
 .top {{ display: flex; gap: 12px; align-items: center; }}
 .ph {{ width: 64px; height: 64px; flex: 0 0 64px; border-radius: 50%; overflow: hidden; }}
 .ph img {{ width: 100%; height: 100%; object-fit: cover; }}
 .nm {{ font-size: 17px; font-weight: 700; color: {D['ink']}; line-height: 1.15; }}
 .sub {{ font-size: 11.5px; color: {D['ink2']}; margin-top: 3px; line-height: 1.4; }}
 .big {{ margin-left: auto; text-align: right; }}
 .big b {{ display: block; font-size: 30px; color: {GOLD}; line-height: 1; }}
 .big span {{ font-size: 10.5px; color: {D['muted']}; text-transform: uppercase; letter-spacing: .06em; }}
 .bar {{ position: relative; height: 16px; background: {D['bg']}; border: 1px solid {D['grid']}; }}
 .bar .ci {{ position: absolute; top: 3px; bottom: 3px; background: rgba(254,190,16,.35); }}
 .bar .pm {{ position: absolute; top: 0; bottom: 0; width: 3px; background: {GOLD}; }}
 .bar .rw {{ position: absolute; top: 0; bottom: 0; width: 2px; background: {NAVY}; }}
 .stats {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; font-size: 10.5px; color: {D['muted']}; text-transform: uppercase; }}
 .stats b {{ display: block; font-size: 15px; color: {D['ink']}; text-transform: none; }}
 .charts {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 6px; min-height: 0; height: 100%; }}
 .charts > div {{ min-height: 0; height: 100%; }}
 #mcharts > div {{ height: 100%; min-height: 0; }}
 #modal {{ position: fixed; inset: 0; background: rgba(5,7,10,.985); display: none; padding: 26px 30px; }}
 #modal.on {{ display: block; }}
 #modal .mh {{ display: flex; align-items: center; gap: 16px; margin-bottom: 8px; }}
 #modal .mh .nm {{ font-size: 24px; }}
 #modal .close {{ margin-left: auto; color: {D['muted']}; font-size: 13px; }}
 #mcharts {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; height: calc(100vh - 140px); }}
 .src {{ position: fixed; right: 14px; bottom: 6px; font-size: 10px; color: {D['muted']}; }}
</style></head><body>
<header><h1>R22 · THE SQUAD: REAL MADRID'S ATTACKERS</h1>
<span class="hint">G+A per 90: red = posterior mean, band = 95% interval, blue = raw rate · radar = percentile among <b id="pn"></b> big-club attackers · click a card</span>
<span id="fk"></span></header>
<div class="grid" id="grid"></div>
<div id="modal"><div class="mh" id="mh"></div><div id="mcharts"><div id="m1"></div><div id="m2"></div><div id="m3"></div></div></div>
<div class="src">Data: Transfermarkt via transfermarkt-datasets; authors' calculations</div>
<script>
const DATA = {json.dumps(data)};
const C = {json.dumps(dict(bg=D['bg'], panel=D['panel'], grid=D['grid'], ink=D['ink'], ink2=D['ink2'], muted=D['muted'],
                           gold=GOLD, navy=NAVY, grey=GREY, mono=mono))};
const SIL = "{SILHOUETTE}";
const $ = id => document.getElementById(id);
const fmt = (x, d = 0) => x === null || x === undefined || Number.isNaN(x) ? '–' : Number(x).toFixed(d);
$('pn').textContent = DATA.pool_n;
$('grid').style.gridTemplateRows = 'repeat(' + Math.max(1, Math.ceil(DATA.cards.length / 3)) + ', minmax(0, 1fr))';
if (DATA.fake) $('fk').innerHTML = '<span class="fake">FAKE DATA</span>';
const AX = {{gridcolor: C.grid, zerolinecolor: C.grid, tickfont: {{color: C.ink2, size: 10}}, titlefont: {{color: C.ink2, size: 11}}}};
const BASE = {{paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {{family: C.mono, color: C.ink, size: 11}},
              showlegend: false}};
const CFG = {{displayModeBar: false, responsive: true, staticPlot: false}};
function sub(c) {{
  const bits = [c.sub_position || c.position, c.country_of_citizenship, c.age ? 'age ' + fmt(c.age) : null,
                c.foot ? c.foot + ' foot' : null, c.height_in_cm ? c.height_in_cm + ' cm' : null].filter(Boolean);
  return bits.join(' · ');
}}
function photo(c) {{ return c.photo ? '<img src="' + c.photo + '" alt="">' : SIL; }}
function radar(div, c, big) {{
  const A = big ? DATA.axes : DATA.axes_short, th = A.concat([A[0]]), r = c.radar.concat([c.radar[0]]);
  Plotly.newPlot(div, [{{type: 'scatterpolar', r: r, theta: th, fill: 'toself', fillcolor: 'rgba(254,190,16,0.25)',
    line: {{color: C.gold, width: 2}}, hovertemplate: '%{{theta}}: %{{r:.0f}}th percentile<extra></extra>'}}],
    Object.assign({{}}, BASE, {{margin: big ? {{l: 100, r: 100, t: 50, b: 40}} : {{l: 34, r: 34, t: 16, b: 16}},
      title: big ? {{text: 'Percentile among big-club attackers', font: {{color: C.gold, size: 14}}, x: 0.02}} : undefined,
      polar: {{bgcolor: C.bg, radialaxis: {{range: [0, 100], showticklabels: big, tickfont: {{color: C.muted, size: 9}}, gridcolor: C.grid,
                linecolor: C.grid}}, angularaxis: {{tickfont: {{color: C.ink2, size: big ? 12 : 8}}, gridcolor: C.grid, linecolor: C.grid}}}}}}), CFG);
}}
function value(div, c, big) {{
  Plotly.newPlot(div, [{{x: c.v_age, y: c.v_val, mode: 'lines', line: {{color: C.gold, width: 2, shape: 'hv'}},
    fill: 'tozeroy', fillcolor: 'rgba(254,190,16,0.08)', hovertemplate: 'age %{{x:.1f}}: €%{{y:.0f}}M<extra></extra>'}}],
    Object.assign({{}}, BASE, {{margin: big ? {{l: 60, r: 20, t: 40, b: 50}} : {{l: 34, r: 6, t: 14, b: 22}},
      title: {{text: big ? 'Market value by age (€M)' : '€M by age', font: {{color: big ? C.gold : C.muted, size: big ? 14 : 9}}, x: 0.02}},
      xaxis: Object.assign({{}}, AX, {{title: big ? 'age' : ''}}), yaxis: Object.assign({{}}, AX, {{rangemode: 'tozero'}})}}), CFG);
}}
function post(div, c) {{
  Plotly.newPlot(div, [
    {{x: DATA.lam, y: DATA.prior, mode: 'lines', line: {{color: C.grey, width: 2, dash: 'dot'}}, name: 'prior (all attackers)'}},
    {{x: DATA.lam, y: c.dens, mode: 'lines', line: {{color: C.gold, width: 3}}, fill: 'tozeroy', fillcolor: 'rgba(254,190,16,0.15)',
      name: 'posterior'}},
    {{x: [c.raw, c.raw], y: [0, Math.max(...c.dens)], mode: 'lines', line: {{color: C.navy, width: 2}}, name: 'raw k/t'}}],
    Object.assign({{}}, BASE, {{showlegend: true, legend: {{orientation: 'h', y: -0.2, font: {{color: C.ink2}}}},
      margin: {{l: 50, r: 20, t: 40, b: 70}},
      title: {{text: 'λ | data ~ Gamma(α+k, β+t): ' + fmt(c.post, 2) + ' [' + fmt(c.lo, 2) + ', ' + fmt(c.hi, 2) + ']',
               font: {{color: C.gold, size: 14}}, x: 0.02}},
      xaxis: Object.assign({{}}, AX, {{title: 'G+A per 90'}}), yaxis: Object.assign({{}}, AX, {{title: 'density'}})}}), CFG);
}}
const xmax = Math.max(...DATA.cards.map(c => Math.max(c.hi, c.raw))) * 1.1;
const pos = x => (100 * x / xmax).toFixed(1) + '%';
DATA.cards.forEach((c, i) => {{
  const el = document.createElement('div'); el.className = 'card';
  el.innerHTML = '<div class="top"><div class="ph">' + photo(c) + '</div><div><div class="nm">' + c.name + '</div><div class="sub">' +
    sub(c) + '</div></div><div class="big"><b>' + fmt(c.post, 2) + '</b><span>G+A / 90</span></div></div>' +
    '<div class="bar" title="95% interval ' + fmt(c.lo, 2) + '–' + fmt(c.hi, 2) + '"><div class="ci" style="left:' + pos(c.lo) +
    ';width:calc(' + pos(c.hi) + ' - ' + pos(c.lo) + ')"></div><div class="rw" style="left:' + pos(c.raw) + '"></div>' +
    '<div class="pm" style="left:' + pos(c.post) + '"></div></div>' +
    '<div class="stats"><div><b>' + c.seasons + '</b>seasons</div><div><b>' + c.apps + '</b>apps</div><div><b>' + c.goals +
    '</b>goals</div><div><b>' + c.assists + '</b>assists</div><div><b>' + (c.fee === null ? 'academy' : '€' + fmt(c.fee) + 'M') +
    '</b>fee</div></div><div class="charts"><div id="r' + i + '"></div><div id="v' + i + '"></div></div>';
  el.onclick = () => openCard(c);
  $('grid').appendChild(el);
  radar('r' + i, c, false); value('v' + i, c, false);
}});
function openCard(c) {{
  $('mh').innerHTML = '<div class="ph" style="width:80px;height:80px;flex-basis:80px">' + photo(c).replace("width='64' height='64'",
    "width='80' height='80'") + '</div><div><div class="nm">' + c.name + '</div><div class="sub">' + sub(c) + ' · ' +
    c.minutes.toLocaleString() + ' league minutes for Real Madrid · peak value €' + fmt(c.peak_value) + 'M</div></div>' +
    '<span class="close">click anywhere or press Esc to close</span>';
  $('modal').classList.add('on');
  radar('m1', c, true); post('m2', c); value('m3', c, true);
}}
$('modal').onclick = () => $('modal').classList.remove('on');
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') $('modal').classList.remove('on'); }});
</script></body></html>"""
    html = OUT / f"{name}.html"
    html.write_text(strip_codes(page), encoding="utf-8")
    _screenshot(html, FIG / f"{name}.png")
    return html


NUM = {}


def run():
    made = [r0_matchday(), r10_mcmc_walk(), r10b_mcmc_step_size(), r11_mle_surface(), r12_permutation_test(),
            r13_galton_board(), r14_sporting_director(), r15_competing_exponentials(), r16_poisson_splitting(),
            r17_weibull_hazard(), r18_bootstrap(), r19_coverage(), r20_bayes_tree(), r21_leverage(), r22_player_cards()]
    for f in made:
        print("  wrote", f.relative_to(FIG.parent))
    p = DATA_PROC / "deck_numbers.json"
    nums = json.loads(p.read_text()) if p.exists() else {}
    nums.update({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in NUM.items()})
    p.write_text(json.dumps(nums, indent=2))
    return made


if __name__ == "__main__":
    run()
