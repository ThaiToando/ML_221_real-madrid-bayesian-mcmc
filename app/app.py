"""Interactive app: live probability simulations.   Run:  streamlit run app/app.py"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from config import DATA_PROC, REAL_MADRID
from src.part5_season_sim import fit_poisson, rates, simulate_season
import plotly.io as pio
from src.style import GOLD as _G, NAVY as _N, GREY as _Y, plotly_template

pio.templates["project"] = plotly_template()      # same terminal / paper look as the figures and the deck
pio.templates.default = "project"

st.set_page_config(page_title="Real Madrid: Does Money Buy Wins?", layout="wide")
GOLD, NAVY, GREY = _G, _N, _Y


@st.cache_data
def load_tables():
    games = pd.read_csv(DATA_PROC / "league_games.csv", parse_dates=["date"])
    signings = pd.read_csv(DATA_PROC / "signings.csv", parse_dates=["transfer_date"])
    players = pd.read_csv(DATA_PROC / "player_seasons.csv")
    return games, signings, players


@st.cache_resource
def model_for(season):
    gs = games[games.season == season].sort_values("date")
    return fit_poisson(gs), gs


games, signings, players = load_tables()
st.title("Real Madrid: Does Money Buy Wins?")
st.caption("ML 221 final project · every number on this page comes from a probability model you can inspect")
tab1, tab2, tab3 = st.tabs(["Live season simulator", "Live match", "Bayesian transfer tracker"])

# ---------------- Tab 1: season simulator (law of large numbers + 'what if')
with tab1:
    c1, c2, c3 = st.columns(3)
    season = c1.selectbox("Season (start year)", sorted(games.season.unique(), reverse=True))
    n_total = c2.slider("Number of simulated seasons", 1_000, 50_000, 10_000, step=1_000)
    boost = c3.slider("Change Real Madrid's attack (%)", -40, 40, 0)
    m, gs = model_for(season)
    names = dict(zip(gs.home_club_id, gs.home_club_name))
    rm_season = players[(players.club_id == REAL_MADRID) & (players.season == season)]
    star = rm_season.sort_values("goals", ascending=False).head(1)
    drop_star = False
    if len(star):
        drop_star = st.checkbox(f"Remove top scorer {star.name.iloc[0]} ({int(star.goals.iloc[0])} league goals)")
    replacement = st.slider("Replacement player produces this share of his output", 0.0, 1.0, 0.5)
    lam_h, lam_a = rates(m, gs)
    factor = 1 + boost / 100
    if drop_star and len(star):
        share = star.goals.iloc[0] / max(rm_season.goals.sum(), 1)
        factor *= 1 - share * (1 - replacement)
    lam_h = np.where(gs.home_club_id == REAL_MADRID, lam_h * factor, lam_h)
    lam_a = np.where(gs.away_club_id == REAL_MADRID, lam_a * factor, lam_a)

    if st.button("Run simulation live", type="primary"):
        left, right = st.columns(2)
        line_spot, hist_spot, txt = left.empty(), right.empty(), st.empty()
        batch, wins, done, all_pts = 1_000, None, 0, []
        history = []
        for b in range(n_total // batch):
            pts, teams, champ = simulate_season(gs, lam_h, lam_a, n_sims=batch, seed=b)
            wins = np.bincount(champ, minlength=len(teams)) + (0 if wins is None else wins)
            done += batch
            rm_i = list(teams).index(REAL_MADRID)
            all_pts.append(pts[:, rm_i])
            history.append(wins / done)
            hist = np.array(history)
            top = np.argsort(-wins)[:4]
            fig = go.Figure()
            for i in top:
                fig.add_scatter(x=np.arange(1, len(hist) + 1) * batch, y=hist[:, i], name=names.get(teams[i], teams[i]),
                                line=dict(color=GOLD if teams[i] == REAL_MADRID else None, width=4 if teams[i] == REAL_MADRID else 2))
            fig.update_layout(title="Title probability as simulations pile up", xaxis_title="simulated seasons",
                              yaxis_title="P(champion)", yaxis_range=[0, 1], height=420)
            line_spot.plotly_chart(fig, width="stretch", key=f"line{b}", theme=None)
            h = go.Figure(go.Histogram(x=np.concatenate(all_pts), marker_color=GOLD, nbinsx=40))
            h.update_layout(title="Real Madrid points across simulated seasons", xaxis_title="points", height=420)
            hist_spot.plotly_chart(h, width="stretch", key=f"hist{b}", theme=None)
            p = wins[rm_i] / done
            se = np.sqrt(p * (1 - p) / done)
            txt.markdown(f"**{done:,} seasons:** P(Real Madrid title) = **{p:.3f}** ± {1.96 * se:.3f} (95% CI shrinks like 1/√n)")
            time.sleep(0.05)

# ---------------- Tab 2: one match, minute by minute, with live win probability
with tab2:
    season2 = st.selectbox("Season", sorted(games.season.unique(), reverse=True), key="s2")
    m2, gs2 = model_for(season2)
    names2 = dict(zip(gs2.home_club_id, gs2.home_club_name))
    ids = sorted(names2, key=lambda i: names2[i])
    c1, c2, c3 = st.columns(3)
    home = c1.selectbox("Home", ids, index=ids.index(REAL_MADRID) if REAL_MADRID in ids else 0, format_func=names2.get)
    away = c2.selectbox("Away", [i for i in ids if i != home], format_func=names2.get)
    speed = c3.slider("Seconds per minute", 0.0, 0.3, 0.08)
    lh, la = rates(m2, pd.DataFrame(dict(home_club_id=[home], away_club_id=[away])))
    st.write(f"Expected goals: **{names2[home]} {lh[0]:.2f} – {la[0]:.2f} {names2[away]}**")
    if st.button("Kick off", type="primary"):
        rng = np.random.default_rng()
        score, board, chart = [0, 0], st.empty(), st.empty()
        rows = []
        for minute in range(0, 91):
            if minute > 0:   # goals in one minute ~ Poisson(lambda / 90): a Poisson process
                score[0] += rng.poisson(lh[0] / 90); score[1] += rng.poisson(la[0] / 90)
            left = (90 - minute) / 90
            # remaining goals are Poisson, so final margin = current margin + Skellam(remaining rates)
            diff = score[0] - score[1]
            if left > 0:
                pw = stats.skellam.sf(-diff, lh[0] * left, la[0] * left)
                pdr = stats.skellam.pmf(-diff, lh[0] * left, la[0] * left)
            else:
                pw, pdr = float(diff > 0), float(diff == 0)
            rows.append(dict(minute=minute, home=pw, draw=pdr, away=1 - pw - pdr))
            board.markdown(f"## {minute}'   {names2[home]} {score[0]} – {score[1]} {names2[away]}")
            d = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_scatter(x=d.minute, y=d.home, stackgroup="one", name=f"{names2[home]} win", line_color=GOLD)
            fig.add_scatter(x=d.minute, y=d.draw, stackgroup="one", name="Draw", line_color=GREY)
            fig.add_scatter(x=d.minute, y=d.away, stackgroup="one", name=f"{names2[away]} win", line_color=NAVY)
            fig.update_layout(xaxis_range=[0, 90], yaxis_range=[0, 1], height=380, yaxis_title="probability",
                              title="Live win probability (conditional on the score so far)")
            chart.plotly_chart(fig, width="stretch", key=f"m{minute}", theme=None)
            time.sleep(speed)

# ---------------- Tab 3: watch the Beta posterior update one signing at a time
with tab3:
    club = st.selectbox("Club", sorted(signings.club.unique()), index=sorted(signings.club.unique()).index("Real Madrid")
                        if "Real Madrid" in set(signings.club) else 0)
    a0 = st.number_input("Prior a (successes)", 0.5, 50.0, 5.0); b0 = st.number_input("Prior b (failures)", 0.5, 50.0, 6.0)
    s = signings[signings.club == club].sort_values("transfer_date")
    if st.button("Reveal signings one by one", type="primary"):
        spot, note = st.empty(), st.empty()
        a, b = a0, b0
        x = np.linspace(0, 1, 300)
        for i, r in enumerate(s.itertuples(), start=1):
            a, b = a + r.success, b + 1 - r.success
            fig = go.Figure()
            fig.add_scatter(x=x, y=stats.beta.pdf(x, a0, b0), name="prior", line=dict(dash="dash", color=GREY))
            fig.add_scatter(x=x, y=stats.beta.pdf(x, a, b), name=f"after {i} signings", line=dict(color=GOLD, width=4))
            fig.update_layout(xaxis_title="true hit rate p", height=380, title=f"Beta({a:.0f}, {b:.0f}) posterior")
            spot.plotly_chart(fig, width="stretch", key=f"b{i}", theme=None)
            lo, hi = stats.beta.ppf([0.025, 0.975], a, b)
            note.markdown(f"**{r.season}: {r.player_name}** (EUR {r.fee_m:.0f}M) → {'hit' if r.success else 'miss'} · "
                          f"95% credible interval [{lo:.2f}, {hi:.2f}]")
            time.sleep(0.6)
