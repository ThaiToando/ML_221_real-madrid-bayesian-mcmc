"""Make short videos for the slides:  python -m src.animate
  figures/anim_lln.mp4        title probability settling down as simulations pile up
  figures/anim_posterior.mp4  Beta posterior for Real Madrid's hit rate, one signing at a time
  figures/anim_match.mp4      one simulated match with live win probability
Each is also saved as a .gif (works in Google Slides / PowerPoint without a video player)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy import stats
from config import DATA_PROC, FIG, REAL_MADRID, SEED
from src.style import GOLD, GREY, INK, NAVY  # noqa: E402  (theme)
from src.part5_season_sim import fit_poisson, rates, simulate_season



def save(anim, name, fps):
    try:
        anim.save(FIG / f"{name}.mp4", writer=FFMpegWriter(fps=fps, bitrate=2400))
    except Exception as e:                                   # no ffmpeg installed -> GIF only
        print("mp4 skipped:", e)
    anim.save(FIG / f"{name}.gif", writer=PillowWriter(fps=fps))
    print("saved", name)


def lln_video(n=20_000):
    g = pd.read_csv(DATA_PROC / "league_games.csv", parse_dates=["date"])
    gs = g[g.season == g.season.max()]
    m = fit_poisson(gs)
    lh, la = rates(m, gs)
    _, teams, champ = simulate_season(gs, lh, la, n_sims=n)
    hit = (teams[champ] == REAL_MADRID).astype(float)
    running = np.cumsum(hit) / np.arange(1, n + 1)
    frames = np.unique(np.geomspace(1, n, 120).astype(int))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    line, = ax.plot([], [], color=GOLD, lw=2.5)
    band = ax.fill_between([], [], [], color=GOLD, alpha=.2)
    txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=13)
    ax.set(xlim=(1, n), ylim=(0, 1), xscale="log", xlabel="simulated seasons", ylabel="P(Real Madrid champion)")
    ax.axhline(running[-1], ls="--", color=INK, lw=1)

    def update(k):
        nonlocal band
        x = np.arange(1, k + 1); p = running[:k]
        se = np.sqrt(np.clip(p * (1 - p), 1e-9, None) / x)
        line.set_data(x, p); band.remove()
        band = ax.fill_between(x, p - 1.96 * se, p + 1.96 * se, color=GOLD, alpha=.2)
        txt.set_text(f"n = {k:,}   estimate = {p[-1]:.3f}")
        return line, txt
    save(FuncAnimation(fig, update, frames=frames, blit=False), "anim_lln", fps=20)
    plt.close(fig)


def posterior_video(a0=5.0, b0=6.0):
    s = pd.read_csv(DATA_PROC / "signings.csv", parse_dates=["transfer_date"])
    s = s[s.to_club_id == REAL_MADRID].sort_values("transfer_date")
    x = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, stats.beta.pdf(x, a0, b0), "--", color=GREY, label="prior")
    line, = ax.plot([], [], color=GOLD, lw=3, label="posterior")
    title = ax.set_title("")
    ax.set(xlim=(0, 1), ylim=(0, 8), xlabel="Real Madrid's true transfer hit rate p", ylabel="density"); ax.legend()
    outcomes = s.success.to_numpy(); labels = (s.player_name + " (" + s.season.astype(str) + ")").to_numpy()

    def update(i):
        a = a0 + outcomes[:i].sum(); b = b0 + i - outcomes[:i].sum()
        line.set_data(x, stats.beta.pdf(x, a, b))
        last = f"{labels[i - 1]}: {'hit' if outcomes[i - 1] else 'miss'}" if i else "before any signings"
        title.set_text(f"Beta({a:.0f}, {b:.0f}) after {i} signings  ·  {last}")
        return line, title
    save(FuncAnimation(fig, update, frames=range(len(s) + 1)), "anim_posterior", fps=2)
    plt.close(fig)


def match_video(home=REAL_MADRID, away=131, seed=SEED):
    g = pd.read_csv(DATA_PROC / "league_games.csv", parse_dates=["date"])
    gs = g[g.season == g.season.max()]
    m = fit_poisson(gs)
    lh, la = (r[0] for r in rates(m, pd.DataFrame(dict(home_club_id=[home], away_club_id=[away]))))
    rng = np.random.default_rng(seed)
    hg = np.r_[0, np.cumsum(rng.poisson(lh / 90, 90))]; ag = np.r_[0, np.cumsum(rng.poisson(la / 90, 90))]
    pw, pdr = [], []
    for t in range(91):
        left, diff = (90 - t) / 90, hg[t] - ag[t]
        if left > 0:
            pw.append(stats.skellam.sf(-diff, lh * left, la * left)); pdr.append(stats.skellam.pmf(-diff, lh * left, la * left))
        else:
            pw.append(float(diff > 0)); pdr.append(float(diff == 0))
    pw, pdr = np.array(pw), np.array(pdr)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set(xlim=(0, 90), ylim=(0, 1), xlabel="minute", ylabel="probability")
    title = ax.set_title("")

    def update(t):
        for c in list(ax.collections):
            c.remove()
        x = np.arange(t + 1)
        ax.stackplot(x, pw[:t + 1], pdr[:t + 1], 1 - pw[:t + 1] - pdr[:t + 1], colors=[GOLD, GREY, NAVY],
                     labels=["Real Madrid win", "draw", "Barcelona win"])
        title.set_text(f"{t}'   Real Madrid {hg[t]} – {ag[t]} Barcelona")
        if t == 0:
            ax.legend(loc="lower left")
        return title,
    save(FuncAnimation(fig, update, frames=range(91)), "anim_match", fps=8)
    plt.close(fig)


def run():
    lln_video(); posterior_video(); match_video()


if __name__ == "__main__":
    run()
