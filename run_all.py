"""Run the whole analysis in order:  python run_all.py
python run_all.py --deck            also builds the animations, the interactive deck visuals (R0-R21) and the deck (docs/)
python run_all.py --deck --manim    also renders the 8 Manim explainer videos (needs manim + Pango + LaTeX, ~2-5 min)
python run_all.py --deck --pdf      also prints docs/deck.pdf (needs playwright)"""
import os as _os
import subprocess as _sp
import sys as _sys
if _os.name == "nt" and _os.environ.get("PYTHONUTF8") != "1":   # Windows: read/write every file as UTF-8 (the deck uses λ, ≈, ↓)
    _os.environ["PYTHONUTF8"] = "1"
    _sys.exit(_sp.call([_sys.executable, *_sys.argv]))
import sys
import matplotlib
matplotlib.use("Agg")    # save charts to files without opening windows
from src import build_tables, part1_value_vs_points, part2_transfers, part3_player_output
from src import part4_value_curve, part5_season_sim, part6_clutch, bayes_compute, part7_figures

for name, step in [("Build tables", build_tables.main), ("Part 1", part1_value_vs_points.run),
                   ("Part 2", part2_transfers.run), ("Part 3", part3_player_output.run),
                   ("Part 4", part4_value_curve.run), ("Part 5", part5_season_sim.run),
                   ("Part 6", part6_clutch.run), ("Part 7", bayes_compute.run),
                   ("Part 7 figures", part7_figures.run)]:
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    step()
if "--deck" in sys.argv:
    from src import animate, deck_sims, deck_visuals, eda, slides
    steps = [("EDA figure", eda.run), ("Animations", animate.run), ("Deck visuals R1-R9", deck_visuals.run),
             ("Simulation lab R0, R10-R21", deck_sims.run)]
    if "--manim" in sys.argv:
        from src import manim_videos
        steps.append(("Manim videos M1-M8", lambda: manim_videos.main([])))
    steps.append(("Slide deck (docs/)", lambda: slides.run(sys.argv[1:])))
    for name, step in steps:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        step()
print("\nDone. Charts are in figures/, tables in data/processed/, the deck in docs/.")
