# GO LIVE: from the fake test build to the real presentation

## Fast path (one command)
Install Python 3.10+ (and Node.js if you can), unzip, open the folder in VS Code, then in its terminal:

    python go_live.py

It installs everything, removes the fake build, downloads the real data, runs all seven parts, the 23 visuals,
the 8 Manim videos (when Manim, Pango and LaTeX are installed), builds docs/index.html, deck.pptx and deck.pdf,
and runs the go-live check. Re-run it any time; finished steps are skipped. The only things left for you are
optional: the players on the cards (PLAYER_CARDS), licensed photos, the Streamlit link. The steps below are what
it does, for when you want to run them by hand.

The zip was built and tested on FAKE data (players are called "Player 25", every slide, chart and video has a
red FAKE DATA mark). This file takes you to the real deck. Do the steps in order. After every step run

    python -m tests.check_real

It prints PASS / FAIL / WARN with the fix for each problem. You are done when it prints READY TO PRESENT.

## You never type player names or stats
Names, positions, nationality, age, foot, height, seasons, apps, goals, assists, fees, market values and every
posterior come from the Transfermarkt tables through the code. When the real data is in, they appear by
themselves on the cards (R22), the player explorer (R2), the slides and the videos. If a number looks wrong, fix
the data step that produced it; never edit the number by hand.

## Step 0 - Set up (once)
```bash
cd real-madrid-probability
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium          # for docs/deck.pdf and the widget screenshots
npm install --prefix docs reveal.js@5.1.0 katex@0.16     # offline deck (needs Node.js)
```
Optional, for the 8 Manim videos: Mac `brew install pango` and MacTeX (`brew install --cask mactex-no-gui`);
Ubuntu `sudo apt-get install libpango1.0-dev texlive-latex-extra dvisvgm`; Windows: use WSL, or skip the videos.
Then `pip install manim`. Without Manim the deck still builds; the video sub-slides are left out.

## Step 1 - Remove every fake file
```bash
rm -f data/raw/* data/processed/*.csv data/processed/deck_numbers.json
rm -f figures/*.png figures/*.mp4 figures/*.gif figures/interactive/R*.html
rm -rf docs/assets docs/index.html docs/deck.pptx docs/deck.pdf
```
(Windows PowerShell: delete the same files in Explorer, or use `Remove-Item`.) Keep `figures/players/README.md`.

## Step 2 - Download the real data (about 200 MB, once)
```bash
python -c "from src.data_io import download_data; download_data()"
python -c "from src.data_io import check_club_ids; print(check_club_ids())"
```
The second command must list all 15 clubs with their real names. If one is missing, fix its id in `config.py`.

## Step 3 - Run the analysis and check the numbers
```bash
python run_all.py
```
Read the printed output part by part. Each person should be able to explain their part's numbers.

## Step 4 - Choose the players on the cards (optional)
By default the cards show the six Real Madrid attackers with the most minutes. To pick them yourself, look up
their ids in `data/processed/p3_rm_attackers.csv` (columns `player_id`, `name`) and set in `config.py`:
```python
PLAYER_CARDS = [123456, 234567, 345678]    # shown in this order; six fit best
```

## Step 5 - Add your licensed photos (optional)
1. Save each photo as `figures/players/<player_id>.jpg` (or .png / .webp), square, at least 300 x 300 px.
2. Add one line per photo to `figures/players/LICENSES.md`, e.g.
   `123456.jpg - Getty Images, licence #ABC123, editorial use, class presentation only`.
3. Only publish photos whose licence allows it (the CC BY / CC BY-SA photos in this repo do, with credit).
Never generate images of real players.

## Step 6 - Build everything
```bash
python run_all.py --deck --manim --pdf     # drop --manim if you skipped Manim
python -m tests.check_real
```
The FAKE DATA badges disappear by themselves: every chart, slide and video checks the data when it is built.
If the check says the videos are stale, run `python -m src.manim_videos`.

## Step 7 - Links and wording
- Deploy the Streamlit app (share.streamlit.io, main file `app/app.py`), then paste its link into `APP_URL`
  at the top of `src/slides.py` and run `python -m src.slides --pdf`.
- Change slide wording only in `slide_list()` in `src/slides.py`. Never type a number there: use `n[...]`.
- To promote an extra (for example the player cards) to a main slide, move its entry from `extras` to `iframe`
  in `slide_list()`.

## Step 8 - Rehearse and submit
- Open `docs/index.html` in Chrome. Press F (full screen), S (speaker notes), K (restart the match clock),
  the down arrow for the extras. Open every extra once on the presentation laptop.
- Go through the checklist in proposal section 20.7.
- Submit `docs/deck.pdf`. Bring `docs/deck.pptx` and the whole `docs/` folder on a USB stick.
