"""README images of the built deck (docs/index.html):   python -m src.screenshots
  docs/img/0N_*.png   1600x900 screenshots of key slides
  docs/img/social.png 1280x640 link-preview card (LinkedIn, Slack, GitHub social preview)
  docs/img/demo.gif   a short animated tour for the top of the README
Needs Playwright with Chromium (pip install playwright, then: playwright install chromium) and Pillow."""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "index.html"
OUT = ROOT / "docs" / "img"
# (file name, slide address in the deck, seconds to let the chart animate)
SHOTS = [("01_cover", "#/0", 2), ("02_regression", "#/5/0/1", 7), ("03_player_cards", "#/12", 6),
         ("04_section_opener", "#/17", 2), ("05_clasico", "#/14/0/2", 7), ("06_takeaways", "#/21/0/4", 3)]
# the animated tour: (slide address, frames, seconds between frames); charts are caught mid-animation
TOUR = [("#/0", 1, 0), ("#/4", 1, 0), ("#/5/0/1", 4, 0.9), ("#/12", 2, 1.5), ("#/14/0/2", 4, 1.0),
        ("#/17", 1, 0), ("#/18/0/2", 4, 0.9), ("#/21/0/4", 1, 0)]


def social_card(cover: Path) -> None:
    """1280x640 (2:1) crop of the cover: title on the left, stadium on the right."""
    im = Image.open(cover).convert("RGB").resize((1280, 720), Image.LANCZOS)
    im.crop((0, 40, 1280, 680)).save(OUT / "social.png", optimize=True)


def main():
    from playwright.sync_api import sync_playwright
    OUT.mkdir(exist_ok=True)
    frames, durations = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        for name, where, wait in SHOTS:
            page.goto(DECK.as_uri() + where)
            page.wait_for_timeout(wait * 1000)
            page.screenshot(path=str(OUT / f"{name}.png"))
            print("  wrote", f"docs/img/{name}.png")
        for where, n, gap in TOUR:
            page.goto(DECK.as_uri() + where)
            page.wait_for_timeout(1500)
            for k in range(n):
                shot = OUT / "_frame.png"
                page.screenshot(path=str(shot))
                frames.append(Image.open(shot).convert("RGB").resize((960, 540), Image.LANCZOS))
                durations.append(1100 if k < n - 1 else 2200)       # hold the last frame of each slide
                page.wait_for_timeout(int(gap * 1000))
        browser.close()
    (OUT / "_frame.png").unlink(missing_ok=True)
    pal = [f.quantize(colors=200, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(OUT / "demo.gif", save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print("  wrote docs/img/demo.gif", f"({(OUT / 'demo.gif').stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
    social_card(OUT / "01_cover.png")
    print("  wrote docs/img/social.png")


if __name__ == "__main__":
    main()
