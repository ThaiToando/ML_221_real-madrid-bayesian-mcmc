"""README screenshots of the built deck (docs/index.html), 1600x900, into docs/img/.   python -m src.screenshots
Needs Playwright with Chromium (pip install playwright, then: playwright install chromium)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "index.html"
OUT = ROOT / "docs" / "img"
# (file name, slide address in the deck, seconds to let the chart animate)
SHOTS = [("01_cover", "#/0", 2), ("02_regression", "#/5/0/1", 7), ("03_player_cards", "#/12", 6),
         ("04_section_opener", "#/17", 2), ("05_clasico", "#/14/0/2", 7), ("06_takeaways", "#/21/0/4", 3)]


def main():
    from playwright.sync_api import sync_playwright
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        for name, where, wait in SHOTS:
            page.goto(DECK.as_uri() + where)
            page.wait_for_timeout(wait * 1000)
            page.screenshot(path=str(OUT / f"{name}.png"))
            print("  wrote", f"docs/img/{name}.png")
        browser.close()


if __name__ == "__main__":
    main()
