"""Pack the finished deck (docs/index.html + assets/ + vendor/) into ONE html file that runs on any computer.

    python -m src.single_file            (after python run_all.py --deck)
Output: docs/Real_Madrid_ML221_Presentation.html. Open it in Chrome or Edge; no internet, no other files.

How it works, in plain steps:
  1 every stylesheet and script from vendor/ is copied into the page (KaTeX keeps only its woff2 fonts);
  2 every photo, GIF, poster and video becomes a data: URL inside the page;
  3 each interactive chart (assets/R*.html) is stored as text and loaded into its frame (srcdoc) when its slide is near;
    the charts share one copy of plotly.min.js, which each frame runs from the parent page.
"""
import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "Real_Madrid_ML221_Presentation.html"   # GitHub Pages serves it as a download
MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "mp4": "video/mp4",
        "woff2": "font/woff2", "webp": "image/webp"}


def data_url(path: Path) -> str:
    return f"data:{MIME[path.suffix[1:].lower()]};base64," + base64.b64encode(path.read_bytes()).decode()


def safe_js(text: str) -> str:
    """Text that can sit inside <script>...</script> without ending it early."""
    return text.replace("</script", "<\\/script").replace("</SCRIPT", "<\\/SCRIPT")


def js_string(text: str) -> str:
    return json.dumps(text).replace("</", "<\\/")


def inline_css(href: str) -> str:
    path = DOCS / href
    css = path.read_text(encoding="utf-8")
    css = re.sub(r"@import[^;]+;", "", css)                                         # theme web fonts: not used
    css = re.sub(r",\s*url\([^)]+\.(?:woff|ttf)\)\s*format\(\"[^\"]+\"\)", "", css)   # keep woff2 only
    css = re.sub(r"url\((fonts/[^)]+\.woff2)\)", lambda m: f"url({data_url(path.parent / m.group(1))})", css)
    return f"<style>{css}</style>"


def main():
    page = (DOCS / "index.html").read_text(encoding="utf-8")
    page = re.sub(r' data-png="[^"]*"', "", page)                                  # static twins are only for the PDF

    # 1 libraries
    page = re.sub(r'<link rel="stylesheet" href="(vendor/[^"]+)">', lambda m: inline_css(m.group(1)), page)
    page = re.sub(r'<script src="(vendor/[^"]+)"></script>',
                  lambda m: f"<script>{safe_js((DOCS / m.group(1)).read_text(encoding='utf-8'))}</script>", page)
    if re.search(r'(src|href)="vendor/', page):
        sys.exit("A vendor file is still linked; rebuild the deck with the local copies (python -m src.slides).")

    # 3 charts: data-src="assets/X.html" -> data-chart="X"; their text goes into one table
    charts = {}
    players = {p.name: data_url(p) for p in (DOCS / "assets" / "players").glob("*.jpg")} if (DOCS / "assets" / "players").exists() else {}

    def chart(m):
        name = m.group(1)
        if name not in charts:
            html = (DOCS / "assets" / f"{name}.html").read_text(encoding="utf-8")
            html = re.sub(r'<script[^>]*src="plotly\.min\.js"[^>]*></script>',      # both tag styles in assets/
                          lambda _: "<script>(0, eval)(parent.__PLOTLY_SRC)</script>", html)
            if "plotly.min.js" in html:
                sys.exit(f"{name}: plotly.min.js still linked")
            for fname, url in players.items():
                html = html.replace(f"players/{fname}", url)
            charts[name] = html
        return f'data-chart="{name}"'
    page = re.sub(r'data-src="assets/([A-Za-z0-9_]+)\.html"', chart, page)

    # 2 media
    def media(m):
        path = DOCS / m.group(0)
        return data_url(path) if path.exists() else m.group(0)
    page = re.sub(r"assets/[A-Za-z0-9_\-]+\.(?:jpg|jpeg|png|gif|mp4|webp)", media, page)
    left = re.findall(r"assets/[^\"' )]+", page)
    if left:
        sys.exit(f"Not packed: {sorted(set(left))[:5]}")

    plotly = (DOCS / "assets" / "plotly.min.js").read_text(encoding="utf-8")
    loader = f"""<script>
window.__PLOTLY_SRC = {js_string(plotly)};
window.__CHARTS = {{{",".join(f"{json.dumps(k)}: {js_string(v)}" for k, v in charts.items())}}};
(function () {{
  // load the charts of the current slide and its neighbours (and their extras) into their frames
  function fill(sl) {{ if (!sl) return;
    sl.querySelectorAll('iframe[data-chart]').forEach(f => {{ if (!f.srcdoc) f.srcdoc = window.__CHARTS[f.dataset.chart]; }}); }}
  function near() {{ const i = Reveal.getIndices();
    [-1, 0, 1].forEach(d => {{ const h = Reveal.getHorizontalSlides()[i.h + d]; fill(h); }}); }}
  Reveal.on('ready', near); Reveal.on('slidechanged', near);
}})();
</script>"""
    page = page.replace("</body></html>", loader + "</body></html>")
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT.name}: {OUT.stat().st_size / 1e6:.1f} MB, {len(charts)} charts packed")


if __name__ == "__main__":
    main()
