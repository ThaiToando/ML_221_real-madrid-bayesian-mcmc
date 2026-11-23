"""The 1280x640 GitHub social preview (repo Settings > General > Social preview):   python -m src.social_card
Writes docs/img/github_social.png: title, method tags and the stadium photo, in the deck's colours (src/style.py)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.style import BG, GRIDC, INK, INK2, MUTED, PANEL, GOLD as RED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img" / "github_social.png"
W, H, PHOTO_W, FADE = 1280, 640, 560, 300
TAGS = ["Bayesian inference", "MCMC", "Monte Carlo", "Poisson", "Regression"]


def rgb(hex_colour: str) -> tuple:
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


def font(bold: bool, size: int):
    for name in (("timesbd.ttf", "LiberationSerif-Bold.ttf") if bold else ("times.ttf", "LiberationSerif-Regular.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    img = Image.new("RGB", (W, H), rgb(BG))
    photo = Image.open(ROOT / "figures" / "backgrounds" / "bernabeu.jpg").convert("RGB")
    photo = photo.resize((int(photo.width * H / photo.height), H))
    photo = photo.crop((photo.width - PHOTO_W, 0, photo.width, H))
    img.paste(photo, (W - PHOTO_W, 0))
    fade = Image.new("L", (PHOTO_W, 1))                                    # brown on the photo's left edge, clear after FADE px
    fade.putdata([max(0, int(255 * (1 - x / FADE))) for x in range(PHOTO_W)])
    fade = fade.resize((PHOTO_W, H))
    img.paste(Image.new("RGB", (PHOTO_W, H), rgb(BG)), (W - PHOTO_W, 0), fade)

    d = ImageDraw.Draw(img)
    red, cream, ink2, muted = rgb(RED), rgb(INK), rgb(INK2), rgb(MUTED)
    d.rectangle((64, 58, 184, 66), fill=red)
    d.text((64, 92), "ML 221 APPLIED PROBABILITY  /  FINAL PROJECT", font=font(True, 22), fill=red)
    d.text((64, 140), "Does money buy wins?", font=font(True, 68), fill=cream)
    d.text((64, 224), "Bayesian inference and MCMC", font=font(True, 50), fill=cream)
    d.text((64, 284), "at Real Madrid", font=font(True, 50), fill=cream)
    x, y, f = 64, 374, font(False, 24)
    for tag in TAGS:
        w = d.textlength(tag, font=f)
        d.rounded_rectangle((x, y, x + w + 28, y + 42), radius=21, outline=red, width=2, fill=rgb(PANEL))
        d.text((x + 14, y + 8), tag, font=f, fill=cream)
        x += w + 40
    d.line((64, 468, 560, 468), fill=rgb(GRIDC), width=2)
    d.text((64, 486), "Do Thai Toan", font=font(True, 32), fill=cream)
    d.text((64, 532), "Live interactive deck  ·  12 La Liga seasons  ·  Transfermarkt data", font=font(False, 24), fill=ink2)
    d.text((64, 594), "Stadium photo: Badefa, CC0, Wikimedia Commons", font=font(False, 16), fill=muted)
    img.save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
