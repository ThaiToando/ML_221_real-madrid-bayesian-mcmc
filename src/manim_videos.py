"""Render the Manim explainers (animations/manim_scenes.py) and copy them next to the other figures.

python -m src.manim_videos                 # all 8 scenes, 1280x720 at 30 fps (a few minutes)
python -m src.manim_videos --quality l     # quick 480p preview
python -m src.manim_videos M2BetaUpdate    # just one
Outputs: figures/manim_<Scene>.mp4 and figures/manim_<Scene>.png (last frame, used in the PDF export of the deck).
Each scene gets its own media folder so parallel renders never share a LaTeX cache.
"""
import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES_FILE = ROOT / "animations" / "manim_scenes.py"
MEDIA = ROOT / "animations" / "media"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT / "animations"))
QUALITY = {"l": ("-ql", "480p15"), "m": ("-qm", "720p30"), "h": ("-qh", "1080p60")}


def render(scene, quality="m"):
    flag, sub = QUALITY[quality]
    media = MEDIA / scene
    cmd = ["manim", flag, "--disable_caching", "--media_dir", str(media), str(SCENES_FILE), scene]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError(f"manim failed on {scene}")
    mp4 = next((media / "videos" / "manim_scenes" / sub).glob(f"{scene}.mp4"))
    out = FIG / f"manim_{scene}.mp4"
    shutil.copy(mp4, out)
    last_frame_png(out, FIG / f"manim_{scene}.png")
    return out


def last_frame_png(mp4, png):
    """Save the video's last frame (used in the PDF). ffmpeg if installed, else PyAV (installed with manim)."""
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-0.3", "-i", str(mp4), "-frames:v", "1", str(png)],
                       check=False)
        return
    import av
    with av.open(str(mp4)) as box:
        frame = None
        for frame in box.decode(video=0):
            pass
    if frame is not None:
        frame.to_image().save(png)


def main(argv=None):
    from manim_scenes import SCENES
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=SCENES)
    ap.add_argument("--quality", default="m", choices=list(QUALITY))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args(argv)
    if shutil.which("manim") is None:
        sys.exit("manim not found: pip install manim (and Pango + LaTeX, see animations/manim_scenes.py)")
    with ThreadPoolExecutor(a.jobs) as ex:
        for out in ex.map(lambda s: render(s, a.quality), a.scenes):
            print("  wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
