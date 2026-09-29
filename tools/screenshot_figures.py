#!/usr/bin/env python3
"""Screenshot every <figure> on a page (light and dark) so diagrams can be checked visually.

Usage: python3 tools/screenshot_figures.py pages/NN-slug.html OUT_DIR
Writes OUT_DIR/figK-light.png and figK-dark.png. Open them with an image viewer (or the Read tool)
and look for clipped, overlapping, or unreadable text.
"""
import glob
import os
import sys

from playwright.sync_api import sync_playwright


def main():
    path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    exe = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome"))
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception:
            b = p.chromium.launch(executable_path=exe[-1])
        for theme in ("light", "dark"):
            pg = b.new_page(viewport={"width": 1400, "height": 1000}, color_scheme=theme, device_scale_factor=1.5)
            pg.goto("file://" + os.path.abspath(path))
            pg.wait_for_timeout(300)
            for i, fig in enumerate(pg.query_selector_all("figure.fig")):
                fig.scroll_into_view_if_needed()
                fig.screenshot(path=os.path.join(outdir, f"fig{i + 1}-{theme}.png"))
            pg.close()
        b.close()
    print("wrote screenshots to", outdir)


if __name__ == "__main__":
    main()
