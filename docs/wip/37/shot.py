import sys, glob, os
from playwright.sync_api import sync_playwright
def find_chrome():
    c = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome"))
    return c[-1] if c else None
url = "file:///home/user/LLM-sttudy/pages/37-reading-list-glossary.html"
with sync_playwright() as pw:
    try: b = pw.chromium.launch()
    except Exception: b = pw.chromium.launch(executable_path=find_chrome())
    for name, w, h in (("phone", 390, 844), ("desk", 1400, 900)):
        pg = b.new_page(viewport={"width": w, "height": h})
        pg.goto(url); pg.wait_for_timeout(600)
        # sections to capture
        for sel in sys.argv[1:]:
            el = pg.query_selector(sel)
            if not el: print("no", sel); continue
            el.scroll_into_view_if_needed()
            fn = f"shots/{name}_{sel.strip('#.').replace(' ','_')}.png"
            pg.screenshot(path=fn, clip=None, full_page=False)
            print(fn)
        print(name, "scrollWidth", pg.evaluate("document.documentElement.scrollWidth"), "innerWidth", pg.evaluate("window.innerWidth"))
        pg.close()
    b.close()
