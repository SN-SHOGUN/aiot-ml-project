"""
build_webapp.py
---------------
Embeds all report/figures/*.png as base64 into webapp/index.html,
producing webapp/dist/index.html — a 100% self-contained file that
runs on ANY computer without a server.

Run: python build_webapp.py
Then open: webapp/dist/index.html in any browser.
"""
import pathlib, base64, shutil, re, sys

ROOT   = pathlib.Path(__file__).parent
SRC    = ROOT / "webapp" / "index.html"
FIGS   = ROOT / "report" / "figures"
DIST   = ROOT / "webapp" / "dist"
OUT    = DIST / "index.html"

DIST.mkdir(parents=True, exist_ok=True)

# ── 1. Read template ──────────────────────────────────────────────────────────
html = SRC.read_text(encoding="utf-8")

# ── 2. Embed all PNG figures as base64 ───────────────────────────────────────
def img_to_b64(path: pathlib.Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

replaced = 0
for fig in sorted(FIGS.glob("*.png")):
    # Match src="../report/figures/FILENAME.png"
    old = f"../report/figures/{fig.name}"
    new = img_to_b64(fig)
    if old in html:
        html = html.replace(old, new)
        replaced += 1
        print(f"  Embedded: {fig.name} ({fig.stat().st_size // 1024} KB)")

print(f"\nEmbedded {replaced}/{len(list(FIGS.glob('*.png')))} figures.")

# ── 3. Update sidebar Streamlit link to point to localhost ────────────────────
html = html.replace(
    'href="http://localhost:8501"',
    'href="http://localhost:8501" title="Run streamlit run app.py first"'
)

# ── 4. Write output ───────────────────────────────────────────────────────────
OUT.write_text(html, encoding="utf-8")
size_kb = OUT.stat().st_size // 1024
print(f"\nOutput: {OUT}")
print(f"Size:   {size_kb} KB ({size_kb//1024} MB)")
print("\nDone! Open webapp/dist/index.html in any browser.")
