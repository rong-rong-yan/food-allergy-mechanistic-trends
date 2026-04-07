import pandas as pd
from pathlib import Path

# =========================================================
# Input / output
# =========================================================
INPUT_FILE = "processed/main_vs_background_yearly.csv"
OUTDIR = Path("processed/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUTDIR / "mechanistic_share_vs_background.svg"

# =========================================================
# Load data
# =========================================================
df = pd.read_csv(INPUT_FILE)
df = df[(df["pub_year"] >= 2015) & (df["pub_year"] <= 2025)].copy()

# Use percent for plotting
df["mechanistic_share_pct"] = 100 * df["mechanistic_share"]

# =========================================================
# SVG settings
# =========================================================
W = 950
H = 600

margin_left = 90
margin_right = 80
margin_top = 70
margin_bottom = 80

plot_x0 = margin_left
plot_y0 = margin_top
plot_w = W - margin_left - margin_right
plot_h = H - margin_top - margin_bottom

years = df["pub_year"].tolist()
x_min, x_max = min(years), max(years)

y_min = 0
y_max = max(30, df["mechanistic_share_pct"].max() + 2)

def x_map(x):
    if x_max == x_min:
        return plot_x0 + plot_w / 2
    return plot_x0 + (x - x_min) / (x_max - x_min) * plot_w

def y_map(y):
    if y_max == y_min:
        return plot_y0 + plot_h / 2
    return plot_y0 + plot_h - (y - y_min) / (y_max - y_min) * plot_h

parts = []
parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
parts.append('<rect width="100%" height="100%" fill="white"/>')

# Title
parts.append(
    f'<text x="{W/2}" y="30" text-anchor="middle" font-size="24" font-family="Arial">'
    'Mechanistic corpus share within early-life food allergy literature'
    '</text>'
)

# Subtitle
parts.append(
    f'<text x="{W/2}" y="55" text-anchor="middle" font-size="15" font-family="Arial" fill="#444">'
    'Main corpus as a percentage of the broader early-life food allergy background'
    '</text>'
)

# Axes
parts.append(f'<line x1="{plot_x0}" y1="{plot_y0 + plot_h}" x2="{plot_x0 + plot_w}" y2="{plot_y0 + plot_h}" stroke="black" stroke-width="2"/>')
parts.append(f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y0 + plot_h}" stroke="black" stroke-width="2"/>')

# Y ticks
n_y_ticks = 6
for i in range(n_y_ticks + 1):
    y_val = y_min + (y_max - y_min) * i / n_y_ticks
    y_px = y_map(y_val)
    parts.append(f'<line x1="{plot_x0}" y1="{y_px}" x2="{plot_x0 + plot_w}" y2="{y_px}" stroke="#dddddd" stroke-width="1"/>')
    parts.append(
        f'<text x="{plot_x0 - 10}" y="{y_px + 5}" text-anchor="end" font-size="14" font-family="Arial">'
        f'{y_val:.0f}%'
        '</text>'
    )

# X ticks
for yr in years:
    x_px = x_map(yr)
    parts.append(f'<line x1="{x_px}" y1="{plot_y0}" x2="{x_px}" y2="{plot_y0 + plot_h}" stroke="#eeeeee" stroke-width="1"/>')
    parts.append(
        f'<text x="{x_px}" y="{plot_y0 + plot_h + 25}" text-anchor="middle" font-size="14" font-family="Arial">'
        f'{yr}'
        '</text>'
    )

# Axis labels
parts.append(
    f'<text x="{plot_x0 + plot_w/2}" y="{H - 20}" text-anchor="middle" font-size="18" font-family="Arial">'
    'Publication year'
    '</text>'
)
parts.append(
    f'<text x="25" y="{plot_y0 + plot_h/2}" text-anchor="middle" font-size="18" font-family="Arial" '
    f'transform="rotate(-90 25 {plot_y0 + plot_h/2})">'
    'Main corpus share (%)'
    '</text>'
)

# Line + points
pts = [(x_map(row["pub_year"]), y_map(row["mechanistic_share_pct"])) for _, row in df.iterrows()]
pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

parts.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="3.5" points="{pts_str}"/>')

for x, y in pts:
    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.8" fill="#1f77b4"/>')

# Optional labels on points
for _, row in df.iterrows():
    x = x_map(row["pub_year"])
    y = y_map(row["mechanistic_share_pct"])
    parts.append(
        f'<text x="{x}" y="{y - 10}" text-anchor="middle" font-size="12" font-family="Arial" fill="#333">'
        f'{row["mechanistic_share_pct"]:.1f}'
        '</text>'
    )

parts.append("</svg>")

with open(OUTFILE, "w", encoding="utf-8") as f:
    f.write("\n".join(parts))

print(f"Saved SVG plot to: {OUTFILE}")