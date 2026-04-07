import pandas as pd
from pathlib import Path
import html

# =========================================================
# Input / output
# =========================================================
INPUT_FILE = "processed/theme_yearly_proportions.csv"
OUTDIR = Path("processed/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUTDIR / "theme_proportions_over_time.svg"

# =========================================================
# Load data
# =========================================================
df = pd.read_csv(INPUT_FILE)
df = df[(df["pub_year"] >= 2015) & (df["pub_year"] <= 2025)].copy()

theme_cols = {
    "theme_tolerance_regulation_prop": "Tolerance / regulation",
    "theme_sensitization_ige_prop": "Sensitization / IgE",
    "theme_microbiome_prop": "Microbiome",
    "theme_barrier_epithelial_prop": "Barrier / epithelial",
}

# =========================================================
# SVG settings
# =========================================================
W = 1000
H = 650

margin_left = 90
margin_right = 220
margin_top = 60
margin_bottom = 80

plot_x0 = margin_left
plot_y0 = margin_top
plot_w = W - margin_left - margin_right
plot_h = H - margin_top - margin_bottom

years = df["pub_year"].tolist()
x_min, x_max = min(years), max(years)

# Find y max across all theme proportions
y_max = 0
for col in theme_cols:
    y_max = max(y_max, df[col].max())

# Add small headroom
y_max = max(y_max, 0.5)
y_min = 0.0

def x_map(x):
    if x_max == x_min:
        return plot_x0 + plot_w / 2
    return plot_x0 + (x - x_min) / (x_max - x_min) * plot_w

def y_map(y):
    if y_max == y_min:
        return plot_y0 + plot_h / 2
    return plot_y0 + plot_h - (y - y_min) / (y_max - y_min) * plot_h

# Simple built-in palette
colors = {
    "theme_tolerance_regulation_prop": "#1f77b4",
    "theme_sensitization_ige_prop": "#d62728",
    "theme_microbiome_prop": "#2ca02c",
    "theme_barrier_epithelial_prop": "#9467bd",
}

# =========================================================
# Build SVG
# =========================================================
parts = []
parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
parts.append('<rect width="100%" height="100%" fill="white"/>')

# Title
parts.append(
    f'<text x="{W/2}" y="30" text-anchor="middle" font-size="24" font-family="Arial">'
    'Theme proportions over time'
    '</text>'
)

# Axes
parts.append(f'<line x1="{plot_x0}" y1="{plot_y0 + plot_h}" x2="{plot_x0 + plot_w}" y2="{plot_y0 + plot_h}" stroke="black" stroke-width="2"/>')
parts.append(f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y0 + plot_h}" stroke="black" stroke-width="2"/>')

# Y grid + labels
n_y_ticks = 5
for i in range(n_y_ticks + 1):
    y_val = y_min + (y_max - y_min) * i / n_y_ticks
    y_px = y_map(y_val)

    parts.append(
        f'<line x1="{plot_x0}" y1="{y_px}" x2="{plot_x0 + plot_w}" y2="{y_px}" '
        f'stroke="#dddddd" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{plot_x0 - 10}" y="{y_px + 5}" text-anchor="end" font-size="14" font-family="Arial">'
        f'{y_val:.2f}'
        '</text>'
    )

# X ticks + labels
for yr in years:
    x_px = x_map(yr)
    parts.append(
        f'<line x1="{x_px}" y1="{plot_y0}" x2="{x_px}" y2="{plot_y0 + plot_h}" '
        f'stroke="#eeeeee" stroke-width="1"/>'
    )
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
    'Proportion within mechanistic corpus'
    '</text>'
)

# Plot lines and points
legend_x = plot_x0 + plot_w + 30
legend_y = plot_y0 + 40
legend_gap = 35

for idx, (col, label) in enumerate(theme_cols.items()):
    color = colors[col]
    pts = []
    for _, row in df.iterrows():
        pts.append((x_map(row["pub_year"]), y_map(row[col])))

    # polyline
    pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    parts.append(
        f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{pts_str}"/>'
    )

    # points
    for x, y in pts:
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>'
        )

    # legend
    ly = legend_y + idx * legend_gap
    parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 28}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
    parts.append(f'<circle cx="{legend_x + 14}" cy="{ly}" r="4.5" fill="{color}"/>')
    parts.append(
        f'<text x="{legend_x + 40}" y="{ly + 5}" font-size="15" font-family="Arial">'
        f'{html.escape(label)}'
        '</text>'
    )

parts.append("</svg>")

with open(OUTFILE, "w", encoding="utf-8") as f:
    f.write("\n".join(parts))

print(f"Saved SVG plot to: {OUTFILE}")