
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# === Font and style settings ===
plt.rcParams["font.family"] = "Arial"
GRID_EDGE = "#000000"
EDGE_LW = 1.5
HEADER_BG = "#E0E0E0"
FONT_SIZE_MAIN = 14
FONT_SIZE_HEADER = 12
BAR_Y_PAD = 0.05
BAR_X_PAD = 0.02
dpi = 300

# === File paths ===
excel_path = os.path.join(".", "Fig 6.xlsx")
sheet_name = "Result"
output_png = "Figure 6 oral.png"
species_order = ["FL", "FG", "SP", "SH", "SF", "SO", "MS", "MG"]

# === Gradient definitions (dark → light) ===
cmap_input = LinearSegmentedColormap.from_list("input_grad", ["#E63946", "#FAD4D4"])
cmap_price = LinearSegmentedColormap.from_list("price_grad", ["#2F80ED", "#BBDFFF"])

# === Data loading ===
df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
df.columns = ["Group", "Item"] + species_order[: len(df.columns) - 2]
df = df[~(df["Group"].isna() & df["Item"].isna())].copy()
df["Group"] = df["Group"].ffill()

# === Normalize values to [0,1] ===
def to_unit_fraction(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, str):
        s = v.strip()
        if s.endswith("%"):
            s = s[:-1]
            try:
                return float(s) / 100.0
            except:
                return np.nan
        try:
            x = float(s)
        except:
            return np.nan
        return x / 100.0 if x > 1.0 else x
    try:
        x = float(v)
    except:
        return np.nan
    return x / 100.0 if x > 1.0 else x

for c in species_order:
    df[c] = df[c].apply(to_unit_fraction).clip(lower=0, upper=1)

groups = df["Group"].astype(str).tolist()
items = df["Item"].astype(str).tolist()
mat = df[species_order].to_numpy(dtype=float)
n_rows, n_cols = mat.shape

# === Figure setup ===
fig_w = max(10, 1.0 * n_cols + 6)
fig_h = max(4, 0.40 * n_rows + 1.2)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(-1.5, n_cols - 0.5)
ax.set_ylim(n_rows - 0.5, -1.5)
ax.set_xticks([])
ax.set_yticks([])

# === Header row ===
for j in range(n_cols):
    ax.add_patch(Rectangle((j - 0.5, -1 - 0.5), 1, 1,
                           facecolor=HEADER_BG, edgecolor=GRID_EDGE, linewidth=EDGE_LW))
    ax.text(j, -1, species_order[j], ha="center", va="center",
            fontsize=FONT_SIZE_HEADER, color="black", fontweight="bold")

# === Left header column ===
for i in range(n_rows):
    ax.add_patch(Rectangle((-1 - 0.5, i - 0.5), 1, 1,
                           facecolor=HEADER_BG, edgecolor=GRID_EDGE, linewidth=EDGE_LW))
    ax.text(-1, i, items[i], ha="center", va="center",
            fontsize=FONT_SIZE_HEADER, color="black", fontweight="bold")

# === Cell grid ===
for i in range(n_rows):
    for j in range(n_cols):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                               facecolor="white", edgecolor=GRID_EDGE, linewidth=EDGE_LW))

# === Bar parameters ===
bar_h = 1 - 2 * BAR_Y_PAD
grad_img = np.linspace(1, 0, 256).reshape(1, -1)

# === Draw bars ===
for i in range(n_rows):
    gname = groups[i].strip().lower()
    cmap = cmap_input if gname == "input" else cmap_price
    for j in range(n_cols):
        v = mat[i, j]
        if not np.isfinite(v) or v <= 0:
            ax.text(j, i, f"{(0 if not np.isfinite(v) else v)*100:.2f}%",
                    va="center", ha="center", fontsize=FONT_SIZE_MAIN, color="black")
            continue
        bar_w = (1 - 2 * BAR_X_PAD) * v
        x0, x1 = j - 0.5 + BAR_X_PAD, j - 0.5 + BAR_X_PAD + bar_w
        y0, y1 = i - 0.5 + BAR_Y_PAD, i - 0.5 + BAR_Y_PAD + bar_h
        ax.imshow(grad_img, extent=(x0, x1, y0, y1),
                  origin="lower", cmap=cmap, vmin=0, vmax=1,
                  aspect="auto", interpolation="bilinear", zorder=2, clip_on=True)
        ax.text(j, i, f"{v*100:.2f}%", va="center", ha="center",
                fontsize=FONT_SIZE_MAIN, color="black", zorder=3)

# === Group separation lines ===
split_idx = [i for i in range(1, n_rows) if groups[i] != groups[i - 1]]
for idx in split_idx:
    ax.plot([-1.5, n_cols - 0.5], [idx - 0.5, idx - 0.5],
            color=GRID_EDGE, lw=EDGE_LW + 0.4, zorder=4)

for s in ax.spines.values():
    s.set_visible(False)

plt.subplots_adjust(left=0.05, right=0.99, top=0.98, bottom=0.06)

# === Save figure ===
plt.savefig(output_png, dpi=dpi)
print(f"Saved: {os.path.abspath(output_png)}")
