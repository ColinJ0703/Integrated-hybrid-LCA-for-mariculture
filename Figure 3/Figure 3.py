import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
plt.rcParams["font.family"] = "Arial"

# === 文件路径 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Figure 3.xlsx")
sheet_name = "Figure 3"
error_file_path = os.path.join(current_dir, "uncertainty analysis", "Output", "MC_Error_Ratio.xlsx")

# === 读取误差棒数据 ===
error_df = pd.read_excel(error_file_path)
error_data = {}
for _, row in error_df.iterrows():
    cat = row['Category']
    case = str(row['Case']).strip()
    lower = row['Lower_Error']
    upper = row['Upper_Error']
    if cat not in error_data:
        error_data[cat] = {}
    error_data[cat][case] = (lower, upper)

# === 读取主数据 ===
df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
df.columns = df.columns.str.strip()
cat_col, case_col = df.columns[0], df.columns[1]
pos_cols = ["Carbon Footprint (Process-based)", "Carbon Footprint (IO-based)"]
df["Total_Carbon"] = df[pos_cols[0]] + df[pos_cols[1]]

cat_order = ["Fish", "Shrimp", "Shellfish", "Macroalgae"]
categories = sorted(df[cat_col].unique(), key=lambda c: cat_order.index(c))
max_cases = df.groupby(cat_col)[case_col].count().max()
x_positions = np.arange(max_cases)

# === 图像设置 ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
axes = axes.flatten()

palette = {
    "Carbon Footprint (Process-based)": "#3484c0",
    "Carbon Footprint (IO-based)": "#ec7166"
}
subplot_labels = ["a)", "b)", "c)", "d)"]

# === 绘图循环 ===
for idx, cat in enumerate(categories):
    ax = axes[idx]
    subset = df[df[cat_col] == cat].reset_index(drop=True)
    cases = subset[case_col].astype(str)
    x = x_positions[:len(cases)]

    proc_vals = subset[pos_cols[0]].values
    io_vals = subset[pos_cols[1]].values
    totals = subset["Total_Carbon"].values

    ax.bar(x, proc_vals, width=0.5, color=palette[pos_cols[0]],
           edgecolor="black", linewidth=0.75, label=pos_cols[0])
    ax.bar(x, io_vals, bottom=proc_vals, width=0.5,
           color=palette[pos_cols[1]], edgecolor="black",
           linewidth=0.75, label=pos_cols[1])

    # 添加误差棒
    lowers, uppers = [], []
    for i, c in enumerate(cases):
        err_pair = error_data.get(cat, {}).get(c, (0, 0))
        lowers.append(err_pair[0])
        uppers.append(err_pair[1])

    ax.errorbar(x, totals, yerr=[lowers, uppers], fmt="none",
                ecolor="black", elinewidth=0.5, capsize=3, zorder=10)

    ax.set_ylim(0, totals.max() * 1.25)
    ax.set_xticks(x)
    disp = (
        cases
        .str.replace(r'(?i)cage\s*culture\s*with\s*north[-–]south\s*transport',
                     'cage culture with north-south transport', regex=True)
        .str.wrap(12).str.replace('\n', '\n')
    )
    ax.set_xticklabels(disp, fontsize=9, ha="center", rotation=30, linespacing=1.3)

    ax.tick_params(axis='y', direction='in', labelsize=12, length=6, width=1)
    ax.tick_params(axis='x', direction='out', labelsize=9)

    # 中上方添加分类标题
    ax.text(0.5, 1.02, cat, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=14)

    ax.set_ylabel(r"Carbon Footprint (kg CO$_2$ eq./FU)", fontsize=12, labelpad=12)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)


fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for idx, ax in enumerate(axes[:len(categories)]):

    y_label = ax.yaxis.label
    bbox = y_label.get_tightbbox(renderer=renderer)
    fig_x = bbox.x0 / fig.bbox.width
    y = ax.get_position().y1 + 0.01

    fig.text(fig_x+0.01, y, subplot_labels[idx], fontsize=14, weight="extra bold",
             ha="right", va="bottom")


for j in range(len(categories), len(axes)):
    fig.delaxes(axes[j])


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, loc='lower center',
           bbox_to_anchor=(0.5, -0.04), frameon=False,
           prop={'size': 12})


out_path = os.path.join(current_dir, "Figure 3_original.png")
plt.savefig(out_path, dpi=600, bbox_inches="tight")
plt.close()
print(f"\nFigure saved in: {out_path}")
