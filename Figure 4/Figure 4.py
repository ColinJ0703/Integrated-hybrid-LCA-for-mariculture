import os
import pandas as pd
import matplotlib
import warnings

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


base_dir = os.path.dirname(os.path.abspath(__file__))


excel_files = {
    "Macroalgae": os.path.join(base_dir, "Macroalgae_ExplicitPaths.xlsx"),
    "Fish": os.path.join(base_dir, "Fish_ExplicitPaths.xlsx"),
    "Shellfish": os.path.join(base_dir, "Shellfish_ExplicitPaths.xlsx"),
    "Shrimp": os.path.join(base_dir, "Shrimp_ExplicitPaths.xlsx")
}

all_data = []


for species, file_path in excel_files.items():
    try:
        df = pd.read_excel(file_path, sheet_name=1, usecols=[0, 1, 2])
        df.columns = ["Department", "Value", "Contribution"]
        df = df[["Department", "Contribution"]].dropna()


        df["Department"] = df["Department"].str.replace(r"^\d+\s*-\s*", "", regex=True)


        df_top5 = df.nlargest(5, "Contribution")
        others_contribution = 1 - df_top5["Contribution"].sum()
        df_others = pd.DataFrame({"Department": ["Others"], "Contribution": [others_contribution]})
        df_final = pd.concat([df_top5, df_others], ignore_index=True)
        df_final["Species"] = species
        all_data.append(df_final)
    except Exception as e:
        print(f" Failed to read: {file_path}, error: {e}")

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)


    pivot_df = final_df.pivot_table(
        index="Species",
        columns="Department",
        values="Contribution",
        aggfunc="sum"
    ).fillna(0)


    department_order = pivot_df.sum().sort_values(ascending=False).index.tolist()
    if "Others" in department_order:
        department_order = [d for d in department_order if d != "Others"] + ["Others"]
    pivot_df = pivot_df[department_order]


    species_order = ["Fish", "Shrimp", "Shellfish",  "Macroalgae"]
    pivot_df = pivot_df.loc[species_order]
    pivot_df *= 100


    custom_cmap = LinearSegmentedColormap.from_list(
        "blue_red_fade",
        [ "#CB181D", "#FDBDBD","#9ECAE1","#3182BD"]
    )
    n = len(pivot_df.columns)
    colors = [custom_cmap(i / (n - 1)) for i in range(n)]

    plt.rcParams['font.family'] = 'Arial'

    def wrap_labels(labels, width=12):
        return ['\n'.join(textwrap.wrap(label, width=width, break_long_words=False)) for label in labels]

    # 绘图
    plt.figure(figsize=(18, 8))
    ax = pivot_df.plot(
        kind="bar",
        stacked=True,
        color=colors,
        width=0.6,
        edgecolor='white',
        linewidth=1.5
    )

    plt.ylabel("Normalized Contribution (%)", fontsize=12)
    ax.set_xticks(range(len(pivot_df.index)))
    ax.set_xticklabels(wrap_labels(pivot_df.index, width=10), rotation=0, ha="center", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xlabel("")
    ax.tick_params(axis='y', direction='in', labelsize=12)
    ax.margins(y=0.02)


    color_map = {dep: col for dep, col in zip(pivot_df.columns, colors)}
    handles = [Patch(facecolor=color_map[dep], label=dep) for dep in pivot_df.columns]


    plt.legend(
        handles=handles,
        title="Industrial Sector",
        title_fontsize=10,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        prop={'size': 10},
        handleheight=1,
        handlelength=1,
        handletextpad=0.5,
        markerfirst=True,
        frameon=False
    )


    plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.22)
    plt.savefig("Figure 4.png", dpi=600, bbox_inches="tight")
    plt.savefig("Figure 4.tif", dpi=600, bbox_inches="tight", format="tiff")
    plt.close()
    print("The figure has been saved as: Figure 4.png")
