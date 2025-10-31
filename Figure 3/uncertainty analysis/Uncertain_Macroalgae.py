import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')


base_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(base_dir, "Macroalgae")
output_folder = os.path.join(base_dir, "output")
os.makedirs(output_folder, exist_ok=True)


file_list = [f for f in os.listdir(folder_path) if f.endswith(".xlsx") and not f.startswith("~$")]


dM_c = pd.read_excel(os.path.join(folder_path, "Macroalgae_Process.xlsx"), sheet_name="Consistency matrix", header=None)
M_c = dM_c.iloc[2:148, 1:11].values.astype(float)

dM_t = pd.read_excel(os.path.join(folder_path, "Macroalgae_Process.xlsx"), sheet_name="Technical coefficient matrix", header=None)
M_t = dM_t.iloc[2:148, 1:11].values.astype(float)

dM_p = pd.read_excel(os.path.join(folder_path, "Macroalgae_Process.xlsx"), sheet_name="Unit price", header=None)


num_simulations = 1000

def process_file(file_name):
    file_path = os.path.join(folder_path, file_name)

    if "mg" in file_name.lower():
        Type = 1
    elif "ms" in file_name.lower():
        Type = 2
    else:
        return file_name, None

    dA = pd.read_excel(file_path, sheet_name="A", header=None)
    A_original = dA.values
    rows = A_original.shape[0]
    rows_Ap = rows - 146

    M_p1 = pd.to_numeric(dM_p.iloc[2, 1:11], errors="coerce").values.reshape(1, 10)
    M_p2 = pd.to_numeric(dM_p.iloc[3, 1:11], errors="coerce").values.reshape(1, 10)

    dy = pd.read_excel(file_path, sheet_name="y", header=None)
    y = dy.values
    dE = pd.read_excel(file_path, sheet_name="E", header=None)
    E = dE.values
    E_T = E.T

    CF_results = np.zeros(num_simulations)

    for i in range(num_simulations):
        A = A_original.copy()
        M_p = M_p1 if Type == 1 else M_p2
        M_p = M_p.reshape(1, 10)

        M_p_left = np.where(M_p == 0, np.nan, M_p * 0.5)
        M_p_right = np.where(M_p == 0, np.nan, M_p * 1.5)
        M_p_sample = np.zeros_like(M_p)

        for idx in range(10):
            if not np.isnan(M_p_left[0, idx]) and not np.isnan(M_p_right[0, idx]):
                M_p_sample[0, idx] = np.random.triangular(M_p_left[0, idx], M_p[0, idx], M_p_right[0, idx])
            else:
                M_p_sample[0, idx] = 0

        Cu = M_c * M_p_sample * M_t
        if Cu.shape == (146, 10):
            A[rows_Ap:rows_Ap + 146, 0:10] = Cu

        I = np.eye(rows)
        D = I - A
        try:
            T = np.linalg.inv(D)
        except np.linalg.LinAlgError:
            continue

        x = np.dot(T, y)
        CF = np.dot(E_T, x)

        CF_results[i] = CF

    return file_name, CF_results

if __name__ == "__main__":
    with Pool() as pool:
        results = pool.map(process_file, file_list)

    CF_dict = {}
    for file_name, CF_array in results:
        if CF_array is not None:
            case_name = file_name.replace(".xlsx", "")
            CF_dict[case_name] = CF_array

    CF_df = pd.DataFrame(CF_dict)
    output_path = os.path.join(output_folder, "MonteCarlo_Results_Macroalgae.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        CF_df.to_excel(writer, sheet_name="MonteCarlo_Results", index=False)

    df_results = pd.read_excel(output_path, sheet_name="MonteCarlo_Results")

    adjustment_values = {}
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            adj_value = pd.read_excel(file_path, sheet_name="Result", usecols="G", skiprows=2, nrows=1, header=None).values[0, 0]
            adjustment_values[file_name.replace(".xlsx", "")] = adj_value
        except:
            adjustment_values[file_name.replace(".xlsx", "")] = 0

    adjusted_df = df_results.copy()
    for case in df_results.columns:
        if case in adjustment_values:
            adjusted_df[case] = df_results[case] - adjustment_values[case]


    means = adjusted_df.mean().values
    errors = 1.96 * adjusted_df.std().values

    num_cases = len(df_results.columns)
    colors = sns.color_palette("coolwarm", num_cases)

    fig, ax = plt.subplots(figsize=(12, 8))
    x_pos = np.arange(num_cases)

    bars = ax.bar(x_pos, means, yerr=errors, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_results.columns, fontsize=10, rotation=0, ha="center", wrap=True)
    ax.set_ylabel("IO-based Carbon Footprint (kg CO2 eq)")

    for bar, err in zip(bars, errors):
        ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=err, fmt='none', color='gray', capsize=5)

    plot_path_bar = os.path.join(output_folder, "Figure 5_MacroalgaeMacroalgae_Adjusted.png")
    plt.savefig(plot_path_bar, bbox_inches='tight')
    print(f"Adjusted Bar plot with 95% CI saved at {plot_path_bar}")
