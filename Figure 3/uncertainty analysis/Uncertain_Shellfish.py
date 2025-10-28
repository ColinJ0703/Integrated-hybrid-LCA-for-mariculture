import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')


base_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(base_dir, "Shellfish")
output_folder = os.path.join(base_dir, "output")
os.makedirs(output_folder, exist_ok=True)


file_list = [f for f in os.listdir(folder_path) if f.endswith(".xlsx") and not f.startswith("~$")]


dM_c = pd.read_excel(os.path.join(folder_path, "Shellfish_Process.xlsx"), sheet_name="Consistency matrix", header=None)
M_c = dM_c.iloc[2:148, 1:12].values.astype(float)

dM_t = pd.read_excel(os.path.join(folder_path, "Shellfish_Process.xlsx"), sheet_name="Technical coefficient matrix", header=None)
M_t = dM_t.iloc[2:148, 1:12].values.astype(float)

dM_p = pd.read_excel(os.path.join(folder_path, "Shellfish_Process.xlsx"), sheet_name="Unit price", header=None)


num_simulations = 1000

def process_file(file_name):
    file_path = os.path.join(folder_path, file_name)

    if "sh" in file_name.lower():
        Type = 1
    elif "so" in file_name.lower():
        Type = 2
    else:
        return file_name, None

    dA = pd.read_excel(file_path, sheet_name="A", header=None)
    A_original = dA.values
    rows = A_original.shape[0]
    rows_Ap = rows - 146

    M_p1 = pd.to_numeric(dM_p.iloc[1, 1:12], errors="coerce").values.reshape(1, 11)
    M_p2 = pd.to_numeric(dM_p.iloc[2, 1:12], errors="coerce").values.reshape(1, 11)

    dy = pd.read_excel(file_path, sheet_name="y", header=None)
    y = dy.values
    dE = pd.read_excel(file_path, sheet_name="E", header=None)
    E = dE.values
    E_T = E.T

    CF_results = np.zeros(num_simulations)

    for i in range(num_simulations):
        A = A_original.copy()

        M_p = M_p1 if Type == 1 else M_p2
        M_p = M_p.reshape(1, 11)

        M_p_left = np.where(M_p == 0, np.nan, M_p * 0.5)
        M_p_right = np.where(M_p == 0, np.nan, M_p * 1.5)

        M_p_sample = np.zeros_like(M_p)
        for idx in range(11):
            if not np.isnan(M_p_left[0, idx]) and not np.isnan(M_p_right[0, idx]):
                M_p_sample[0, idx] = np.random.triangular(M_p_left[0, idx], M_p[0, idx], M_p_right[0, idx])
            else:
                M_p_sample[0, idx] = 0

        Cu = M_c * M_p_sample * M_t
        if Cu.shape == (146, 11):
            A[rows_Ap:rows_Ap + 146, 0:11] = Cu

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
    output_path = os.path.join(output_folder, "MonteCarlo_Results_Shellfish.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        CF_df.to_excel(writer, sheet_name="MonteCarlo_Results", index=False)

    df_results = pd.read_excel(output_path, sheet_name="MonteCarlo_Results")
    case_names = df_results.columns.tolist()

    adjustments = {}
    for case in case_names:
        file_path = os.path.join(folder_path, case + ".xlsx")
        try:
            df_adjust = pd.read_excel(file_path, sheet_name="Result", usecols="G", skiprows=2, nrows=1, header=None)
            adjustments[case] = df_adjust.iloc[0, 0]
        except Exception as e:
            adjustments[case] = 0

    for case in case_names:
        df_results[case] -= adjustments[case]

    means = df_results.mean().values
    errors = 1.96 * df_results.std().values

    num_cases = len(case_names)
    colors = sns.color_palette("coolwarm", num_cases)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(num_cases)

    bars = ax.bar(x_pos, means, yerr=errors, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([label.replace(' ', '\n') for label in case_names], fontsize=10, rotation=0, ha="center")
    ax.set_ylabel("IO-based Carbon Footprint (kg CO2 eq)")

    for bar, err in zip(bars, errors):
        ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=err, fmt='none', color='gray', capsize=5)

    plot_path_bar = os.path.join(output_folder, "Figure 5_Shellfish_Adjusted.png")
    plt.savefig(plot_path_bar, bbox_inches='tight')
    print(f"Adjusted Bar plot with 95% CI saved at {plot_path_bar}")
