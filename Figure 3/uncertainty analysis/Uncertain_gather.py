import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import textwrap


matplotlib.use('Agg')


plt.rcParams['font.family'] = 'Arial'


base_dir = os.path.dirname(os.path.abspath(__file__))
mc_folder_path = os.path.join(base_dir, "output")

category_paths = {
    "Fish": os.path.join(base_dir, "Fish"),
    "Shrimp": os.path.join(base_dir, "Shrimp"),
    "Shellfish": os.path.join(base_dir, "Shellfish"),
    "Macroalgae": os.path.join(base_dir, "Macroalgae")
}


file_list = [
    "MonteCarlo_Results_Fish.xlsx",
    "MonteCarlo_Results_Shrimp.xlsx",
    "MonteCarlo_Results_Shellfish.xlsx",
    "MonteCarlo_Results_Macroalgae.xlsx"
]


custom_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
subplot_labels = ['a)', 'b)', 'c)', 'd)']  # 子图序号


error_ratio_records = []


dfs = {}
for file_name in file_list:
    df = pd.read_excel(os.path.join(mc_folder_path, file_name), sheet_name="MonteCarlo_Results")

    category = file_name.replace("MonteCarlo_Results_", "").replace(".xlsx", "")
    category_path = category_paths[category]

    bar_values = {}
    error_values = {}

    for case in df.columns:
        case_file = os.path.join(category_path, f"{case}.xlsx")

        if not os.path.exists(case_file):
            print(f"Case not found:{case_file}, skip.")
            continue

        try:
            result_df = pd.read_excel(case_file, sheet_name="Result", header=None)
            original_value = result_df.iloc[1, 6] if result_df.shape[1] > 6 else np.nan
            io_cf_value = result_df.iloc[2, 6] if result_df.shape[1] > 6 else np.nan

            if pd.isna(io_cf_value) or pd.isna(original_value):
                print(f"G2 or G3 in case {case}is NaN, skip.")
                continue

            bar_height = original_value - io_cf_value
            lower_error = np.percentile(df[case], 2.5) - original_value
            upper_error = np.percentile(df[case], 97.5) - original_value

            bar_values[case] = bar_height
            error_values[case] = [abs(lower_error), abs(upper_error)]


            if bar_height != 0:
                error_ratio_records.append({
                    "Category": category,
                    "Case": case,
                    "Bar_Height": bar_height,
                    "Lower_Error": abs(lower_error),
                    "Upper_Error": abs(upper_error),
                    "Lower_Error_Ratio": abs(lower_error) / abs(bar_height),
                    "Upper_Error_Ratio": abs(upper_error) / abs(bar_height)
                })

        except Exception as e:
            print(f"Fail to read {case_file} , error message:{e}")
            continue

    if not bar_values or not error_values:
        print(f"Data in {category} invalid, skip.")
        continue

    dfs[category] = (bar_values, error_values)


if error_ratio_records:
    error_ratio_df = pd.DataFrame(error_ratio_records)
    error_ratio_path = os.path.join(mc_folder_path, "MC_Error_Ratio.xlsx")
    error_ratio_df.to_excel(error_ratio_path, index=False)
    print(f"The error ratio table has been saved to {error_ratio_path}")

