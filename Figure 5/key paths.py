import numpy as np
import pandas as pd
import os

def generate_explicit_path_contributions(
    file_path, sheet_name, start_row, output_prefix,
    max_order=3, threshold=1e-6
):
    output_folder = "./output_key paths"
    mapping_path = "./Conversion of row to column.xlsx"
    os.makedirs(output_folder, exist_ok=True)


    A = pd.read_excel(file_path, sheet_name="A", header=None).values
    y = pd.read_excel(file_path, sheet_name="y", header=None).values.flatten()
    E = pd.read_excel(file_path, sheet_name="E", header=None).values.flatten()
    n = A.shape[0]


    mapping_df = pd.read_excel(mapping_path, sheet_name=sheet_name, header=None)
    id_to_name = dict(zip(mapping_df.iloc[:, 0], mapping_df.iloc[:, 1].astype(str).str.strip()))
    get_name = lambda idx: id_to_name.get(idx, '未知')  # ❗️去除前缀数字和横杠


    paths = []
    current_paths = [[j] for j in range(n) if y[j] != 0]

    for order in range(1, max_order + 1):
        next_paths = []
        for path in current_paths:
            last_node = path[-1]
            for i in range(n):
                if A[i, last_node] == 0:
                    continue
                new_path = path + [i]

                coeff = 1.0
                for k in range(len(new_path) - 1):
                    coeff *= A[new_path[k + 1], new_path[k]]

                j = new_path[0]
                i_end = new_path[-1]
                contrib = E[i_end] * coeff * y[j]

                if contrib < threshold:
                    continue

                path_names = [get_name(p) for p in new_path]
                path_dict = {f"Node_{i}": name for i, name in enumerate(path_names)}
                path_dict.update({
                    "Order": order,
                    "Contribution": contrib
                })
                paths.append(path_dict)
                next_paths.append(new_path)
        current_paths = next_paths

    df_paths = pd.DataFrame(paths)
    max_path_len = max([len([col for col in row if col.startswith("Node_")]) for row in df_paths.to_dict(orient="records")], default=0)
    for i in range(max_path_len):
        col = f"Node_{i}"
        if col not in df_paths.columns:
            df_paths[col] = ""

    exclusion_keywords = {
        "Fish": ["配合饲料", "柴油", "木板", "钢铁", "塑料", "货船运输", "幼苗", "电力", "增氧机", "水泥", "砖头", "鼓风机", "发电机", "水车", "搅拌机", "自然收支", "鱼类养殖"],
        "Shrimp": ["配合饲料", "电力", "塑料", "水泥", "砖头", "水车", "漂白粉", "虾苗", "自然收支过程", "对虾养殖"],
        "Shellfish": ["龙须菜", "工作船", "塑料", "电力", "柴油", "货车运输", "海运", "饲料船运", "幼苗", "自然收支过程", "贝类养殖"],
        "Seaweed": ["塑料（聚乙烯）", "聚丙烯绳", "桩", "浮球", "矿泉水瓶", "海绵", "工作船", "柴油", "自然收支", "藻类养殖"]
    }
    excluded_keywords = exclusion_keywords.get(sheet_name, [])

    def is_excluded_terminal(row):
        if row["Order"] != 1:
            return False
        terminal_node = row.get(f"Node_{row['Order']}")
        return any(kw in terminal_node for kw in excluded_keywords)

    df_paths = df_paths[~df_paths.apply(is_excluded_terminal, axis=1)]
    df_paths = df_paths.sort_values(by="Contribution", ascending=False)
    total_contrib = df_paths["Contribution"].sum()
    df_paths["Share_in_Total"] = df_paths["Contribution"] / total_contrib

    node_cols = [f"Node_{i}" for i in range(max_path_len)]
    output_cols = ["Order"] + node_cols + ["Contribution", "Share_in_Total"]
    df_paths = df_paths[output_cols]

    def find_responsible_department(row):
        for i in range(max_path_len):
            node = row.get(f"Node_{i}", "")
            if not any(kw in node for kw in excluded_keywords):
                return node
        return "Unknown"

    df_paths["Department"] = df_paths.apply(find_responsible_department, axis=1)

    df_dept = (
        df_paths.groupby("Department")["Contribution"]
        .sum()
        .reset_index()
        .rename(columns={"Contribution": "Total_Contribution"})
        .sort_values(by="Total_Contribution", ascending=False)
    )
    df_dept["Share_in_Total"] = df_dept["Total_Contribution"] / total_contrib

    output_file = os.path.join(output_folder, f"{output_prefix}_ExplicitPaths.xlsx")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_paths.to_excel(writer, sheet_name="PathContribution", index=False)
        df_dept.to_excel(writer, sheet_name="DepartmentContribution", index=False)

    print(f"{output_prefix} finish, write in: {output_file}")

# === 示例运行 ===
if __name__ == "__main__":
    generate_explicit_path_contributions(
        "./Seaweed Average.xlsx", "Seaweed", 10, "Seaweed", max_order=3
    )
    generate_explicit_path_contributions(
        "./Fish Average.xlsx", "Fish", 17, "Fish", max_order=3
    )
    generate_explicit_path_contributions(
        "./Shellfish Average .xlsx", "Shellfish", 11, "Shellfish", max_order=3
    )
    generate_explicit_path_contributions(
        "./Shrimp Average.xlsx", "Shrimp", 10, "Shrimp", max_order=3
    )
