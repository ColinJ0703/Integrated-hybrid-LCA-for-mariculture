# GSA_simulation.py
import warnings
import os
import time
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

base_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(base_dir, "output")
os.makedirs(output_folder, exist_ok=True)

CONFIG = {
    "Fish": {
        "folder": "Fish",
        "param_file": "Fish_Cu.xlsx",
        "mc_range": (5, 151, 3, 20),
        "mt_range": (3, 149, 1, 18),
        "mp_rows": {"fg": 3, "fl": 4},
        "a_rows": 17, "a_col": 16,
        "cu_insert_start": 10,
        "sheet_mc": "Step1_concordance matrix2",
        "sheet_mt": "Step3_Technical coeffi matrix",
        "sheet_mp": "Step2_unit prices",
        "col_end_inclusive": False
    },
    "Shrimp": {
        "folder": "Shrimp",
        "param_file": "Shrimp_Process.xlsx",
        "mc_range": (2, 148, 1, 10),
        "mt_range": (2, 148, 1, 10),
        "mp_rows": {"sp": 2},
        "a_rows": 10, "a_col": 9,
        "cu_insert_start": 10,
        "sheet_mc": "Consistency matrix",
        "sheet_mt": "Technical coefficient matrix",
        "sheet_mp": "Unit price",
        "col_end_inclusive": True
    },
    "Shellfish": {
        "folder": "Shellfish",
        "param_file": "Shellfish_Process.xlsx",
        "mc_range": (2, 148, 1, 11),
        "mt_range": (2, 148, 1, 11),
        "mp_rows": {"sh": 1, "so": 2, "shf": 1},
        "a_rows": 11, "a_col": 10,
        "cu_insert_start": 10,
        "sheet_mc": "Consistency matrix",
        "sheet_mt": "Technical coefficient matrix",
        "sheet_mp": "Unit price",
        "col_end_inclusive": True
    },
    "Macroalgae": {
        "folder": "Macroalgae",
        "param_file": "Macroalgae_Process.xlsx",
        "mc_range": (2, 148, 1, 10),
        "mt_range": (2, 148, 1, 10),
        "mp_rows": {"sg": 2, "ss": 3},
        "a_rows": 10, "a_col": 9,
        "cu_insert_start": 10,
        "sheet_mc": "Consistency matrix",
        "sheet_mt": "Technical coefficient matrix",
        "sheet_mp": "Unit price",
        "col_end_inclusive": True
    }
}


N = 2 ** int(np.log2(10000))

def _col_slice(col_start: int, col_end: int, inclusive: bool) -> slice:

    return slice(col_start, col_end + 1 if inclusive else col_end)

def triangular_mapping(samples, means):
    mapped = []
    for i in range(len(means)):
        c = means[i]
        if c == 0 or np.isnan(c):
            mapped.append(np.full(samples.shape[0], 0.0))
            continue
        a = c * 0.5
        b = c * 1.5
        u = samples[:, i]
        threshold = (c - a) / (b - a) if (b - a) != 0 else 0.5
        mapped_i = np.where(
            u < threshold,
            a + np.sqrt(u * (b - a) * (c - a)),
            b - np.sqrt((1 - u) * (b - a) * (b - c))
        )
        mapped.append(mapped_i)
    return np.array(mapped).T

def solve_lca(A, y):
    try:
        D = np.eye(A.shape[0]) - A
        x = np.linalg.solve(D, y)
        return np.sum(x)
    except np.linalg.LinAlgError:
        return np.nan

def run_one(params, A_original, y, A_vars, M_p, M_c, M_t, config):
    A_perturbed = A_original.copy()
    A_perturbed[:config["a_rows"], config["a_col"]] = A_vars * params[:len(A_vars)]
    M_p_sample = M_p * params[len(A_vars):].reshape(M_p.shape)
    Cu = M_c * M_p_sample * M_t
    Cu_shape = Cu.shape
    A_target_shape = A_perturbed[config["cu_insert_start"]:config["cu_insert_start"] + Cu_shape[0], 0:Cu_shape[1]].shape
    if Cu_shape == A_target_shape:
        A_perturbed[config["cu_insert_start"]:config["cu_insert_start"] + Cu_shape[0], 0:Cu_shape[1]] = Cu
    return solve_lca(A_perturbed, y)

def lca_model(file_name, config, M_c, M_t, dM_p):
    folder_path = os.path.join(base_dir, config["folder"])
    file_path = os.path.join(folder_path, file_name)
    try:
        xls = pd.ExcelFile(file_path)
        if not {"A", "y"}.issubset(set(xls.sheet_names)):
            print(f"{file_name} is missing sheets A or y, skipping.")
            return None, None
        A_original = pd.read_excel(xls, sheet_name="A", header=None).fillna(0).values
        y = pd.read_excel(xls, sheet_name="y", header=None).values
    except Exception as e:
        print(f"{file_name} failed to open: {e}")
        return None, None

    A_vars = A_original[:config["a_rows"], config["a_col"]].flatten()


    M_p = None
    try:
        if "M_p" in xls.sheet_names:
            M_p_df = pd.read_excel(xls, sheet_name="M_p", header=None)
            M_p_array = M_p_df.dropna(axis=1, how='all').values
            if M_p_array.shape == M_c.shape:
                M_p = M_p_array
                print(f"{file_name} successfully read M_p sheet, shape matches {M_p.shape}")
            else:
                print(f"{file_name} M_p sheet dimension {M_p_array.shape} does not match M_c {M_c.shape}, fallback to config")
    except Exception as e:
        print(f"{file_name} unable to read M_p sheet: {e}, fallback to config")


    if M_p is None:
        matched = False
        col_start, col_end = config["mc_range"][2], config["mc_range"][3]
        inclusive = bool(config.get("col_end_inclusive", False))
        col_sl = _col_slice(col_start, col_end, inclusive)

        ncols = M_c.shape[1]
        for key, row in config["mp_rows"].items():
            if key in file_name.lower():
                row_slice = dM_p.iloc[row, col_sl]

                row_vals = pd.to_numeric(row_slice, errors="coerce").astype(float).values
                row_vals = np.nan_to_num(row_vals, nan=0.0)

                if row_vals.size != ncols:
                    row_all = pd.to_numeric(dM_p.iloc[row, :], errors="coerce").astype(float).values
                    row_all = np.nan_to_num(row_all, nan=0.0)
                    if row_all.size >= ncols:
                        row_vals = row_all[-ncols:]
                    else:
                        raise ValueError(f"{file_name}: M_p row numeric length {row_all.size} < needed {ncols}")
                M_p = row_vals.reshape(1, ncols)
                matched = True
                print(f"{file_name} using configured {key} row as M_p (cols {col_start}:{col_end}{' inclusive' if inclusive else ''}), shape: {M_p.shape}")
                break
        if not matched:
            print(f"{file_name} does not match any config key, skipping.")
            return None, None

    M_p_vars = M_p.flatten()
    k = len(A_vars) + len(M_p_vars)

    problem = {
        'num_vars': k,
        'names': [f"A_{i}" for i in range(len(A_vars))] + [f"M_p_{i}" for i in range(len(M_p_vars))],
        'bounds': [[0, 1]] * k
    }

    unit_samples = sobol.sample(problem, N, calc_second_order=False)
    means = np.concatenate([A_vars, M_p_vars])
    if np.any(means == 0):
        print("Warning: zero found in parameter means, using 0 instead")
    param_values = triangular_mapping(unit_samples, means)

    print(f"{file_name} simulation started, total {N} runs...")
    CF_results = Parallel(n_jobs=6)(
        delayed(run_one)(params, A_original, y, A_vars, M_p, M_c, M_t, config)
        for params in tqdm(param_values, desc=f"Running {file_name}")
    )
    CF_results = np.array(CF_results)
    CF_results = CF_results[~np.isnan(CF_results)]

    if len(CF_results) < 100:
        print(f"{file_name} only {len(CF_results)} valid results, skipping analysis.")
        return None, None

    print(f"{file_name} completed, valid samples: {len(CF_results)}, range: {np.min(CF_results):.2f} ~ {np.max(CF_results):.2f}")
    return CF_results, problem

def process_all(config_key):
    config = CONFIG[config_key]
    folder_path = os.path.join(base_dir, config["folder"])
    param_file = os.path.join(folder_path, config["param_file"])

    dM_c = pd.read_excel(param_file, sheet_name=config["sheet_mc"], header=None)
    dM_t = pd.read_excel(param_file, sheet_name=config["sheet_mt"], header=None)
    dM_p = pd.read_excel(param_file, sheet_name=config["sheet_mp"], header=None)

    r0, r1, c0, c1 = config["mc_range"]
    inclusive = bool(config.get("col_end_inclusive", False))
    col_sl = _col_slice(c0, c1, inclusive)


    M_c = dM_c.iloc[r0:r1, col_sl].values.astype(float)
    mt_r0, mt_r1, mt_c0, mt_c1 = config["mt_range"]
    mt_col_sl = _col_slice(mt_c0, mt_c1, inclusive)
    M_t = dM_t.iloc[mt_r0:mt_r1, mt_col_sl].values.astype(float)

    file_list = [f for f in os.listdir(folder_path) if f.endswith(".xlsx") and not f.startswith("~$")]
    for file_name in file_list:
        print(f"\nProcessing: {file_name} ({config_key})")
        CF_results, problem = lca_model(file_name, config, M_c, M_t, dM_p)
        if CF_results is None:
            continue
        try:
            Si = sobol_analyze.analyze(problem, CF_results, calc_second_order=False, print_to_console=False)
            ST_sum = np.sum(Si["ST"])
            total_variance = np.var(CF_results)
            Si["ST"] = Si["ST"] / ST_sum
            Si["Total Variance"] = total_variance
        except Exception as e:
            print(f"Sobol analysis failed: {e}")
            continue
        output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_Sobol.xlsx")
        df = pd.DataFrame({"ST": Si["ST"], "Total Variance": Si["Total Variance"]}, index=problem["names"])
        df.to_excel(output_path)
        print(f"Output saved to: {output_path}")


SPECIES_CODE_MAP = {
    "FG": "grouper",
    "FL": "large_yellow_croaker",
    "SP": "shrimp_penaeus_vannamei",
    "SO": "oyster",
    "SH": "blacklip_abalone",
    "SHF": "greenlip_abalone_hybrid",
    "SG": "kelp",
    "SS": "gracilaria",
}
FALLBACK_KEYWORDS = {
    "grouper": "grouper",
    "yellow croaker": "large_yellow_croaker",
    "croaker": "large_yellow_croaker",
    "oyster": "oyster",
    "abalone": "blacklip_abalone",
    "shrimp": "shrimp_penaeus_vannamei",
    "kelp": "kelp",
    "gracilaria": "gracilaria",
}

def infer_species_from_filename(xlsx_name: str) -> str | None:
    stem = os.path.splitext(os.path.basename(xlsx_name))[0]
    stem = re.sub(r"_Sobol$", "", stem, flags=re.IGNORECASE)
    for k in sorted(SPECIES_CODE_MAP.keys(), key=len, reverse=True):
        if stem.upper().startswith(k):
            return SPECIES_CODE_MAP[k]
    m = re.match(r"^\s*([A-Za-z]+)", stem)
    if m:
        code = m.group(1).upper()
        if code in SPECIES_CODE_MAP:
            return SPECIES_CODE_MAP[code]
    low = stem.lower()
    for kw, sp in FALLBACK_KEYWORDS.items():
        if kw in low:
            return sp
    return None

def read_sobol_file(fp: str) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(fp, sheet_name=0, index_col=0)
        keep = []
        for c in df.columns:
            lc = str(c).strip().lower()
            if lc == "st" or "variance" in lc:
                keep.append(c)
        keep = list(dict.fromkeys(keep))
        if not keep:
            return None
        return df[keep]
    except Exception:
        return None

def aggregate_species_means():
    files = [f for f in os.listdir(output_folder) if f.endswith(".xlsx") and f.lower().endswith("_sobol.xlsx")]
    if not files:
        print("No *_Sobol.xlsx found, skip aggregation.")
        return

    buckets = defaultdict(list)
    for fname in files:
        species = infer_species_from_filename(fname)
        if not species:
            print(f"Skip (species unknown): {fname}")
            continue
        df = read_sobol_file(os.path.join(output_folder, fname))
        if df is None or df.empty:
            print(f"Skip (invalid structure or empty): {fname}")
            continue
        buckets[species].append(df)

    if not buckets:
        print("No data to aggregate.")
        return

    combined_path = os.path.join(output_folder, "Sobol_means_by_species.xlsx")
    with pd.ExcelWriter(combined_path, engine="openpyxl") as writer:
        for species, dfs in buckets.items():
            all_idx = dfs[0].index
            for df in dfs[1:]:
                all_idx = all_idx.union(df.index)
            aligned = [df.reindex(all_idx) for df in dfs]

            out = pd.DataFrame(index=all_idx)
            if any("ST" == c for df in aligned for c in df.columns):
                out["ST_mean"] = pd.concat([d[["ST"]] for d in aligned if "ST" in d.columns], axis=1).mean(axis=1, skipna=True)
            if any("Total Variance" == c for df in aligned for c in df.columns):
                out["Total Variance_mean"] = pd.concat([d[["Total Variance"]] for d in aligned if "Total Variance" in d.columns], axis=1).mean(axis=1, skipna=True)
            out["n_files"] = len(dfs)

            def _sort_key(s: str):
                s = str(s)
                if s.startswith("A_"):
                    try:
                        return (0, int(s.split("_")[1]))
                    except Exception:
                        return (0, 9999999)
                if s.startswith("M_p_"):
                    try:
                        return (1, int(s.split("_")[2]))
                    except Exception:
                        return (1, 9999999)
                return (2, s)
            out = out.sort_index(key=lambda idx: idx.map(_sort_key))

            out.to_excel(writer, sheet_name=species, index=True)

    print(f"Aggregated means written to: {combined_path}")

if __name__ == "__main__":
    start = time.perf_counter()
    for category in ["Fish", "Shrimp", "Shellfish", "Macroalgae"]:
        process_all(category)

    aggregate_species_means()

    total_time = time.perf_counter() - start
    print(f"\nAll simulations completed, total runtime: {total_time:.2f} seconds")
