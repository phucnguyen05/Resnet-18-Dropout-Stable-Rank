import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== CONFIG ======
EXCEL_FILE = "results.xlsx"
OUTPUT_DIR = "results_plots"
TOPK = 100  # number of singular values to consider
NORMALIZATION = "energy"  # choose: "energy" or "max"
# ====================

def parse_svals(s):
    """Convert '2211.1,862.5,...' -> np.array([...])"""
    if isinstance(s, str):
        return np.array([float(v) for v in s.split(",") if v.strip() != ""])
    elif isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=float)
    else:
        return np.array([])

def normalize_svals(svals, mode="energy"):
    """Normalize singular values either by total energy or by max"""
    svals = svals[:TOPK]
    if len(svals) == 0:
        return np.zeros(TOPK)
    if mode == "energy":
        denom = np.sqrt(np.sum(svals ** 2)) + 1e-8
    elif mode == "max":
        denom = svals[0] + 1e-8
    else:
        raise ValueError("mode must be 'energy' or 'max'")
    return svals / denom

def compute_auc(svals):
    """Compute area under curve of normalized singular values"""
    trapz = getattr(np, "trapezoid", np.trapz)  # backward compatibility
    return trapz(svals[:TOPK], dx=1)

# ====== LOAD DATA ======
df = pd.read_excel(EXCEL_FILE)

# Ensure proper column names
required_cols = {"dropout", "seed", "epoch", "layer", "singular_values"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

# ====== PARSE + NORMALIZE + COMPUTE AUC ======
df["svals"] = df["singular_values"].apply(parse_svals)
df["svals_norm"] = df["svals"].apply(lambda x: normalize_svals(x, NORMALIZATION))
df["AUC"] = df["svals_norm"].apply(compute_auc)

# ====== AGGREGATE ======
auc_summary = (
    df.groupby(["dropout", "layer"])["AUC"]
      .agg(["mean", "std"])
      .reset_index()
)

# ====== SAVE RESULTS ======
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_csv = os.path.join(OUTPUT_DIR, "auc_summary.csv")
auc_summary.to_csv(output_csv, index=False)
print(f"✅ Saved AUC summary to: {output_csv}")

# ====== PLOT 1: SPECTRAL DECAY ======
for layer in df["layer"].unique():
    plt.figure(figsize=(7,5))
    for dropout, subdf in df[df["layer"] == layer].groupby("dropout"):
        # Stack spectra across seeds
        svals_stack = np.stack(subdf["svals_norm"].values)
        mean_svals = svals_stack.mean(axis=0)
        std_svals = svals_stack.std(axis=0)
        
        plt.semilogy(mean_svals[:TOPK], label=f"dropout={dropout}")
        plt.fill_between(
            np.arange(TOPK),
            mean_svals[:TOPK] - std_svals[:TOPK],
            mean_svals[:TOPK] + std_svals[:TOPK],
            alpha=0.2
        )
    plt.title(f"{layer} — Mean normalized singular values ({NORMALIZATION} norm)")
    plt.xlabel("Index (i)")
    plt.ylabel("Normalized σᵢ")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.4)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"spectral_decay_{layer}.png")
    plt.savefig(out_path, dpi=300)
    print(f"📊 Saved plot: {out_path}")
    plt.close()

# ====== PLOT 2: AUC VS DROPOUT ======
for layer in df["layer"].unique():
    layer_auc = auc_summary[auc_summary["layer"] == layer]
    plt.figure(figsize=(6,4))
    plt.errorbar(layer_auc["dropout"], layer_auc["mean"], 
                 yerr=layer_auc["std"], fmt="-o", capsize=4)
    plt.title(f"AUC of top-{TOPK} normalized singular values vs Dropout ({layer})")
    plt.xlabel("Dropout rate")
    plt.ylabel("AUC (normalized spectrum)")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"auc_vs_dropout_{layer}.png")
    plt.savefig(out_path, dpi=300)
    print(f"📈 Saved plot: {out_path}")
    plt.close()

print("✅ All results saved successfully.")
