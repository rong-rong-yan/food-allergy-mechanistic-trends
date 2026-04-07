import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

# =========================================================
# Input files
# =========================================================
THEME_FILE = "processed/theme_yearly_counts.csv"
YEARLY_FILE = "processed/yearly_publication_counts.csv"

OUTFILE_PROPORTIONS = "processed/theme_yearly_proportions.csv"
OUTFILE_MODEL_SUMMARY = "processed/theme_model_summary.csv"

# =========================================================
# Load and merge
# =========================================================
theme_df = pd.read_csv(THEME_FILE)
yearly_df = pd.read_csv(YEARLY_FILE)

df = pd.merge(theme_df, yearly_df, on="pub_year", how="inner")

# Keep only 2015-2025
df = df[(df["pub_year"] >= 2015) & (df["pub_year"] <= 2025)].copy()

# Center year
df["year_centered"] = df["pub_year"] - df["pub_year"].mean()
df["year_centered_sq"] = df["year_centered"] ** 2

theme_cols = [
    "theme_tolerance_regulation",
    "theme_sensitization_ige",
    "theme_microbiome",
    "theme_barrier_epithelial",
]

# =========================================================
# Compute proportions and save
# =========================================================
for col in theme_cols:
    prop_col = col + "_prop"
    df[prop_col] = df[col] / df["n_publications"]

df.to_csv(OUTFILE_PROPORTIONS, index=False)
print(f"Saved theme proportions to: {OUTFILE_PROPORTIONS}")
print()
print(df[["pub_year", "n_publications"] + theme_cols + [c + "_prop" for c in theme_cols]])
print()

# =========================================================
# Helper: likelihood-ratio test
# =========================================================
def lr_test(model_small, model_large):
    lr_stat = 2 * (model_large.llf - model_small.llf)
    df_diff = model_large.df_model - model_small.df_model
    p_value = chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value

# =========================================================
# Helper: fit models for one theme
# =========================================================
def fit_theme_models(data, theme_col, denom_col="n_publications"):
    tmp = data.copy()

    # successes and proportions
    tmp["successes"] = tmp[theme_col]
    tmp["failures"] = tmp[denom_col] - tmp[theme_col]
    tmp["share"] = tmp["successes"] / tmp[denom_col]

    # Model 0: intercept only
    X0 = pd.DataFrame({"const": np.ones(len(tmp))})
    m0 = sm.GLM(
        tmp["share"],
        X0,
        family=sm.families.Binomial(),
        var_weights=tmp[denom_col]
    ).fit()

    # Model 1: linear
    X1 = sm.add_constant(tmp[["year_centered"]])
    m1 = sm.GLM(
        tmp["share"],
        X1,
        family=sm.families.Binomial(),
        var_weights=tmp[denom_col]
    ).fit()

    # Model 2: quadratic
    X2 = sm.add_constant(tmp[["year_centered", "year_centered_sq"]])
    m2 = sm.GLM(
        tmp["share"],
        X2,
        family=sm.families.Binomial(),
        var_weights=tmp[denom_col]
    ).fit()

    # LR tests
    lr_0_vs_1 = lr_test(m0, m1)
    lr_1_vs_2 = lr_test(m1, m2)
    lr_0_vs_2 = lr_test(m0, m2)

    # Extract coefficients
    beta1 = m1.params.get("year_centered", np.nan)
    se1 = m1.bse.get("year_centered", np.nan)
    p1 = m1.pvalues.get("year_centered", np.nan)

    beta2_lin = m2.params.get("year_centered", np.nan)
    beta2_quad = m2.params.get("year_centered_sq", np.nan)
    p2_lin = m2.pvalues.get("year_centered", np.nan)
    p2_quad = m2.pvalues.get("year_centered_sq", np.nan)

    result_row = {
        "theme": theme_col,
        "overall_mean_share": tmp["successes"].sum() / tmp[denom_col].sum(),

        "linear_beta": beta1,
        "linear_or_per_year": np.exp(beta1),
        "linear_p": p1,

        "quadratic_beta_linear_term": beta2_lin,
        "quadratic_beta_sq_term": beta2_quad,
        "quadratic_p_linear_term": p2_lin,
        "quadratic_p_sq_term": p2_quad,

        "lr_intercept_vs_linear_stat": lr_0_vs_1[0],
        "lr_intercept_vs_linear_df": lr_0_vs_1[1],
        "lr_intercept_vs_linear_p": lr_0_vs_1[2],

        "lr_linear_vs_quadratic_stat": lr_1_vs_2[0],
        "lr_linear_vs_quadratic_df": lr_1_vs_2[1],
        "lr_linear_vs_quadratic_p": lr_1_vs_2[2],

        "lr_intercept_vs_quadratic_stat": lr_0_vs_2[0],
        "lr_intercept_vs_quadratic_df": lr_0_vs_2[1],
        "lr_intercept_vs_quadratic_p": lr_0_vs_2[2],
    }

    return result_row, m0, m1, m2, tmp

# =========================================================
# Fit models for all themes
# =========================================================
summary_rows = []

for theme in theme_cols:
    print("=====================================================")
    print(f"THEME: {theme}")
    print("=====================================================")

    row, m0, m1, m2, tmp = fit_theme_models(df, theme)
    summary_rows.append(row)

    print("\n--- Yearly data ---")
    print(tmp[["pub_year", "n_publications", "successes", "share"]])

    print("\n--- Intercept-only model ---")
    print(m0.summary())

    print("\n--- Linear model ---")
    print(m1.summary())

    print("\n--- Quadratic model ---")
    print(m2.summary())

    print("\n--- Key tests ---")
    print(f"Intercept vs linear:    p = {row['lr_intercept_vs_linear_p']:.6g}")
    print(f"Linear vs quadratic:    p = {row['lr_linear_vs_quadratic_p']:.6g}")
    print(f"Intercept vs quadratic: p = {row['lr_intercept_vs_quadratic_p']:.6g}")

    print("\n--- Quick effect summary ---")
    print(f"Overall mean share: {row['overall_mean_share']:.4%}")
    print(f"Linear OR per year: {row['linear_or_per_year']:.4f}")
    print(f"Linear trend p: {row['linear_p']:.6g}")
    print(f"Quadratic term p: {row['quadratic_p_sq_term']:.6g}")
    print()

# =========================================================
# Save model summary table
# =========================================================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTFILE_MODEL_SUMMARY, index=False)

print("=====================================================")
print(f"Saved model summary to: {OUTFILE_MODEL_SUMMARY}")
print("=====================================================")
print(summary_df)

# =========================================================
# Optional interpretation helper
# =========================================================
print("\nINTERPRETATION HINT:")
print("- If intercept vs linear is significant: evidence of a linear trend.")
print("- If linear vs quadratic is significant: curved pattern fits better than straight line.")
print("- If intercept vs quadratic is significant: theme share changes over time overall.")
print("- Negative quadratic term often suggests rise-then-plateau or rise-then-fall.")