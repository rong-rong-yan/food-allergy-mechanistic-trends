import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

# =========================================================
# Load yearly comparison table
# Must contain:
#   pub_year, main_count, background_count
# =========================================================
df = pd.read_csv("processed/main_vs_background_yearly.csv")

# Keep only 2015-2025
df = df[(df["pub_year"] >= 2015) & (df["pub_year"] <= 2025)].copy()

# Basic variables
df["share"] = df["main_count"] / df["background_count"]
df["failures"] = df["background_count"] - df["main_count"]

# Center year for stability
df["year_centered"] = df["pub_year"] - df["pub_year"].mean()
df["year_centered_sq"] = df["year_centered"] ** 2

print("Input data:")
print(df[["pub_year", "main_count", "background_count", "share"]])
print()

# =========================================================
# Helper function: likelihood-ratio test
# =========================================================
def lr_test(model_small, model_large, name_small="small", name_large="large"):
    """
    Likelihood-ratio test comparing nested GLM models.
    """
    lr_stat = 2 * (model_large.llf - model_small.llf)
    df_diff = model_large.df_model - model_small.df_model
    p_value = chi2.sf(lr_stat, df_diff)
    print(f"=== Likelihood-ratio test: {name_small} vs {name_large} ===")
    print(f"LR statistic = {lr_stat:.4f}")
    print(f"df = {df_diff}")
    print(f"p-value = {p_value:.6g}")
    print()
    return lr_stat, df_diff, p_value


# =========================================================
# 1) Binomial GLM: intercept only
# Tests whether a constant proportion fits all years
# =========================================================
X0 = pd.DataFrame({"const": np.ones(len(df))})

m0 = sm.GLM(
    df["share"],
    X0,
    family=sm.families.Binomial(),
    var_weights=df["background_count"]
).fit()

print("=== Model 0: intercept only ===")
print(m0.summary())
print()

overall_share = df["main_count"].sum() / df["background_count"].sum()
print(f"Overall pooled mechanistic share: {overall_share:.4%}")
print()

# =========================================================
# 2) Binomial GLM: linear trend in year
# share ~ year
# =========================================================
X1 = sm.add_constant(df[["year_centered"]])

m1 = sm.GLM(
    df["share"],
    X1,
    family=sm.families.Binomial(),
    var_weights=df["background_count"]
).fit()

print("=== Model 1: linear year trend ===")
print(m1.summary())
print()

beta1 = m1.params["year_centered"]
se1 = m1.bse["year_centered"]
or1 = np.exp(beta1)
ci1_low = np.exp(beta1 - 1.96 * se1)
ci1_high = np.exp(beta1 + 1.96 * se1)

print(f"Odds ratio per 1-year increase: {or1:.4f}")
print(f"95% CI: ({ci1_low:.4f}, {ci1_high:.4f})")
print(f"P-value for linear year term: {m1.pvalues['year_centered']:.6g}")
print()

# Compare intercept-only vs linear
lr_test(m0, m1, "intercept-only", "linear trend")

# =========================================================
# 3) Binomial GLM: quadratic trend
# share ~ year + year^2
# Useful if pattern is rise then plateau or curved
# =========================================================
X2 = sm.add_constant(df[["year_centered", "year_centered_sq"]])

m2 = sm.GLM(
    df["share"],
    X2,
    family=sm.families.Binomial(),
    var_weights=df["background_count"]
).fit()

print("=== Model 2: quadratic year trend ===")
print(m2.summary())
print()

beta_lin = m2.params["year_centered"]
beta_quad = m2.params["year_centered_sq"]

print(f"Linear term p-value: {m2.pvalues['year_centered']:.6g}")
print(f"Quadratic term p-value: {m2.pvalues['year_centered_sq']:.6g}")
print()

# Compare linear vs quadratic
lr_test(m1, m2, "linear trend", "quadratic trend")

# Also compare intercept-only vs quadratic
lr_test(m0, m2, "intercept-only", "quadratic trend")

# =========================================================
# 4) Binomial GLM: year as categorical
# share ~ C(year)
# Tests whether proportions differ by year at all
# =========================================================
year_dummies = pd.get_dummies(df["pub_year"].astype(str), prefix="year", drop_first=True)
X3 = sm.add_constant(year_dummies).astype(float)

m3 = sm.GLM(
    df["share"],
    X3,
    family=sm.families.Binomial(),
    var_weights=df["background_count"]
).fit()

print("=== Model 3: year as categorical ===")
print(m3.summary())
print()

# Compare intercept-only vs categorical
lr_test(m0, m3, "intercept-only", "categorical year")

# Compare linear vs categorical
lr_test(m1, m3, "linear trend", "categorical year")

# Compare quadratic vs categorical
lr_test(m2, m3, "quadratic trend", "categorical year")

# =========================================================
# 5) Predicted values from each model
# Useful for inspection / plotting later
# =========================================================
pred_df = df[["pub_year", "main_count", "background_count", "share", "year_centered", "year_centered_sq"]].copy()

pred_df["pred_intercept_only"] = m0.predict(X0)
pred_df["pred_linear"] = m1.predict(X1)
pred_df["pred_quadratic"] = m2.predict(X2)
pred_df["pred_categorical"] = m3.predict(X3)

pred_df.to_csv("processed/model_predicted_shares.csv", index=False)

print("Saved predicted shares to: processed/model_predicted_shares.csv")
print()
print(pred_df[[
    "pub_year",
    "share",
    "pred_intercept_only",
    "pred_linear",
    "pred_quadratic",
    "pred_categorical"
]])

# =========================================================
# 6) Simple interpretation guide
# =========================================================
print("\n================ INTERPRETATION GUIDE ================\n")
print("Model 0 (intercept only):")
print("  Assumes the mechanistic share is constant across years.\n")

print("Model 1 (linear trend):")
print("  Tests whether mechanistic share changes linearly with year.\n")

print("Model 2 (quadratic trend):")
print("  Tests whether the share follows a curved pattern,")
print("  such as increase-then-plateau or rise-then-fall.\n")

print("Model 3 (categorical year):")
print("  Tests whether proportions differ by year at all,")
print("  without assuming linearity.\n")

print("Likelihood-ratio tests:")
print("  - intercept-only vs linear: is there evidence of a linear trend?")
print("  - linear vs quadratic: is curvature helpful beyond a straight line?")
print("  - intercept-only vs categorical: do shares vary across years at all?")
print("  - linear/quadratic vs categorical: does a more flexible year-by-year fit help?")
print()