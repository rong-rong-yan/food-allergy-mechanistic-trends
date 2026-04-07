import pandas as pd
import re
from pathlib import Path

MAIN_FILE = "processed/pubmed_articles_clean.csv"
BACKGROUND_FILE = "background_pubmed_raw_export.csv"

OUTDIR = Path("processed")
OUTDIR.mkdir(parents=True, exist_ok=True)

COMPARE_FILE = OUTDIR / "main_vs_background_yearly.csv"


def extract_year(pub_date: str) -> str:
    if not isinstance(pub_date, str):
        return ""
    m = re.search(r"\b(19|20)\d{2}\b", pub_date)
    return m.group(0) if m else ""


# Main corpus
main_df = pd.read_csv(MAIN_FILE, dtype=str).fillna("")
if "pub_year" not in main_df.columns:
    if "pub_date" in main_df.columns:
        main_df["pub_year"] = main_df["pub_date"].apply(extract_year)
    else:
        raise ValueError("Main file must contain either 'pub_year' or 'pub_date'.")

main_df = main_df[main_df["pub_year"].str.match(r"^(19|20)\d{2}$", na=False)].copy()
main_df["pub_year"] = main_df["pub_year"].astype(int)
main_df = main_df[(main_df["pub_year"] >= 2015) & (main_df["pub_year"] <= 2025)]

main_yearly = (
    main_df.groupby("pub_year")
    .size()
    .reset_index(name="main_count")
)

# Background corpus
bg_df = pd.read_csv(BACKGROUND_FILE, dtype=str).fillna("")
if "PubDate" not in bg_df.columns:
    raise ValueError("Background raw export must contain 'PubDate'.")

bg_df["pub_year"] = bg_df["PubDate"].apply(extract_year)
bg_df = bg_df[bg_df["pub_year"].str.match(r"^(19|20)\d{2}$", na=False)].copy()
bg_df["pub_year"] = bg_df["pub_year"].astype(int)
bg_df = bg_df[(bg_df["pub_year"] >= 2015) & (bg_df["pub_year"] <= 2025)]

background_yearly = (
    bg_df.groupby("pub_year")
    .size()
    .reset_index(name="background_count")
)

# Merge and compute share
years = pd.DataFrame({"pub_year": list(range(2015, 2026))})

compare_df = (
    years
    .merge(main_yearly, on="pub_year", how="left")
    .merge(background_yearly, on="pub_year", how="left")
    .fillna(0)
)

compare_df["main_count"] = compare_df["main_count"].astype(int)
compare_df["background_count"] = compare_df["background_count"].astype(int)

compare_df["mechanistic_share"] = compare_df.apply(
    lambda row: row["main_count"] / row["background_count"]
    if row["background_count"] > 0 else 0,
    axis=1
)
compare_df["mechanistic_share_pct"] = 100 * compare_df["mechanistic_share"]

compare_df.to_csv(COMPARE_FILE, index=False)

print(f"Saved comparison table to: {COMPARE_FILE}")
print(compare_df)