import pandas as pd
import re
from pathlib import Path


# =========================================================
# 1. File paths
# =========================================================
# Change this to your own file location
INPUT_FILE = "pubmed_raw_export.csv"

# Output folder
OUTDIR = Path("processed")
OUTDIR.mkdir(parents=True, exist_ok=True)

CLEAN_FILE = OUTDIR / "pubmed_articles_clean.csv"
YEARLY_FILE = OUTDIR / "yearly_publication_counts.csv"
THEME_YEARLY_FILE = OUTDIR / "theme_yearly_counts.csv"


# =========================================================
# 2. Read raw PubMed export
# =========================================================
# Read everything as string first so we don't accidentally lose text
df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

print(f"Loaded {len(df)} rows.")
print("Columns:")
print(df.columns.tolist())


# =========================================================
# 3. Keep only columns we actually need
# =========================================================
# Adjust this list if your file uses slightly different names
keep_cols = [
    "PMID",
    "ArticleTitle",
    "Abstract",
    "JournalTitle",
    "PubDate",
    "AuthorList_Fullnames",
    "AuthorList_Fullnames_with_affiliation",
    "PublicationTypeList",
    "MeshHeadingList",
    "KeywordList",
    "ArticleIdList",
    "ELocationID",
    "Language",
    "Journal_Country",
]

existing_keep_cols = [c for c in keep_cols if c in df.columns]
df = df[existing_keep_cols].copy()


# =========================================================
# 4. Rename columns to cleaner names
# =========================================================
rename_map = {
    "PMID": "pmid",
    "ArticleTitle": "title",
    "Abstract": "abstract",
    "JournalTitle": "journal",
    "PubDate": "pub_date",
    "AuthorList_Fullnames": "authors",
    "AuthorList_Fullnames_with_affiliation": "authors_affiliations",
    "PublicationTypeList": "publication_types",
    "MeshHeadingList": "mesh_terms",
    "KeywordList": "keywords",
    "ArticleIdList": "article_ids",
    "ELocationID": "elocation_id",
    "Language": "language",
    "Journal_Country": "journal_country",
}
df = df.rename(columns=rename_map)


# =========================================================
# 5. Extract publication year
# =========================================================
def extract_year(pub_date: str) -> str:
    """
    Try to find a 4-digit year from PubDate.
    Examples:
    - 2022-Mar
    - 2017-Feb
    - 2021
    """
    if not isinstance(pub_date, str):
        return ""
    match = re.search(r"\b(19|20)\d{2}\b", pub_date)
    return match.group(0) if match else ""


df["pub_year"] = df["pub_date"].apply(extract_year)


# =========================================================
# 6. Extract DOI
# =========================================================
def extract_doi(article_ids: str, elocation_id: str) -> str:
    """
    Try to get DOI from ArticleIdList first, then ELocationID.
    """
    for text in [article_ids, elocation_id]:
        if not isinstance(text, str):
            continue

        # Common DOI pattern
        match = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9a-z]+', text)
        if match:
            return match.group(0)

    return ""


df["doi"] = df.apply(
    lambda row: extract_doi(
        row.get("article_ids", ""),
        row.get("elocation_id", "")
    ),
    axis=1
)


# =========================================================
# 7. Build combined text field
# =========================================================
# This is the main text we'll use for keyword/theme analysis
df["text"] = (
    df["title"].fillna("").astype(str).str.strip()
    + " "
    + df["abstract"].fillna("").astype(str).str.strip()
).str.strip()

df["text_lower"] = df["text"].str.lower()


# =========================================================
# 8. Basic cleaning flags
# =========================================================
df["has_abstract"] = df["abstract"].str.strip().ne("")
df["is_review"] = df["publication_types"].str.contains("Review", case=False, na=False)


# =========================================================
# 9. Theme buckets
# =========================================================
# These are simple keyword-based buckets.
# You can refine them later.

THEME_BUCKETS = {
    "theme_tolerance_regulation": [
        r"\btolerance\b",
        r"\boral tolerance\b",
        r"\btreg\b",
        r"\bregulatory t\b",
        r"\bimmune tolerance\b",
    ],
    "theme_sensitization_ige": [
        r"\bsensitization\b",
        r"\ballergic sensitization\b",
        r"\bige\b",
        r"\bimmunoglobulin e\b",
        r"\banaphylaxis\b",
        r"\bmast cell\b",
    ],
    "theme_microbiome": [
        r"\bmicrobiome\b",
        r"\bmicrobiota\b",
        r"\bgut microbiota\b",
        r"\bdysbiosis\b",
        r"\bcolonization\b",
        r"\bcommensal\b",
    ],
    "theme_barrier_epithelial": [
        r"\bbarrier\b",
        r"\bepithelial\b",
        r"\bskin barrier\b",
        r"\bmucosal\b",
        r"\bpermeability\b",
        r"\bepicutaneous\b",
    ],
}


def contains_any_pattern(text: str, patterns: list[str]) -> int:
    if not isinstance(text, str):
        return 0
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return 1
    return 0


for col, patterns in THEME_BUCKETS.items():
    df[col] = df["text"].apply(lambda x: contains_any_pattern(x, patterns))


# =========================================================
# 10. Optional: early-life relevance flag
# =========================================================
# This is a loose screening flag, not a final truth label.
EARLY_LIFE_PATTERNS = [
    r"\bearly life\b",
    r"\binfant\b",
    r"\binfants\b",
    r"\binfancy\b",   # typo-resistant, harmless
    r"\bnewborn\b",
    r"\bneonatal\b",
    r"\bcord blood\b",
    r"\bmaternal\b",
    r"\bpregnancy\b",
    r"\bprenatal\b",
    r"\bpostnatal\b",
    r"\bchildhood\b",
    r"\bchild\b",
    r"\bchildren\b",
]

FOOD_ALLERGY_PATTERNS = [
    r"\bfood allergy\b",
    r"\bfood allergies\b",
    r"\bfood hypersensitivity\b",
    r"\bige-mediated food allergy\b",
    r"\bnon-ige-mediated\b",
]

df["mentions_early_life"] = df["text"].apply(lambda x: contains_any_pattern(x, EARLY_LIFE_PATTERNS))
df["mentions_food_allergy"] = df["text"].apply(lambda x: contains_any_pattern(x, FOOD_ALLERGY_PATTERNS))

# A simple relevance flag
df["likely_relevant"] = (
    (df["mentions_early_life"] == 1) &
    (df["mentions_food_allergy"] == 1)
).astype(int)


# =========================================================
# 11. Reorder useful columns
# =========================================================
front_cols = [
    "pmid",
    "title",
    "abstract",
    "journal",
    "pub_date",
    "pub_year",
    "doi",
    "authors",
    "authors_affiliations",
    "publication_types",
    "mesh_terms",
    "keywords",
    "language",
    "journal_country",
    "has_abstract",
    "is_review",
    "mentions_early_life",
    "mentions_food_allergy",
    "likely_relevant",
    "theme_tolerance_regulation",
    "theme_sensitization_ige",
    "theme_microbiome",
    "theme_barrier_epithelial",
    "text",
]

existing_front_cols = [c for c in front_cols if c in df.columns]
other_cols = [c for c in df.columns if c not in existing_front_cols]
df = df[existing_front_cols + other_cols]


# =========================================================
# 12. Save clean article-level dataset
# =========================================================
df.to_csv(CLEAN_FILE, index=False)
print(f"Saved clean dataset to: {CLEAN_FILE}")


# =========================================================
# 13. Yearly publication counts
# =========================================================
yearly_counts = (
    df[df["pub_year"].str.match(r"^(19|20)\d{2}$", na=False)]
    .groupby("pub_year")
    .size()
    .reset_index(name="n_publications")
    .sort_values("pub_year")
)

yearly_counts.to_csv(YEARLY_FILE, index=False)
print(f"Saved yearly counts to: {YEARLY_FILE}")


# =========================================================
# 14. Theme counts by year
# =========================================================
theme_cols = [
    "theme_tolerance_regulation",
    "theme_sensitization_ige",
    "theme_microbiome",
    "theme_barrier_epithelial",
]

theme_yearly = (
    df[df["pub_year"].str.match(r"^(19|20)\d{2}$", na=False)]
    .groupby("pub_year")[theme_cols]
    .sum()
    .reset_index()
    .sort_values("pub_year")
)

theme_yearly.to_csv(THEME_YEARLY_FILE, index=False)
print(f"Saved theme-by-year counts to: {THEME_YEARLY_FILE}")


# =========================================================
# 15. Quick summary prints
# =========================================================
print("\nQuick summary:")
print(f"Total papers: {len(df)}")
print(f"Papers with abstracts: {df['has_abstract'].sum()}")
print(f"Likely relevant papers: {df['likely_relevant'].sum()}")

print("\nTheme totals:")
for col in theme_cols:
    print(f"{col}: {df[col].sum()}")