"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Run `python -m src.data_prep` to rebuild data/house_sale_clean.csv from the
raw Kaggle dump. Every cleaning decision is listed in `clean()` and printed
as a row-count log so it can be copied into the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/house_sale.csv"          # raw Kaggle file (binaaz-sale-project)
CLEAN_PATH = "data/house_sale_clean.csv"   # output of clean()

# Columns that LEAK the target (arithmetic functions of price).
LEAKAGE_COLUMNS: list[str] = [
    "total_price",   # equals price in 99.99% of rows
    "unit_price",    # price / area  ("3 450 AZN/m²")
]

# Columns that are not known before the price is set (post-listing signals).
POST_LISTING_COLUMNS: list[str] = [
    "views",         # accumulates after posting and depends on the price
    "updated",
]

# Identifiers, URLs, personal data, scrape metadata — no generalisable signal.
ID_COLUMNS: list[str] = [
    "id_x", "id_y", "estate_id", "rel_url", "estate_rel_url_x",
    "estate_rel_url_y", "estate_rel_url", "estate_details_id_x",
    "estate_details_id_y", "img_url", "owner_name", "owner_title",
    "shop_name", "shop_title", "address", "datetime_scrape_x",
    "datetime_scrape_y", "day_x", "day_y", "hour_x", "hour_y", "city_when",
]

# Redundant or near-empty columns.
REDUNDANT_COLUMNS: list[str] = [
    "currency_x", "currency_y",   # always AZN
    "repair", "bill_of_sale", "mortgage",  # duplicates of Təmir/Çıxarış/İpoteka
    "Binanın növü",               # 99.8% missing
    "attributes",                 # text summary of rooms/area/floor (parsed separately)
    "description", "extra_info",  # free text — optional, not used
]

# Residential categories kept (Azerbaijani -> English).
CATEGORY_MAP = {
    "Yeni tikili": "new_build",
    "Köhnə tikili": "old_build",
    "Həyət evi/Bağ evi": "house",
}

# Rough bounding box of Azerbaijan, used to drop bad coordinates.
LAT_RANGE = (38.3, 42.0)
LNG_RANGE = (44.7, 50.7)

# Outlier rules (fixed domain thresholds, NOT data quantiles -> no test leakage).
PRICE_RANGE = (10_000, 5_000_000)     # AZN
AREA_RANGE = (15, 1_000)              # m²
PPM2_RANGE = (200, 15_000)            # AZN per m², sanity check only
MAX_ROOMS = 15
MAX_LAND_SOT = 100                    # 1 sot = 100 m²; 100 sot = 1 ha


# --- Parsing helpers ----------------------------------------------------------
def _parse_number(s: pd.Series) -> pd.Series:
    """'145 m²' -> 145.0, '1.3 sot' -> 1.3, '3 450' -> 3450.0."""
    num = (s.astype("string")
             .str.replace(" ", "", regex=False)
             .str.extract(r"([\d\s]+(?:[.,]\d+)?)")[0]
             .str.replace(r"\s", "", regex=True)
             .str.replace(",", ".", regex=False))
    return pd.to_numeric(num, errors="coerce")


def _parse_floor(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'7 / 9' -> (7, 9)."""
    parts = s.astype("string").str.extract(r"(-?\d+)\s*/\s*(\d+)")
    return (pd.to_numeric(parts[0], errors="coerce"),
            pd.to_numeric(parts[1], errors="coerce"))


# --- Pipeline -----------------------------------------------------------------
def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame (UTF-8). Return it unmodified."""
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean the raw frame. Steps (each logged with the remaining row count):
      1. drop duplicate listings (same ad scraped from several pages),
      2. keep residential categories only (new build, old build, house),
      3. keep Baku only (99.7% of residential listings),
      4. parse text numbers: area, floor / total floors, land area,
      5. encode yes/no columns, translate names to English,
      6. drop rows with missing essentials,
      7. drop outliers by fixed domain rules,
      8. drop leakage / identifier / redundant columns.
    """
    log = []

    def step(name, frame):
        log.append((name, len(frame)))
        return frame

    df = step("raw", df.copy())

    # 1. duplicates: the same listing URL appears on several scraped pages
    df = df.sort_values("datetime_scrape_x").drop_duplicates("estate_rel_url_x", keep="last")
    df = step("drop duplicate listings", df)

    # 2. residential only
    df = df[df["Kateqoriya"].isin(CATEGORY_MAP)]
    df = step("keep residential categories", df)

    # 3. Baku only
    df = df[df["city"] == "bakı"]
    df = step("keep Baku", df)

    # 4–5. parse and translate
    out = pd.DataFrame(index=df.index)
    out["price"] = pd.to_numeric(df["price"], errors="coerce")
    out["category"] = df["Kateqoriya"].map(CATEGORY_MAP)
    out["area_m2"] = _parse_number(df["Sahə"])
    out["rooms"] = pd.to_numeric(df["Otaq sayı"], errors="coerce")
    floor, total_floors = _parse_floor(df["Mərtəbə"])
    out["floor"] = floor
    out["total_floors"] = total_floors
    out["land_sot"] = _parse_number(df["Torpaq sahəsi"])  # 1 sot = 100 m²
    out["is_repaired"] = df["Təmir"].map({"var": 1, "yoxdur": 0})
    out["has_bill_of_sale"] = df["Çıxarış"].map({"var": 1, "yoxdur": 0})
    out["has_mortgage"] = df["İpoteka"].eq("var").astype(int)    # missing = no
    out["is_agency"] = df["products_label"].eq("Agentlik").astype(int)
    out["is_complex"] = df["products_label"].eq("Kompleks").astype(int)
    out["is_vip"] = df["vip"].notna().astype(int)
    out["is_featured"] = df["featured"].notna().astype(int)
    out["location"] = df["location"].astype("string").str.strip()
    out["lat"] = df["lat"]
    out["lng"] = df["lng"]

    # Structural fills (not statistics -> no leakage):
    #   houses have no floor in a building -> floor 0, total_floors 1 (own building)
    #   apartments have no land plot       -> land_sot 0
    is_house = out["category"].eq("house")
    out.loc[is_house, "floor"] = out.loc[is_house, "floor"].fillna(0)
    out.loc[is_house, "total_floors"] = out.loc[is_house, "total_floors"].fillna(1)
    out.loc[~is_house, "land_sot"] = out.loc[~is_house, "land_sot"].fillna(0)
    df = step("parse & translate", out)

    # 6. missing essentials
    essential = ["price", "area_m2", "rooms", "floor", "total_floors",
                 "land_sot", "is_repaired", "has_bill_of_sale", "location"]
    df = df.dropna(subset=essential)
    df = step("drop rows with missing essentials", df)

    # 7. outliers (fixed rules)
    df = df[df["price"].between(*PRICE_RANGE)]
    df = step(f"price in {PRICE_RANGE} AZN", df)
    df = df[df["area_m2"].between(*AREA_RANGE)]
    df = step(f"area in {AREA_RANGE} m²", df)
    ppm2 = df["price"] / df["area_m2"]          # used for filtering only, not a feature
    df = df[ppm2.between(*PPM2_RANGE)]
    df = step(f"price per m² in {PPM2_RANGE}", df)
    df = df[df["rooms"].between(1, MAX_ROOMS)]
    df = step(f"rooms in [1, {MAX_ROOMS}]", df)
    df = df[df["land_sot"] <= MAX_LAND_SOT]
    df = step(f"land <= {MAX_LAND_SOT} sot", df)
    df = df[(df["floor"] <= df["total_floors"]) & (df["floor"] >= 0)]
    df = step("0 <= floor <= total_floors", df)
    df = df[df["lat"].between(*LAT_RANGE) & df["lng"].between(*LNG_RANGE)]
    df = step("coordinates inside Azerbaijan", df)

    # 8. leakage / ids / redundant columns were never copied into `out`;
    #    assert to be safe.
    for c in LEAKAGE_COLUMNS + POST_LISTING_COLUMNS + ID_COLUMNS:
        assert c not in df.columns, c

    int_cols = ["rooms", "floor", "total_floors", "is_repaired", "has_bill_of_sale"]
    df[int_cols] = df[int_cols].astype(int)
    df = df.reset_index(drop=True)

    if verbose:
        print("Cleaning log (rows remaining):")
        for name, n in log:
            print(f"  {name:<38s} {n:>7d}")
    return df


def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot / ordinal — your choice, justify it),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    raise NotImplementedError("TODO: build X, y")


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).  The threshold must come from TRAIN only
    (no peeking at val/test) to avoid leakage.
    """
    raise NotImplementedError("TODO: derive the price-tier label")


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    raise NotImplementedError("TODO: split the data")


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    raise NotImplementedError("TODO: standardize using train stats")


if __name__ == "__main__":
    clean_df = clean(load_raw())
    clean_df.to_csv(CLEAN_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {clean_df.shape[0]} rows x {clean_df.shape[1]} cols -> {CLEAN_PATH}")