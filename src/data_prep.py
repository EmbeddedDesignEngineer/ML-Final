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
             .str.extract(r"([-+]?\d[\d\s]*(?:[.,]\d+)?)")[0]
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
    for column in ("currency_x", "currency_y"):
        if column in df:
            currencies = df[column].dropna().astype(str).str.strip().str.upper()
            if not currencies.isin(["AZN", "₼", ""]).all():
                raise ValueError(f"Unsupported currency in {column}; prices must be AZN")

    # 1. duplicates: the same listing URL appears on several scraped pages
    df = df.sort_values("datetime_scrape_x")
    has_url = df["estate_rel_url_x"].notna() & df["estate_rel_url_x"].astype(str).str.strip().ne("")
    df = df[~has_url | ~df.duplicated("estate_rel_url_x", keep="last")]
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
    out["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    out["lng"] = pd.to_numeric(df["lng"], errors="coerce")

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
    df = df[df["land_sot"].between(0, MAX_LAND_SOT)]
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


# --- Features, split, label, scaling ----------------------------------------
CITY_CENTRE = (40.3714, 49.8378)   # Fountain Square, Baku (lat, lng)
MIN_LOCATION_COUNT = 200           # rarer districts are grouped as "other"
NUMERIC_FEATURES = [
    "area_m2", "rooms", "floor", "total_floors", "land_sot",
    "is_repaired", "has_bill_of_sale", "has_mortgage", "is_agency",
    "is_complex", "is_vip", "is_featured", "lat", "lng",
]


def _haversine_km(lat, lng, lat0, lng0):
    """Great-circle distance in km from (lat, lng) arrays to one point."""
    lat, lng, lat0, lng0 = map(np.radians, (lat, lng, lat0, lng0))
    a = (np.sin((lat - lat0) / 2) ** 2
         + np.cos(lat) * np.cos(lat0) * np.sin((lng - lng0) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


class FeatureEncoder:
    """
    Learns the encoding rules from the TRAINING rows only, then applies the
    same saved rules to any split (fit on train, transform train/val/test).

    Rules learned in fit():
      - categories_ : fixed domain list from CATEGORY_MAP (not read from data),
      - locations_  : districts with >= MIN_LOCATION_COUNT listings IN TRAIN;
                      every other district (rare in train, or unseen) -> "other".
    No statistic is ever taken from validation or test rows.
    """

    def __init__(self, min_location_count: int = MIN_LOCATION_COUNT):
        self.min_location_count = min_location_count

    def fit(self, df_train: pd.DataFrame) -> "FeatureEncoder":
        self.categories_ = list(CATEGORY_MAP.values())
        counts = df_train["location"].value_counts()
        self.locations_ = sorted(counts[counts >= self.min_location_count].index)
        self.feature_names_ = (NUMERIC_FEATURES + ["dist_centre_km", "floor_ratio"]
                               + [f"cat_{c}" for c in self.categories_]
                               + [f"loc_{l}" for l in self.locations_] + ["loc_other"])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "locations_"):
            raise RuntimeError("FeatureEncoder.fit(train_df) must be called first")
        num = df[NUMERIC_FEATURES].to_numpy(dtype=float)
        dist = _haversine_km(df["lat"].to_numpy(), df["lng"].to_numpy(), *CITY_CENTRE)
        ratio = (df["floor"] / df["total_floors"].clip(lower=1)).to_numpy(dtype=float)

        cat = df["category"].to_numpy()
        cat_oh = np.column_stack([(cat == c) for c in self.categories_]).astype(float)

        loc = df["location"].to_numpy()
        known = np.isin(loc, self.locations_)
        loc_oh = np.column_stack([(loc == l) for l in self.locations_] + [~known]).astype(float)

        X = np.column_stack([num, dist, ratio, cat_oh, loc_oh])
        assert X.shape[1] == len(self.feature_names_)
        return X


def make_features(df: pd.DataFrame, encoder: FeatureEncoder):
    """
    Turn a cleaned frame (one split) into a numeric design matrix using an
    encoder that was fitted on the TRAINING split only:
      - numeric / 0-1 columns as they are,
      - dist_centre_km: distance to Fountain Square (fixed point, no data stats),
      - floor_ratio: floor / total_floors (row-level),
      - one-hot `category` (fixed list of 3),
      - one-hot `location` for districts frequent in TRAIN, the rest -> "other".
        One-hot (not ordinal) because districts have no natural order.
    Return (X float ndarray, y price ndarray, feature_names list[str]).
    """
    return (encoder.transform(df),
            df["price"].to_numpy(dtype=float),
            list(encoder.feature_names_))


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
      - premium = 1 (price > threshold), standard = 0.
    Pass threshold=None ONLY with the TRAINING prices: the median is computed
    from them. For val/test, pass the threshold returned by the train call.
    Return (y_tier int ndarray, threshold float).
    """
    y_price = np.asarray(y_price, dtype=float)
    if y_price.ndim != 1 or not len(y_price) or not np.isfinite(y_price).all() or np.any(y_price <= 0):
        raise ValueError("Prices must be a non-empty 1-D array of finite positive values")
    if threshold is None:
        threshold = float(np.median(y_price))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Tier threshold must be finite and positive")
    return (y_price > threshold).astype(int), float(threshold)


def _split_indices(y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Row indices for train / validation / test (default 70/15/15),
    stratified on price deciles so every split has the same price
    distribution (and the same premium/standard balance for Task B).
    Like sklearn's stratified split, this only decides WHICH split a row
    goes to; no feature, label or scaling statistic comes from val/test.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Split targets must be a non-empty finite 1-D array")
    if (not np.isfinite([val_size, test_size]).all() or min(val_size, test_size) < 0
            or val_size + test_size >= 1):
        raise ValueError("Split sizes must be non-negative and sum to less than one")
    rng = np.random.default_rng(seed)
    edges = np.quantile(y, np.linspace(0, 1, 11)[1:-1])
    strata = np.searchsorted(edges, y, side="right")

    tr, va, te = [], [], []
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        n_te = int(round(len(idx) * test_size))
        n_va = int(round(len(idx) * val_size))
        te.append(idx[:n_te]); va.append(idx[n_te:n_te + n_va]); tr.append(idx[n_te + n_va:])
    return tuple(rng.permutation(np.concatenate(p)) for p in (tr, va, te))


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split of ready arrays into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    For the real data use get_splits(): it splits the ROWS first and only
    then fits the encoder on train.
    """
    X = np.asarray(X); y = np.asarray(y, dtype=float)
    tr, va, te = _split_indices(y, val_size, test_size, seed)
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits. Constant columns
    (std 0) are left centred, not divided.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0
    return tuple((X - mean) / std for X in (X_tr, *others))


def get_splits(seed: int = SEED, path: str = CLEAN_PATH) -> dict:
    """
    One call for everyone (team interface contract). Order matters:
      1. split the cleaned ROWS into train / val / test,
      2. fit the FeatureEncoder on the train rows only,
      3. transform all three splits with that same encoder,
      4. tier threshold = median TRAIN price, reused for val/test.
    Returns:
      X_tr, X_val, X_te   raw features (float)  -> trees
      y_tr, y_val, y_te   price in AZN          -> Task A (use np.log for log-price)
      t_tr, t_val, t_te   0/1 price tier         -> Task B
      tier_threshold      median TRAIN price
      feature_names, encoder
    For the SVM call standardize(X_tr, X_val, X_te) yourself (train mean/std).
    """
    df = pd.read_csv(path, encoding="utf-8")
    tr, va, te = _split_indices(df["price"].to_numpy(), seed=seed)
    df_tr, df_val, df_te = (df.iloc[i].reset_index(drop=True) for i in (tr, va, te))

    encoder = FeatureEncoder().fit(df_tr)
    X_tr, y_tr, names = make_features(df_tr, encoder)
    X_val, y_val, _ = make_features(df_val, encoder)
    X_te, y_te, _ = make_features(df_te, encoder)

    t_tr, thr = make_tier_label(y_tr)
    t_val, _ = make_tier_label(y_val, thr)
    t_te, _ = make_tier_label(y_te, thr)
    return dict(X_tr=X_tr, X_val=X_val, X_te=X_te,
                y_tr=y_tr, y_val=y_val, y_te=y_te,
                t_tr=t_tr, t_val=t_val, t_te=t_te,
                tier_threshold=thr, feature_names=names, encoder=encoder)


if __name__ == "__main__":
    clean_df = clean(load_raw())
    clean_df.to_csv(CLEAN_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {clean_df.shape[0]} rows x {clean_df.shape[1]} cols -> {CLEAN_PATH}")

    sp = get_splits()
    print(f"Features: {len(sp['feature_names'])}")
    print(f"Split sizes: train {len(sp['y_tr'])}, val {len(sp['y_val'])}, test {len(sp['y_te'])}")
    print(f"Tier threshold (train median): {sp['tier_threshold']:,.0f} AZN")
    print("Premium share: " + ", ".join(f"{k} {sp[k].mean():.3f}" for k in ("t_tr", "t_val", "t_te")))
