import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder


def _mode_or_nan(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan


class CustomImputer(BaseEstimator, TransformerMixin):

    KNN_FEATURE_COLS = [
        "BOROUGH",
        "LOG_TOTAL_UNITS_TMP",
        "BC_LETTER_FREQ_TMP",
        "LAND SQUARE FEET",
        "GROSS SQUARE FEET",
    ]

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def _build_knn_matrix(self, X, land, gross, bc_letter_freq_map):
        log_total_units = np.log1p(X["TOTAL UNITS"].astype(float))
        bc_letter = X["BUILDING CLASS AT TIME OF SALE"].astype(str).str[0]
        bc_letter_freq = bc_letter.map(bc_letter_freq_map).fillna(0.0)

        matrix = pd.DataFrame(
            {
                "BOROUGH": X["BOROUGH"].astype(float),
                "LOG_TOTAL_UNITS_TMP": log_total_units,
                "BC_LETTER_FREQ_TMP": bc_letter_freq,
                "LAND SQUARE FEET": land,
                "GROSS SQUARE FEET": gross,
            }
        )
        return matrix[self.KNN_FEATURE_COLS]

    def fit(self, X, y=None):
        X = X.copy()

        zip_known = X.dropna(subset=["ZIP CODE"])
        self.zip_mode_bb_ = zip_known.groupby(["BOROUGH", "BLOCK"])["ZIP CODE"].agg(_mode_or_nan).to_dict()
        self.zip_mode_b_ = zip_known.groupby("BOROUGH")["ZIP CODE"].agg(_mode_or_nan).to_dict()

        year_known = X.dropna(subset=["YEAR BUILT"])
        self.year_mode_bbb_ = (
            year_known.groupby(["BOROUGH", "BLOCK", "BUILDING CLASS CATEGORY"], observed=True)["YEAR BUILT"]
            .agg(_mode_or_nan)
            .to_dict()
        )
        self.year_mode_bb_ = year_known.groupby(["BOROUGH", "BLOCK"])["YEAR BUILT"].agg(_mode_or_nan).to_dict()
        self.year_mode_b_ = year_known.groupby("BOROUGH")["YEAR BUILT"].agg(_mode_or_nan).to_dict()

        self.bc_present_fallback_ = _mode_or_nan(X["BUILDING CLASS AT PRESENT"].dropna())

        self.tax_present_fallback_ = _mode_or_nan(X["TAX CLASS AT PRESENT"].dropna())

        land = X["LAND SQUARE FEET"].replace(0, np.nan)
        gross = X["GROSS SQUARE FEET"].replace(0, np.nan)

        bc_letter_train = X["BUILDING CLASS AT TIME OF SALE"].astype(str).str[0]
        self.bc_letter_freq_map_ = bc_letter_train.value_counts(normalize=True).to_dict()

        knn_matrix = self._build_knn_matrix(X, land, gross, self.bc_letter_freq_map_)

        self.sqft_pipeline_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNNImputer(n_neighbors=self.n_neighbors)),
            ]
        )
        self.sqft_pipeline_.fit(knn_matrix)

        return self

    def transform(self, X):
        X = X.copy()

        for col in [
            "NEIGHBORHOOD",
            "BUILDING CLASS CATEGORY",
            "BUILDING CLASS AT PRESENT",
            "BUILDING CLASS AT TIME OF SALE",
            "TAX CLASS AT PRESENT",
            "APARTMENT NUMBER",
        ]:
            if col in X.columns:
                X[col] = X[col].astype(object)

        keys_bb = list(zip(X["BOROUGH"], X["BLOCK"]))
        fill_bb = pd.Series([self.zip_mode_bb_.get(k, np.nan) for k in keys_bb], index=X.index)
        X["ZIP CODE"] = X["ZIP CODE"].fillna(fill_bb)

        fill_b = X["BOROUGH"].map(self.zip_mode_b_)
        X["ZIP CODE"] = X["ZIP CODE"].fillna(fill_b)

        keys_bbb = list(zip(X["BOROUGH"], X["BLOCK"], X["BUILDING CLASS CATEGORY"]))
        fill_bbb = pd.Series([self.year_mode_bbb_.get(k, np.nan) for k in keys_bbb], index=X.index)
        X["YEAR BUILT"] = X["YEAR BUILT"].fillna(fill_bbb)

        keys_bb2 = list(zip(X["BOROUGH"], X["BLOCK"]))
        fill_bb2 = pd.Series([self.year_mode_bb_.get(k, np.nan) for k in keys_bb2], index=X.index)
        X["YEAR BUILT"] = X["YEAR BUILT"].fillna(fill_bb2)

        fill_b2 = X["BOROUGH"].map(self.year_mode_b_)
        X["YEAR BUILT"] = X["YEAR BUILT"].fillna(fill_b2)

        X["BUILDING CLASS AT PRESENT"] = X["BUILDING CLASS AT PRESENT"].fillna(
            X["BUILDING CLASS AT TIME OF SALE"]
        )
        X["BUILDING CLASS AT PRESENT"] = X["BUILDING CLASS AT PRESENT"].fillna(self.bc_present_fallback_)

        X["TAX CLASS AT PRESENT"] = X["TAX CLASS AT PRESENT"].fillna(
            X["TAX CLASS AT TIME OF SALE"].astype(str)
        )
        X["TAX CLASS AT PRESENT"] = X["TAX CLASS AT PRESENT"].fillna(self.tax_present_fallback_)

        land = X["LAND SQUARE FEET"].replace(0, np.nan)
        gross = X["GROSS SQUARE FEET"].replace(0, np.nan)

        X["LAND_MISSING"] = land.isna().astype(int)
        X["GROSS_MISSING"] = gross.isna().astype(int)

        knn_matrix = self._build_knn_matrix(X, land, gross, self.bc_letter_freq_map_)
        imputed_scaled = self.sqft_pipeline_.transform(knn_matrix)
        imputed_raw = self.sqft_pipeline_.named_steps["scaler"].inverse_transform(imputed_scaled)
        imputed_raw = pd.DataFrame(imputed_raw, columns=self.KNN_FEATURE_COLS, index=X.index)

        X["LAND SQUARE FEET"] = imputed_raw["LAND SQUARE FEET"].clip(lower=1.0)
        X["GROSS SQUARE FEET"] = imputed_raw["GROSS SQUARE FEET"].clip(lower=1.0)

        X["ZIP CODE"] = X["ZIP CODE"].astype(int).astype(str)

        return X


class customFeatures(BaseEstimator, TransformerMixin):

    RARE_BC_CATEGORY_THRESHOLD = 200

    NEIGHBORHOOD_MAPPING = {
        "CITY ISLAND-PELHAM STRIP": "other_2",
        "EAST RIVER": "other_2",
        "FRESH KILLS": "other_5",
        "VAN CORTLANDT PARK": "other_2",
        "ROSSVILLE-PORT MOBIL": "other_5",
        "CO-OP CITY": "other_2",
        "DONGAN HILLS-OLD TOWN": "other_5",
        "ROSSVILLE-RICHMOND VALLEY": "other_5",
        "AIRPORT LA GUARDIA": "other_4",
        "EMERSON HILL": "other_5",
        "RICHMONDTOWN-LIGHTHS HILL": "other_5",
        "JAMAICA BAY": "other_4",
        "FLUSHING MEADOW PARK": "other_4",
        "CONCORD-FOX HILLS": "other_5",
        "ARROCHAR-SHORE ACRES": "other_5",
        "NEPONSIT": "other_4",
        "JAVITS CENTER": "other_1",
        "OAKWOOD": "other_5",
        "TODT HILL": "other_5",
        "HARLEM-WEST": "other_1",
        "NEW BRIGHTON-ST. GEORGE": "other_5",
        "LITTLE ITALY": "other_1",
        "FIELDSTON": "other_2",
    }

    DROP_COLUMNS = [
        "SALE PRICE",
        "LOG PRICE",
        "BLOCK",
        "LOT",
        "APARTMENT NUMBER",
        "SALE DATE",
        "YEAR BUILT",
        "RESIDENTIAL UNITS",
        "TOTAL UNITS",
        "COMMERCIAL UNITS",
        "LAND SQUARE FEET",
        "GROSS SQUARE FEET",
        "TAX CLASS AT PRESENT",
        "BUILDING CLASS AT PRESENT",
        "BUILDING CLASS CATEGORY",
    ]

    def _fix_sunnyside(self, X):
        mask = (X["NEIGHBORHOOD"] == "SUNNYSIDE") & (X["BOROUGH"] == 5)
        X.loc[mask, "NEIGHBORHOOD"] = "SUNNYSIDE 2"
        return X

    def fit(self, X, y=None):
        X = X.copy()
        X["NEIGHBORHOOD"] = X["NEIGHBORHOOD"].astype(object)
        X = self._fix_sunnyside(X)
        X["NEIGHBORHOOD"] = X["NEIGHBORHOOD"].replace(self.NEIGHBORHOOD_MAPPING)

        bc_category = X["BUILDING CLASS CATEGORY"].astype(object).str.strip()
        bc_counts = bc_category.value_counts()
        self.bc_category_keep_ = set(bc_counts[bc_counts >= self.RARE_BC_CATEGORY_THRESHOLD].index)

        land = X["LAND SQUARE FEET"].astype(float)
        gross = X["GROSS SQUARE FEET"].astype(float)
        far = (gross / land.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        self.far_median_ = far.median()

        return self

    def transform(self, X):
        X = X.copy()

        for col in ["NEIGHBORHOOD", "BUILDING CLASS CATEGORY", "BUILDING CLASS AT PRESENT",
                    "BUILDING CLASS AT TIME OF SALE", "APARTMENT NUMBER"]:
            if col in X.columns:
                X[col] = X[col].astype(object)

        X = self._fix_sunnyside(X)
        X["NEIGHBORHOOD"] = X["NEIGHBORHOOD"].replace(self.NEIGHBORHOOD_MAPPING)

        bc_category = X["BUILDING CLASS CATEGORY"].str.strip()
        X["BC_CATEGORY_GROUPED"] = np.where(
            bc_category.isin(self.bc_category_keep_),
            bc_category,
            "OTHER",
        )

        X["BC_LETTER"] = X["BUILDING CLASS AT TIME OF SALE"].astype(str).str[0]
        X["BC_CHANGED_SINCE_SALE"] = (
            X["BUILDING CLASS AT PRESENT"] != X["BUILDING CLASS AT TIME OF SALE"]
        ).astype(int)

        X["HAS_APARTMENT_NUMBER"] = X["APARTMENT NUMBER"].notna().astype(int)

        X["IS_CONDO"] = (X["LOT"] >= 1000).astype(int)

        X["MONTH_YEAR"] = X["SALE DATE"].dt.to_period("M").astype(str)
        X["IS_WEEKEND"] = (X["SALE DATE"].dt.dayofweek >= 5).astype(int)

        X["AGE"] = (X["SALE DATE"].dt.year - X["YEAR BUILT"].astype(float)).astype(float)

        X["LOG_TOTAL_UNITS"] = np.log1p(X["TOTAL UNITS"].astype(float))
        X["HAS_COMMERCIAL_UNIT"] = (X["COMMERCIAL UNITS"].astype(float) > 0).astype(int)
        X["LOG_COMM_UNITS"] = np.log1p(X["COMMERCIAL UNITS"].astype(float))

        land = X["LAND SQUARE FEET"].astype(float)
        gross = X["GROSS SQUARE FEET"].astype(float)
        X["LOG_LAND_SQFT"] = np.log1p(land)
        X["LOG_GROSS_SQFT"] = np.log1p(gross)

        far = (gross / land.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        X["FAR"] = far.fillna(self.far_median_)

        X = X.drop(columns=self.DROP_COLUMNS, errors="ignore")

        return X


def get_preprocessor():
    onehot_cols = [
        "BOROUGH",
        "BC_CATEGORY_GROUPED",
        "BC_LETTER",
        "TAX CLASS AT TIME OF SALE",
    ]

    target_enc_cols = [
        "NEIGHBORHOOD",
        "ZIP CODE",
        "BUILDING CLASS AT TIME OF SALE",
    ]

    ordinal_cols = ["MONTH_YEAR"]

    numeric_cols = [
        "AGE",
        "LOG_TOTAL_UNITS",
        "LOG_COMM_UNITS",
        "LOG_LAND_SQFT",
        "LOG_GROSS_SQFT",
        "FAR",
    ]

    binary_cols = [
        "IS_WEEKEND",
        "HAS_APARTMENT_NUMBER",
        "IS_CONDO",
        "BC_CHANGED_SINCE_SALE",
        "LAND_MISSING",
        "GROSS_MISSING",
        "HAS_COMMERCIAL_UNIT",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                onehot_cols,
            ),
            (
                "target",
                TargetEncoder(
                    smooth="auto",
                    random_state=42
                ),
                target_enc_cols,
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_cols,
            ),
            ("numeric", "passthrough", numeric_cols),
            ("binary", "passthrough", binary_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def get_full_pipeline():
    return Pipeline([
        ("imputer", CustomImputer(n_neighbors=5)),
        ("features", customFeatures()),
        ("preprocessor", get_preprocessor()),
    ])