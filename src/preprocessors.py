
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

# class CVTargetEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, cols=None, n_splits=5, random_state=42):
#         self.cols = cols or []
#         self.n_splits = n_splits
#         self.random_state = random_state
#         self.global_mean = None
#         self.encoding_maps = {}

#     def fit(self, X, y):
#         self.global_mean = y.mean()
#         self.encoding_maps = {
#             col: y.groupby(X[col]).mean()
#             for col in self.cols
#         }
#         return self

#     def transform(self, X):
#         X_encoded = X.copy()
#         for col in self.cols:
#             X_encoded[col + "_te"] = X_encoded[col].map(self.encoding_maps.get(col, {})).fillna(self.global_mean)
#         return X_encoded.drop(columns=self.cols)

# class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.zip_mode_by_city = {}
#         self.freq_maps = {}
#         self.medians = {}
#         self.modes = {}

#     def fit(self, X, y=None):
#         if 'city' in X and 'zip_code' in X:
#             valid = X.dropna(subset=['city'])
#             self.zip_mode_by_city = valid.groupby('city')['zip_code'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
#         for col in ['status']:
#             if col in X:
#                 self.freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
#         for col in ['house_size', 'bath', 'bed', 'acre_lot', 'days_since_prev_sale']:
#             if col in X:
#                 self.medians[col] = X[col].median()
#         for col in ['street', 'brokered_by', 'city', 'state', 'status']:
#             if col in X:
#                 self.modes[col] = X[col].mode().iloc[0] if not X[col].mode().empty else None
#         return self

#     def transform(self, X):
#         X = X.copy()
#         if 'zip_code' in X and 'city' in X:
#             X['zip_code'] = X.apply(lambda row: self.zip_mode_by_city.get(row['city'], np.nan) if pd.isna(row['zip_code']) else row['zip_code'], axis=1)
#         if 'prev_sold_date' in X:
#             X['prev_sold_date'] = pd.to_datetime(X['prev_sold_date'], errors='coerce')
#             X['days_since_prev_sale'] = (pd.Timestamp.today().normalize() - X['prev_sold_date']).dt.days

#         for col in ['prev_sold_date', 'house_size', 'bath', 'bed', 'acre_lot', 'days_since_prev_sale']:
#             if col in X:
#                 X[f'{col}_is_missing'] = X[col].isna().astype(int)

#         for col in self.medians:
#             if col in X:
#                 X[col] = X[col].fillna(self.medians[col])
#         for col in self.modes:
#             if col in X:
#                 X[col] = X[col].fillna(self.modes[col] if self.modes[col] is not None else 'unknown_category')

#         for col in ['acre_lot', 'bath', 'bed', 'days_since_prev_sale', 'house_size']:
#             if col in X:
#                 X[f'{col}_log'] = np.log1p(X[col].clip(lower=0))

#         if 'bed' in X and 'bath' in X:
#             X['total_rooms'] = X['bed'] + X['bath']
#             X['bed_to_bath_ratio'] = X['bed'] / (X['bath'] + 1e-5)
#             X['house_size_per_room'] = X['house_size'] / X['total_rooms'].replace(0, 1e-5)
#         if 'house_size' in X and 'acre_lot' in X:
#             X['house_size_to_lot_ratio'] = X['house_size'] / X['acre_lot'].replace(0, 1e-5)
#         for col in self.freq_maps:
#             if col in X:
#                 X[f'{col}_freq'] = X[col].map(self.freq_maps[col]).fillna(0)

#         return X.select_dtypes(include=np.number)


class CVTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, n_splits=5, random_state=42):
        self.cols = cols or []
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean = None
        self.encoding_maps = {}

    def fit(self, X, y):
        X_copy = X.copy() 
        y_copy = y.copy()
        self.global_mean = y_copy.mean()
        self.encoding_maps = {}

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for col in self.cols:
            self.encoding_maps[col] = y_copy.groupby(X_copy[col]).mean()

        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in self.cols:
            X_encoded[col + "_te"] = X_encoded[col].map(self.encoding_maps.get(col, {})).fillna(self.global_mean)

        return X_encoded.drop(columns=[c for c in self.cols if c in X_encoded.columns])

# --- Feature Engineering ---
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.zip_mode_by_city = {}
        self.freq_maps = {}

    def fit(self, X, y=None):
        X_copy = X.copy()

        # Determine zip_mode_by_city for imputation
        if 'city' in X_copy.columns and 'zip_code' in X_copy.columns:
            # Filter out NaN cities before grouping if necessary, or mode will be NaN
            valid_x_copy = X_copy.dropna(subset=['city'])
            self.zip_mode_by_city = (
                valid_x_copy.groupby('city')['zip_code']
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
                .to_dict()
            )

        # Determine frequency maps
        for col in ['status']:
            if col in X_copy.columns and not X_copy[col].isna().all():
                self.freq_maps[col] = X_copy[col].value_counts(normalize=True).to_dict() 

        # Store medians for numerical imputation
        self.medians = {}
        for col in ['house_size', 'bath', 'bed', 'acre_lot', 'days_since_prev_sale']:
            if col in X_copy.columns:
                self.medians[col] = X_copy[col].median()

        # Store modes for categorical imputation
        self.modes = {}
        for col in ['street', 'brokered_by', 'city', 'state', 'status']:
            if col in X_copy.columns and not X_copy[col].isna().all():
                self.modes[col] = X_copy[col].mode().iloc[0]
            elif col in X_copy.columns: # If all values are NaN, mode() will be empty
                self.modes[col] = None # Or a reasonable default like 'missing'

        return self

    def transform(self, X):
        X_transformed = X.copy()

        # Handle zip_code imputation based on city mode (learned in fit)
        if 'zip_code' in X_transformed.columns and 'city' in X_transformed.columns:
             X_transformed['zip_code'] = X_transformed.apply(
                 lambda row: self.zip_mode_by_city.get(row['city'], np.nan)
                 if pd.isna(row['zip_code']) else row['zip_code'], axis=1)

        # Date feature engineering
        if 'prev_sold_date' in X_transformed.columns:
            X_transformed['prev_sold_date'] = pd.to_datetime(X_transformed['prev_sold_date'], errors='coerce')

            X_transformed['days_since_prev_sale'] = (
                pd.Timestamp.today().normalize() - X_transformed['prev_sold_date']
            ).dt.days

        for col in ['prev_sold_date', 'house_size', 'bath', 'bed', 'acre_lot', 'days_since_prev_sale']:
            if col in X_transformed.columns:
                X_transformed[f'{col}_is_missing'] = X_transformed[col].isna().astype(int)

        for col in ['house_size', 'bath', 'bed', 'acre_lot', 'days_since_prev_sale']:
            if col in X_transformed.columns and col in self.medians and not pd.isna(self.medians[col]):
                X_transformed[col] = X_transformed[col].fillna(self.medians[col])

        for col in ['street', 'brokered_by', 'city', 'state', 'status']:
            if col in X_transformed.columns and col in self.modes and not pd.isna(self.modes[col]):
                X_transformed[col] = X_transformed[col].fillna(self.modes[col])
            elif col in X_transformed.columns and X_transformed[col].isna().any(): # If mode was None (all NaNs in fit)
                X_transformed[col] = X_transformed[col].fillna('unknown_category') # Default placeholder

        for col in ['acre_lot', 'bath', 'bed', 'days_since_prev_sale', 'house_size']:
            if col in X_transformed.columns:
                # Ensure values are non-negative before log1p
                X_transformed[f'{col}_log'] = np.log1p(X_transformed[col].clip(lower=0))

        # Interaction features
        if 'bed' in X_transformed.columns and 'bath' in X_transformed.columns:
            X_transformed['total_rooms'] = X_transformed['bed'] + X_transformed['bath']
            X_transformed['bed_to_bath_ratio'] = X_transformed['bed'] / (X_transformed['bath'] + 1e-5)
            X_transformed['house_size_per_room'] = X_transformed['house_size'] / (X_transformed['total_rooms'].replace(0, 1e-5))

        if 'house_size' in X_transformed.columns and 'acre_lot' in X_transformed.columns:
            X_transformed['house_size_to_lot_ratio'] = X_transformed['house_size'] / (X_transformed['acre_lot'].replace(0, 1e-5))

        # Frequency encoding
        for col in ['status']:
            if col in X_transformed.columns and col in self.freq_maps:
                X_transformed[f'{col}_freq'] = X_transformed[col].map(self.freq_maps[col]).fillna(0) # Fill unknown with 0 frequency

    
        drop_cols = [
            'street', 'brokered_by', 'city', 'zip_code', 'state',
            'prev_sold_date', 'price', 'status' 
        ]
        X_clean = X_transformed.drop(columns=[c for c in drop_cols if c in X_transformed.columns], errors='ignore')

        numeric_cols = X_clean.select_dtypes(include=np.number).columns.tolist()
        return X_clean[numeric_cols]
