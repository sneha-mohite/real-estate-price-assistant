import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from joblib import dump
from preprocessors import CVTargetEncoder, CustomFeatureEngineer

df = pd.read_csv("data/usa-real-estate-dataset.csv")
df = df.dropna(subset=['price'])
df['price'] = df['price'].clip(df['price'].quantile(0.01), df['price'].quantile(0.99))

X = df.drop(columns=['price'])
y = np.log1p(df['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('target_encode', CVTargetEncoder(cols=['city', 'zip_code', 'brokered_by', 'state'])),
    ('features', CustomFeatureEngineer()),
    ('model', XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        random_state=42,
        # tree_method='gpu_hist'
        tree_method='hist', 
        device='gpu'# cuda if no gpu is available
    ))
])

pipeline.fit(X_train, y_train)
dump(pipeline, "models/real_estate_model_xgboost_gpu_pipeline.joblib")
print("✅ Model trained and saved.")

y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"\n--- Model Performance ---")
print(f"Train RMSE: {rmse_train:.4f}")
print(f"Test RMSE: {rmse_test:.4f}")
print(f"Train R2 Score: {r2_train:.4f}")
print(f"Test R2 Score: {r2_test:.4f}")
