"""Reproduce the AQI estimation experiment using the India city-day dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "city_day.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Load and clean data using the same rules as the analysis notebook.
df = pd.read_csv(DATA)
df = df.dropna(subset=["AQI"]).copy()
df = df.drop(columns=["AQI_Bucket", "Xylene"])
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

feature_cols = ["City", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Date"]
X = df[feature_cols].copy()
y = df["AQI"].copy()
X["Year"] = X["Date"].dt.year
X["Month"] = X["Date"].dt.month
X["Day"] = X["Date"].dt.day
X = X.drop(columns=["Date"])

# Chronological split by unique date, matching the notebook.
unique_dates = sorted(df["Date"].unique())
split_date = unique_dates[int(len(unique_dates) * 0.80)]
train_mask = df["Date"] < split_date
test_mask = df["Date"] >= split_date
X_train, X_test = X.loc[train_mask].copy(), X.loc[test_mask].copy()
y_train, y_test = y.loc[train_mask].copy(), y.loc[test_mask].copy()

numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, objective="reg:squarederror", random_state=42, n_jobs=-1),
}

rows, predictions = [], {}
for name, model in models.items():
    model.fit(X_train_p, y_train)
    pred = model.predict(X_test_p)
    predictions[name] = pred
    rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred), "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2 Score": r2_score(y_test, pred)})

metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
metrics.to_csv(RESULTS / "model_performance.csv", index=False)
print(f"Original rows: {len(pd.read_csv(DATA)):,}")
print(f"Cleaned rows: {len(df):,}; train: {len(X_train):,}; test: {len(X_test):,}")
print(f"Train dates: {df.loc[train_mask, 'Date'].min().date()} to {df.loc[train_mask, 'Date'].max().date()}")
print(f"Test dates: {df.loc[test_mask, 'Date'].min().date()} to {df.loc[test_mask, 'Date'].max().date()}")
print("\nModel performance (lower MAE/RMSE and higher R² are better):")
print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# Generate reproducible visual summaries.
fig, ax = plt.subplots(figsize=(9, 5))
metrics.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar", ax=ax)
ax.set_title("AQI Model Error Comparison"); ax.set_ylabel("Error (AQI points)"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=15); fig.tight_layout(); fig.savefig(FIGURES / "model_error_comparison.png", dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
metrics.set_index("Model")["R2 Score"].plot(kind="bar", ax=ax)
ax.set_title("R² Score Comparison"); ax.set_ylabel("R² score"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=15); fig.tight_layout(); fig.savefig(FIGURES / "r2_comparison.png", dpi=300); plt.close(fig)

best = metrics.iloc[0]["Model"]
pred = predictions[best]
fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(y_test, pred, alpha=.35, s=12); lo=min(float(y_test.min()),float(pred.min())); hi=max(float(y_test.max()),float(pred.max())); ax.plot([lo,hi],[lo,hi], linestyle="--"); ax.set_xlabel("Actual AQI"); ax.set_ylabel("Predicted AQI"); ax.set_title(f"Actual vs Predicted — {best}"); fig.tight_layout(); fig.savefig(FIGURES / "actual_vs_predicted.png", dpi=300); plt.close(fig)

residual = y_test.to_numpy() - pred
fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(residual, bins=50); ax.set_xlabel("Residual (actual − predicted)"); ax.set_ylabel("Frequency"); ax.set_title(f"Residual Distribution — {best}"); fig.tight_layout(); fig.savefig(FIGURES / "residual_distribution.png", dpi=300); plt.close(fig)

print(f"\nBest model by RMSE: {best}")
print(f"Saved metrics to: {RESULTS / 'model_performance.csv'}")
print(f"Saved figures to: {FIGURES}")
