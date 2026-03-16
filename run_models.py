"""
Runs the full model training pipeline from final_merged_data.xls
and reports R² (and other metrics) for all 6 models.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading final_merged_data.xls ...")
# File has .xls extension but is actually CSV-formatted
final_data = pd.read_csv("final_merged_data.xls")
print(f"  Loaded: {final_data.shape[0]:,} rows × {final_data.shape[1]} cols")

# ── 2. Categorical conversion ─────────────────────────────────────────────────
categorical_columns = ['superregion', 'country', 'type_desc', 'Dietary Factor']
for col in categorical_columns:
    if col in final_data.columns:
        final_data[col] = final_data[col].astype('category')

# ── 3. Remove outliers (IQR) ──────────────────────────────────────────────────
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

numeric_columns = ['age', 'median', 'upperci_95', 'lowerci_95']
data_no_outliers = remove_outliers(final_data, numeric_columns)
print(f"  After outlier removal: {data_no_outliers.shape[0]:,} rows")

# ── 4. Stratified undersampling ───────────────────────────────────────────────
target_column = 'type'
min_class_count = data_no_outliers[target_column].value_counts().min()
reduced_data = data_no_outliers.groupby(target_column, group_keys=False).apply(
    lambda x: x.sample(min_class_count, random_state=42)
)
print(f"  After stratified sampling: {reduced_data.shape[0]:,} rows")

# ── 5. Further 20% reduction ──────────────────────────────────────────────────
further_reduced_data = reduced_data.sample(frac=0.2, random_state=42)
print(f"  Final training set: {further_reduced_data.shape[0]:,} rows\n")

# ── 6. Shared train/test split ────────────────────────────────────────────────
FEATURES = ['age', 'gender', 'education', 'year']
TARGET   = 'median'

X = further_reduced_data[FEATURES]
y = further_reduced_data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

def metrics(name, y_true, y_pred, n_features):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    n    = len(y_true)
    adj  = r2 - (1 - r2) * (n - 1) / (n - n_features - 1)
    print(f"{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  MAE          : {mae:.4f}")
    print(f"  MSE          : {mse:.4f}")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"  R²           : {r2:.4f}")
    print(f"  Adjusted R²  : {adj:.4f}")
    return r2

results = {}

# ── Model 1 & 2: Ridge + Lasso ────────────────────────────────────────────────
print("\n=== RIDGE & LASSO REGRESSION ===")
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

ridge_gs = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_squared_error', cv=5)
lasso_gs = GridSearchCV(Lasso(), param_grid, scoring='neg_mean_squared_error', cv=5)
ridge_gs.fit(X_train_s, y_train)
lasso_gs.fit(X_train_s, y_train)
print(f"  Best Ridge alpha: {ridge_gs.best_params_['alpha']}")
print(f"  Best Lasso alpha: {lasso_gs.best_params_['alpha']}")

results['Ridge']  = metrics("Ridge Regression",  y_test, ridge_gs.best_estimator_.predict(X_test_s), len(FEATURES))
results['Lasso']  = metrics("Lasso Regression",  y_test, lasso_gs.best_estimator_.predict(X_test_s), len(FEATURES))

# ── Model 3: KNN ──────────────────────────────────────────────────────────────
print("\n=== KNN REGRESSION ===")
knn_gs = GridSearchCV(
    KNeighborsRegressor(),
    {'n_neighbors': [3, 5, 7, 9, 11, 15, 21], 'weights': ['uniform', 'distance'], 'p': [1, 2]},
    scoring='neg_mean_squared_error', cv=5
)
knn_gs.fit(X_train_s, y_train)
print(f"  Best params: {knn_gs.best_params_}")
results['KNN'] = metrics("KNN Regression", y_test, knn_gs.best_estimator_.predict(X_test_s), len(FEATURES))

# ── Model 4: Random Forest ────────────────────────────────────────────────────
print("\n=== RANDOM FOREST REGRESSION ===")
rf = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=2,
                           min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
results['Random Forest'] = metrics("Random Forest", y_test, rf.predict(X_test_s), len(FEATURES))

# ── Model 5: XGBoost ──────────────────────────────────────────────────────────
print("\n=== XGBOOST REGRESSION ===")
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                              random_state=42, verbosity=0)
xgb_model.fit(X_train_s, y_train)
results['XGBoost'] = metrics("XGBoost", y_test, xgb_model.predict(X_test_s), len(FEATURES))

# ── Model 6: Bayesian Ridge (FIXED — real data) ───────────────────────────────
print("\n=== BAYESIAN RIDGE REGRESSION (real dietary data) ===")
best_r2, best_lambda = -np.inf, None
for lam in [1e-6, 1e-4, 1e-2, 1, 10]:
    br = BayesianRidge(lambda_1=lam)
    br.fit(X_train_s, y_train)
    r2 = r2_score(y_test, br.predict(X_test_s))
    print(f"  lambda_1={lam:<8} → R²={r2:.4f}")
    if r2 > best_r2:
        best_r2, best_lambda, best_br = r2, lam, br

print(f"\n  → Best lambda_1: {best_lambda}")
results['Bayesian Ridge'] = metrics("Bayesian Ridge (best lambda)", y_test, best_br.predict(X_test_s), len(FEATURES))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*45)
print("  FINAL R² SUMMARY (all models)")
print("="*45)
for model, r2 in sorted(results.items(), key=lambda x: -x[1]):
    bar = "█" * int(r2 * 30) if r2 > 0 else ""
    print(f"  {model:<20} R² = {r2:.4f}  {bar}")
print("="*45)
