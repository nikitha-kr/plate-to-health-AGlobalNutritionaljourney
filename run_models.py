"""
Runs the full model training pipeline from final_merged_data.xls.
Compares two feature sets:
  - Baseline : age, gender, education, year  (original)
  - Improved : baseline + Dietary Factor (OHE) + type + region + superregion (OHE)
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

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading final_merged_data.xls ...")
final_data = pd.read_csv("final_merged_data.xls")
print(f"  Loaded: {final_data.shape[0]:,} rows × {final_data.shape[1]} cols")

# ── 2. Categorical typing ──────────────────────────────────────────────────────
for col in ['superregion', 'country', 'type_desc', 'Dietary Factor']:
    if col in final_data.columns:
        final_data[col] = final_data[col].astype('category')

# ── 3. Remove outliers (IQR) ──────────────────────────────────────────────────
def remove_outliers(df, columns):
    for col in columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

data_no_outliers = remove_outliers(final_data, ['age', 'median', 'upperci_95', 'lowerci_95'])
print(f"  After outlier removal: {data_no_outliers.shape[0]:,} rows")

# ── 4. Stratified undersampling by 'type' ─────────────────────────────────────
# Use concat instead of groupby.apply to preserve all columns (pandas 3.x compat)
min_class = data_no_outliers['type'].value_counts().min()
reduced_data = pd.concat([
    grp.sample(min_class, random_state=42)
    for _, grp in data_no_outliers.groupby('type')
])
print(f"  After stratified sampling: {reduced_data.shape[0]:,} rows")

# ── 5. Further 20% reduction ──────────────────────────────────────────────────
further_reduced_data = reduced_data.sample(frac=0.2, random_state=42)
print(f"  Final training set: {further_reduced_data.shape[0]:,} rows\n")

TARGET = 'median'

# ── 6. Build improved feature matrix ──────────────────────────────────────────
# One-hot encode Dietary Factor (26 categories → 25 dummy cols, drop_first avoids multicollinearity)
# One-hot encode superregion (7 regions → 6 dummy cols)
# Keep numeric: age, gender, education, year, type, region
dietary_dummies    = pd.get_dummies(further_reduced_data['Dietary Factor'], prefix='df',    drop_first=True)
superregion_dummies = pd.get_dummies(further_reduced_data['superregion'],   prefix='sreg',  drop_first=True)

numeric_features = further_reduced_data[['age', 'gender', 'education', 'year', 'type', 'region']].reset_index(drop=True)
X_improved = pd.concat(
    [numeric_features,
     dietary_dummies.reset_index(drop=True),
     superregion_dummies.reset_index(drop=True)],
    axis=1
)
y = further_reduced_data[TARGET].reset_index(drop=True)
print(f"  Improved feature matrix: {X_improved.shape[1]} features "
      f"(6 numeric + {dietary_dummies.shape[1]} dietary OHE + {superregion_dummies.shape[1]} superregion OHE)\n")

# ── 7. Shared train/test split (same seed → comparable test set) ───────────────
X_train, X_test, y_train, y_test = train_test_split(X_improved, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

n_features = X_improved.shape[1]

# ── Helper ─────────────────────────────────────────────────────────────────────
def report(name, y_true, y_pred, n_feat):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    n    = len(y_true)
    adj  = r2 - (1 - r2) * (n - 1) / (n - n_feat - 1)
    print(f"  {'─'*43}")
    print(f"  {name}")
    print(f"  {'─'*43}")
    print(f"  MAE         : {mae:.4f}")
    print(f"  RMSE        : {rmse:.4f}")
    print(f"  R²          : {r2:.4f}")
    print(f"  Adjusted R² : {adj:.4f}")
    return r2

results = {}

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 47)
print("  IMPROVED FEATURE SET  (with Dietary Factor)")
print("=" * 47)

# ── Model 1 & 2: Ridge + Lasso ────────────────────────────────────────────────
print("\n--- Ridge & Lasso ---")
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
ridge_gs = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
lasso_gs = GridSearchCV(Lasso(), param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
ridge_gs.fit(X_train_s, y_train)
lasso_gs.fit(X_train_s, y_train)
print(f"  Best Ridge alpha: {ridge_gs.best_params_['alpha']}  |  Best Lasso alpha: {lasso_gs.best_params_['alpha']}")
results['Ridge'] = report("Ridge Regression", y_test, ridge_gs.best_estimator_.predict(X_test_s), n_features)
results['Lasso'] = report("Lasso Regression", y_test, lasso_gs.best_estimator_.predict(X_test_s), n_features)

# ── Model 3: KNN ──────────────────────────────────────────────────────────────
print("\n--- KNN ---")
knn_gs = GridSearchCV(
    KNeighborsRegressor(),
    {'n_neighbors': [3, 5, 7, 9, 11, 15, 21], 'weights': ['uniform', 'distance'], 'p': [1, 2]},
    scoring='neg_mean_squared_error', cv=5, n_jobs=-1
)
knn_gs.fit(X_train_s, y_train)
print(f"  Best params: {knn_gs.best_params_}")
results['KNN'] = report("KNN Regression", y_test, knn_gs.best_estimator_.predict(X_test_s), n_features)

# ── Model 4: Random Forest ────────────────────────────────────────────────────
print("\n--- Random Forest ---")
rf = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=2,
                           min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
results['Random Forest'] = report("Random Forest", y_test, rf.predict(X_test_s), n_features)

# ── Model 5: XGBoost ──────────────────────────────────────────────────────────
print("\n--- XGBoost ---")
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                              random_state=42, verbosity=0)
xgb_model.fit(X_train_s, y_train)
results['XGBoost'] = report("XGBoost", y_test, xgb_model.predict(X_test_s), n_features)

# ── Model 6: Bayesian Ridge ───────────────────────────────────────────────────
print("\n--- Bayesian Ridge ---")
best_r2, best_lambda, best_br = -np.inf, None, None
for lam in [1e-6, 1e-4, 1e-2, 1, 10]:
    br = BayesianRidge(lambda_1=lam)
    br.fit(X_train_s, y_train)
    r2 = r2_score(y_test, br.predict(X_test_s))
    print(f"  lambda_1={lam:<8} → R²={r2:.4f}")
    if r2 > best_r2:
        best_r2, best_lambda, best_br = r2, lam, br
print(f"  → Best lambda_1: {best_lambda}")
results['Bayesian Ridge'] = report("Bayesian Ridge (best lambda)", y_test, best_br.predict(X_test_s), n_features)

# ── Feature importances (Random Forest) ───────────────────────────────────────
print("\n--- Top 10 Feature Importances (Random Forest) ---")
importances = pd.Series(rf.feature_importances_, index=X_improved.columns)
top10 = importances.nlargest(10)
for feat, imp in top10.items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# ── Final summary ──────────────────────────────────────────────────────────────
BASELINE = {'Ridge': 0.0008, 'Lasso': 0.0008, 'KNN': -0.0539,
            'Random Forest': -0.0015, 'XGBoost': 0.0011, 'Bayesian Ridge': 0.0007}

print("\n" + "=" * 60)
print(f"  {'MODEL':<20} {'BASELINE R²':>12} {'IMPROVED R²':>12} {'LIFT':>8}")
print("=" * 60)
for model, r2 in sorted(results.items(), key=lambda x: -x[1]):
    base = BASELINE.get(model, 0)
    lift = r2 - base
    bar  = "█" * int(r2 * 40) if r2 > 0 else ""
    print(f"  {model:<20} {base:>12.4f} {r2:>12.4f} {lift:>+8.4f}  {bar}")
print("=" * 60)
