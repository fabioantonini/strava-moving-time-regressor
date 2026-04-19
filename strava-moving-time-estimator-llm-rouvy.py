#!/usr/bin/env python
# coding: utf-8

"""
Strava/Rouvy Moving Time Predictor
Predicts virtual cycling ride time from route parameters (distance, elevation, grade).
Pipeline: data filtering → feature engineering → outlier removal (IsolationForest)
         → model comparison → GridSearchCV tuning → sklearn Pipeline export
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("seaborn not installed — heatmaps will be skipped (pip install seaborn to enable)")

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, IsolationForest
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ── 1. Load & Filter ──────────────────────────────────────────────────────────

data = pd.read_csv('activities.csv')
routes = data[data['Activity Type'] == 'Virtual Ride'].copy()

for col in ['Moving Time', 'Distance', 'Elevation Gain', 'Max Grade']:
    routes[col] = pd.to_numeric(routes[col], errors='coerce')

routes = routes[routes['Moving Time'] >= 180]
routes = routes[['Moving Time', 'Distance', 'Elevation Gain', 'Max Grade']].dropna()

print(f"Routes after filtering: {len(routes)}")


# ── 2. Correlation Matrix (before engineering) ────────────────────────────────

correlation_matrix = routes.corr()
if HAS_SEABORN:
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix — Base Features")
    plt.tight_layout()
    plt.savefig("correlation_base.png", dpi=100)
    plt.show()

print("\nCorrelation with Moving Time:")
print(correlation_matrix["Moving Time"].sort_values(ascending=False).to_string())


# ── 3. Feature Engineering ────────────────────────────────────────────────────

routes['Distance * Elevation Gain'] = routes['Distance'] * routes['Elevation Gain']
routes['Total Climbing Effort'] = (
    routes['Elevation Gain'] / routes['Distance']
).replace([np.inf, -np.inf], 0).fillna(0)
routes['Distance^2'] = routes['Distance'] ** 2
routes['Elevation Gain^2'] = routes['Elevation Gain'] ** 2
routes['Max Grade^2'] = routes['Max Grade'] ** 2

FEATURE_COLS = [
    'Distance', 'Elevation Gain', 'Max Grade',
    'Distance * Elevation Gain', 'Total Climbing Effort',
    'Distance^2', 'Elevation Gain^2', 'Max Grade^2',
]

print(f"\nFeature engineering done. Shape: {routes.shape}")


# ── 4. Outlier Removal (Isolation Forest) ─────────────────────────────────────

iso = IsolationForest(contamination=0.05, random_state=42)
mask = iso.fit_predict(routes[FEATURE_COLS]) == 1
n_removed = (~mask).sum()
routes = routes[mask].reset_index(drop=True)
print(f"Outliers removed: {n_removed} → remaining: {len(routes)}")


# ── 5. Correlation Matrix (after engineering) ─────────────────────────────────

corr_eng = routes[FEATURE_COLS + ['Moving Time']].corr()
if HAS_SEABORN:
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr_eng, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix — Engineered Features")
    plt.tight_layout()
    plt.savefig("correlation_engineered.png", dpi=100)
    plt.show()

print("\nCorrelation with Moving Time (engineered):")
print(corr_eng["Moving Time"].sort_values(ascending=False).to_string())


# ── 6. Train / Test Split ─────────────────────────────────────────────────────

X = routes[FEATURE_COLS]
y = routes['Moving Time']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")


# ── 7. Model Comparison ───────────────────────────────────────────────────────

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """Wrap model in Pipeline, fit, evaluate with R², RMSE, MAE, CV R²±std."""
    pipe = Pipeline([('scaler', MinMaxScaler()), ('model', model)])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    r2   = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    cv   = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring='r2')

    print(
        f"  {name:<28}  R²={r2:.3f}  "
        f"RMSE={rmse:.0f}s ({rmse/60:.1f}min)  "
        f"MAE={mae:.0f}s  "
        f"CV R²={cv.mean():.3f}±{cv.std():.3f}"
    )
    return pipe, r2


candidates = [
    ("Random Forest",
     RandomForestRegressor(
         n_estimators=100, max_depth=10,
         min_samples_leaf=2, min_samples_split=5, random_state=42
     )),
    ("Gradient Boosting",
     GradientBoostingRegressor(
         n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
     )),
    ("Ridge Regression",
     Ridge(alpha=1.0)),
    ("SVR (RBF)",
     SVR(kernel='rbf', C=100, gamma=0.1, epsilon=10)),
    ("MLP Neural Network",
     MLPRegressor(
         hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
     )),
]

print("\n── Model Comparison ──")
results = []
for name, model in candidates:
    pipe, r2 = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results.append((name, pipe, r2))

best_name, _, best_r2 = max(results, key=lambda x: x[2])
print(f"\nBest baseline model: {best_name} (R²={best_r2:.3f})")


# ── 8. Gradient Boosting — Hyperparameter Tuning ─────────────────────────────

print("\n── Tuning Gradient Boosting ──")

gb_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', GradientBoostingRegressor(random_state=42)),
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth':    [3, 4, 5],
    'model__learning_rate': [0.05, 0.1],
}

grid_search = GridSearchCV(
    gb_pipeline, param_grid,
    cv=5, scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

tuned_pipeline = grid_search.best_estimator_
y_pred_tuned   = tuned_pipeline.predict(X_test)

r2_t   = r2_score(y_test, y_pred_tuned)
rmse_t = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
mae_t  = mean_absolute_error(y_test, y_pred_tuned)
cv_t   = cross_val_score(tuned_pipeline, X_train, y_train, cv=5, scoring='r2')

print(f"Best params : {grid_search.best_params_}")
print(
    f"Tuned GB    : R²={r2_t:.3f}  "
    f"RMSE={rmse_t:.0f}s ({rmse_t/60:.1f}min)  "
    f"MAE={mae_t:.0f}s  "
    f"CV R²={cv_t.mean():.3f}±{cv_t.std():.3f}"
)


# ── 9. Feature Importance ────────────────────────────────────────────────────

gb_model = tuned_pipeline.named_steps['model']
importances = pd.Series(gb_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
importances.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title("Feature Importance — Tuned Gradient Boosting")
plt.ylabel("Importance")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("feature_importance_gb.png", dpi=100)
plt.show()

print("\nFeature Importance:")
print(importances.to_string())


# ── 10. Predictions vs Actual ────────────────────────────────────────────────

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_tuned, alpha=0.6, color='steelblue', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title(f"Predicted vs Actual Moving Time — Tuned GB (R²={r2_t:.3f})")
plt.xlabel("Actual Moving Time (s)")
plt.ylabel("Predicted Moving Time (s)")
plt.grid(True)
plt.tight_layout()
plt.savefig("predictions_vs_actual.png", dpi=100)
plt.show()


# ── 11. Save Pipeline ────────────────────────────────────────────────────────

joblib.dump(tuned_pipeline, 'pipeline_gb.pkl')
print("\nSaved: pipeline_gb.pkl  (MinMaxScaler + GradientBoosting in one object)")


# ── 12. Prediction Helper ────────────────────────────────────────────────────

def predict_moving_time(distance_km: float, elevation_gain_m: float, max_grade_pct: float,
                        pipeline=tuned_pipeline) -> float:
    """
    Predict moving time in seconds from route parameters available before the ride.

    Args:
        distance_km       : Route distance in kilometres
        elevation_gain_m  : Total elevation gain in metres
        max_grade_pct     : Maximum gradient in percent

    Returns:
        Predicted moving time in seconds
    """
    d, e, g = distance_km, elevation_gain_m, max_grade_pct
    row = {
        'Distance':                  d,
        'Elevation Gain':            e,
        'Max Grade':                 g,
        'Distance * Elevation Gain': d * e,
        'Total Climbing Effort':     e / d if d > 0 else 0,
        'Distance^2':                d ** 2,
        'Elevation Gain^2':          e ** 2,
        'Max Grade^2':               g ** 2,
    }
    seconds = pipeline.predict(pd.DataFrame([row]))[0]
    print(f"Route {d:.1f}km  +{e:.0f}m  {g:.1f}% → {seconds/60:.1f} min  ({seconds:.0f}s)")
    return seconds


print("\n── Example Predictions ──")
predict_moving_time(32.6, 226.0, 9.0)   # Brunnen-style flat
predict_moving_time(50.0, 800.0, 8.0)   # Medium mountain
predict_moving_time(100.0, 2000.0, 12.0)  # Hard gran fondo
