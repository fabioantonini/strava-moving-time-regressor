# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML regression project that predicts cycling moving time from route parameters (distance, elevation gain, grade). Data comes from a personal Strava export (`activities.csv`, 1764 activities, 86 columns) and is filtered to cycling rides only.

## Running the Code

No build system. Run notebooks with Jupyter or execute the standalone script directly:

```bash
python strava-moving-time-estimator-llm-rouvy.py
```

Key dependencies (install manually if missing): `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `gradio`, `joblib`.

## Notebooks

| Notebook | Purpose |
|---|---|
| `strava-moving-time-estimator.ipynb` | Baseline: data exploration, multiple model comparison (Linear, Ridge, DT, RF, Poly, SVM) |
| `strava-moving-time-estimator-llm.ipynb` | Extended EDA, correlation/distribution analysis |
| `strava-moving-time-estimator-llm-rouvy.ipynb` | Latest/best: Gradient Boosting, 95.5% R², full feature engineering pipeline |
| `strava-moving-time-gradio-ui.ipynb` | Gradio web UI for interactive predictions |

## Architecture & Data Flow

1. **Input**: `activities.csv` (raw Strava export)
2. **Cleaning**: filter to bike rides, drop short/fake routes, Isolation Forest outlier removal
3. **Features used**: `Distance`, `Elevation Gain`, `Max Grade` (Average Grade dropped — weak correlation)
4. **Feature engineering**: interaction term `Distance × Elevation Gain`, derived `Pace` and `Gradient`, polynomial features (Distance², ElevGain²)
5. **Scaling**: `MinMaxScaler` → saved as `scaler.pkl`
6. **Best model**: Gradient Boosting (R²=0.955) → saved as `best_random_forest_model.pkl` / `tuned_random_forest_model.pkl`
7. **UI**: Gradio notebook wraps the pickled model+scaler for interactive use

## Key Findings (do not re-derive)

- Most predictive feature: `Distance × Elevation Gain` interaction term
- Outlier removal (Isolation Forest) reduced MSE from 866k → 334k
- Gradient Boosting outperforms Random Forest; Decision Tree severely overfits
- Cross-validation mean R²: 93.7%; held-out R²: 95.5%

## Pending Work (from todo.txt)

- TensorFlow model variant
- Verify RMSE standard deviation and linearity conditions
- Pipeline cycling over polynomial degrees
- Rename `activities` → `routes` throughout
- Sklearn Pipeline with scaler + imputer
- Ensemble methods
- SGD with early stopping
- Train final model on full dataset
