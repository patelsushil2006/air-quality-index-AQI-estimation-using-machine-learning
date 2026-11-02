# AQI Estimation Using Machine Learning

A data analytics and machine-learning case study that estimates the Air Quality Index (AQI) from pollutant measurements and city/date information in the India city-day air-quality dataset.

> **Scope:** This project estimates AQI for held-out observations. It is **not** a future-AQI forecasting system.

## Project objectives
- Explore air-quality measurements and missing-data patterns.
- Clean and prepare the dataset for supervised regression.
- Compare Linear Regression, Random Forest, Gradient Boosting and XGBoost.
- Evaluate models with MAE, RMSE and R².
- Visualize model performance and prediction errors.

## Dataset
The project uses `data/city_day.csv`, sourced from the Kaggle India Air Quality Data collection:
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

The source file contains 29,531 rows and 16 columns. After removing rows without AQI, dropping the derived `AQI_Bucket` field and removing `Xylene` (61.32% missing), the notebook reports 24,850 rows and 14 columns. Remaining predictor missing values are imputed within a preprocessing pipeline.

## Methods
1. Parse dates and sort records chronologically.
2. Use pollutant concentrations, city and calendar components (year, month, day) as inputs; use AQI as the target.
3. Split by unique dates: training dates before the 80%-position split date and test dates on/after it. The actual split is 16,018 training records and 8,832 test records.
4. Impute numeric values with training medians; impute City with the training-set mode; one-hot encode City. Fit preprocessing only on training data.
5. Train and compare four regression models.

## Recorded notebook results
| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | 18.4283 | **31.4081** | **0.9087** |
| Random Forest | 17.7693 | 32.3495 | 0.9032 |
| XGBoost | **17.7473** | 35.0770 | 0.8862 |
| Linear Regression | 24.9152 | 37.3474 | 0.8709 |

Gradient Boosting has the lowest RMSE and highest R² in the recorded experiment; XGBoost has the lowest MAE. The figures in `figures/` and the original experiment outputs in `results/` are included for review.

## Repository structure
```text
AQI_Project_GitHub/
├── AQI_Prediction_ML.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── city_day.csv
├── src/
│   └── train_aqi_models.py
├── figures/
│   ├── figure_1_error_comparison.png
│   ├── figure_2_r2_comparison.png
│   ├── figure_3_actual_vs_predicted.png
│   ├── figure_4_residual_analysis.png
│   └── figure_5_aqi_time_series.png
└── results/
    ├── final_model_performance.csv
    ├── model_performance_metrics.csv
    └── experiment_summary.txt
```

## Run locally
Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train_aqi_models.py
```

Open `AQI_Prediction_ML.ipynb` in Jupyter or upload the repository to Google Colab to inspect the complete exploratory analysis and model evaluation workflow.

## Important interpretation and limitations
- The dataset has substantial missingness; the preprocessing pipeline imputes remaining missing predictor values.
- AQI values are highly skewed, and prediction errors are larger for extreme AQI observations. Interpret aggregate metrics alongside the AQI-range analysis in the notebook.
- The chronological split is based on dates shared across cities. Multiple cities recorded on a split date can fall on the same side of the date boundary; this is not a strict city-wise or unseen-city evaluation.
- This is an educational case study, not a certified public-health monitoring tool.

## Team / author
Add your team members, roll number(s), department and faculty details here before publishing if required by your course.

## License and dataset acknowledgement
The dataset remains subject to the terms set by its original Kaggle publisher. Retain the source attribution and check the dataset page's license before redistribution. The analysis code in this repository is provided for academic use.


## Important note about the two supplied datasets

This repository contains the exact notebook uploaded for this project (`AQI_Prediction_ML.ipynb`) and the exact additional dataset archive supplied by the student (`data/user_provided_AirQualityUCI/air+quality.zip`), including its extracted CSV and XLSX files.

**Dataset compatibility:** The notebook's code loads `city_day.csv` and expects the India city-day dataset with fields such as `City`, `Date`, pollutant measurements, and target `AQI`. The separately attached `AirQualityUCI.csv` / `.xlsx` is a different UCI hourly air-sensor dataset (semicolon-separated, with fields such as `CO(GT)`, `PT08.S1(CO)`, `T`, `RH`, and `AH`) and does not contain the same `AQI` target column. Therefore, it is included as the student's original attached dataset for completeness, but it is **not the dataset used by the current notebook**. The notebook's required `data/city_day.csv` is also included and is the dataset used by the experiment.

Do not claim that the UCI `AirQualityUCI` files were used to generate the notebook's recorded AQI results. If the faculty specifically requires the attached UCI dataset to be the dataset analyzed, the notebook and report must be redesigned for that dataset before submission.

## Files supplied and verification

- `AQI_Prediction_ML.ipynb`: replaced with the exact notebook file uploaded in the current conversation.
- `data/city_day.csv`: dataset used by the notebook and the recorded India AQI experiment.
- `data/user_provided_AirQualityUCI/air+quality.zip`: exact dataset ZIP uploaded in the current conversation.
- `data/user_provided_AirQualityUCI/AirQualityUCI.csv` and `.xlsx`: extracted copies of the uploaded dataset archive.
- `Project_Report.pdf`: project report included in the repository.

Before submission, open the notebook and run it from top to bottom in an environment with the listed dependencies. Check that the data path points to `data/city_day.csv` when running locally; the notebook currently uses a Google Colab path (`/content/city_day.csv`).
