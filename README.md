# Insurance Charges Prediction

Machine learning project for predicting medical insurance charges using Random Forest and XGBoost regression models.

## Project Overview

The goal of this project is to predict individual medical insurance costs based on personal and health-related features such as:

- age
- sex
- BMI
- number of children
- smoking status
- region

The project includes:
- exploratory data analysis (EDA)
- preprocessing pipelines
- feature engineering
- model training and evaluation
- cross-validation
- feature importance analysis

---

# Dataset

The dataset contains information about medical insurance beneficiaries and their corresponding insurance charges.

### Features
- `age` — age of primary beneficiary
- `sex` — gender
- `bmi` — body mass index
- `children` — number of dependents
- `smoker` — smoking status
- `region` — residential area in the US
- `charges` — medical insurance cost (target variable)

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

# Machine Learning Workflow

## 1. Exploratory Data Analysis
Performed:
- distribution analysis
- correlation analysis
- outlier inspection
- feature relationship visualization

## 2. Data Preprocessing
Implemented preprocessing using:
- `Pipeline`
- `ColumnTransformer`
- `OneHotEncoder`
- `SimpleImputer`

Target variable was log-transformed using:

```python
np.log1p()
```

to reduce skewness.

## 3. Models
The following regression models were trained:
- Random Forest Regressor
- XGBoost Regressor

## 4. Model Evaluation
Models were evaluated using:
- RMSE
- MAPE
- Cross-validation

## 5. Feature Importance
Feature importance analysis was performed for model interpretability.

---

# Results

The models showed strong predictive performance on the dataset.

Main findings:
- smoking status had the strongest impact on insurance charges
- age and BMI were also highly influential features
- region and sex had relatively low predictive importance

---

# Repository Structure

```text
├── insurance_regression.ipynb
├── insurance.csv
└── README.md
```

---

# Future Improvements

Possible future improvements:
- SHAP analysis
- hyperparameter tuning
- model deployment with Streamlit or FastAPI
- saving/loading trained models

---

# Author

Aryna Kachaharava
