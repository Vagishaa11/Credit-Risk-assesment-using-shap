# Credit Risk Assessment & Explainable AI using SHAP

An end-to-end Machine Learning application that predicts the probability of loan default and classifies applicants into High Risk or Low Risk categories. The project combines XGBoost, probability calibration, threshold optimization, SHAP-based explainability, and FastAPI deployment to make credit risk predictions more interpretable and accessible.

## Project Overview

Financial institutions need to assess the likelihood of borrowers defaulting on their loans. Traditional credit assessment approaches may struggle to identify complex patterns in applicant financial information.

This project develops a machine learning-based Credit Risk Assessment System that analyzes applicant details and predicts the probability of loan default. It also focuses on model interpretability, helping users understand which financial factors influence credit risk.


## Live Demo
 

**Live Web Application:** https://credit-risk-assesment-using-shap-w08f.onrender.com


## Key Features

- **Credit Risk Prediction:** Predicts the probability of loan default using a trained XGBoost model.
- **Model Comparison:** Evaluates Logistic Regression and calibrated XGBoost.
- **Probability Calibration:** Improves the reliability of predicted probabilities.
- **Threshold Optimization:** Uses an optimized decision threshold instead of relying on the default 0.5 cutoff.
- **Explainable AI:** Uses SHAP to understand feature contributions and model predictions.
- **REST API:** Provides real-time predictions through FastAPI.
- **Input Validation:** Uses Pydantic models to validate loan application data.
- **Web Interface:** Serves the frontend through FastAPI static file hosting.
- **Deployment:** Includes Render deployment configuration.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Model Evaluation | ROC-AUC, PR-AUC, Precision, Recall, F1-score |
| Backend | FastAPI |
| Data Validation | Pydantic |
| Model Serialization | Joblib |
| Server | Uvicorn |
| Deployment | Render |
| Development | Jupyter Notebook, VS Code |

## Machine Learning Workflow

The project follows the following workflow:

1. **Data Collection:** Uses a loan applicant dataset containing financial and personal attributes.
2. **Exploratory Data Analysis:** Examines distributions, missing values, class imbalance, and relationships between features.
3. **Data Preprocessing:** Handles data cleaning and prepares categorical and numerical features.
4. **Model Training:** Trains and compares Logistic Regression and XGBoost.
5. **Model Calibration:** Applies probability calibration to improve probability estimates.
6. **Model Evaluation:** Evaluates performance using classification metrics, ROC curves, and Precision-Recall analysis.
7. **Threshold Optimization:** Selects a decision threshold based on the model evaluation process.
8. **Explainability:** Uses SHAP to interpret feature importance and individual predictions.
9. **Deployment:** Loads the saved model and threshold into a FastAPI application for real-time inference.

## Dataset Information

The project uses a credit risk dataset containing borrower information and loan characteristics.

**Dataset statistics:**

| Metric | Value |
|---|---:|
| Total Records | 31,522 |
| Training Records | 25,217 |
| Testing Records | 6,305 |
| Default Rate | 21.59% |

**Important features include:**

- `person_age` – Age of the applicant.
- `person_income` – Annual income.
- `person_home_ownership` – Home ownership status.
- `person_emp_length` – Employment length.
- `loan_intent` – Purpose of the loan.
- `loan_grade` – Assigned loan grade.
- `loan_amnt` – Loan amount.
- `loan_int_rate` – Interest rate.
- `loan_percent_income` – Loan amount as a percentage of income.
- `cb_person_default_on_file` – Historical default indicator.
- `cb_person_cred_hist_length` – Credit history length.

**Target variable:** `loan_status`

The target represents whether an applicant has defaulted on a loan.

## Model Performance

The saved model evaluation report contains the following test-set results:

| Metric | Logistic Regression | Calibrated XGBoost |
|---|---:|---:|
| Accuracy | 81.93% | 93.88% |
| Precision | 55.80% | 96.65% |
| Recall | 78.71% | 74.23% |
| F1-Score | 65.31% | 83.97% |
| ROC-AUC | 87.51% | 95.25% |
| PR-AUC | 72.47% | 91.00% |

**Optimized Decision Threshold:** `0.6952`

The calibrated XGBoost model outperformed Logistic Regression across the reported accuracy, precision, F1-score, ROC-AUC, and PR-AUC metrics.

The threshold is used to convert the predicted default probability into a binary classification.

## Explainable AI using SHAP

SHAP (SHapley Additive exPlanations) is used to interpret model predictions and understand how individual features contribute to credit risk.

It helps answer questions such as:

- Which financial attributes influence default predictions?
- How does an applicant's income affect predicted risk?
- How do loan amount and interest rate contribute to predictions?
- Which features have the greatest influence on the model?

SHAP provides a way to understand model behavior rather than treating the prediction as a black box.

## Application Architecture

```text
        Loan Applicant
              |
              v
       Web Interface
              |
              v
        FastAPI Backend
              |
              v
       Pydantic Validation
              |
              v
       Input DataFrame
              |
              v
      Trained XGBoost Model
              |
              v
      Default Probability
              |
              v
      Optimized Threshold
              |
              v
       Risk Classification
              |
              v
      High Risk / Low Risk
```

## API Documentation

The application exposes a REST API for credit risk prediction.

### Prediction Endpoint

**POST** `/predict`

The endpoint accepts applicant information and returns the predicted default probability and risk category.

**Example Request:**

```json
{
  "person_age": 25,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 3.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

**Example Response:**

```json
{
  "default_probability": 0.82,
  "default_prediction": 1,
  "threshold": 0.6952,
  "Result": "High Risk"
}
```

*The response values above are illustrative, not a recorded prediction for this sample.*

### API Response Explanation

- `default_probability`: Predicted probability of loan default.
- `default_prediction`: Binary classification (1 = Default, 0 = No Default).
- `threshold`: Optimized classification threshold.
- `Result`: Final risk classification.

Interactive API documentation is available at `/docs` when the application is running.

## Project Structure

```text
Credit-Risk-Assessment-using-SHAP/
│
├── static/                    # Frontend application files
│
├── Credit_risk.ipynb          # EDA, model training and analysis
│
├── credit_risk_dataset.csv    # Credit risk dataset
│
├── credit_risk_model.pkl      # Trained ML model
│
├── best_threshold.pkl         # Optimized decision threshold
│
├── model_report.json          # Model evaluation results
│
├── main.py                    # FastAPI application
│
├── requirements.txt           # Project dependencies
│
├── runtime.txt                # Python runtime configuration
│
├── render.yaml                # Render deployment configuration
│
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Vagishaa11/Credit-Risk-assesment-using-shap.git

cd Credit-Risk-assesment-using-shap
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
uvicorn main:app --reload
```

### 5. Access the Application

Open the following URLs in your browser:

- Web Application: `http://127.0.0.1:8000`
- API Documentation: `http://127.0.0.1:8000/docs`

## Deployment

The project includes a `render.yaml` configuration file for deployment on Render.


## Future Improvements

- Add JWT-based authentication and role-based access control.
- Integrate MySQL or PostgreSQL for storing loan applications and prediction history.
- Add detailed individual SHAP explanations to the web interface.
- Implement model monitoring and data drift detection.
- Introduce automated model retraining pipelines.
- Add Docker containerization and CI/CD workflows.

## Disclaimer

This project is developed for educational and demonstration purposes. Its predictions should not be used as the sole basis for real-world lending decisions. Actual credit decisions require additional validation, regulatory compliance, fairness assessment, and human oversight.

## Author

**Vagisha Bhavsar**

GitHub: [@Vagishaa11](https://github.com/Vagishaa11)

Project Repository: [Credit Risk Assessment using SHAP](https://github.com/Vagishaa11/Credit-Risk-assesment-using-shap)
