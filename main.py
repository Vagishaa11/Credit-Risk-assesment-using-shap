from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not on every request
    ml_model["model"] = joblib.load(BASE_DIR / "credit_risk_model.pkl")
    ml_model["threshold"] = float(joblib.load(BASE_DIR / "best_threshold.pkl"))
    yield
    ml_model.clear()


app = FastAPI(title="Credit Risk API", lifespan=lifespan)


# FIX 9: strict validation (ranges and allowed values)
# FIX 8: loan_percent_income is NOT asked from the user anymore.
class LoanApplication(BaseModel):
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0, le=1_000_000)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(ge=0, le=60)
    loan_intent: Literal[
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
        "HOMEIMPROVEMENT", "DEBTCONSOLIDATION",
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(gt=0, le=1_000_000)
    loan_int_rate: float = Field(gt=0, le=40)
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(ge=0, le=80)

    @model_validator(mode="after")
    def check_consistency(self):
        if self.person_emp_length > self.person_age:
            raise ValueError("person_emp_length cannot be greater than person_age")
        if self.cb_person_cred_hist_length > self.person_age:
            raise ValueError("cb_person_cred_hist_length cannot be greater than person_age")
        return self


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: LoanApplication):
    row = data.model_dump()  # FIX 9: .dict() is deprecated in Pydantic v2

    # FIX 8: calculate it on the server, same formula as in training
    row["loan_percent_income"] = row["loan_amnt"] / row["person_income"]

    input_df = pd.DataFrame([row])

    probability = float(ml_model["model"].predict_proba(input_df)[:, 1][0])
    threshold = ml_model["threshold"]
    prediction = int(probability >= threshold)  # same ">=" rule as in the notebook

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }


# Mount the frontend last. If the static folder is missing, "/" opens the API docs instead of a 404.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")