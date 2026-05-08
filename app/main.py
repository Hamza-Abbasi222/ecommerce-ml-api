# ================================================
# E-Commerce Quantity Prediction - FastAPI
# Features: Validation, Auth, Health, Batch, Async
# ================================================

# Add this import at the top


from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, field_validator
from typing import List
import joblib
import pandas as pd
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------
# App setup
# -----------------------------------------------
app = FastAPI(
    title="E-Commerce Quantity Predictor",
    description="Predicts order quantity based on invoice and product info.",
    version="3.0"
)
# Add this RIGHT AFTER app = FastAPI(...)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins including GitHub Pages
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------
# Load model from project root (works from any folder)
# -----------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "ecommerce_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model not found at: {MODEL_PATH}\n"
        "Run train.py first to generate ecommerce_model.pkl"
    )

pipeline = joblib.load(MODEL_PATH)
print(f"✅ Model loaded from: {os.path.abspath(MODEL_PATH)}")

# -----------------------------------------------
# API Key Authentication
# HOW TO SET YOUR KEY:
#   Windows:  set API_KEY=mysecretkey123
#   Mac/Linux: export API_KEY=mysecretkey123
# If not set, defaults to "dev-key-123" for local testing
# -----------------------------------------------
VALID_API_KEY = os.environ.get("API_KEY", "dev-key-123")

def verify_api_key(x_api_key: str = Header(..., description="Your API key")):
    """
    FastAPI runs this function automatically before any protected route.
    The caller must include header:  X-API-Key: your-key-here
    """
    if x_api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Set header: X-API-Key: <your-key>"
        )

# -----------------------------------------------
# Valid countries (model was trained on these)
# -----------------------------------------------
VALID_COUNTRIES = [
    "United Kingdom", "France", "Germany", "EIRE", "Spain",
    "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia",
    "Norway", "Italy", "Channel Islands", "Finland", "Cyprus",
    "Sweden", "Austria", "Denmark", "Poland", "Japan",
    "Iceland", "USA", "Singapore", "Lebanon", "United Arab Emirates",
    "Israel", "Saudi Arabia", "Brazil", "Czech Republic", "Canada",
    "Unspecified", "Greece", "Malta", "RSA", "Bahrain",
    "Hong Kong", "Lithuania", "European Community"
]

# -----------------------------------------------
# Input Schema with Pydantic Validation
# -----------------------------------------------
class InputData(BaseModel):
    UnitPrice  : float
    CustomerID : float
    InvoiceNo  : str
    InvoiceDate: str
    Country    : str

    @field_validator("UnitPrice")
    @classmethod
    def validate_unit_price(cls, v):
        if v <= 0:
            raise ValueError("UnitPrice must be greater than 0.")
        if v > 100000:
            raise ValueError("UnitPrice seems too high (> 100,000). Please check.")
        return v

    @field_validator("CustomerID")
    @classmethod
    def validate_customer_id(cls, v):
        if v <= 0:
            raise ValueError("CustomerID must be a positive number.")
        return v

    @field_validator("InvoiceNo")
    @classmethod
    def validate_invoice_no(cls, v):
        cleaned = v.lstrip("C").strip()
        if not cleaned.isdigit():
            raise ValueError(
                f"InvoiceNo '{v}' is invalid. Must be numeric like '536365'. "
                "Cancellation invoices starting with 'C' are not supported."
            )
        return v

    @field_validator("InvoiceDate")
    @classmethod
    def validate_invoice_date(cls, v):
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
            try:
                datetime.strptime(v, fmt)
                return v
            except ValueError:
                continue
        raise ValueError(
            f"InvoiceDate '{v}' is invalid. "
            "Use format: 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD'."
        )

    @field_validator("Country")
    @classmethod
    def validate_country(cls, v):
        if v not in VALID_COUNTRIES:
            raise ValueError(
                f"Country '{v}' not recognized. "
                f"Examples: {', '.join(VALID_COUNTRIES[:5])} ..."
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "UnitPrice"  : 2.55,
                "CustomerID" : 17850.0,
                "InvoiceNo"  : "536365",
                "InvoiceDate": "2024-12-01 08:30:00",
                "Country"    : "United Kingdom"
            }
        }

# -----------------------------------------------
# Helper: build model-ready DataFrame from input
# -----------------------------------------------
def build_input_df(data: InputData) -> pd.DataFrame:
    dt = None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(data.InvoiceDate, fmt)
            break
        except ValueError:
            continue

    invoice_num = float(data.InvoiceNo.lstrip("C").strip())

    return pd.DataFrame([{
        "UnitPrice"    : data.UnitPrice,
        "CustomerID"   : data.CustomerID,
        "InvoiceNo_num": invoice_num,
        "Month"        : dt.month,
        "Day"          : dt.day,
        "Hour"         : dt.hour,
        "Country"      : data.Country
    }])


# ===============================================
# ROUTES
# ===============================================

# -----------------------------------------------
# 1. Home — no auth needed (public)
# -----------------------------------------------
@app.get("/")
async def home():
    return {
        "message": "E-Commerce ML API is running ✅",
        "version": "3.0",
        "docs"   : "/docs"
    }

# -----------------------------------------------
# 2. Health — no auth needed (public)
#    Checks if the model file exists and is loaded.
#    Useful for monitoring tools and portfolio demos.
# -----------------------------------------------
@app.get("/health")
async def health():
    return {
        "status"      : "healthy",
        "model_loaded": pipeline is not None,
        "features"    : ["UnitPrice", "CustomerID", "InvoiceNo", "InvoiceDate", "Country"],
        "version"     : "3.0",
        "checked_at"  : str(datetime.now())
    }

# -----------------------------------------------
# 3. Predict — PROTECTED with API key
#    Single prediction from one input.
#    async def used here because this is I/O ready
#    (easy to add DB logging later without refactor)
# -----------------------------------------------
@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(data: InputData):
    try:
        input_df   = build_input_df(data)
        prediction = pipeline.predict(input_df)[0]

        # Linear regression can predict negatives — clamp to 0
        prediction = max(0.0, float(prediction))

        return {
            "predicted_quantity": round(prediction, 2),
            "input_received": {
                "UnitPrice"  : data.UnitPrice,
                "CustomerID" : data.CustomerID,
                "InvoiceNo"  : data.InvoiceNo,
                "InvoiceDate": data.InvoiceDate,
                "Country"    : data.Country
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# -----------------------------------------------
# 4. Batch Predict — PROTECTED with API key
#    Send a list of inputs, get a list of predictions.
#    Max 100 items per request to prevent abuse.
# -----------------------------------------------
@app.post("/batch-predict", dependencies=[Depends(verify_api_key)])
async def batch_predict(items: List[InputData]):
    if len(items) > 100:
        raise HTTPException(
            status_code=400,
            detail="Max 100 items per batch request."
        )

    results = []
    for item in items:
        try:
            df   = build_input_df(item)
            pred = pipeline.predict(df)[0]
            pred = max(0.0, float(pred))
            results.append({
                "predicted_quantity": round(pred, 2),
                "status"            : "ok"
            })
        except Exception as e:
            # Don't fail the whole batch — log this item's error and continue
            results.append({
                "predicted_quantity": None,
                "status"            : f"error: {str(e)}"
            })

    return {
        "predictions": results,
        "total"      : len(results),
        "successful" : sum(1 for r in results if r["status"] == "ok"),
        "failed"     : sum(1 for r in results if r["status"] != "ok")
    }