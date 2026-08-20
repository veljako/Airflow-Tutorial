import io
import glob
import joblib
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
import pyarrow.parquet as pq

app = FastAPI(
    title="NYC Property Price Prediction API",
    description="Production API za procenu vrednosti nekretnina u Njujorku pomocu XGBoost modela.",
    version="1.0.0",
)


matching_files = glob.glob("models/tren_model/*")
if matching_files:
    MODEL_PATH = matching_files[0] 
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"Uspešno učitan model: {MODEL_PATH}")
    except Exception as e:
        model_pipeline = None
        print(f"Greška pri učitavanju modela: {e}")
else:
    model_pipeline = None
    print("Greška: Nije pronađen nijedan model u folderu models/tren_model/")


class PropertyInput(BaseModel):
    BOROUGH: int = Field(..., examples=[1], description="1-Manhattan, 2-Bronx, 3-Brooklyn, 4-Queens, 5-Staten Island")
    NEIGHBORHOOD: str = Field(..., examples=["MIDTOWN EAST"])
    BUILDING_CLASS_CATEGORY: str = Field(..., examples=["01 ONE FAMILY DWELLINGS"], alias="BUILDING CLASS CATEGORY")
    TAX_CLASS_AT_PRESENT: str = Field(default="1", examples=["1"], alias="TAX CLASS AT PRESENT")
    BLOCK: int = Field(..., examples=[1000], alias="BLOCK")
    LOT: int = Field(default=1, examples=[20], alias="LOT")
    EASE_MENT: str | None = Field(default=None, examples=[""], alias="EASE-MENT")
    BUILDING_CLASS_AT_PRESENT: str | None = Field(default="A1", examples=["A1"], alias="BUILDING CLASS AT PRESENT")
    ADDRESS: str = Field(default="123 MAIN ST", examples=["500 PARK AVENUE"], alias="ADDRESS")
    APARTMENT_NUMBER: str | None = Field(default=None, examples=["4B"], alias="APARTMENT NUMBER")
    ZIP_CODE: int = Field(..., examples=[10022], alias="ZIP CODE")

    RESIDENTIAL_UNITS: int = Field(..., examples=[1], alias="RESIDENTIAL UNITS")
    COMMERCIAL_UNITS: int = Field(..., examples=[0], alias="COMMERCIAL UNITS")
    TOTAL_UNITS: int = Field(..., examples=[1], alias="TOTAL UNITS")
    LAND_SQUARE_FEET: float = Field(..., examples=[1500.0], alias="LAND SQUARE FEET")
    GROSS_SQUARE_FEET: float = Field(..., examples=[2000.0], alias="GROSS SQUARE FEET")

    YEAR_BUILT: int = Field(..., examples=[1990], alias="YEAR BUILT")
    TAX_CLASS_AT_TIME_OF_SALE: int = Field(..., examples=[1], alias="TAX CLASS AT TIME OF SALE")
    BUILDING_CLASS_AT_TIME_OF_SALE: str = Field(..., examples=["A1"], alias="BUILDING CLASS AT TIME OF SALE")
    SALE_DATE: str = Field(..., examples=["2023-05-15"], alias="SALE DATE", description="YYYY-MM-DD format")

    model_config = {
        "populate_by_name": True
    }


def _run_prediction(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df_copy = df.copy()
    df_copy["SALE DATE"] = pd.to_datetime(df_copy["SALE DATE"])
    log_preds = model_pipeline.predict(df_copy)
    dollar_preds = np.expm1(log_preds)
    return log_preds, dollar_preds


@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": model_pipeline is not None,
    }


@app.post("/predict")
def predict_price(payload: PropertyInput):
    if not model_pipeline:
        raise HTTPException(
            status_code=500, detail="Model (xgboost_model.joblib) nije uspesno ucitan."
        )

    try:
        input_dict = payload.model_dump(by_alias=True)
        input_df = pd.DataFrame([input_dict])

        log_prediction, dollar_prediction = _run_prediction(input_df)

        return {
            "predicted_price_usd": round(float(dollar_prediction[0]), 2),
            "log_scale_value": round(float(log_prediction[0]), 4),
            "currency": "USD",
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Greska prilikom obrade i predikcije: {str(e)}"
        )


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not model_pipeline:
        raise HTTPException(
            status_code=500, detail="Model (xgboost_model.joblib) nije uspesno ucitan."
        )

    if not file.filename.endswith(".parquet"):
        raise HTTPException(
            status_code=400, detail="Poslati fajl mora biti u .parquet formatu."
        )

    try:
        contents = await file.read()
        table = pq.read_table(io.BytesIO(contents))
        df = table.to_pandas(types_mapper=None)

        log_preds, dollar_preds = _run_prediction(df)

        results = [
            {
                "row_index": idx,
                "predicted_price_usd": round(float(dollar_pred), 2),
                "log_scale_value": round(float(log_pred), 4),
            }
            for idx, (log_pred, dollar_pred) in enumerate(zip(log_preds, dollar_preds))
        ]

        return {
            "total_rows": len(results),
            "currency": "USD",
            "predictions": results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Greska prilikom obrade Parquet fajla: {str(e)}"
        )