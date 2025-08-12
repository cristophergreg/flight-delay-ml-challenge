from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import DelayModel

class Flight(BaseModel):
    """Single flight payload used for prediction.

    Attributes:
        OPERA (str): Airline/operator. Must exist in the training catalog.
        TIPOVUELO (str): Flight type, either 'N' (domestic) or 'I' (international).
        MES (int): Month number in the range 1..12.
    """
    OPERA: str
    TIPOVUELO: str  # "N" o "I"
    MES: int        # 1..12

class PredictRequest(BaseModel):
    """Batch request schema for the /predict endpoint.

    Attributes:
        flights (List[Flight]): Non-empty list of flights to score.
    """
    flights: List[Flight]

class PredictResponse(BaseModel):
    """Response schema for the /predict endpoint.

    Attributes:
        predict (List[int]): Predicted labels (0/1) for each input flight.
    """
    predict: List[int]

app = FastAPI(title="Flight Delay API")

# Train the model once at startup (single-shot bootstrap).
_train_df = pd.read_csv("data/data.csv")
_model = DelayModel()
_X_train, _y_train = _model.preprocess(_train_df, target_column="delay")
_model.fit(_X_train, _y_train)

# Domain catalogs for input validation.
_VALID_OPERAS = set(_train_df["OPERA"].astype(str).unique())
_VALID_TIPOS = {"N", "I"}
_VALID_MESES = set(range(1, 13))

def _validate_flight(f: Flight) -> None:
    """Validate a single Flight against domain constraints.

    Checks:
        - OPERA must be in the training catalog.
        - TIPOVUELO must be one of {'N', 'I'}.
        - MES must be an integer within 1..12.

    Raises:
        HTTPException: With status_code=400 when any constraint fails.
    """
    errs = []
    if f.OPERA not in _VALID_OPERAS:
        errs.append("Invalid OPERA")
    if f.TIPOVUELO not in _VALID_TIPOS:
        errs.append("Invalid TIPOVUELO (must be 'N' or 'I')")
    if f.MES not in _VALID_MESES:
        errs.append("Invalid MES (must be 1..12)")
    if errs:
        # Tests esperan 400 ante errores de dominio
        raise HTTPException(status_code=400, detail=", ".join(errs))

@app.get("/health", status_code=200)
async def get_health() -> dict:
    """Liveness probe to verify the service is running.

    Returns:
        dict: {"status": "OK"} when the service is healthy.
    """
    return {
        "status": "OK"
    }

@app.post("/predict", response_model=PredictResponse, status_code=200)
async def post_predict(req: PredictRequest) -> dict:
    """Batch prediction endpoint.

    Validates each incoming flight and returns a list of predicted labels.
    The underlying model is trained at process startup and reused here.

    Args:
        req (PredictRequest): Batch request with at least one Flight.

    Returns:
        dict: {"predict": List[int]} where each element is 0 (on-time) or 1 (delay).

    Raises:
        HTTPException:
            - 400 if the payload is empty or any flight fails domain validation.
    """
    if not req.flights:
        raise HTTPException(status_code=400, detail="Empty 'flights' payload")

    # Validate each flight
    for f in req.flights:
        _validate_flight(f)

    # To DataFrame → features → predictions
    df = pd.DataFrame([f.dict() for f in req.flights])
    X = _model.preprocess(df)

    preds = _model.predict(X)
    return {"predict": preds}