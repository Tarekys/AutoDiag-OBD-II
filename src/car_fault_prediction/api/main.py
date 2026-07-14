import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from car_fault_prediction.services.prediction_service import (
    preprocess_and_predict_from_df,
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "API Ready to receive data and analyze faults "}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be CSV format")

        dataframe = pd.read_csv(file.file)

        if dataframe.empty:
            raise HTTPException(status_code=400, detail="empty file")

        predictions, dataframe_with_results = preprocess_and_predict_from_df(dataframe)

        if predictions is None:
            raise HTTPException(status_code=500, detail="An error occurred during prediction")

        return {
            "status": "success",
            "results": dataframe_with_results.to_dict(orient="records"),
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Errors: {str(error)}")
