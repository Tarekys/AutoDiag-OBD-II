from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
import io
from predictor import preprocess_and_predict_from_df

from pydantic import BaseModel
from typing import List, Dict, Any
import joblib



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
    return {"message": " API جاهز لاستقبال البيانات وتحليل الأعطال"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # قراءة الملف إلى DataFrame
        df = pd.read_csv(file.file)

        predictions, df_with_results = preprocess_and_predict_from_df(df)
        
        if predictions is None:
            return {"error": "حدث خطأ أثناء التنبؤ"}
        
        return {
            "status": "success",
            "results": df_with_results.to_dict(orient="records")
        }
    except Exception as e:
        return {"error": f"فشل في معالجة الملف: {str(e)}"}
