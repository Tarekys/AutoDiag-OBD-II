import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from predictor import preprocess_and_predict_from_df

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
    return {"message": "API جاهز لاستقبال البيانات وتحليل الأعطال"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="الملف يجب أن يكون بصيغة CSV")

        df = pd.read_csv(file.file)


        if df.empty:
            raise HTTPException(status_code=400, detail="الملف فارغ")
            
        predictions, df_with_results = preprocess_and_predict_from_df(df)

        # التحقق من نجاح التنبؤ
        if predictions is None:
            raise HTTPException(status_code=500, detail="حدث خطأ أثناء التنبؤ")
            
        return {
            "status": "success",
            "results": df_with_results.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل في معالجة الملف: {str(e)}")
