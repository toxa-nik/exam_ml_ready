from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import os
import sqlalchemy

app = FastAPI(title="EdTech MLOps Gateway", version="2.0")

model = None

try:
    API_KEY = os.environ["API_KEY"]
    DB_USER = os.environ["DB_USER"]
    DB_PASSWORD = os.environ["DB_PASSWORD"]
    DB_HOST = os.environ.get("POSTGRES_HOST", "postgres")
    DB_NAME = os.environ.get("POSTGRES_DB", "ml_database")
except KeyError as e:
    raise RuntimeError(f"Не задана обязательная переменная окружения {e}. Запуск остановлен.")
    
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)    

engine = sqlalchemy.create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}")

class StudentRecord(BaseModel):
    user_id: int
    duration_days: int
    event_dropout: int
    actions_total: float
    LMS_ratio_weekend: float
    lesson_activity_user_lessons_nunique: float
    SOC_peer_activity_ratio: float

class BatchIngestRequest(BaseModel):
    data: List[StudentRecord]

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Доступ запрещен. Неверный API ключ.")

@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "/shared_models/cox_model.pkl")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Модель успешно загружена!")
    except FileNotFoundError:
        print("Модель не найдена. Запустите пайплайн в Dagster.")

@app.get("/health")
def health_check():
    return {"status": "Up"}

@app.post("/ingest")
def ingest_data(payload: BatchIngestRequest, api_key: str = Depends(get_api_key)):
    """Принимает батч данных и складывает в PostgreSQL."""
    try:
        df = pd.DataFrame([record.dict() for record in payload.data])
        df.to_sql("raw_students", engine, if_exists="append", index=False)
        
        return {
            "status": "success", 
            "inserted_rows": len(df),
            "message": "Батч успешно загружен в Feature Store."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_churn(student: StudentRecord, api_key: str = Depends(get_api_key)):
    """Возвращает индекс риска (Hazard Score) для одного студента."""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель еще не обучена. Загрузите данные и запустите Dagster.")
    
    df = pd.DataFrame([student.dict()])
    features_df = df.drop(columns=["user_id", "duration_days", "event_dropout"], errors='ignore')
    
    hazard_score = model.predict_partial_hazard(features_df).iloc[0]
    return {"user_id": student.user_id, "hazard_score": float(hazard_score)}