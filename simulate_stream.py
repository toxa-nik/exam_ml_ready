import pandas as pd
import requests
import time
import os
import sys

API_URL = "http://localhost:8000/ingest"
API_KEY = "my_secure_api_key_2026"

def stream_data(file_path):
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден!")
        return

    df = pd.read_csv(file_path)
    chunk_size = 20
    headers = {"X-API-Key": API_KEY}

    print(f"Начинаем стриминг {len(df)} строк в {API_URL}...")
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        payload = {"data": chunk.to_dict(orient="records")}
        
        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            print(f"[{i}/{len(df)}] Отправлено: {len(chunk)} строк. Статус: {response.status_code}")
        except Exception as e:
            print(f"Ошибка соединения: {e}")
            
        time.sleep(3)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python3 simulate_stream.py data/mlops_historical_data.csv")
    else:
        stream_data(sys.argv[1])