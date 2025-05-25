import pymysql
import pandas as pd
from joblib import load
from datetime import datetime, timedelta
import pytz
from flask import Flask, jsonify

# Flask 애플리케이션 초기화
app = Flask(__name__)
KST = pytz.timezone("Asia/Seoul")  # 한국 표준시 (KST) 설정

# DB 접속 정보 설정
DB_CONFIG = {
    'host': 'localhost',
    'user': 'solar_user',
    'password': 'solar_pass_2025',
    'db': 'solar_forecast_muan',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor  # 결과를 딕셔너리로 반환
}

# 1. measurement 테이블에서 누적 발전량 데이터를 시간 순으로 불러오는 함수
def load_measurements():
    conn = pymysql.connect(**DB_CONFIG)
    query = """
        SELECT measured_at, cumulative_kwh
        FROM measurement
        WHERE cumulative_kwh IS NOT NULL
        ORDER BY measured_at
    """
    df = pd.read_sql(query, conn, parse_dates=['measured_at'])  # measured_at을 datetime으로 파싱
    conn.close()
    return df

# 2. 예측 결과를 forecast_arima 테이블에 저장하는 함수
def save_forecast_to_db(forecast_date, predicted_kwh):
    conn = pymysql.connect(**DB_CONFIG)
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO forecast_arima (forecast_date, predicted_kwh, created_at)
            VALUES (%s, %s, NOW())  -- 현재 시간 기준으로 저장
        """, (forecast_date, predicted_kwh))
    conn.commit()
    conn.close()

# 3. 저장된 ARIMA 모델을 불러와서 예측 수행 및 결과 저장
def run_arima_forecast():
    df = load_measurements()  # 실측 누적 발전량 불러오기
    if df.empty:
        return None, "❌ 실측 데이터가 없습니다."

    model = load("arima_model.pkl")  # 저장된 모델 불러오기
    forecast = model.predict(n_periods=1)  # 익일 발전량 예측 (1일치)
    forecast_date = df["measured_at"].iloc[-1].date() + timedelta(days=1)  # 마지막 데이터 기준 다음 날
    predicted_kwh = float(forecast[0])  # 예측 결과 실수형 변환

    save_forecast_to_db(forecast_date, predicted_kwh)  # 예측 결과 DB에 저장
    return forecast_date, predicted_kwh

# 4. 루트 경로에 접속 시 예측을 실행하고 결과를 HTML로 보여주는 라우트
@app.route("/")
def index():
    try:
        forecast_date, predicted_kwh = run_arima_forecast()
        if forecast_date is None:
            return f"<h1>예측 실패</h1><p>{predicted_kwh}</p>", 400
        return f"""
            <h1>ARIMA 예측 결과</h1>
            <p><strong>예측 일자:</strong> {forecast_date}</p>
            <p><strong>예측 발전량 (kWh):</strong> {predicted_kwh:.2f}</p>
        """
    except Exception as e:
        # 예외 발생 시 브라우저에서 상세 오류 확인 가능
        return f"<h1>500 내부 오류 발생</h1><pre>{e}</pre>", 500

# 5. 메인 실행 진입점
# 해당 파일이 직접 실행될 때만 Flask 서버를 시작
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)