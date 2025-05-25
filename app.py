import pymysql
import pandas as pd
from joblib import load
from datetime import datetime, timedelta
import pytz
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler

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
    'cursorclass': pymysql.cursors.DictCursor  # SELECT 결과를 딕셔너리 형태로 반환
}

# 1. 실측 누적 발전량 데이터를 DB에서 시간순으로 불러오는 함수
def load_measurements():
    conn = pymysql.connect(**DB_CONFIG)
    query = """
        SELECT measured_at, cumulative_kwh
        FROM measurement
        WHERE cumulative_kwh IS NOT NULL
        ORDER BY measured_at
    """
    df = pd.read_sql(query, conn, parse_dates=['measured_at'])
    conn.close()
    return df

# 2. 예측 결과를 forecast_arima 테이블에 저장하는 함수
def save_forecast_to_db(forecast_date, predicted_kwh):
    conn = pymysql.connect(**DB_CONFIG)
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO forecast_arima (forecast_date, predicted_kwh, created_at)
            VALUES (%s, %s, NOW())
        """, (forecast_date, predicted_kwh))
    conn.commit()
    conn.close()

# 3. 저장된 ARIMA 모델을 불러와서 익일 발전량을 예측하는 함수
def run_arima_forecast():
    df = load_measurements()
    if df.empty:
        return None, "❌ 실측 데이터가 없습니다."

    try:
        model = load("arima_model.pkl")
        forecast = model.predict(n_periods=1)  # 1일치 예측
        forecast_date = df["measured_at"].iloc[-1].date() + timedelta(days=1)
        predicted_kwh = float(forecast[0])

        save_forecast_to_db(forecast_date, predicted_kwh)
        return forecast_date, predicted_kwh
    except Exception as e:
        return None, f"❌ 예측 오류 발생: {e}"

# 4. 루트 라우트: 웹 브라우저에서 접속 시 예측을 수행하고 결과를 HTML로 반환
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
        return f"<h1>500 내부 오류 발생</h1><pre>{e}</pre>", 500

# 5. 스케줄러: 매일 오전 7시 00분에 ARIMA 예측 자동 실행
def start_scheduler():
    scheduler = BackgroundScheduler(timezone=KST)
    scheduler.add_job(run_arima_forecast, 'cron', hour=7, minute=0)
    scheduler.start()

# 6. 직접 실행 시 콘솔에 예측 결과 출력 및 스케줄러 시작
if __name__ == "__main__":
    print("✅ ARIMA 백엔드 시작 및 스케줄러 등록")
    start_scheduler()  # 07:00 자동 실행 스케줄러 시작
    app.run(host="0.0.0.0", port=5001)
