from flask import Flask, request, render_template_string
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import create_engine, text
import pandas as pd
from joblib import load
from datetime import datetime, timedelta
import pytz
import traceback

# Flask 초기화
app = Flask(__name__)
KST = pytz.timezone("Asia/Seoul")

# SQLAlchemy 기반 DB 엔진
DB_URL = "mysql+pymysql://solar_user:solar_pass_2025@localhost/solar_forecast_muan"
engine = create_engine(DB_URL)

# 1. 실측 누적 발전량 불러오기
def load_measurements():
    query = """
        SELECT measured_at, cumulative_kwh
        FROM measurement
        WHERE cumulative_kwh IS NOT NULL
        ORDER BY measured_at
    """
    df = pd.read_sql(query, engine)
    df['measured_at'] = pd.to_datetime(df['measured_at'], format="%Y-%m-%d %H:%M:%S")
    df.set_index("measured_at", inplace=True)
    return df

# 2. 예측 결과 저장
def save_forecast_to_db(forecast_date, predicted_kwh):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO forecast_arima (forecast_date, predicted_kwh, created_at)
            VALUES (:forecast_date, :predicted_kwh, NOW())
        """), {
            'forecast_date': forecast_date,
            'predicted_kwh': predicted_kwh
        })

# 3. ARIMA 예측 실행
def run_arima_forecast():
    df = load_measurements()
    if df.empty:
        return None, "\u274c 실측 데이터가 없습니다."

    try:
        model = load("arima_model.pkl")
        forecast = model.predict(n_periods=1)
        forecast_value = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
        forecast_date = df.index[-1].date() + timedelta(days=1)
        predicted_kwh = float(forecast_value)
        save_forecast_to_db(forecast_date, predicted_kwh)
        return forecast_date, predicted_kwh
    except Exception as e:
        return None, f"\u274c 예측 오류 발생:<br><pre>{traceback.format_exc()}</pre>"

# 4. 루트 라우트 - 수동 예측 버튼 포함
@app.route("/", methods=["GET", "POST"])
def index():
    forecast_date = predicted_kwh = message = None
    if request.method == "POST":
        forecast_date, predicted_kwh = run_arima_forecast()
        if forecast_date is None:
            message = predicted_kwh

    html = f"""
        <h1>ARIMA 예측 시스템</h1>
        <form method="post">
            <button type="submit">수동 예측 실행</button>
        </form>
        {f"<p><strong>예측 일자:</strong> {forecast_date}</p>" if forecast_date else ""}
        {f"<p><strong>예측 발전량 (kWh):</strong> {float(predicted_kwh):.2f}</p>" if isinstance(predicted_kwh, (int, float)) else ""}
        {f"<p style='color:red'>{message}</p>" if message else ""}
    """
    return render_template_string(html)

# 5. 스케줄러: 매일 오전 7시 자동 실행
def start_scheduler():
    scheduler = BackgroundScheduler(timezone=KST)
    scheduler.add_job(run_arima_forecast, 'cron', hour=7, minute=30)
    scheduler.start()

# 6. 실행 시작
if __name__ == "__main__":
    print("\u2705 ARIMA 예측 서버 실행 & 스케줄러 등록됨")
    start_scheduler()
    app.run(host="0.0.0.0", port=5000)

