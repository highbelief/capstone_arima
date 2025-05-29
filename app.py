from flask import Flask, request, render_template_string
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import create_engine, text
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import traceback

# Flask 앱 및 시간대
app = Flask(__name__)
KST = pytz.timezone("Asia/Seoul")

# DB 연결
DB_URL = "mysql+pymysql://solar_user:solar_pass_2025@localhost/solar_forecast_muan"
engine = create_engine(DB_URL)

# 실측 데이터 로딩
def load_measurements():
    query = """
        SELECT measured_at, cumulative_mwh,
               forecast_irradiance_wm2, forecast_temperature_c, forecast_wind_speed_ms
        FROM measurement
        WHERE cumulative_mwh IS NOT NULL
        ORDER BY measured_at
    """
    df = pd.read_sql(query, engine, parse_dates=['measured_at'])
    df.set_index('measured_at', inplace=True)
    return df

# 예측 결과 저장
def save_forecast_to_db(forecast_date, predicted_mwh, actual_mwh=None, rmse=None, mae=None, mape=None):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO forecast_arima (forecast_date, predicted_mwh, actual_mwh, rmse, mae, mape, created_at)
            VALUES (:forecast_date, :predicted_mwh, :actual_mwh, :rmse, :mae, :mape, NOW())
        """), {
            'forecast_date': forecast_date,
            'predicted_mwh': predicted_mwh,
            'actual_mwh': actual_mwh,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        })

# ARIMA 예측
def run_arima_forecast():
    df = load_measurements()
    if df.empty:
        return None, "❌ 실측 데이터가 없습니다."

    try:
        # 일일 발전량 집계
        daily_power = df['cumulative_mwh'].resample('D').agg(lambda x: x.max() - x.min())
        daily_power = daily_power[daily_power > 1000]  # 이상치 제거
        log_power = np.log1p(daily_power)

        # 외생 변수 집계
        weather = df.resample('D').agg({
            'forecast_irradiance_wm2': 'mean',
            'forecast_temperature_c': 'mean',
            'forecast_wind_speed_ms': 'mean'
        })
        weather = np.log1p(weather).bfill().ffill()

        today = datetime.now(KST).replace(hour=0, minute=0, second=0, microsecond=0)
        forecast_date = today + timedelta(days=1)

        model = SARIMAX(log_power, exog=weather.loc[log_power.index],
                        order=(2,1,2), seasonal_order=(1,1,1,7))
        model_fit = model.fit(disp=False)

        if forecast_date not in weather.index:
            exog_next = weather.iloc[[-1]]
        else:
            exog_next = weather.loc[[forecast_date]]

        log_forecast = model_fit.forecast(steps=1, exog=exog_next)
        predicted_mwh = float(np.expm1(log_forecast.iloc[0]))

        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT MAX(cumulative_mwh) - MIN(cumulative_mwh) AS actual
                FROM measurement
                WHERE DATE(measured_at) = :date
            """), {'date': forecast_date.date()})
            row = result.mappings().fetchone()
            actual_mwh = row['actual'] if row and row['actual'] is not None else None

        rmse = mae = mape = None
        if actual_mwh is not None:
            rmse = np.sqrt((predicted_mwh - actual_mwh)**2)
            mae = abs(predicted_mwh - actual_mwh)
            mape = abs((predicted_mwh - actual_mwh) / (actual_mwh + 1e-6)) * 100

        save_forecast_to_db(forecast_date, predicted_mwh, actual_mwh, rmse, mae, mape)
        return forecast_date, predicted_mwh

    except Exception:
        return None, f"❌ 예측 오류:<br><pre>{traceback.format_exc()}</pre>"

# 웹 라우트
@app.route("/", methods=["GET", "POST"])
def index():
    forecast_date = predicted_mwh = message = None
    if request.method == "POST":
        forecast_date, predicted_mwh = run_arima_forecast()
        if forecast_date is None:
            message = predicted_mwh

    html = f"""
        <h2>ARIMA 예측 시스템</h2>
        <form method="post">
            <button type="submit">수동 예측 실행</button>
        </form>
        {f"<p>📅 예측 일자: {forecast_date}</p>" if forecast_date else ""}
        {f"<p>🔮 예측 발전량 (MWh): {float(predicted_mwh):.2f}</p>" if isinstance(predicted_mwh, (int, float)) else ""}
        {f"<p style='color:red'>{message}</p>" if message else ""}
    """
    return render_template_string(html)

# 스케줄러
def start_scheduler():
    scheduler = BackgroundScheduler(timezone=KST)
    scheduler.add_job(run_arima_forecast, 'cron', hour=7, minute=30)
    scheduler.start()

# 실행
if __name__ == "__main__":
    print("✅ 예측 서버 실행 중... (포트 5000)")
    start_scheduler()
    app.run(host="0.0.0.0", port=5000)
