# 📘 ARIMA Backend for Solar Forecast

이 프로젝트는 저장된 ARIMA 모델을 활용하여 무안군 태양광 발전소의 익일 누적 발전량을 예측하고, 예측 결과를 MySQL 데이터베이스에 저장하는 Flask 기반의 백엔드 시스템입니다.

---

## 🚀 주요 기능

* `arima_model.pkl` 로드 및 예측 수행
* MySQL 데이터베이스에서 실측 발전량 데이터 로드
* ARIMA 예측 결과를 `forecast_arima` 테이블에 저장
* `/` 접속 시 예측 결과를 HTML 형식으로 출력

---

## 📦 프로젝트 구조

```plaintext
capstone_arima/
├── app.py                # Flask 서버 및 예측 처리
├── arima_model.pkl       # 사전 학습된 ARIMA 모델
├── requirements.txt      # Python 의존성 목록
```

---

## ⚙️ 환경 설정

### Python 패키지 설치

```bash
pip install -r requirements.txt
```

### MySQL 사용자 및 DB 구성

```sql
CREATE DATABASE IF NOT EXISTS solar_forecast_muan DEFAULT CHARACTER SET utf8mb4;
CREATE USER IF NOT EXISTS 'solar_user'@'localhost' IDENTIFIED BY 'solar_pass_2025';
GRANT ALL PRIVILEGES ON solar_forecast_muan.* TO 'solar_user'@'localhost';
FLUSH PRIVILEGES;
```

### 테이블 구조

```sql
CREATE TABLE measurement (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,                   -- 고유 식별자 (자동 증가)
    measured_at DATETIME NOT NULL,                          -- 예보 기준 시간 (오늘 or 내일 시각)

    power_kw FLOAT,                                         -- 실측 발전량 (kW)
    cumulative_kwh FLOAT,                                   -- 누적 발전량 (kWh)
    irradiance_wm2 FLOAT,                                   -- 실측 일사량 (W/m²)
    temperature_c FLOAT,                                    -- 실측 기온 (℃)
    wind_speed_ms FLOAT,                                    -- 실측 풍속 (m/s)

    forecast_irradiance_wm2 FLOAT,                          -- 예보 일사량 (W/m²)
    forecast_temperature_c FLOAT,                           -- 예보 기온 (℃)
    forecast_wind_speed_ms FLOAT,                           -- 예보 풍속 (m/s)

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP           -- 데이터 삽입 시각 (자동 기록)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ARIMA 예측 결과 테이블 (익일 예측)
CREATE TABLE IF NOT EXISTS forecast_arima (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    forecast_date DATE NOT NULL,
    predicted_kwh FLOAT NOT NULL,
    actual_kwh FLOAT,
    rmse FLOAT,
    mae FLOAT,
    mape FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- SARIMA 예측 결과 테이블 (2~7일 누적 예측)
CREATE TABLE IF NOT EXISTS forecast_sarima (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    forecast_start DATE NOT NULL,
    forecast_end DATE NOT NULL,
    predicted_kwh FLOAT NOT NULL,
    actual_kwh FLOAT,
    rmse FLOAT,
    mae FLOAT,
    mape FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🌐 실행 방법

```bash
python app.py
```

실행 후 브라우저에서 `http://localhost:5000` 으로 접속하면 ARIMA 예측 결과를 확인할 수 있습니다.

---

## 📌 예측 방식

* DB에서 `measurement` 테이블의 누적 발전량 데이터를 시간순으로 불러옵니다.
* `arima_model.pkl`을 로드하여 익일 발전량을 예측합니다.
* 예측 결과를 `forecast_arima` 테이블에 저장하고 웹 페이지로 표시합니다.

---

## 🧪 예시 출력 (HTML)

```
ARIMA 예측 결과
예측 일자: 2024-12-12
예측 발전량 (kWh): 153.47
```

---

## 📬 문의

* 담당자: 국립목포대학교 공과대학 융합소프트웨어학과 스포트라이트팀 팀장
* 피드백 및 기능 추가 요청은 GitHub Issue 또는 메일로 남겨주세요.
