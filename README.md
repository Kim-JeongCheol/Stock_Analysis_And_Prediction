# Stock_Analysis_And_Prediction
Web Scrapping과 LSTM을 활용한 주가 예측&분석 프로젝트

## 🔍 목차

1. [프로젝트 개요](#-프로젝트-개요)  
2. [기술 스택](#-기술-스택)  
3. [아키텍처](#-아키텍처)  
4. [데이터](#-데이터)  
5. [전처리](#-전처리)  
6. [모델링](#-모델링)  
7. [결과 및 평가](#-결과-및-평가)  
8. [사용 방법](#-사용-방법)  
9. [향후 계획](#-향후-계획)  

---

## 📋 프로젝트 개요

- **목표**: 과거 주가 데이터를 기반으로 다음 날 시가(open price)를 예측  
- **역할**:  
  - 웹 스크래퍼(`scraper`) 개발  
  - 데이터 전처리(`preprocessor`) 파이프라인 설계  
  - LSTM 모델(`LSTMModel`) 구현 및 학습  
  - 결과 시각화(`Visualizer`) 자동화  
- **성과**:  
  - MAPE **3.5%** 달성  
  - 자동 리포팅용 그래프 2종 저장 (전체/최근 확대)  

---

## 🛠 기술 스택

- **Language**: Python 3.8+  
- **Web Scraping**: `urllib`, `BeautifulSoup`  
- **Data Processing**: `pandas`, `numpy`, `scikit-learn`  
- **Modeling**: `tensorflow`/`keras` (LSTM)  
- **Visualization**: `matplotlib`  
- **Version Control**: Git, GitHub  
- **Documentation**: Markdown  

---

## 🏗 아키텍처

![Workflow](docs/images/workflow.png)

1. **Data Collection**  
   - `scraper.fetch(code, start_page, end_page)`  
   - 네이버 금융 HTML 파싱 → 날짜·종가·시가·고가·저가·거래량 추출  

2. **Data Preprocessing**  
   - `preprocessor.process()`  
     - 문자열 정제(쉼표, % 제거) → `float` 변환  
     - `StandardScaler`로 스케일링  
     - 날짜 `datetime` 변환 및 정렬  
   - `preprocessor.create_sequences(data, seq_len, pred_steps)`  
     - LSTM 입력용 슬라이딩 윈도우 시퀀스 생성  

3. **Modeling**  
   - `LSTMModel`  
     - 64-unit LSTM → 32-unit LSTM → Dense(1)  
     - `optimizer=Adam(lr)`, `loss='mse'`  
     - `train()` 메서드로 `validation_split=0.1` 적용  

4. **Visualization**  
   - `Visualizer.plot_loss(history)`  
     - 학습/검증 손실 곡선 저장(`loss.png`)  
   - `Visualizer.plot_predictions(dates, actual, pred, zoom_len)`  
     - 전체 예측(`predictions_full.png`), 최근 N일 확대(`predictions_zoom.png`)  

---

## 💾 데이터

- **소스**: 네이버 금융 일별 시세 (`https://finance.naver.com/item/sise_day.naver`)  
- **포맷**:  
  | date       | close  | volatility | open   | high   | low    | volume    |
  |------------|--------|------------|--------|--------|--------|-----------|
  | 2024-11-01 | 65,500 | 1.23%      | 65,200 | 65,700 | 64,800 | 12,345,678 |
- **페이징**: 기본 1~50페이지 (최대 250일치)까지 수집  
- **저장**: 메모리 내 `DataFrame`, 필요시 `CSV`로 내보내기 가능  

---

## 🔄 전처리

1. **문자열 정제**  
   ```python
   df['volatility'] = df['volatility'].str.extract(r'([\d\.]+)').astype(float)
   df[['close','open','high','low','volume']] = df[['close','open','high','low','volume']].astype(float)
