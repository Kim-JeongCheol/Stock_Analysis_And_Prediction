# Stock Analysis And Prediction
웹 스크래핑과 LSTM을 활용한 주가 예측&분석 프로젝트

---

## 🔍 목차

<details>
<summary>목차 보기 (클릭)</summary>

1. [프로젝트 개요](#-프로젝트-개요)  
2. [기술 스택](#-기술-스택)   
3. [데이터 수집](#-데이터-수집)  
4. [데이터 전처리](#-데이터-전처리)  
5. [모델링](#-모델링)  
6. [결과 및 평가](#-결과-및-평가)  
7. [사용 방법](#-사용-방법)  
8. [향후 계획](#-향후-계획)  

</details>

---

## 📋 프로젝트 개요

**목표)**  
- 과거 주가 데이터를 활용해 다음 날 시가를 예측하고, 이를 통해 의사결정에 도움을 주는 모델 개발

**역할)**  
- 웹 스크래핑: 네이버 금융에서 일별 주가 데이터 자동 수집  
- 데이터 전처리: 결측치 처리와 스케일링, 시퀀스 생성 파이프라인 설계  
- 모델 개발: LSTM 네트워크 구현 및 하이퍼파라미터 튜닝  
- 결과 시각화: 학습·검증 손실과 예측 결과를 자동으로 플롯하고 저장

**성과)**  
- MAPE(평균 절대 백분율 오차) 약 1% 달성  
- 전체 예측 그래프와 최근 N일 확대 그래프 자동 저장 기능 구현  

---

## 🛠 기술 스택

- **언어**: Python 3.8+  
- **웹 스크래핑**: urllib (*Request*, *urlopen*), BeautifulSoup  
- **데이터 전처리**: pandas, numpy, scikit-learn (*StandardScaler*)  
- **모델링**: TensorFlow/Keras (*Sequential*, *LSTM*, *Dense*, *Adam*)  
- **모델 평가**: scikit-learn (*mean_squared_error*, *r2_score*, *mean_absolute_percentage_error*)  
- **시각화**: Matplotlib 

---

## 📊 데이터 수집

- **소스**: 네이버 금융 일별 시세 (`https://finance.naver.com/item/sise_day.naver`)  
- **형태**:  
  | date       | close  | volatility | open   | high   | low    | volume    |
  |------------|--------|------------|--------|--------|--------|-----------|
  | 2024-11-01 | 65,500 | 1.23%      | 65,200 | 65,700 | 64,800 | 12,345,678 |
- **페이징**: 기본 1~50페이지 (최대 250일치)까지 수집  
- **저장**: 메모리 내 `DataFrame`, 필요시 `CSV`로 내보내기 가능  

---

## 🧹 데이터 전처리

1. **문자열 정제**  
   - `volatility`에서 `%` 제거 후 숫자만 추출  
     ```
     df['volatility'] = df['volatility'].str.extract(r'([\d\.]+)').astype(float)
     ```
   - 나머지 피처(`close`, `open`, `high`, `low`, `volume`)를 `float(실수)` 형태로 변환  
     ```
     df[['close','open','high','low','volume']] = df[['close','open','high','low','volume']].astype(float)
     ```

2. **날짜 처리 & 정렬**  
   - `date`를 `datetime(날짜)` 타입으로 변환  
     ```
     df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
     ```
   - 오름차순 정렬 및 인덱스 재설정  
     ```
     df.sort_values('date', inplace=True)
     df.reset_index(drop=True, inplace=True)
     ```

3. **스케일링**  
   - `StandardScaler`로 정규화
     ```
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(df[features].values)
     ```

4. **시퀀스 생성**  
   - **시퀀스 생성** (7 거래일 단위: `seq_len = 7`, 다음 날 예측: `pred_steps = 1`)
     ```
     X, y = preprocessor.create_sequences(X_scaled, seq_len=7, pred_steps=1)
     ```

---

## 🧠 모델링

- **하이퍼파라미터**  
  | 파라미터      | 값    |
  |---------------|------|
  | seq_len       | 7    |
  | batch_size    | 32   |
  | epochs        | 30   |
  | learning_rate | 0.01 |

- **학습 코드**  
  ```python
  model = LSTMModel(seq_len = 7, n_features = X.shape[2], lr = 0.01)
  history = model.train(trainX, trainY, epochs = 30, batch_size = 32)

---

## 📊 결과 및 평가

- **평가지표**  
  | 지표  | 값            |
  |------|--------------|
  | MSE  | 875048.88 |
  | R²   | 0.98       |
  | MAPE | 0.97%        |

- **그래프 예시**
![학습 손실](images/loss.png) 
![전체 예측](images/predictions.png)  
![최근 확대](images/predictions_zoom.png)

- **주요 인사이트**  
  - 평균적으로 실제 시가 대비 **0.97%** 오차 발생 → 매우 높은 정확도  
  - 급등락 구간에서는 예측 오차가 다소 커지는 경향 확인
