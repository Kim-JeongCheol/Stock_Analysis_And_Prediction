"""
Stock Analysis And Prediction
=============================

Author : Jeong-Cheol Kim  |  Version : 2024.11.08
"""
# 1. 사전 작업

# 패키지 임포트
import numpy as np  
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import tensorflow as tf
from urllib.request import Request, urlopen 
from bs4 import BeautifulSoup 
from sklearn.preprocessing import StandardScaler  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense  
from tensorflow.keras.optimizers import Adam  
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# 한글 폰트 설정 (맑은 고딕 폰트 사용)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

# 이미지 저장 디렉토리 설정 (결과 그래프 저장 경로)
img_dir = r"C:\Users\김정철\Desktop\취업 자료\깃허브\Stock_Analysis_And_Prediction\images"
os.makedirs(img_dir, exist_ok = True)

# 랜덤 시드 고정 (재현성)
seed = 20241108
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)



# 2. 데이터 수집 클래스 정의 (Web Scrapping)
class scraper:
    def __init__(self, code: str, start_page: int = 15, end_page: int = 114):
        # code : 네이버 금융 종목 코드
        # start_page : 시작 페이지 번호
        # end_page : 끝 페이지 번호
        self.code = code
        self.start_page = start_page
        self.end_page = end_page

    def fetch(self) -> pd.DataFrame:
        # 네이버 금융 일별 시세 데이터를 페이지 단위로 수집하여 데이터프레임으로 반환
        dataset = []
        
        # start_page에서 end_page까지 반복
        for page in range(self.start_page, self.end_page + 1):
            url = f'https://finance.naver.com/item/sise_day.naver?code={self.code}&page={page}'
            req = Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0') # 봇 탐지를 피하기 위해 User-Agent 지정
            with urlopen(req) as response:
                soup = BeautifulSoup(response, 'html.parser') # HTML 파싱
            
            # 가격 테이블의 각 행(row)을 선택 (hover 시 스타일 적용 CSS selector)
            rows = soup.find_all('tr', {'onmouseover': 'mouseOver(this)'})
            for row in rows:
                cols = row.find_all('td') # 각 열(td) 요소 추출
                if len(cols) >= 7:
                    # 텍스트만 추출 & 쉼표 제거
                    values = [cols[i].get_text().strip().replace(',', '') for i in range(7)]
                    dataset.append(values)
        
        # 2차원 리스트를 DataFrame으로 변환
        df = pd.DataFrame(dataset, columns=['date', 'close', 'volatility', 'open', 'high', 'low', 'volume'])
        return df



# 3. 데이터 전처리 클래스 정의
class preprocessor:
    def __init__(self, df: pd.DataFrame):
        # df: 수집된 원본 데이터프레임
        self.df = df.copy() 
        self.scaler = StandardScaler()
        self.features = ['volatility', 'close', 'open', 'high', 'low', 'volume']

    def process(self):
        # 등락률(%) 문자열에서 숫자만 추출하여 실수 타입으로 변환
        self.df['volatility'] = self.df['volatility'].str.extract(r"(\d+)").astype(float)
        
        # 나머지 숫자형 컬럼들 실수 타입으로 변환
        for col in ['close', 'open', 'high', 'low', 'volume']:
            self.df[col] = self.df[col].astype(float)
            
        # 날짜 컬럼 파싱 및 오름차순 정렬
        self.df['date'] = pd.to_datetime(self.df['date'], format = '%Y.%m.%d')
        self.df.sort_values('date', inplace = True)
        self.df.reset_index(drop = True, inplace = True)
        
        # 스케일링 전용 행렬 생성
        x = self.df[self.features].values
        x_scaled = self.scaler.fit_transform(x)
        self.df_scaled = x_scaled
        return self.df, x_scaled, self.scaler

    @staticmethod
    def create_sequences(data: np.ndarray, seq_len: int = 5, pred_steps: int = 1):
        # LSTM 입력용 시퀀스(X)와 라벨(y) 생성
        # data : 스케일링된 numpy 배열
        # seq_len : 과거 시퀀스 길이 (5 거래일)
        # param pred_steps : 예측 시점(다음 날)
        x, y = [], []
        for i in range(seq_len, len(data) - pred_steps + 1):
            x.append(data[i - seq_len:i])
            y.append(data[i + pred_steps - 1, 2])
        return np.array(x), np.array(y)



# 4. 데이터 모델링 클래스 정의 (LSTM 모델)
class LSTMModel:
    def __init__(self, seq_len: int, n_features: int, lr: float = 0.01):
        # seq_len : 시퀀스 길이
        # param n_features : 입력 피처 개수
        # param lr : 학습률 
        self.seq_len = seq_len
        self.n_features = n_features
        self.lr = lr
        self._build_model()

    def _build_model(self):
        # 모델 구조 및 컴파일
        self.model = Sequential([
            LSTM(64, input_shape = (self.seq_len, self.n_features), return_sequences = True),
            LSTM(32),
            Dense(1)  
        ])
        self.model.compile(optimizer = Adam(learning_rate = self.lr), loss = 'mse')

    def train(self, trainX: np.ndarray, trainY: np.ndarray, epochs: int = 30, batch_size: int = 32):
        # 모델 학습
        return self.model.fit(
            trainX, trainY,
            epochs = epochs,
            batch_size = batch_size,
            validation_split = 0.1,
            verbose = 1
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 예측
        return self.model.predict(X)
    
    

# 5. 데이터 시각화 클래스 정의
class Visualizer:
    @staticmethod
    def plot_loss(history):
        # 학습 / 검증 손실을 그림으로 출력 및 저장
        plt.figure(figsize = (10, 5))
        plt.plot(history.history['loss'], label = '학습 손실')
        plt.plot(history.history['val_loss'], label = '검증 손실')
        plt.title('Epoch별 손실 추이')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(img_dir, 'loss.png')
        plt.savefig(save_path, dpi = 150)
        plt.show()

    @staticmethod
    def plot_predictions(dates, actual, pred, zoom_len: int = None):
        # 실제 vs 예측 가격 비교 그래프 출력 및 저장
        plt.figure(figsize = (10, 5))
        plt.plot(dates, actual, label = '실제 시가')
        plt.plot(dates, pred, linestyle = '--', label = '예측 시가')
        plt.title('실제 시가 vs 예측 시가')
        plt.xlabel('날짜')
        plt.ylabel('원')
        plt.legend()
        plt.tight_layout()
        full_path = os.path.join(img_dir, 'predictions.png')
        plt.savefig(full_path, dpi = 150)
        plt.show()
        
        # 최근 zoom_len 거래일만 표시
        if zoom_len:
            plt.figure(figsize = (10, 5))
            plt.plot(dates[-zoom_len:], actual[-zoom_len:], marker = 'o', label = '실제 시가')
            plt.plot(dates[-zoom_len:], pred[-zoom_len:], marker = 'x', linestyle = '--', label = '예측 시가')
            plt.title(f'최근 {zoom_len}거래일')
            plt.xlabel('날짜')
            plt.ylabel('원')
            plt.legend()
            plt.tight_layout()
            zoom_path = os.path.join(img_dir, 'predictions_zoom.png')
            plt.savefig(zoom_path, dpi = 150)
            plt.show() 



# 6. 실행
if __name__ == '__main__':
    # 데이터 수집
    scrap = scraper(code = '005930')
    raw_df = scrap.fetch()

    # 데이터 전처리
    preprocess = preprocessor(raw_df)
    df, scaled, scaler = preprocess.process()
    
    # 학습 / 테스트 데이터 분할 (70% 학습용, 30% 테스트용)
    n_train = int(0.7 * len(scaled))  
    train_sc, test_sc = scaled[:n_train], scaled[n_train:]
    
    # 시퀀스 생성
    dates = df['date']
    trainX, trainY = preprocess.create_sequences(train_sc)
    testX, testY = preprocess.create_sequences(test_sc)

    # 모델 학습
    model = LSTMModel(seq_len = 5, n_features=scaled.shape[1])
    history = model.train(trainX, trainY)

    # 손실 시각화 및 저장
    Visualizer.plot_loss(history)

    # 테스트 예측 및 결과 복원 (스케일링 역변환)
    y_pred_sc = model.predict(testX)
    mean_open = scaler.mean_[2]
    std_open = np.sqrt(scaler.var_[2])
    testY_orig = testY * std_open + mean_open
    y_pred_orig = y_pred_sc.flatten() * std_open + mean_open
    
    # 예측 결과 시각화
    Visualizer.plot_predictions(dates[n_train + 7:], testY_orig, y_pred_orig, zoom_len = 10)

    # 성능 평가 지표 출력
    mse = mean_squared_error(testY_orig, y_pred_orig)
    r2 = r2_score(testY_orig, y_pred_orig)
    mape = mean_absolute_percentage_error(testY_orig, y_pred_orig)
    print(f"[Result] MSE: {mse:.2f}, R2: {r2:.2f}, MAPE: {mape:.2%}")

    # 다음 날 시가 예측
    last_seq = testX[-1].reshape(1,7,scaled.shape[1])
    next_sc = model.predict(last_seq)
    next_price = next_sc.flatten()[0] * std_open + mean_open
    print(f"다음 날 예측 시가: {next_price:.2f}원")