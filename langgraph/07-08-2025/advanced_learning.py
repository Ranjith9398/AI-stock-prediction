import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import logging
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = "advanced_models"
DATA_DIR = "advanced_data"
WINDOW_SIZE = 60
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
MIN_LR = 1e-6

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

class AdvancedStockLearner:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        nltk.download('vader_lexicon', quiet=True)
        
    def fetch_market_data(self, ticker, period="5y"):
        """Fetch comprehensive market data including historical prices and company info."""
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            # Fetch company info
            info = stock.info
            
            # Fetch institutional holders
            institutional_holders = stock.institutional_holders
            
            # Fetch major holders
            major_holders = stock.major_holders
            
            # Fetch earnings
            earnings = stock.earnings
            
            # Fetch balance sheet
            balance_sheet = stock.balance_sheet
            
            # Fetch cash flow
            cash_flow = stock.cashflow
            
            return {
                'historical_data': hist,
                'company_info': info,
                'institutional_holders': institutional_holders,
                'major_holders': major_holders,
                'earnings': earnings,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")
            return None

    def fetch_market_news(self, ticker):
        """Fetch and analyze market news and sentiment."""
        try:
            # Fetch news from multiple sources
            news_sources = [
                f"https://finviz.com/quote.ashx?t={ticker}",
                f"https://www.marketwatch.com/investing/stock/{ticker}",
                f"https://seekingalpha.com/symbol/{ticker}"
            ]
            
            news_data = []
            for source in news_sources:
                try:
                    response = requests.get(source, headers={'User-Agent': 'Mozilla/5.0'})
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract news headlines and content
                    headlines = soup.find_all(['h1', 'h2', 'h3', 'h4'])
                    for headline in headlines:
                        if headline.text.strip():
                            sentiment = self.sentiment_analyzer.polarity_scores(headline.text)
                            news_data.append({
                                'headline': headline.text.strip(),
                                'sentiment': sentiment['compound'],
                                'source': source
                            })
                except Exception as e:
                    logger.warning(f"Error fetching news from {source}: {e}")
                    continue
            
            return news_data
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []

    def compute_advanced_features(self, data):
        """Compute advanced technical and fundamental features."""
        try:
            df = data.copy()
            
            # Technical Indicators
            # Trend Indicators
            df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
            df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
            df['SMA_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()
            df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
            df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
            
            # Momentum Indicators
            df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
            df['Stoch_K'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
            df['Stoch_D'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch_signal()
            
            # Volatility Indicators
            bb = BollingerBands(close=df['Close'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Mid'] = bb.bollinger_mavg()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
            df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
            
            # Volume Indicators
            df['VWAP'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
            
            # Price Patterns
            df['Doji'] = ((df['Close'] - df['Open']).abs() / (df['High'] - df['Low']) < 0.1).astype(int)
            df['Engulfing'] = ((df['Close'] > df['Open']) & 
                             (df['Close'].shift(1) < df['Open'].shift(1)) & 
                             (df['Close'] > df['Open'].shift(1)) & 
                             (df['Open'] < df['Close'].shift(1))).astype(int)
            
            # Returns and Volatility
            df['Daily_Return'] = df['Close'].pct_change()
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
            
            # Market Regime Features
            df['Trend_Strength'] = abs(df['SMA_20'] - df['SMA_50']) / df['SMA_50']
            df['Volatility_Regime'] = df['Volatility'].rolling(window=20).mean()
            df['Volume_Trend'] = df['Volume'].rolling(window=20).mean() / df['Volume'].rolling(window=50).mean()
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
        except Exception as e:
            logger.error(f"Error computing advanced features: {e}")
            return data

    def create_advanced_model(self, input_shape):
        """Create an advanced model architecture with attention mechanism."""
        model = Sequential([
            # Input LSTM layer with attention
            Bidirectional(LSTM(256, return_sequences=True,
                             kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                             recurrent_regularizer=l1_l2(l1=0.01, l2=0.01)),
                         input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            # Attention layer
            Attention(),
            
            # Second LSTM layer
            Bidirectional(LSTM(128, return_sequences=True,
                             kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                             recurrent_regularizer=l1_l2(l1=0.01, l2=0.01))),
            BatchNormalization(),
            Dropout(0.3),
            
            # GRU layer for long-term dependencies
            GRU(64, return_sequences=True,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                recurrent_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Final LSTM layer
            LSTM(32, kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                 recurrent_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers
            Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.1),
            
            # Output layer
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='huber_loss',
            metrics=['mae', 'mse']
        )
        
        return model

    def prepare_training_data(self, data, news_data=None):
        """Prepare training data with advanced preprocessing."""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = [], []
            for i in range(WINDOW_SIZE, len(scaled_data)):
                X.append(scaled_data[i - WINDOW_SIZE:i])
                y.append(scaled_data[i, 0])  # Predicting 'Close' price
            
            X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
            
            # Add news sentiment if available
            if news_data:
                sentiment_features = np.array([news['sentiment'] for news in news_data[-len(X):]])
                X = np.concatenate([X, sentiment_features.reshape(-1, 1)], axis=1)
            
            return X, y
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None

    def train_model(self, ticker):
        """Train the model with advanced learning capabilities."""
        try:
            # Fetch comprehensive market data
            market_data = self.fetch_market_data(ticker)
            if not market_data:
                return False
            
            # Fetch market news
            news_data = self.fetch_market_news(ticker)
            
            # Compute advanced features
            historical_data = market_data['historical_data']
            processed_data = self.compute_advanced_features(historical_data)
            
            # Prepare training data
            X, y = self.prepare_training_data(processed_data, news_data)
            if X is None or y is None:
                return False
            
            # Create and train model
            self.model = self.create_advanced_model((WINDOW_SIZE, X.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=REDUCE_LR_PATIENCE,
                    min_lr=MIN_LR
                ),
                ModelCheckpoint(
                    filepath=os.path.join(MODEL_DIR, f'{ticker}_model.h5'),
                    monitor='val_loss',
                    save_best_only=True
                ),
                TensorBoard(
                    log_dir=os.path.join(MODEL_DIR, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
                )
            ]
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                callbacks=callbacks,
                verbose=1
            )
            
            # Plot training history
            self.plot_training_history(history, ticker)
            
            # Save scaler
            np.save(os.path.join(MODEL_DIR, f'{ticker}_scaler.npy'),
                   {'min_': self.scaler.min_, 'scale_': self.scaler.scale_})
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {e}")
            return False

    def plot_training_history(self, history, ticker):
        """Plot and save training history."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss - {ticker}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'Model MAE - {ticker}')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, f'{ticker}_training_history.png'))
        plt.close()

def main():
    """Main function to run advanced learning."""
    # List of stocks to analyze
    stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech
        'JPM', 'BAC', 'WFC', 'GS', 'MS',          # Finance
        'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY',       # Healthcare
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',        # Energy
        'PG', 'KO', 'PEP', 'WMT', 'COST'          # Consumer
    ]
    
    learner = AdvancedStockLearner()
    
    for stock in stocks:
        logger.info(f"Starting advanced learning for {stock}")
        success = learner.train_model(stock)
        if success:
            logger.info(f"Successfully trained model for {stock}")
        else:
            logger.error(f"Failed to train model for {stock}")

if __name__ == "__main__":
    main() 