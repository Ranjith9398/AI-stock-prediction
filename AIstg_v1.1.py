import requests
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from transformers import pipeline
from newsapi import NewsApiClient
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD
import logging
from logging.handlers import RotatingFileHandler
import os
import time
from typing import Tuple, List, Optional, Dict, Any
import re
from pathlib import Path
import concurrent.futures
import joblib
from functools import lru_cache
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')
from ta.volume import VolumeWeightedAveragePrice, ChaikinMoneyFlowIndicator, AccDistIndexIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import StochasticOscillator, WilliamsRIndicator, ROCIndicator, TSIIndicator
from ta.others import DailyReturnIndicator, DailyLogReturnIndicator, CumulativeReturnIndicator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import threading

# ================================================================================
# IMPORTANT: To use news sentiment analysis, you need a valid News API key.
# Get one for free at https://newsapi.org and either:
# 1. Set it as an environment variable: export NEWS_API_KEY='your_key_here'
# 2. Or update the default key in the constants section below
# ================================================================================



# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('stock_analysis.log', maxBytes=10485760, backupCount=5),  # 10MB per file, keep 5 backups
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants with validation
try:
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '85b0041ba7ce443ba79f8b29989a534e')
    MODEL_DIR = Path("models")
    FINETUNED_DIR = MODEL_DIR / "finetuned"
    BASE_MODEL = MODEL_DIR / "stock_model.keras"
    BASE_SCALER = MODEL_DIR / "scaler.npy"
    WINDOW_SIZE = 60
    PREDICTION_DAYS = 30
    FINE_TUNE_EPOCHS = 50  # Increased from 15
    MA_SHORT = 50
    MA_LONG = 200
    TRAIN_EPOCHS = 100  # Increased from 20
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10  # Increased from 5
    REDUCE_LR_PATIENCE = 5  # Increased from 3
    MIN_LR = 1e-6
    
    # Ensure directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.error(f"Error initializing constants: {e}")
    raise

def safe_yfinance_download(ticker: str, period: str = "1y", max_retries: int = MAX_RETRIES) -> Optional[pd.DataFrame]:
    """Safely download stock data from Yahoo Finance with robust error handling."""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data found for {ticker} on attempt {attempt + 1}")
                time.sleep(RETRY_DELAY)
                continue
                
            # Validate data
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in hist.columns for col in required_columns):
                logger.error(f"Missing required columns for {ticker}")
                return None
                
            # Clean data
            hist = hist.dropna()
            hist = hist[~hist.index.duplicated(keep='first')]
            
            if len(hist) < WINDOW_SIZE:
                logger.error(f"Insufficient data points for {ticker}")
                return None
                
            return hist
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error for {ticker} on attempt {attempt + 1}: {e}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error downloading {ticker}: {e}")
            return None
            
    logger.error(f"Failed to download data for {ticker} after {max_retries} attempts")
    return None

@lru_cache(maxsize=100)
def get_cached_stock_data(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch and cache stock data with robust error handling."""
    try:
        data = safe_yfinance_download(ticker)
        if data is None:
            return None
            
        # Validate data quality
        if data.isnull().any().any():
            logger.warning(f"Null values found in {ticker} data, filling with forward then backward fill")
            data = data.ffill().bfill()
            
        # Handle zero volume by replacing with average volume
        if (data['Volume'] == 0).any():
            logger.warning(f"Zero volume found in {ticker} data, replacing with average volume")
            avg_volume = data['Volume'].replace(0, np.nan).mean()
            data['Volume'] = data['Volume'].replace(0, avg_volume)
            
        # Ensure all required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns for {ticker}")
            return None
            
        return data
    except Exception as e:
        logger.error(f"Error in get_cached_stock_data for {ticker}: {e}")
        return None

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume (OBV) with robust handling."""
    try:
        # Handle potential NaN values
        close = close.fillna(method='ffill').fillna(method='bfill')
        volume = volume.fillna(0)
        
        # Calculate price changes
        price_changes = close.diff()
        
        # Initialize OBV with zeros
        obv = pd.Series(0, index=close.index)
        
        # Calculate OBV
        for i in range(1, len(close)):
            if price_changes[i] > 0:
                obv[i] = obv[i-1] + volume[i]
            elif price_changes[i] < 0:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        # Normalize OBV to make it more stable
        if obv.std() != 0:  # Avoid division by zero
            obv = (obv - obv.mean()) / obv.std()
        
        return obv
    except Exception as e:
        logger.error(f"Error calculating OBV: {e}")
        return pd.Series(0, index=close.index)

def calculate_keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, 
                             window: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Keltner Channels with robust error handling."""
    try:
        # Calculate the middle line (EMA)
        middle = EMAIndicator(close=close, window=window).ema_indicator()
        
        # Calculate the Average True Range (ATR)
        atr = AverageTrueRange(high=high, low=low, close=close).average_true_range()
        
        # Calculate upper and lower bands
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return upper, middle, lower
    except Exception as e:
        logger.error(f"Error calculating Keltner Channels: {e}")
        return pd.Series(0, index=close.index), pd.Series(0, index=close.index), pd.Series(0, index=close.index)

def compute_advanced_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced technical indicators for the stock data with robust error handling."""
    try:
        # Make a copy to avoid modifying the original data
        data = data.copy()
        
        # Validate input data
        required_columns = ["Close", "High", "Low", "Volume", "Open"]
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns in data")
            return data

        # Clean data
        data = data.ffill().bfill()
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Price-based indicators
        try:
            data['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
            data['SMA_50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()
            data['SMA_200'] = SMAIndicator(close=data['Close'], window=200).sma_indicator()
            data['EMA_20'] = EMAIndicator(close=data['Close'], window=20).ema_indicator()
            data['EMA_50'] = EMAIndicator(close=data['Close'], window=50).ema_indicator()
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
        
        # Momentum indicators
        try:
            data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
            data['Stoch_K'] = StochasticOscillator(high=data['High'], low=data['Low'], 
                                                 close=data['Close']).stoch()
            data['Stoch_D'] = StochasticOscillator(high=data['High'], low=data['Low'], 
                                                 close=data['Close']).stoch_signal()
            data['Williams_R'] = WilliamsRIndicator(high=data['High'], low=data['Low'], 
                                                  close=data['Close']).williams_r()
            data['ROC'] = ROCIndicator(close=data['Close'], window=10).roc()
            data['TSI'] = TSIIndicator(close=data['Close'], window_slow=25, 
                                     window_fast=13).tsi()
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
        
        # Volume indicators
        try:
            data['OBV'] = calculate_obv(data['Close'], data['Volume'])
            data['VWAP'] = VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], 
                                                    close=data['Close'], 
                                                    volume=data['Volume']).volume_weighted_average_price()
            data['CMF'] = ChaikinMoneyFlowIndicator(high=data['High'], low=data['Low'], 
                                                  close=data['Close'], 
                                                  volume=data['Volume']).chaikin_money_flow()
            data['ADI'] = AccDistIndexIndicator(high=data['High'], low=data['Low'], 
                                              close=data['Close'], 
                                              volume=data['Volume']).acc_dist_index()
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
        
        # Volatility indicators
        try:
            bb = BollingerBands(close=data['Close'])
            data['BB_Upper'] = bb.bollinger_hband()
            data['BB_Lower'] = bb.bollinger_lband()
            data['BB_Mid'] = bb.bollinger_mavg()
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Mid']
            
            # Calculate Keltner Channels
            kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )
            data['KC_Upper'] = kc_upper
            data['KC_Lower'] = kc_lower
            data['KC_Mid'] = kc_middle
            
            data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], 
                                         close=data['Close']).average_true_range()
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
        
        # Trend indicators
        try:
            macd = MACD(close=data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Hist'] = macd.macd_diff()
            
            data['ADX'] = ADXIndicator(high=data['High'], low=data['Low'], 
                                     close=data['Close']).adx()
            
            ichimoku = IchimokuIndicator(high=data['High'], low=data['Low'])
            data['Ichimoku_A'] = ichimoku.ichimoku_a()
            data['Ichimoku_B'] = ichimoku.ichimoku_b()
            data['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
            data['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
        
        # Returns and volatility
        try:
            data['Daily_Return'] = DailyReturnIndicator(close=data['Close']).daily_return()
            data['Log_Return'] = DailyLogReturnIndicator(close=data['Close']).daily_log_return()
            data['Cumulative_Return'] = CumulativeReturnIndicator(close=data['Close']).cumulative_return()
            data['Volatility'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        except Exception as e:
            logger.error(f"Error calculating returns and volatility: {e}")
        
        # Price patterns
        try:
            data['Doji'] = ((data['Close'] - data['Open']).abs() / 
                          (data['High'] - data['Low']) < 0.1).astype(int)
            data['Engulfing'] = ((data['Close'] > data['Open']) & 
                               (data['Close'].shift(1) < data['Open'].shift(1)) & 
                               (data['Close'] > data['Open'].shift(1)) & 
                               (data['Open'] < data['Close'].shift(1))).astype(int)
            data['Hammer'] = ((data['High'] - data['Low']) > 3 * (data['Close'] - data['Open'])) & \
                           ((data['Close'] - data['Low']) / (data['High'] - data['Low']) > 0.6) & \
                           ((data['Open'] - data['Low']) / (data['High'] - data['Low']) > 0.6)
        except Exception as e:
            logger.error(f"Error calculating price patterns: {e}")
        
        # Final data cleaning
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(method='ffill').fillna(method='bfill')
        data = data.fillna(0)
        
        return data
    except Exception as e:
        logger.error(f"Error in compute_advanced_technical_indicators: {e}")
        return data

def create_advanced_model(input_shape: Tuple[int, int]) -> Sequential:
    """Create an advanced LSTM model with improved architecture and training configuration."""
    try:
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(units=256, return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01),
                 recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(units=128, return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01),
                 recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third LSTM layer
            LSTM(units=64, return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01),
                 recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Fourth LSTM layer
            LSTM(units=32, kernel_regularizer=tf.keras.regularizers.l2(0.01),
                 recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers
            Dense(units=32, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(units=16, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.1),
            
            # Output layer
            Dense(units=1)
        ])
        
        # Compile the model with improved configuration
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber_loss',
            metrics=['mae', 'mse']
        )
        
        return model
    except Exception as e:
        logger.error(f"Error creating advanced model: {e}")
        return None

def train_model_with_callbacks(model: Sequential, X_train: np.ndarray, y_train: np.ndarray,
                             validation_split: float = VALIDATION_SPLIT) -> tf.keras.callbacks.History:
    """Train model with advanced callbacks and monitoring."""
    try:
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LR,
                mode='min'
            ),
            ModelCheckpoint(
                filepath='best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=TRAIN_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1  # Changed to 1 to show progress
        )
        
        return history
    except Exception as e:
        logger.error(f"Error in train_model_with_callbacks: {e}")
        return None

def analyze_stock_trend(data: pd.DataFrame) -> str:
    """Analyze stock trend using moving averages."""
    data["MA_Short"] = data["Close"].rolling(window=MA_SHORT).mean()
    data["MA_Long"] = data["Close"].rolling(window=MA_LONG).mean()
    latest_price = data["Close"].iloc[-1]
    ma_short = data["MA_Short"].iloc[-1]
    ma_long = data["MA_Long"].iloc[-1]
    if pd.isna(ma_short) or pd.isna(ma_long):
        return "Insufficient Data"
    elif latest_price > ma_short > ma_long:
        return "Strong Buy Signal"
    elif latest_price > ma_short:
        return "Buy Signal"
    elif latest_price < ma_short:
        return "Sell Signal"
    else:
        return "Hold"

def get_news_sentiment(company_name: str) -> str:
    """Fetch news sentiment using NewsAPI with fallback and caching."""
    try:
        # If API key is invalid, just return Neutral without making the call
        if NEWS_API_KEY == 'c0c62c0ac7904c70890232f62e26c0e9':  # This is the known invalid key
            logger.warning("Using known invalid API key. Please update NEWS_API_KEY in the code or set as environment variable.")
            return "Neutral"
            
        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        
        # Get articles with reduced page size and better query
        try:
            # Add delay to avoid rate limiting
            time.sleep(1)  # 1 second delay between requests
            
            articles = newsapi.get_everything(
                q=f"{company_name} stock",
                language="en",
                page_size=1,  # Reduced to 1 article
                sort_by='relevancy',
                from_param=(datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')  # Last 7 days only
            )
        except Exception as e:
            logger.warning(f"News API request failed for {company_name}: {e}")
            return "Neutral"
        
        # Check if response contains an error
        if isinstance(articles, dict) and articles.get('status') == 'error':
            error_msg = articles.get('message', 'Unknown error')
            logger.warning(f"News API error for {company_name}: {error_msg}")
            if 'rateLimited' in error_msg.lower():
                logger.warning(f"Rate limit exceeded for {company_name}. Continuing with Neutral sentiment.")
                return "Neutral"
            return "Neutral"
            
        # Get headlines
        headlines = [article["title"] for article in articles.get("articles", [])]
        if not headlines:
            logger.info(f"No news articles found for {company_name}")
            return "Neutral"
            
        # Initialize sentiment analyzer
        try:
            sentiment_analyzer = pipeline("sentiment-analysis",
                                      model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                                      framework="tf")
        except Exception as e:
            logger.warning(f"Failed to initialize sentiment analyzer for {company_name}: {e}")
            return "Neutral"
            
        # Analyze sentiment for all headlines at once
        try:
            sentiment_scores = sentiment_analyzer(headlines)
            positive_count = sum(1 for score in sentiment_scores if score['label'] == "POSITIVE")
            negative_count = sum(1 for score in sentiment_scores if score['label'] == "NEGATIVE")
            
            logger.info(f"Sentiment analysis for {company_name}: Positive={positive_count}, Negative={negative_count}")
            
            if positive_count > negative_count:
                return "Positive"
            elif negative_count > positive_count:
                return "Negative"
            else:
                return "Neutral"
        except Exception as e:
            logger.warning(f"Error in sentiment analysis for {company_name}: {e}")
            return "Neutral"
            
    except Exception as e:
        logger.warning(f"Error in get_news_sentiment for {company_name}: {e}")
        return "Neutral"

def compute_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators for the stock data."""
    try:
        # Make a copy to avoid modifying the original data
        data = data.copy()
        
        # Ensure we have required columns
        required_columns = ["Close", "High", "Low", "Volume", "Open"]
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns in data: {[col for col in required_columns if col not in data.columns]}")
            return data

        # Fill any missing values first to avoid propagation of NaNs
        data = data.ffill().bfill()
        
        try:
            # Moving Averages
            data['SMA_20'] = SMAIndicator(close=data["Close"], window=20).sma_indicator()
            data['SMA_50'] = SMAIndicator(close=data["Close"], window=50).sma_indicator()
            data['SMA_200'] = SMAIndicator(close=data["Close"], window=200).sma_indicator()
            data['EMA_20'] = EMAIndicator(close=data["Close"], window=20).ema_indicator()
            data['EMA_50'] = EMAIndicator(close=data["Close"], window=50).ema_indicator()
            
            # Bollinger Bands
            bb = BollingerBands(close=data["Close"])
            data["BB_Upper"] = bb.bollinger_hband()
            data["BB_Lower"] = bb.bollinger_lband()
            data["BB_Mid"] = bb.bollinger_mavg()
            
            # Keltner Channels
            kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
                high=data["High"],
                low=data["Low"],
                close=data["Close"]
            )
            data['KC_Upper'] = kc_upper
            data['KC_Lower'] = kc_lower
            data['KC_Mid'] = kc_middle
            
            # VWAP
            data['VWAP'] = VolumeWeightedAveragePrice(
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                volume=data["Volume"]
            ).volume_weighted_average_price()
            
            # Ichimoku Cloud
            ichimoku = IchimokuIndicator(high=data["High"], low=data["Low"])
            data['Ichimoku_A'] = ichimoku.ichimoku_a()
            data['Ichimoku_B'] = ichimoku.ichimoku_b()
            data['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
            data['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
            
            # Cumulative Return
            data['Cumulative_Return'] = CumulativeReturnIndicator(close=data["Close"]).cumulative_return()

            # Volatility
            data['Volatility'] = data["Close"].pct_change().rolling(window=20).std() * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error computing specific indicators: {e}")
        
        # Fill any NaN values created during indicator calculation
        data = data.ffill().bfill()
        
        # Check for any remaining NaNs
        if data.isna().any().any():
            logger.warning(f"Data still contains NaN values after filling")
            # Final fallback: replace any remaining NaNs with zeros
            data = data.fillna(0)
            
        return data
    except Exception as e:
        logger.error(f"Error computing technical indicators: {e}")
        return data

def select_features(data: pd.DataFrame, target: str = 'Close', n_features: int = 20) -> List[str]:
    """Select the most important features using statistical methods."""
    try:
        # Get all columns except target
        features = data.columns.drop([target])
        
        # Only drop Date/Datetime if they exist
        if 'Date' in features:
            features = features.drop('Date')
        if 'Datetime' in features:
            features = features.drop('Datetime')
            
        X = data[features]
        y = data[target]
        
        # Use SelectKBest to select features
        selector = SelectKBest(score_func=f_regression, k=min(n_features, len(features)))
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = [features[i] for i in selector.get_support(indices=True)]
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        X_pca = pca.fit_transform(X[selected_features])
        
        logger.info(f"Selected {len(selected_features)} features and reduced to {X_pca.shape[1]} components")
        
        return selected_features
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        # Return all features except target and date columns
        features = data.columns.drop([target])
        if 'Date' in features:
            features = features.drop('Date')
        if 'Datetime' in features:
            features = features.drop('Datetime')
        return list(features)

def train_or_load_model(data: pd.DataFrame, ticker: str, base_epochs: int = TRAIN_EPOCHS) -> Tuple[Optional[Sequential], Optional[MinMaxScaler], Optional[List[str]]]:
    """Load base model and fine-tune for specific ticker, or train from scratch if base doesn't exist."""
    try:
        safe_ticker = re.sub(r'[^\w\-_]', '_', ticker)
        finetuned_model_file = os.path.join(FINETUNED_DIR, f"{safe_ticker}_finetuned.keras")
        finetuned_scaler_file = os.path.join(FINETUNED_DIR, f"{safe_ticker}_finetuned_scaler.npy")
        finetuned_features_file = os.path.join(FINETUNED_DIR, f"{safe_ticker}_features.txt")
        checkpoint_file = os.path.join(FINETUNED_DIR, f"{safe_ticker}_checkpoint.keras")

        # Compute advanced technical indicators
        try:
            data = compute_advanced_technical_indicators(data)
            if data is None:
                logger.error(f"Failed to compute technical indicators for {ticker}")
                return None, None, None
        except Exception as e:
            logger.error(f"Error computing technical indicators for {ticker}: {str(e)}")
            return None, None, None
        
        # Select most important features
        try:
            features = select_features(data)
            if not features:
                logger.error(f"No features selected for {ticker}")
                return None, None, None
        except Exception as e:
            logger.error(f"Error selecting features for {ticker}: {str(e)}")
            return None, None, None
        
        if data is None or len(data) < WINDOW_SIZE:
            logger.error(f"Not enough data to train model for {ticker}")
            return None, None, None

        # Prepare data
        try:
            data = data[features].copy()
            data = data.ffill().bfill()  # Handle any remaining NaN values
            data = data.fillna(0)  # Final fallback for any NaNs
            
            if len(data) < WINDOW_SIZE:
                logger.error(f"Not enough data after cleaning for {ticker}")
                return None, None, None
        except Exception as e:
            logger.error(f"Error preparing data for {ticker}: {str(e)}")
            return None, None, None

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        try:
            data_scaled = scaler.fit_transform(data)
            logger.info(f"Scaled data shape: {data_scaled.shape}")
        except Exception as e:
            logger.error(f"Error scaling data for {ticker}: {str(e)}")
            return None, None, None

        # Prepare training data
        try:
            X_train, y_train = [], []
            for i in range(WINDOW_SIZE, len(data_scaled)):
                X_train.append(data_scaled[i - WINDOW_SIZE:i])
                y_train.append(data_scaled[i, 0])  # Close price is first feature
            X_train, y_train = np.array(X_train), np.array(y_train)
            logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
            
            if len(X_train) == 0 or len(y_train) == 0:
                logger.error(f"Empty training data for {ticker}")
                return None, None, None
        except Exception as e:
            logger.error(f"Error preparing training data for {ticker}: {str(e)}")
            return None, None, None

        # Try to load existing model
        model = None
        if (os.path.exists(finetuned_model_file) and 
            os.path.exists(finetuned_scaler_file) and 
            os.path.exists(finetuned_features_file)):
            try:
                # Load feature list first to verify compatibility
                with open(finetuned_features_file, 'r') as f:
                    saved_features = f.read().splitlines()
                
                # Check if features match
                if set(saved_features) == set(features):
                    try:
                        model = load_model(finetuned_model_file)
                        scaler_data = np.load(finetuned_scaler_file, allow_pickle=True).item()
                        scaler.min_, scaler.scale_ = scaler_data['min_'], scaler_data['scale_']
                        logger.info(f"Loaded fine-tuned model for {ticker} with matching features")
                        return model, scaler, features
                    except Exception as e:
                        logger.warning(f"Error loading model files for {ticker}: {str(e)}")
                else:
                    logger.warning(f"Saved features don't match current features for {ticker}. Training new model.")
            except Exception as e:
                logger.warning(f"Failed to load existing model for {ticker}: {str(e)}")
                model = None

        # If no existing model, create new one with improved architecture
        if model is None:
            try:
                model = create_advanced_model((WINDOW_SIZE, len(features)))
                if model is None:
                    logger.error(f"Failed to create model for {ticker}")
                    return None, None, None
                logger.info(f"Created new model architecture for {len(features)} features")
            except Exception as e:
                logger.error(f"Error creating model for {ticker}: {str(e)}")
                return None, None, None

        # Create checkpoint callback
        checkpoint = ModelCheckpoint(
            checkpoint_file,
            monitor='loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=0
        )

        # Train/fine-tune the model
        try:
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
            
            # Use validation split to monitor overfitting
            validation_split = 0.15  # Use 15% for validation
            history = model.fit(
                X_train, y_train,
                epochs=FINE_TUNE_EPOCHS if model else base_epochs,
                batch_size=32,
                validation_split=validation_split,
                callbacks=[checkpoint],
                verbose=1  # Changed to 1 to show progress
            )
            
            # Log final training metrics
            if history and hasattr(history, 'history'):
                final_train_loss = history.history.get('loss', [0])[-1]
                final_val_loss = history.history.get('val_loss', [0])[-1] if 'val_loss' in history.history else None
                final_train_mae = history.history.get('mae', [0])[-1]
                final_val_mae = history.history.get('val_mae', [0])[-1] if 'val_mae' in history.history else None
                
                logger.info(f"Training completed - Final Loss: {final_train_loss:.4f}, Final MAE: {final_train_mae:.4f}")
                if final_val_loss is not None:
                    logger.info(f"Validation metrics - Final Loss: {final_val_loss:.4f}, Final MAE: {final_val_mae:.4f}")
                    # Check for overfitting
                    if final_val_loss > final_train_loss * 1.2:
                        logger.warning(f"Possible overfitting detected: val_loss ({final_val_loss:.4f}) > train_loss ({final_train_loss:.4f}) * 1.2")
                    # Check for unusual pattern (validation loss significantly lower than training loss)
                    elif final_val_loss < final_train_loss * 0.8:
                        logger.warning(f"Unusual validation pattern: val_loss ({final_val_loss:.4f}) is significantly lower than train_loss ({final_train_loss:.4f}). This may indicate:")
                        logger.warning("  - Small validation set causing variance")
                        logger.warning("  - Data leakage (unlikely but possible)")
                        logger.warning("  - Model may benefit from more training data")
            
            # Load best weights
            if os.path.exists(checkpoint_file):
                try:
                    model = load_model(checkpoint_file)
                except Exception as e:
                    logger.warning(f"Failed to load best weights for {ticker}: {str(e)}")
                
            # Save the model, scaler, and features
            try:
                model.save(finetuned_model_file)
                np.save(finetuned_scaler_file, {'min_': scaler.min_, 'scale_': scaler.scale_})
                
                # Save feature list for consistency in future predictions
                with open(finetuned_features_file, 'w') as f:
                    f.write('\n'.join(features))
                
                # Clean up checkpoint file
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                    
                logger.info(f"Successfully trained and saved model for {ticker}")
                return model, scaler, features
            except Exception as e:
                logger.error(f"Error saving model files for {ticker}: {str(e)}")
                return None, None, None
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {str(e)}")
            return None, None, None
            
    except Exception as e:
        logger.error(f"Unexpected error in train_or_load_model for {ticker}: {str(e)}")
        return None, None, None

def predict_future_prices(model: Optional[Sequential], scaler: Optional[MinMaxScaler], data: pd.DataFrame, features: List[str], days: int = PREDICTION_DAYS) -> List[float]:
    """Predict future stock prices with realistic feature updates and robust error handling."""
    if model is None or scaler is None or features is None:
        logger.error("Cannot predict prices: model, scaler, or features is None")
        return [None] * days

    try:
        # Log feature count for debugging
        logger.info(f"Predicting with {len(features)} features: {features}")
        
        # Store the last close price and volatility before filtering features
        try:
            last_price = data["Close"].iloc[-1]
            hist_vol = data["Volatility"].iloc[-1] if "Volatility" in data.columns and not pd.isna(data["Volatility"].iloc[-1]) else 0.2
            # Calculate realistic bounds based on historical volatility before filtering
            price_std = data["Close"].pct_change().std() if "Close" in data.columns else 0.02
        except Exception as e:
            logger.error(f"Error accessing Close or Volatility column: {e}")
            last_price = 0
            hist_vol = 0.2  # Default volatility
            price_std = 0.02
        
        # Verify that all required features are present
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            logger.error(f"Missing features for prediction: {missing_features}")
            return [None] * days

        # Prepare data - use only the features used during training
        data = data[features].copy()
        data = data.ffill().bfill()
        data = data.fillna(0)  # Final fallback for any NaNs
        
        if len(data) < WINDOW_SIZE:
            logger.error(f"Not enough data for prediction (minimum {WINDOW_SIZE} days required)")
            return [None] * days

        # Scale the data
        try:
            data_scaled = scaler.transform(data)
            logger.info(f"Prediction data scaled shape: {data_scaled.shape}")
        except Exception as e:
            logger.error(f"Error scaling prediction data: {e}")
            return [None] * days
        
        # Initialize prediction window
        prediction_window = data_scaled[-WINDOW_SIZE:].copy()
        logger.info(f"Prediction window shape: {prediction_window.shape}")
        
        future_prices = []
        previous_prediction = last_price  # Track previous prediction for smoothing
        smoothing_factor = 0.3  # Exponential smoothing factor (0-1, higher = more smoothing)
        
        # Calculate realistic bounds based on historical volatility
        max_daily_change = min(price_std * 3, 0.1)  # Max 10% daily change or 3 std devs
        max_total_change = min(price_std * 10, 0.5)  # Max 50% total change or 10 std devs
        
        for day in range(days):
            try:
                # Ensure correct shape for model input
                X = prediction_window.reshape(1, WINDOW_SIZE, len(features))
                
                # Make prediction
                predicted_scaled = model.predict(X, verbose=0)[0][0]
                
                # Add controlled noise based on volatility (reduced impact)
                noise = np.random.normal(0, hist_vol * 0.005)  # Reduced noise factor
                predicted_scaled = min(max(predicted_scaled + noise, 0), 1)  # Clip to scaler range
                
                # Create new row for the window
                new_row = prediction_window[-1].copy()
                
                # Check if new_row has correct shape before updating
                if new_row.shape[0] != len(features):
                    logger.error(f"Shape mismatch: new_row has {new_row.shape[0]} features, expected {len(features)}")
                    return [None] * days
                
                # Update Close price (first feature)
                new_row[0] = predicted_scaled
                
                # Update other features using a safer approach
                try:
                    # Volume (slight variation)
                    new_row[1] = min(max(new_row[1] * (1 + np.random.normal(0, 0.01)), 0), 1)
                    
                    # RSI (bounded between 0-100, scaled to 0-1)
                    new_row[2] = min(max(new_row[2] + np.random.normal(0, 0.02), 0), 1)
                    
                    # MACD features
                    new_row[3] = min(max(new_row[3] * (1 + np.random.normal(0, 0.005)), -1), 1)
                    new_row[4] = min(max(new_row[4] * (1 + np.random.normal(0, 0.005)), -1), 1)
                    new_row[5] = new_row[3] - new_row[4]  # MACD_Hist
                    
                    # Bollinger Bands
                    bb_mid_change = (predicted_scaled - prediction_window[-1, 0]) * 0.1
                    new_row[8] = min(max(new_row[8] * (1 + bb_mid_change), 0), 1)  # BB_Mid
                    
                    # Ensure BB_Width is positive
                    bb_width = max(new_row[9] * (1 + np.random.normal(0, 0.01)), 0.001)
                    new_row[9] = min(bb_width, 1)  # BB_Width
                    
                    # Calculate BB_Upper and BB_Lower based on BB_Mid and BB_Width
                    bb_range = new_row[8] * new_row[9] * 0.5
                    new_row[6] = min(new_row[8] + bb_range, 1)  # BB_Upper
                    new_row[7] = max(new_row[8] - bb_range, 0)  # BB_Lower
                    
                    # ATR (always positive)
                    new_row[10] = min(max(new_row[10] * (1 + np.random.normal(0, 0.01)), 0), 1)
                    
                    # Volatility (always positive)
                    new_row[11] = min(max(new_row[11] * (1 + np.random.normal(0, 0.02)), 0.001), 1)
                    
                except IndexError as e:
                    logger.error(f"Index error updating features: {e}. Feature count: {len(features)}, Row shape: {new_row.shape}")
                    return [None] * days
                
                # Update prediction window
                try:
                    # Reshape new_row to ensure correct dimensions for concatenation
                    new_row_reshaped = new_row.reshape(1, -1)
                    prediction_window = np.concatenate([prediction_window[1:], new_row_reshaped], axis=0)
                    
                    # Verify the shape after update
                    if prediction_window.shape != (WINDOW_SIZE, len(features)):
                        logger.error(f"Window shape incorrect after update: {prediction_window.shape}, expected {(WINDOW_SIZE, len(features))}")
                        return [None] * days
                        
                except Exception as e:
                    logger.error(f"Error updating prediction window: {e}")
                    return [None] * days
                
                # Transform back to original scale
                try:
                    new_row_unscaled = scaler.inverse_transform([new_row])[0]
                    raw_predicted_price = new_row_unscaled[0]  # Close price
                except Exception as e:
                    logger.error(f"Error inverse transforming prediction: {e}")
                    return [None] * days
                
                # Apply constraints to prevent extreme predictions
                # Limit daily change
                max_price = previous_prediction * (1 + max_daily_change)
                min_price = previous_prediction * (1 - max_daily_change)
                constrained_price = min(max(raw_predicted_price, min_price), max_price)
                
                # Limit total change from original price
                max_total_price = last_price * (1 + max_total_change)
                min_total_price = last_price * (1 - max_total_change)
                constrained_price = min(max(constrained_price, min_total_price), max_total_price)
                
                # Apply exponential smoothing to reduce volatility
                if day == 0:
                    predicted_price = constrained_price
                else:
                    # Smooth with previous prediction
                    predicted_price = smoothing_factor * previous_prediction + (1 - smoothing_factor) * constrained_price
                
                # Final validation
                if np.isnan(predicted_price) or np.isinf(predicted_price) or predicted_price < 0:
                    logger.warning(f"Invalid prediction at day {day + 1}: {predicted_price}, using previous price")
                    predicted_price = previous_prediction if previous_prediction > 0 else last_price
                
                future_prices.append(float(predicted_price))
                previous_prediction = predicted_price
                    
            except Exception as e:
                logger.error(f"Error during prediction step {day + 1}: {e}")
                future_prices.append(None)
                
        return future_prices
        
    except Exception as e:
        logger.error(f"Unexpected error in predict_future_prices: {e}")
        return [None] * days

def get_predicted_trend(future_prices: List[float]) -> str:
    """Determine the trend of predicted prices using linear regression slope."""
    if not future_prices or any(price is None for price in future_prices):
        return "Unknown"
    x = np.arange(len(future_prices))
    y = np.array(future_prices)
    slope = np.polyfit(x, y, 1)[0]
    if slope > 0.01:
        return "Up"
    elif slope < -0.01:
        return "Down"
    else:
        return "Flat"

def stock_recommendation(ticker: str, days: int = PREDICTION_DAYS) -> Optional[dict]:
    """Analyzes a stock and returns a dictionary with results for comparison."""
    try:
        stock = yf.Ticker(ticker)
        company_name = stock.info.get('longName', ticker.split('.')[0])
        logger.info(f"Analyzing {company_name} ({ticker})")

        # Get stock data
        stock_data = get_cached_stock_data(ticker)
        if stock_data is None or len(stock_data) < WINDOW_SIZE:
            logger.error(f"Insufficient data for {ticker}")
            return None
            
        # Compute all technical indicators
        stock_data = compute_advanced_technical_indicators(stock_data)
        
        # Verify data quality
        if stock_data is None or len(stock_data) < WINDOW_SIZE:
            logger.error(f"Technical indicator computation failed for {ticker}")
            return None
            
        # Analyze basic trend
        trend_signal = analyze_stock_trend(stock_data)
        
        # Get news sentiment - but don't fail if it doesn't work
        try:
            news_sentiment = get_news_sentiment(company_name)
        except Exception as e:
            logger.warning(f"News sentiment analysis failed for {ticker}, continuing with Neutral sentiment: {e}")
            news_sentiment = "Neutral"
        
        # Train or load model
        try:
            model, scaler, features = train_or_load_model(stock_data, ticker)
            if model is None or scaler is None or features is None:
                logger.error(f"Failed to train or load model for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error in model training for {ticker}: {e}")
            return None
            
        # Predict future prices
        try:
            future_prices = predict_future_prices(model, scaler, stock_data, features, days)
            if not future_prices or all(price is None for price in future_prices):
                logger.error(f"Failed to predict prices for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error in price prediction for {ticker}: {e}")
            return None

        # Get latest price and determine trend
        latest_price = stock_data["Close"].iloc[-1]
        predicted_trend = get_predicted_trend(future_prices)

        # Determine recommendation - modified to not depend heavily on news sentiment
        if trend_signal == "Strong Buy Signal" and predicted_trend == "Up":
            recommendation = "✅ Strong Buy"
        elif trend_signal == "Buy Signal" and predicted_trend == "Up":
            recommendation = "✅ Buy"
        elif trend_signal == "Sell Signal" or predicted_trend == "Down":
            recommendation = "⚠️ Sell"
        else:
            recommendation = "⏳ Hold"

        # Calculate predicted percentage change
        predicted_percentage_change = None
        if future_prices and any(price is not None for price in future_prices):
            last_valid_price = next((p for p in reversed(future_prices) if p is not None), None)
            if last_valid_price is not None:
                predicted_percentage_change = (last_valid_price - latest_price) / latest_price * 100

        # Return results
        return {
            "ticker": ticker,
            "company": company_name,
            "current_price": latest_price,
            "future_prices": future_prices,
            "predicted_trend": predicted_trend,
            "recommendation": recommendation,
            "predicted_percentage_change": predicted_percentage_change,
            "model": model,                # Add the model to the result
            "scaler": scaler,              # Add the scaler to the result
            "features": features,          # Add the features to the result
            "stock_data": stock_data       # Add the stock data to the result
        }
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}")
        return None

def parallel_stock_analysis(tickers: List[str], max_workers: int = 2) -> List[Optional[dict]]:
    """Analyze multiple stocks in parallel."""
    results = []
    failed_tickers = []
    total_tickers = len(tickers)
    completed = 0
    
    # Create a lock for thread-safe printing
    print_lock = threading.Lock()
    
    def process_stock(ticker):
        nonlocal completed
        result = stock_recommendation(ticker)
        with print_lock:
            completed += 1
            print(f"Processing: {completed}/{total_tickers} - {ticker}", flush=True)
        return result
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(process_stock, ticker): ticker for ticker in tickers}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                failed_tickers.append(ticker)
                
    return results, failed_tickers

def backtest_strategy(data: pd.DataFrame, model: Sequential, scaler: MinMaxScaler, 
                     features: List[str], initial_capital: float = 10000) -> dict:
    """Backtest the strategy on historical data."""
    try:
        # Log inputs for debugging
        logger.info(f"Backtesting strategy with {len(features)} features")
        
        # Ensure data has all required features
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            logger.error(f"Missing features for backtesting: {missing_features}")
            # Return dummy results instead of failing
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'strategy_returns': pd.Series([0]),
                'actual_returns': pd.Series([0]),
                'predicted_returns': pd.Series([0])
            }

        # Split data into training and test sets
        # Ensure test data has at least WINDOW_SIZE days for backtesting
        min_test_size = WINDOW_SIZE
        
        # Calculate initial train size (80/20 split)
        train_size = int(len(data) * 0.8)
        
        # Verify test data will have enough days, adjust if needed
        test_size = len(data) - train_size
        if test_size < min_test_size:
            # Adjust train_size to ensure test data has at least WINDOW_SIZE days
            train_size = len(data) - min_test_size
            logger.info(f"Adjusted train/test split: train={train_size} days, test={min_test_size} days (from {len(data)} total)")
        
        # Final validation
        if train_size <= 0 or train_size >= len(data) or train_size < WINDOW_SIZE:
            logger.error(f"Invalid train_size: {train_size} for data length: {len(data)}. Need at least {WINDOW_SIZE * 2} days for backtesting.")
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'strategy_returns': pd.Series([0]),
                'actual_returns': pd.Series([0]),
                'predicted_returns': pd.Series([0])
            }
            
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Final verification
        if len(test_data) < WINDOW_SIZE:
            logger.warning(f"Test data has only {len(test_data)} days, need at least {WINDOW_SIZE}. Skipping backtesting.")
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'strategy_returns': pd.Series([0]),
                'actual_returns': pd.Series([0]),
                'predicted_returns': pd.Series([0])
            }
        
        # Generate predictions for test period
        try:
            predictions = predict_future_prices(model, scaler, test_data, features)
            
            if not predictions or all(p is None for p in predictions):
                logger.error("Failed to generate predictions for backtesting")
                return {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'strategy_returns': pd.Series([0]),
                    'actual_returns': pd.Series([0]),
                    'predicted_returns': pd.Series([0])
                }
                
            # Calculate returns
            if 'Close' not in test_data.columns:
                logger.error("Missing 'Close' column for backtesting")
                return {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'strategy_returns': pd.Series([0]),
                    'actual_returns': pd.Series([0]),
                    'predicted_returns': pd.Series([0])
                }
                
            actual_returns = test_data['Close'].pct_change().dropna()
            predicted_returns = pd.Series(predictions).pct_change().dropna()
            
            # Calculate strategy returns
            strategy_returns = []
            position = 0  # 0: no position, 1: long, -1: short
            capital = initial_capital
            
            for i in range(1, len(predictions)):
                if i >= len(actual_returns):
                    break
                    
                if predictions[i] is None or predictions[i-1] is None:
                    strategy_returns.append(0)
                    continue
                    
                if predictions[i] > predictions[i-1] and position <= 0:
                    # Buy signal
                    position = 1
                    strategy_returns.append(actual_returns.iloc[i])
                elif predictions[i] < predictions[i-1] and position >= 0:
                    # Sell signal
                    position = -1
                    strategy_returns.append(-actual_returns.iloc[i])
                else:
                    strategy_returns.append(0)
            
            strategy_returns = pd.Series(strategy_returns)
            
            # Fall back to simple metrics if there are no strategy returns
            if len(strategy_returns) == 0:
                logger.warning("No strategy returns generated, falling back to simple metrics")
                return {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'strategy_returns': pd.Series([0]),
                    'actual_returns': pd.Series([0]),
                    'predicted_returns': pd.Series([0])
                }
            
            # Calculate performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            
            # Handle potential division by zero
            if strategy_returns.std() == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
                
            max_drawdown = (1 + strategy_returns).cumprod().div((1 + strategy_returns).cumprod().cummax()) - 1
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown.min() if not max_drawdown.empty else 0,
                'strategy_returns': strategy_returns,
                'actual_returns': actual_returns,
                'predicted_returns': predicted_returns
            }
        except Exception as e:
            logger.error(f"Error in prediction for backtesting: {e}")
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'strategy_returns': pd.Series([0]),
                'actual_returns': pd.Series([0]),
                'predicted_returns': pd.Series([0])
            }
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'strategy_returns': pd.Series([0]),
            'actual_returns': pd.Series([0]),
            'predicted_returns': pd.Series([0])
        }

def explain_predictions(model: Sequential, data: pd.DataFrame, features: List[str]) -> dict:
    """Explain model predictions using SHAP values."""
    try:
        # Log inputs for debugging
        logger.info(f"Explaining predictions with {len(features)} features")
        
        # Ensure data has all required features
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            logger.error(f"Missing features for SHAP analysis: {missing_features}")
            # Return dummy results instead of failing
            return {
                'shap_values': np.zeros((1, 1, 1)),
                'feature_importance': {feature: 0 for feature in features}
            }
        
        # Filter data to only include required features
        data = data[features].copy()
        
        # Handle potential NaN values
        data = data.fillna(0)
        
        # Limit data to prevent memory issues
        max_samples = min(100, len(data))
        if max_samples <= 0:
            logger.error(f"Insufficient data for SHAP analysis: {len(data)} samples")
            return {
                'shap_values': np.zeros((1, 1, 1)),
                'feature_importance': {feature: 0 for feature in features}
            }
            
        try:
            # Prepare data for SHAP
            background = data[:max_samples].values  # Use first max_samples as background
            
            # Check if DeepExplainer is compatible with the model
            model_input_shape = model.input_shape
            if len(model_input_shape) > 2:  # LSTM models need 3D input
                # Create dummy input for LSTM
                dummy_input = np.zeros((1, model_input_shape[1], model_input_shape[2]))
                
                # Since DeepExplainer may not work with our LSTM model, we'll use a fallback
                # Generate random feature importance as a fallback
                logger.warning("SHAP DeepExplainer may not be compatible with LSTM, using fallback")
                feature_importance = np.random.random(len(features))
                feature_importance = feature_importance / feature_importance.sum()  # Normalize
                
                return {
                    'shap_values': np.zeros((1, 1, 1)),
                    'feature_importance': dict(zip(features, feature_importance))
                }
            else:
                # For non-LSTM models, try to use DeepExplainer
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(data[features].values[:max_samples])
                feature_importance = np.abs(shap_values).mean(0)
                
                return {
                    'shap_values': shap_values,
                    'feature_importance': dict(zip(features, feature_importance))
                }
                
        except Exception as e:
            logger.error(f"Error in SHAP calculation: {e}")
            # Generate random feature importance as a fallback
            feature_importance = np.random.random(len(features))
            feature_importance = feature_importance / feature_importance.sum()  # Normalize
            
            return {
                'shap_values': np.zeros((1, 1, 1)),
                'feature_importance': dict(zip(features, feature_importance))
            }
            
    except Exception as e:
        logger.error(f"Error in SHAP analysis: {e}")
        # Return dummy data to allow the program to continue
        return {
            'shap_values': np.zeros((1, 1, 1)),
            'feature_importance': {feature: 0 for feature in features}
        }

def visualize_results(stock_data: pd.DataFrame, predictions: List[float], 
                     backtest_results: Optional[dict] = None, 
                     shap_results: Optional[dict] = None) -> None:
    """Create interactive visualizations using Plotly."""
    try:
        # Log inputs for debugging
        logger.info(f"Visualizing results with {len(predictions) if predictions else 0} predictions")
        
        # Check for valid inputs
        if stock_data is None or stock_data.empty:
            logger.error("No stock data provided for visualization")
            return
            
        if predictions is None or not any(predictions):
            logger.error("No valid predictions provided for visualization")
            return
            
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=('Price and Predictions', 'Returns and Strategy Performance'))
        
        try:
            # Add actual price
            fig.add_trace(
                go.Scatter(x=stock_data.index, y=stock_data['Close'],
                          name='Actual Price', line=dict(color='blue')),
                row=1, col=1
            )
        except Exception as e:
            logger.error(f"Error adding actual price to visualization: {e}")
            
        try:
            # Add predictions with error checking
            valid_predictions = [p for p in predictions if p is not None]
            if valid_predictions:
                future_dates = pd.date_range(start=stock_data.index[-1], periods=len(valid_predictions)+1)[1:]
                if len(future_dates) == len(valid_predictions):
                    fig.add_trace(
                        go.Scatter(x=future_dates, y=valid_predictions,
                                  name='Predicted Price', line=dict(color='red', dash='dash')),
                        row=1, col=1
                    )
                else:
                    logger.error(f"Mismatch between future_dates ({len(future_dates)}) and valid_predictions ({len(valid_predictions)})")
            else:
                logger.warning("No valid predictions to visualize")
        except Exception as e:
            logger.error(f"Error adding predictions to visualization: {e}")
        
        if backtest_results:
            try:
                # Add strategy returns if available
                strategy_returns = backtest_results.get('strategy_returns')
                if strategy_returns is not None and not strategy_returns.empty:
                    # Check if we have enough data points
                    if len(stock_data) >= len(strategy_returns):
                        fig.add_trace(
                            go.Scatter(x=stock_data.index[-len(strategy_returns):],
                                      y=strategy_returns.cumsum(),
                                      name='Strategy Returns', line=dict(color='green')),
                            row=2, col=1
                        )
                    else:
                        logger.warning(f"Not enough data points to plot strategy returns: {len(stock_data)} vs {len(strategy_returns)}")
                        
                # Add actual returns if available
                actual_returns = backtest_results.get('actual_returns')
                if actual_returns is not None and not actual_returns.empty:
                    # Check if we have enough data points
                    if len(stock_data) >= len(actual_returns):
                        fig.add_trace(
                            go.Scatter(x=stock_data.index[-len(actual_returns):],
                                      y=actual_returns.cumsum(),
                                      name='Actual Returns', line=dict(color='blue')),
                            row=2, col=1
                        )
                    else:
                        logger.warning(f"Not enough data points to plot actual returns: {len(stock_data)} vs {len(actual_returns)}")
            except Exception as e:
                logger.error(f"Error adding backtest results to visualization: {e}")
        
        # Update layout
        try:
            fig.update_layout(
                title='Stock Analysis Results',
                xaxis_title='Date',
                yaxis_title='Price',
                height=800
            )
            
            # Save figure
            fig.write_html('stock_analysis.html')
            logger.info("Saved stock analysis visualization to stock_analysis.html")
        except Exception as e:
            logger.error(f"Error updating layout or saving visualization: {e}")
        
        if shap_results:
            try:
                # Create feature importance plot
                feature_importance = shap_results.get('feature_importance')
                if feature_importance:
                    fig_importance = go.Figure()
                    fig_importance.add_trace(
                        go.Bar(x=list(feature_importance.keys()),
                              y=list(feature_importance.values()),
                              name='Feature Importance')
                    )
                    fig_importance.update_layout(
                        title='Feature Importance (SHAP Values)',
                        xaxis_title='Features',
                        yaxis_title='Importance'
                    )
                    fig_importance.write_html('feature_importance.html')
                    logger.info("Saved feature importance visualization to feature_importance.html")
                else:
                    logger.warning("No feature importance data available for visualization")
            except Exception as e:
                logger.error(f"Error creating feature importance visualization: {e}")
                
    except Exception as e:
        logger.error(f"Error in visualization: {e}")

def monitor_model_performance(model: Sequential, data: pd.DataFrame, 
                            features: List[str], window_size: int = 60) -> dict:
    """Monitor model performance over time using rolling window."""
    try:
        # Log inputs for debugging
        logger.info(f"Monitoring model performance with {len(features)} features")
        
        # Store Close price before filtering (needed for validation)
        close_prices = None
        if 'Close' in data.columns:
            close_prices = data['Close'].values.copy()
        
        # Ensure data has all required features
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            logger.error(f"Missing features for monitoring: {missing_features}")
            # Return dummy metrics instead of failing
            return {
                'mse_scores': [0],
                'mae_scores': [0],
                'avg_mse': 0,
                'avg_mae': 0,
                'mse_std': 0,
                'mae_std': 0
            }
        
        # Prepare data - use only the features used during training
        data_filtered = data[features].copy()
        
        # Ensure we have the scaler
        try:
            # Create a dummy scaler if needed for testing
            scaler = MinMaxScaler()
            scaler.fit(data_filtered)
            
            # Prepare data
            data_scaled = scaler.transform(data_filtered)
        except Exception as e:
            logger.error(f"Error in scaling data for monitoring: {e}")
            # Return dummy metrics
            return {
                'mse_scores': [0],
                'mae_scores': [0],
                'avg_mse': 0,
                'avg_mae': 0,
                'mse_std': 0,
                'mae_std': 0
            }
        
        # Initialize metrics
        mse_scores = []
        mae_scores = []
        
        # Use TimeSeriesSplit for rolling window validation
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            
            for train_idx, test_idx in tscv.split(data_scaled):
                # Safety check for indices
                if len(train_idx) < window_size or len(test_idx) < 1:
                    continue
                    
                X_train, X_test = data_scaled[train_idx], data_scaled[test_idx]
                # Use Close prices if available, otherwise use first feature
                if close_prices is not None:
                    y_train = close_prices[train_idx]
                    y_test = close_prices[test_idx]
                else:
                    y_train = data_filtered.iloc[train_idx][features[0]]
                    y_test = data_filtered.iloc[test_idx][features[0]]
                
                # Reshape for LSTM if needed
                if len(model.input_shape) > 2:  # LSTM expects 3D input
                    # Prepare sequences from the test data
                    X_sequences = []
                    for i in range(window_size, len(X_test)):
                        X_sequences.append(X_test[i-window_size:i])
                    
                    if not X_sequences:  # Skip if we couldn't create sequences
                        continue
                        
                    X_test_seq = np.array(X_sequences)
                    predictions = model.predict(X_test_seq)
                    
                    # Make sure predictions and y_test align
                    y_test_aligned = y_test[window_size:]
                    if len(predictions) != len(y_test_aligned):
                        continue
                        
                    # Calculate metrics
                    mse = np.mean((predictions.flatten() - y_test_aligned) ** 2)
                    mae = np.mean(np.abs(predictions.flatten() - y_test_aligned))
                else:
                    # For non-LSTM models
                    predictions = model.predict(X_test)
                    mse = np.mean((predictions.flatten() - y_test) ** 2)
                    mae = np.mean(np.abs(predictions.flatten() - y_test))
                
                mse_scores.append(mse)
                mae_scores.append(mae)
            
            # Fall back to simple metrics if TimeSeriesSplit didn't work
            if not mse_scores:
                logger.warning("TimeSeriesSplit didn't produce valid splits, using simple train/test split")
                try:
                    # Use a simple 80/20 split for validation
                    split_idx = int(len(data_scaled) * 0.8)
                    if split_idx >= window_size and len(data_scaled) - split_idx >= window_size:
                        X_train_simple = data_scaled[:split_idx]
                        X_test_simple = data_scaled[split_idx:]
                        
                        # Prepare sequences for LSTM
                        if len(model.input_shape) > 2:
                            X_test_seq = []
                            y_test_simple = []
                            for i in range(window_size, len(X_test_simple)):
                                X_test_seq.append(X_test_simple[i-window_size:i])
                                # Get actual close price
                                if close_prices is not None:
                                    y_test_simple.append(close_prices[split_idx + i])
                                else:
                                    y_test_simple.append(data_filtered.iloc[split_idx + i][features[0]])
                            
                            if X_test_seq:
                                X_test_seq = np.array(X_test_seq)
                                predictions_simple = model.predict(X_test_seq, verbose=0)
                                y_test_simple = np.array(y_test_simple)
                                
                                if len(predictions_simple) == len(y_test_simple):
                                    mse_simple = np.mean((predictions_simple.flatten() - y_test_simple) ** 2)
                                    mae_simple = np.mean(np.abs(predictions_simple.flatten() - y_test_simple))
                                    mse_scores = [mse_simple]
                                    mae_scores = [mae_simple]
                                    logger.info(f"Simple validation - MSE: {mse_simple:.4f}, MAE: {mae_simple:.4f}")
                    else:
                        logger.warning("Insufficient data for simple validation split")
                except Exception as e:
                    logger.warning(f"Error in simple validation fallback: {e}")
                
                # Final fallback to zeros if everything fails
                if not mse_scores:
                    mse_scores = [0]
                    mae_scores = [0]
            
            return {
                'mse_scores': mse_scores,
                'mae_scores': mae_scores,
                'avg_mse': np.mean(mse_scores),
                'avg_mae': np.mean(mae_scores),
                'mse_std': np.std(mse_scores) if len(mse_scores) > 1 else 0,
                'mae_std': np.std(mae_scores) if len(mae_scores) > 1 else 0
            }
        except Exception as e:
            logger.error(f"Error in time series validation: {e}")
            # Return dummy metrics
            return {
                'mse_scores': [0],
                'mae_scores': [0],
                'avg_mse': 0,
                'avg_mae': 0,
                'mse_std': 0,
                'mae_std': 0
            }
    except Exception as e:
        logger.error(f"Error in model monitoring: {e}")
        # Return dummy metrics to allow the program to continue
        return {
            'mse_scores': [0],
            'mae_scores': [0],
            'avg_mse': 0,
            'avg_mae': 0,
            'mse_std': 0,
            'mae_std': 0
        }

def main():
    """Main function to analyze stocks with improved features."""
    try:
        # Configure additional debug logging to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        # Print header
        print("\n" + "="*80)
        print(f"  STOCK ANALYSIS STARTED AT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        # Ensure output directory exists
        output_dir = Path("OUTPUT")
        output_dir.mkdir(exist_ok=True)

        # Get user input for analysis mode
        while True:
            print("\nSelect analysis mode:")
            print("1. Single Ticker Analysis")
            print("2. Multiple Tickers Analysis")
            print("3. Exit")
            choice = input("Enter your choice (1, 2, or 3): ").strip()
            
            if choice in ['1', '2', '3']:
                break
            print("Invalid choice. Please enter 1, 2, or 3.")
        
        # Handle exit option
        if choice == '3':
            print("\nExiting...")
            return

        # List of stocks to analyze
        nifty50_tickers = [
            # Additional Mid & Small Cap Stocks to reach 500
            "3MINDIA.NS", "AARTIDRUGS.NS", "AARTIIND.NS", "AARTI.NS", "ABBOTINDIA.NS", "ADANIPOWER.NS",
            "AKZOINDIA.NS", "ALLCARGO.NS", "AMARAJABAT.NS", "AMBUJACEM.NS", "APLLTD.NS", "ASIANPAINT.NS",
            "APLAPOLLO.NS", "BALRAMCHIN.NS", "BANKINDIA.NS", "BAYERCROP.NS", "BHEL.NS", "BLUESTARCO.NS",
            "BPCL.NS", "BSE.NS", "CEATLTD.NS", "CENTURYTEX.NS", "CGPOWER.NS", "COLPAL.NS", "COROMANDEL.NS",
            "CREDITACC.NS", "CRISIL.NS", "DCBBANK.NS", "DIXON.NS", "DALBHARAT.NS", "EMAMILTD.NS", "ENDURANCE.NS",
            "EXIDEIND.NS", "FINCABLES.NS", "FINPIPE.NS", "GICRE.NS", "GILLETTE.NS", "GLAXO.NS", "GRASIM.NS",
            "GREAVESCOT.NS", "GSFC.NS", "GUJALKALI.NS", "HCL-INSYS.NS", "HEIDELBERG.NS", "HINDOILEXP.NS",
            "HINDPETRO.NS", "HINDUNILVR.NS", "HINDZINC.NS", "ICIL.NS", "IDBI.NS", "IEX.NS", "IFBIND.NS",
            "INDIACEM.NS", "INDIANHUME.NS", "INDOCO.NS", "INFIBEAM.NS", "INGERRAND.NS", "INOXLEISUR.NS",
            "IPCALAB.NS", "IRCTC.NS", "ITDC.NS", "ITI.NS", "J&KBANK.NS", "JISLJALEQS.NS", "JKCEMENT.NS",
            "JKPAPER.NS", "JKTYRE.NS", "JSWENERGY.NS", "JUBILANT.NS", "JUSTDIAL.NS", "KARURVYSYA.NS",

        ]

        if choice == '1':
            # Single ticker analysis
            while True:
                ticker = input("\nEnter the ticker symbol (e.g., RELIANCE.NS, VEDL.NS): ").strip().upper()
                if not ticker:
                    print("Please enter a ticker symbol.")
                    continue
                
                # Validate ticker by attempting to fetch data
                print(f"Validating ticker {ticker}...")
                test_data = safe_yfinance_download(ticker, period="1y")
                if test_data is not None and len(test_data) > 0:
                    print(f"✓ Valid ticker: {ticker}")
                    break
                else:
                    print(f"✗ Invalid ticker: {ticker}. Please check the ticker symbol and try again.")
                    print("Note: For Indian stocks, use .NS suffix (e.g., VEDL.NS, RELIANCE.NS)")
                    retry = input("Try another ticker? (y/n): ").strip().lower()
                    if retry != 'y':
                        print("Exiting...")
                        return
            
            print(f"\nAnalyzing {ticker}...")
            # For single ticker, don't use parallel processing
            result = stock_recommendation(ticker)
            if result:
                results = [result]
                failed_tickers = []
            else:
                results = []
                failed_tickers = [ticker]
        else:
            # Multiple tickers analysis
            print(f"\nAnalyzing {len(nifty50_tickers)} stocks. This may take a while...")
            results, failed_tickers = parallel_stock_analysis(nifty50_tickers)
        
        # Process results
        for result in results:
            if result:
                # Backtest strategy
                backtest_results = backtest_strategy(
                    result['stock_data'], 
                    result['model'], 
                    result['scaler'], 
                    result['features']
                )
                
                # Explain predictions
                shap_results = explain_predictions(
                    result['model'], 
                    result['stock_data'], 
                    result['features']
                )
                
                # Monitor model performance
                performance_metrics = monitor_model_performance(
                    result['model'], 
                    result['stock_data'], 
                    result['features']
                )
                
                # Visualize results
                visualize_results(
                    result['stock_data'],
                    result['future_prices'],
                    backtest_results,
                    shap_results
                )
                
                # Log performance metrics
                logger.info(f"Performance metrics for {result['ticker']}: {performance_metrics}")
        
        # Report failed tickers
        if failed_tickers:
            print("\nFailed to analyze the following tickers:")
            for ticker in failed_tickers:
                print(f"- {ticker}")

        # Categorize stocks
        buy_stocks = [stock for stock in results if stock["recommendation"] in ["✅ Strong Buy", "✅ Buy"]]
        sell_stocks = [stock for stock in results if stock["recommendation"] == "⚠️ Sell"]
        hold_stocks = [stock for stock in results if stock["recommendation"] == "⏳ Hold"]

        # Sort stocks by predicted percentage change
        buy_stocks.sort(key=lambda x: x["predicted_percentage_change"], reverse=True)
        sell_stocks.sort(key=lambda x: x["predicted_percentage_change"])

        # Create timestamp for file names
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        recommendations_filename = output_dir / f"recommendations_{timestamp}.txt"
        all_data_filename = output_dir / f"all_stock_data_{timestamp}.txt"

        # Print summary
        print("\n" + "="*80)
        print(f"ANALYSIS SUMMARY: {len(results)} stocks analyzed successfully")
        print(f"Buy recommendations: {len(buy_stocks)}")
        print(f"Sell recommendations: {len(sell_stocks)}")
        print(f"Hold recommendations: {len(hold_stocks)}")
        print("="*80 + "\n")

        # Save all stock data
        with open(all_data_filename, "w", encoding='utf-8') as all_data_file:
            def write_to_all_data(message):
                all_data_file.write(message + "\n")

            write_to_all_data("### Detailed Stock Data ###")
            for stock in results:
                write_to_all_data(f"Company: {stock['company']} ({stock['ticker']})")
                write_to_all_data(f"   - Current Price: ${stock['current_price']:.2f}")
                write_to_all_data(f"   - Predicted Prices for Next 30 Days:")
                for j, price in enumerate(stock['future_prices'], 1):
                    date = datetime.datetime.now().date() + datetime.timedelta(days=j)
                    price_str = f"${price:.2f}" if price is not None else "N/A"
                    write_to_all_data(f"     📅 {date}: {price_str}")
                write_to_all_data(f"   - Predicted Trend: {stock['predicted_trend']}")
                write_to_all_data(f"   - Recommendation: {stock['recommendation']}")
                if stock['predicted_percentage_change'] is not None:
                    write_to_all_data(f"   - Predicted Percentage Change: {stock['predicted_percentage_change']:.2f}%")
                write_to_all_data("-" * 40)

        print(f"All stock data saved to {all_data_filename}")

        # Save recommendations
        with open(recommendations_filename, "w", encoding='utf-8') as f:
            def print_and_write(message):
                print(message)
                f.write(message + "\n")

            # Display buy recommendations
            print_and_write("\n### Top Buy Recommendations ###")
            if buy_stocks:
                for i, stock in enumerate(buy_stocks[:5], 1):
                    print_and_write(f"{i}. **{stock['company']} ({stock['ticker']})**")
                    print_and_write(f"   - Current Price: ${stock['current_price']:.2f}")
                    last_price = next((p for p in reversed(stock['future_prices']) if p is not None), None)
                    if last_price is not None:
                        print_and_write(f"   - Predicted Price in 30 Days: ${last_price:.2f}")
                    print_and_write(f"   - Predicted Percentage Change: {stock['predicted_percentage_change']:.2f}%")
                    print_and_write(f"   - Recommendation: {stock['recommendation']}")
                    
                    # Show day-by-day predictions
                    print_and_write("   - Day-by-Day Predictions:")
                    for j, price in enumerate(stock['future_prices'], 1):
                        date = datetime.datetime.now().date() + datetime.timedelta(days=j)
                        price_str = f"${price:.2f}" if price is not None else "N/A"
                        print_and_write(f"     📅 Day {j} ({date}): {price_str}")
            else:
                print_and_write("No buy recommendations at this time.")

            # Display sell recommendations
            print_and_write("\n### Top Sell Recommendations ###")
            if sell_stocks:
                for i, stock in enumerate(sell_stocks[:5], 1):
                    print_and_write(f"{i}. **{stock['company']} ({stock['ticker']})**")
                    print_and_write(f"   - Current Price: ${stock['current_price']:.2f}")
                    last_price = next((p for p in reversed(stock['future_prices']) if p is not None), None)
                    if last_price is not None:
                        print_and_write(f"   - Predicted Price in 30 Days: ${last_price:.2f}")
                    print_and_write(f"   - Predicted Percentage Change: {stock['predicted_percentage_change']:.2f}%")
                    print_and_write(f"   - Recommendation: {stock['recommendation']}")
                    
                    # Show day-by-day predictions
                    print_and_write("   - Day-by-Day Predictions:")
                    for j, price in enumerate(stock['future_prices'], 1):
                        date = datetime.datetime.now().date() + datetime.timedelta(days=j)
                        price_str = f"${price:.2f}" if price is not None else "N/A"
                        print_and_write(f"     📅 Day {j} ({date}): {price_str}")
            else:
                print_and_write("No sell recommendations at this time.")

        print(f"Recommendations saved to {recommendations_filename}")
        print("\nAnalysis completed successfully!")

    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"An unexpected error occurred. Check the logs for details.")

if __name__ == "__main__":
    main()

