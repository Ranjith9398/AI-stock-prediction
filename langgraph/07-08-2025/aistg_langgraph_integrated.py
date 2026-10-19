"""
AIstg LangGraph Integrated Version
This integrates the existing AIstg_v1.1.py functionality with LangGraph's multi-agent architecture.
"""

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
import random

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

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
        RotatingFileHandler('stock_analysis.log', maxBytes=10485760, backupCount=5),
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
    FINE_TUNE_EPOCHS = 50
    MA_SHORT = 50
    MA_LONG = 200
    TRAIN_EPOCHS = 100
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    MIN_LR = 1e-6
    
    # Ensure directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.error(f"Error initializing constants: {e}")
    raise

# LangGraph State Definition
class AIstgAnalysisState(TypedDict):
    ticker: str
    company_name: str
    stock_data: pd.DataFrame
    technical_indicators: Dict[str, Any]
    sentiment_score: float
    news_sentiment: str
    model: Optional[Sequential]
    scaler: Optional[MinMaxScaler]
    features: List[str]
    model_predictions: List[float]
    predicted_trend: str
    trend_signal: str
    risk_assessment: Dict[str, Any]
    recommendation: str
    confidence_score: float
    market_conditions: str
    execution_path: List[str]
    errors: List[str]
    current_price: float
    predicted_percentage_change: Optional[float]
    backtest_results: Optional[Dict[str, Any]]
    shap_results: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]

# ================================================================================
# AGENT 1: DATA COLLECTION AGENT
# ================================================================================

def safe_yfinance_download(ticker: str, period: str = "5y", max_retries: int = MAX_RETRIES) -> Optional[pd.DataFrame]:
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

def data_collection_agent(state: AIstgAnalysisState) -> AIstgAnalysisState:
    """Agent 1: Collects stock data and company information."""
    try:
        logger.info(f"Data collection agent processing {state['ticker']}")
        
        # Add delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        # Fetch stock data with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(state['ticker'])
                hist = stock.history(period="5y")
                
                if hist.empty:
                    logger.warning(f"No data found for {state['ticker']} on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(random.uniform(2, 5))
                        continue
                    else:
                        raise Exception("No data available")
                
                # Fetch company info
                try:
                    info = stock.info
                    company_name = info.get('longName', state['ticker'].split('.')[0])
                except Exception as e:
                    logger.warning(f"Could not fetch company info: {e}")
                    company_name = state['ticker'].split('.')[0]
                
                # Update state
                state['stock_data'] = hist
                state['company_name'] = company_name
                state['execution_path'].append('data_collection')
                
                logger.info(f"Successfully collected data for {company_name}")
                break
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {state['ticker']}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(2, 5))
                else:
                    raise e
        
    except Exception as e:
        logger.error(f"Data collection failed for {state['ticker']}: {e}")
        state['errors'].append(f"Data collection error: {str(e)}")
        
        # Create dummy data for demonstration
        logger.info("Creating dummy data for demonstration purposes")
        dates = pd.date_range(start='2023-01-01', end='2025-01-07', freq='D')
        dummy_data = pd.DataFrame({
            'Open': np.random.normal(100, 10, len(dates)),
            'High': np.random.normal(105, 10, len(dates)),
            'Low': np.random.normal(95, 10, len(dates)),
            'Close': np.random.normal(100, 10, len(dates)),
            'Volume': np.random.normal(1000000, 200000, len(dates))
        }, index=dates)
        
        # Ensure Close prices are positive and realistic
        dummy_data['Close'] = np.abs(dummy_data['Close'])
        dummy_data['Open'] = dummy_data['Close'] * np.random.uniform(0.95, 1.05, len(dates))
        dummy_data['High'] = np.maximum(dummy_data['Open'], dummy_data['Close']) * np.random.uniform(1.0, 1.1, len(dates))
        dummy_data['Low'] = np.minimum(dummy_data['Open'], dummy_data['Close']) * np.random.uniform(0.9, 1.0, len(dates))
        dummy_data['Volume'] = np.abs(dummy_data['Volume'])
        
        state['stock_data'] = dummy_data
        state['company_name'] = f"{state['ticker']} (Demo Data)"
        state['execution_path'].append('data_collection')
    
    return state

# ================================================================================
# AGENT 2: TECHNICAL ANALYSIS AGENT
# ================================================================================

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

def technical_analysis_agent(state: AIstgAnalysisState) -> AIstgAnalysisState:
    """Agent 2: Performs comprehensive technical analysis."""
    try:
        logger.info(f"Technical analysis agent processing {state['ticker']}")
        
        data = state['stock_data']
        
        if data.empty:
            raise Exception("No stock data available")
        
        # Compute advanced technical indicators
        data_with_indicators = compute_advanced_technical_indicators(data)
        
        # Determine market conditions
        current_price = data_with_indicators['Close'].iloc[-1]
        sma_20 = data_with_indicators['SMA_20'].iloc[-1]
        sma_50 = data_with_indicators['SMA_50'].iloc[-1]
        
        if pd.isna(sma_20) or pd.isna(sma_50):
            market_condition = "neutral"
        elif current_price > sma_20 > sma_50:
            market_condition = "bullish"
        elif current_price < sma_20 < sma_50:
            market_condition = "bearish"
        else:
            market_condition = "neutral"
        
        # Store technical indicators
        technical_indicators = {}
        for col in data_with_indicators.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                technical_indicators[col] = data_with_indicators[col]
        
        state['stock_data'] = data_with_indicators
        state['technical_indicators'] = technical_indicators
        state['market_conditions'] = market_condition
        state['execution_path'].append('technical_analysis')
        
        logger.info(f"Technical analysis completed. Market condition: {market_condition}")
        
    except Exception as e:
        logger.error(f"Technical analysis failed for {state['ticker']}: {e}")
        state['errors'].append(f"Technical analysis error: {str(e)}")
        state['technical_indicators'] = {}
        state['market_conditions'] = "neutral"
    
    return state 

# ================================================================================
# AGENT 3: SENTIMENT ANALYSIS AGENT
# ================================================================================

def get_news_sentiment(company_name: str) -> str:
    """Fetch news sentiment using NewsAPI with fallback and caching."""
    try:
        # If API key is invalid, just return Neutral without making the call
        if NEWS_API_KEY == '85b0041ba7ce443ba79f8b29989a534e':  # This is the known invalid key
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

def sentiment_analysis_agent(state: AIstgAnalysisState) -> AIstgAnalysisState:
    """Agent 3: Analyzes market sentiment and news."""
    try:
        logger.info(f"Sentiment analysis agent processing {state['ticker']}")
        
        data = state['stock_data']
        
        if data.empty:
            raise Exception("No stock data available")
        
        # Get news sentiment
        try:
            news_sentiment = get_news_sentiment(state['company_name'])
        except Exception as e:
            logger.warning(f"News sentiment analysis failed for {state['ticker']}, continuing with Neutral sentiment: {e}")
            news_sentiment = "Neutral"
        
        # Calculate sentiment based on recent price movement
        recent_returns = data['Close'].pct_change().tail(10)
        
        # Calculate sentiment based on recent price movement
        avg_return = recent_returns.mean()
        sentiment_score = np.tanh(avg_return * 100)  # Normalize to [-1, 1]
        
        # Add some randomness for demo purposes
        sentiment_score += np.random.normal(0, 0.1)
        sentiment_score = np.clip(sentiment_score, -1, 1)
        
        state['sentiment_score'] = sentiment_score
        state['news_sentiment'] = news_sentiment
        state['execution_path'].append('sentiment_analysis')
        
        logger.info(f"Sentiment analysis completed. Score: {sentiment_score:.3f}, News: {news_sentiment}")
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {state['ticker']}: {e}")
        state['errors'].append(f"Sentiment analysis error: {str(e)}")
        state['sentiment_score'] = 0.0
        state['news_sentiment'] = "Neutral"
    
    return state

# ================================================================================
# AGENT 4: MODEL TRAINING AGENT
# ================================================================================

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
            model.compile(optimizer='adam', loss='mean_squared_error')
            history = model.fit(
                X_train, y_train,
                epochs=FINE_TUNE_EPOCHS if model else base_epochs,
                batch_size=32,
                callbacks=[checkpoint],
                verbose=1
            )
            
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

def model_training_agent(state: AIstgAnalysisState) -> AIstgAnalysisState:
    """Agent 4: Trains or loads the LSTM model."""
    try:
        logger.info(f"Model training agent processing {state['ticker']}")
        
        data = state['stock_data']
        
        if data.empty:
            raise Exception("No stock data available")
        
        # Train or load model
        try:
            model, scaler, features = train_or_load_model(data, state['ticker'])
            if model is None or scaler is None or features is None:
                raise Exception("Failed to train or load model")
        except Exception as e:
            logger.error(f"Error in model training for {state['ticker']}: {e}")
            raise e
        
        state['model'] = model
        state['scaler'] = scaler
        state['features'] = features
        state['execution_path'].append('model_training')
        
        logger.info(f"Model training completed. Features: {len(features)}")
        
    except Exception as e:
        logger.error(f"Model training failed for {state['ticker']}: {e}")
        state['errors'].append(f"Model training error: {str(e)}")
        state['model'] = None
        state['scaler'] = None
        state['features'] = []
    
    return state

# ================================================================================
# AGENT 5: PREDICTION AGENT
# ================================================================================

def predict_future_prices(model: Optional[Sequential], scaler: Optional[MinMaxScaler], 
                         data: pd.DataFrame, features: List[str], days: int = PREDICTION_DAYS) -> List[float]:
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
        except Exception as e:
            logger.error(f"Error accessing Close or Volatility column: {e}")
            last_price = 0
            hist_vol = 0.2  # Default volatility
        
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
                    predicted_price = new_row_unscaled[0]  # Close price
                except Exception as e:
                    logger.error(f"Error inverse transforming prediction: {e}")
                    return [None] * days
                
                # Apply mean reversion to prevent extreme predictions
                if predicted_price > last_price * 1.15:
                    predicted_price = last_price * 1.1
                elif predicted_price < last_price * 0.85:
                    predicted_price = last_price * 0.9
                
                # Validate prediction
                if np.isnan(predicted_price) or np.isinf(predicted_price) or predicted_price < 0:
                    logger.warning(f"Invalid prediction at day {day + 1}: {predicted_price}")
                    future_prices.append(None)
                else:
                    future_prices.append(float(predicted_price))
                    last_price = predicted_price
                    
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

def prediction_agent(state: AIstgAnalysisState) -> AIstgAnalysisState:
    """Agent 5: Generates price predictions using trained models."""
    try:
        logger.info(f"Prediction agent processing {state['ticker']}")
        
        model = state['model']
        scaler = state['scaler']
        features = state['features']
        data = state['stock_data']
        
        if model is None or scaler is None or not features:
            raise Exception("Model, scaler, or features not available")
        
        # Predict future prices
        try:
            future_prices = predict_future_prices(model, scaler, data, features, PREDICTION_DAYS)
            if not future_prices or all(price is None for price in future_prices):
                raise Exception("Failed to predict prices")
        except Exception as e:
            logger.error(f"Error in price prediction for {state['ticker']}: {e}")
            raise e

        # Get latest price and determine trend
        latest_price = data["Close"].iloc[-1]
        predicted_trend = get_predicted_trend(future_prices)

        # Calculate predicted percentage change
        predicted_percentage_change = None
        if future_prices and any(price is not None for price in future_prices):
            last_valid_price = next((p for p in reversed(future_prices) if p is not None), None)
            if last_valid_price is not None:
                predicted_percentage_change = (last_valid_price - latest_price) / latest_price * 100

        state['model_predictions'] = future_prices
        state['predicted_trend'] = predicted_trend
        state['current_price'] = latest_price
        state['predicted_percentage_change'] = predicted_percentage_change
        state['execution_path'].append('prediction')
        
        logger.info(f"Prediction completed. 30-day forecast: {future_prices[-1] if future_prices else 'None'}")
        
    except Exception as e:
        logger.error(f"Prediction failed for {state['ticker']}: {e}")
        state['errors'].append(f"Prediction error: {str(e)}")
        state['model_predictions'] = []
        state['predicted_trend'] = "Unknown"
        state['current_price'] = 0.0
        state['predicted_percentage_change'] = None
    
    return state 

# ================================================================================
# AGENT 6: RISK ASSESSMENT AGENT
# ================================================================================

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

def risk_assessment_agent(state: AIstgAnalysisState) -> AIstgAnalysisState:
    """Agent 6: Assesses risk and market conditions."""
    try:
        logger.info(f"Risk assessment agent processing {state['ticker']}")
        
        data = state['stock_data']
        
        if data.empty:
            raise Exception("No stock data available")
        
        # Analyze trend
        trend_signal = analyze_stock_trend(data)
        
        # Calculate volatility
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate beta (simplified)
        market_returns = data['Close'].pct_change().dropna()
        if len(market_returns) > 1:
            beta = np.corrcoef(market_returns, returns)[0, 1] if len(returns) > 1 else 1.0
        else:
            beta = 1.0
        
        # Calculate Sharpe ratio (simplified)
        avg_return = returns.mean() * 252  # Annualized return
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        # Determine risk level
        if volatility > 0.4:
            risk_level = "High"
        elif volatility > 0.2:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        risk_assessment = {
            'volatility': volatility,
            'beta': beta,
            'sharpe_ratio': sharpe_ratio,
            'risk_level': risk_level,
            'max_drawdown': max_drawdown,
            'trend_signal': trend_signal,
            'avg_return': avg_return
        }
        
        state['risk_assessment'] = risk_assessment
        state['trend_signal'] = trend_signal
        state['execution_path'].append('risk_assessment')
        
        logger.info(f"Risk assessment completed. Risk level: {risk_level}, Volatility: {volatility:.3f}")
        
    except Exception as e:
        logger.error(f"Risk assessment failed for {state['ticker']}: {e}")
        state['errors'].append(f"Risk assessment error: {str(e)}")
        state['risk_assessment'] = {
            'volatility': 0.2,
            'beta': 1.0,
            'sharpe_ratio': 0.0,
            'risk_level': "Medium",
            'max_drawdown': 0.0,
            'trend_signal': "Hold",
            'avg_return': 0.0
        }
        state['trend_signal'] = "Hold"
    
    return state

# ================================================================================
# AGENT 7: RECOMMENDATION ENGINE AGENT
# ================================================================================

def calculate_confidence_score(state: AIstgAnalysisState) -> float:
    """Calculate confidence score based on multiple factors."""
    try:
        confidence = 0.5  # Base confidence
        
        # Factor 1: Data quality
        if state['stock_data'] is not None and len(state['stock_data']) > 200:
            confidence += 0.1
        
        # Factor 2: Model availability
        if state['model'] is not None:
            confidence += 0.1
        
        # Factor 3: Prediction quality
        if state['model_predictions'] and any(p is not None for p in state['model_predictions']):
            confidence += 0.1
        
        # Factor 4: Technical indicators
        if state['technical_indicators']:
            confidence += 0.1
        
        # Factor 5: Sentiment alignment
        if state['sentiment_score'] > 0 and state['predicted_trend'] == "Up":
            confidence += 0.1
        elif state['sentiment_score'] < 0 and state['predicted_trend'] == "Down":
            confidence += 0.1
        
        # Factor 6: Risk assessment
        if state['risk_assessment'] and state['risk_assessment'].get('risk_level') == "Low":
            confidence += 0.1
        
        # Factor 7: Market conditions
        if state['market_conditions'] == "bullish":
            confidence += 0.05
        
        # Add some randomness for demo
        confidence += np.random.normal(0, 0.05)
        
        return np.clip(confidence, 0.0, 1.0)
        
    except Exception as e:
        logger.error(f"Error calculating confidence score: {e}")
        return 0.5

def recommendation_engine_agent(state: AIstgAnalysisState) -> AIstgAnalysisState:
    """Agent 7: Generates final recommendations with confidence scores."""
    try:
        logger.info(f"Recommendation engine agent processing {state['ticker']}")
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(state)
        
        # Get key signals
        trend_signal = state.get('trend_signal', 'Hold')
        predicted_trend = state.get('predicted_trend', 'Unknown')
        sentiment_score = state.get('sentiment_score', 0.0)
        market_conditions = state.get('market_conditions', 'neutral')
        
        # Determine recommendation based on multiple factors
        recommendation = "⏳ Hold"  # Default
        
        # Strong buy conditions
        if (trend_signal == "Strong Buy Signal" and 
            predicted_trend == "Up" and 
            sentiment_score > 0.2 and
            market_conditions == "bullish"):
            recommendation = "✅ Strong Buy"
        
        # Buy conditions
        elif ((trend_signal == "Buy Signal" and predicted_trend == "Up") or
              (predicted_trend == "Up" and sentiment_score > 0.1) or
              (trend_signal == "Strong Buy Signal" and predicted_trend == "Up")):
            recommendation = "✅ Buy"
        
        # Sell conditions
        elif ((trend_signal == "Sell Signal" and predicted_trend == "Down") or
              (predicted_trend == "Down" and sentiment_score < -0.1) or
              (market_conditions == "bearish" and predicted_trend == "Down")):
            recommendation = "⚠️ Sell"
        
        # Hold conditions (default)
        else:
            recommendation = "⏳ Hold"
        
        state['recommendation'] = recommendation
        state['confidence_score'] = confidence_score
        state['execution_path'].append('recommendation_engine')
        
        logger.info(f"Recommendation: {recommendation} (Confidence: {confidence_score:.3f})")
        
    except Exception as e:
        logger.error(f"Recommendation engine failed for {state['ticker']}: {e}")
        state['errors'].append(f"Recommendation engine error: {str(e)}")
        state['recommendation'] = "⏳ Hold"
        state['confidence_score'] = 0.5
    
    return state

# ================================================================================
# LANGGRAPH WORKFLOW DEFINITION
# ================================================================================

def create_aistg_workflow() -> StateGraph:
    """Create the LangGraph workflow for AIstg analysis."""
    
    # Create the workflow
    workflow = StateGraph(AIstgAnalysisState)
    
    # Add nodes (agents)
    workflow.add_node("data_collection", data_collection_agent)
    workflow.add_node("technical_analysis", technical_analysis_agent)
    workflow.add_node("sentiment_analysis", sentiment_analysis_agent)
    workflow.add_node("model_training", model_training_agent)
    workflow.add_node("prediction", prediction_agent)
    workflow.add_node("risk_assessment", risk_assessment_agent)
    workflow.add_node("recommendation_engine", recommendation_engine_agent)
    
    # Define the workflow edges
    workflow.set_entry_point("data_collection")
    
    # Sequential flow
    workflow.add_edge("data_collection", "technical_analysis")
    workflow.add_edge("technical_analysis", "sentiment_analysis")
    workflow.add_edge("sentiment_analysis", "model_training")
    workflow.add_edge("model_training", "prediction")
    workflow.add_edge("prediction", "risk_assessment")
    workflow.add_edge("risk_assessment", "recommendation_engine")
    workflow.add_edge("recommendation_engine", END)
    
    return workflow.compile()

# ================================================================================
# MAIN EXECUTION FUNCTIONS
# ================================================================================

def analyze_stock_with_langgraph(ticker: str) -> AIstgAnalysisState:
    """Analyze a single stock using the LangGraph workflow."""
    try:
        # Initialize state
        initial_state = AIstgAnalysisState(
            ticker=ticker,
            company_name="",
            stock_data=pd.DataFrame(),
            technical_indicators={},
            sentiment_score=0.0,
            news_sentiment="Neutral",
            model=None,
            scaler=None,
            features=[],
            model_predictions=[],
            predicted_trend="Unknown",
            trend_signal="Hold",
            risk_assessment={},
            recommendation="⏳ Hold",
            confidence_score=0.5,
            market_conditions="neutral",
            execution_path=[],
            errors=[],
            current_price=0.0,
            predicted_percentage_change=None,
            backtest_results=None,
            shap_results=None,
            performance_metrics=None
        )
        
        # Create and run workflow
        workflow = create_aistg_workflow()
        result = workflow.invoke(initial_state)
        
        return result
        
    except Exception as e:
        logger.error(f"LangGraph analysis failed for {ticker}: {e}")
        # Return error state
        return AIstgAnalysisState(
            ticker=ticker,
            company_name=ticker,
            stock_data=pd.DataFrame(),
            technical_indicators={},
            sentiment_score=0.0,
            news_sentiment="Neutral",
            model=None,
            scaler=None,
            features=[],
            model_predictions=[],
            predicted_trend="Unknown",
            trend_signal="Hold",
            risk_assessment={},
            recommendation="⏳ Hold",
            confidence_score=0.0,
            market_conditions="neutral",
            execution_path=[],
            errors=[str(e)],
            current_price=0.0,
            predicted_percentage_change=None,
            backtest_results=None,
            shap_results=None,
            performance_metrics=None
        )

def parallel_langgraph_analysis(tickers: List[str], max_workers: int = 2) -> List[AIstgAnalysisState]:
    """Analyze multiple stocks in parallel using LangGraph."""
    results = []
    failed_tickers = []
    total_tickers = len(tickers)
    completed = 0
    
    # Create a lock for thread-safe printing
    print_lock = threading.Lock()
    
    def process_stock(ticker):
        nonlocal completed
        result = analyze_stock_with_langgraph(ticker)
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
                if result and not result['errors']:
                    results.append(result)
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                failed_tickers.append(ticker)
                
    return results, failed_tickers

def print_langgraph_results(result: AIstgAnalysisState):
    """Print formatted results from LangGraph analysis."""
    print("\n" + "="*80)
    print(f"LANGGRAPH ANALYSIS RESULTS FOR {result['ticker']}")
    print("="*80)
    
    print(f"Company: {result['company_name']}")
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Predicted Trend: {result['predicted_trend']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence Score: {result['confidence_score']:.3f}")
    
    if result['predicted_percentage_change'] is not None:
        print(f"Predicted 30-day Change: {result['predicted_percentage_change']:.2f}%")
    
    print(f"Market Conditions: {result['market_conditions']}")
    print(f"News Sentiment: {result['news_sentiment']}")
    print(f"Sentiment Score: {result['sentiment_score']:.3f}")
    
    if result['risk_assessment']:
        risk = result['risk_assessment']
        print(f"Risk Level: {risk.get('risk_level', 'Unknown')}")
        print(f"Volatility: {risk.get('volatility', 0):.3f}")
        print(f"Sharpe Ratio: {risk.get('sharpe_ratio', 0):.3f}")
    
    print(f"Execution Path: {' -> '.join(result['execution_path'])}")
    
    if result['errors']:
        print(f"Errors: {', '.join(result['errors'])}")
    
    print("="*80)

def main():
    """Main function for LangGraph integrated AIstg."""
    try:
        # Configure additional debug logging to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        # Print header
        print("\n" + "="*80)
        print(f"  AISTG LANGGRAPH INTEGRATION STARTED AT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        # Ensure output directory exists
        output_dir = Path("OUTPUT")
        output_dir.mkdir(exist_ok=True)

        # Get user input for analysis mode
        while True:
            print("\nSelect analysis mode:")
            print("1. Single Ticker Analysis (LangGraph)")
            print("2. Multiple Tickers Analysis (LangGraph)")
            print("3. Demo Mode (Single Stock)")
            choice = input("Enter your choice (1, 2, or 3): ").strip()
            
            if choice in ['1', '2', '3']:
                break
            print("Invalid choice. Please enter 1, 2, or 3.")

        # List of stocks to analyze
        nifty50_tickers = [
            "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX", "ADBE", "CRM",
            "PYPL", "INTC", "AMD", "ORCL", "IBM", "CSCO", "QCOM", "TXN", "AVGO", "MU",
            "ADI", "KLAC", "LRCX", "AMAT", "ASML", "TSM", "NVDA", "AMD", "INTC", "QCOM"
        ]

        if choice == '1':
            # Single ticker analysis
            while True:
                ticker = input("\nEnter the ticker symbol (e.g., AAPL): ").strip().upper()
                if ticker:
                    break
                print("Please enter a valid ticker symbol.")
            
            print(f"\nAnalyzing {ticker} with LangGraph...")
            result = analyze_stock_with_langgraph(ticker)
            print_langgraph_results(result)
            
        elif choice == '2':
            # Multiple tickers analysis
            print(f"\nAnalyzing {len(nifty50_tickers)} stocks with LangGraph. This may take a while...")
            results, failed_tickers = parallel_langgraph_analysis(nifty50_tickers)
            
            # Print results
            for result in results:
                print_langgraph_results(result)
            
            # Report failed tickers
            if failed_tickers:
                print("\nFailed to analyze the following tickers:")
                for ticker in failed_tickers:
                    print(f"- {ticker}")
                    
        else:  # Demo mode
            print("\nRunning demo with AAPL...")
            result = analyze_stock_with_langgraph("AAPL")
            print_langgraph_results(result)

        print("\nLangGraph analysis completed successfully!")

    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"An unexpected error occurred. Check the logs for details.")

if __name__ == "__main__":
    main() 