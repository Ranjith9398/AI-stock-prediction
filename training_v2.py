import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.optimizers import Adam
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, ChaikinMoneyFlowIndicator
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Optional eager execution (disabled by default for performance)
# Uncomment the following line only for debugging
# tf.config.run_functions_eagerly(True)
logger.info(f"Eager execution enabled: {tf.executing_eagerly()}")

# Check and configure GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info(f"GPU detected: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Avoid pre-allocating all GPU memory
        logger.info("GPU configured with memory growth.")
    except RuntimeError as e:
        logger.error(f"Error configuring GPU: {e}")
else:
    logger.info("No GPU detected. Using CPU.")

# Parse arguments
parser = argparse.ArgumentParser(description="Train or load a stock prediction model.")
parser.add_argument('--retrain', action='store_true', help='Force retraining even if model exists')
args = parser.parse_args()

# Constants
MODEL_FILE = "stock_model.h5"
SCALER_FILE = "scaler.npy"
WINDOW_SIZE = 60
DATA_DIR = "/home/ranjith/AIstg/training/archive/doned"
PERFORMANCE_THRESHOLD = 0.01
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
MIN_LR = 1e-6

def compute_advanced_technical_indicators(df):
    """Compute advanced technical indicators for better pattern recognition."""
    try:
        # Price-based indicators
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['SMA_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()

        # Momentum indicators
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['Stoch_K'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
        df['Stoch_D'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch_signal()
        df['Williams_R'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close']).williams_r()
        df['ROC'] = ROCIndicator(close=df['Close'], window=10).roc()

        # Volume indicators
        df['VWAP'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
        df['CMF'] = ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).chaikin_money_flow()

        # Volatility indicators
        bb = BollingerBands(close=df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
        df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()

        # Trend indicators
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        df['ADX'] = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()

        # Ichimoku Cloud
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'])
        df['Ichimoku_A'] = ichimoku.ichimoku_a()
        df['Ichimoku_B'] = ichimoku.ichimoku_b()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()

        # Price patterns
        df['Doji'] = ((df['Close'] - df['Open']).abs() / (df['High'] - df['Low']) < 0.1).astype(int)
        df['Engulfing'] = ((df['Close'] > df['Open']) & 
                          (df['Close'].shift(1) < df['Open'].shift(1)) & 
                          (df['Close'] > df['Open'].shift(1)) & 
                          (df['Open'] < df['Close'].shift(1))).astype(int)
        df['Hammer'] = ((df['High'] - df['Low']) > 3 * (df['Close'] - df['Open'])) & \
                      ((df['Close'] - df['Low']) / (df['High'] - df['Low']) > 0.6) & \
                      ((df['Open'] - df['Low']) / (df['High'] - df['Low']) > 0.6)

        # Returns and volatility
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)

        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df
    except Exception as e:
        logger.error(f"Error computing technical indicators: {e}")
        return df

def create_advanced_model(input_shape):
    """Create an advanced model architecture for better pattern recognition."""
    model = Sequential([
        # First Bidirectional LSTM layer
        Bidirectional(LSTM(256, return_sequences=True, 
                          kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                          recurrent_regularizer=l1_l2(l1=0.01, l2=0.01)),
                     input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        # Second Bidirectional LSTM layer
        Bidirectional(LSTM(128, return_sequences=True,
                          kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                          recurrent_regularizer=l1_l2(l1=0.01, l2=0.01))),
        BatchNormalization(),
        Dropout(0.3),

        # GRU layer for capturing long-term dependencies
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
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.1),

        # Output layer
        Dense(1)
    ])

    # Compile with custom learning rate
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='huber_loss',  # More robust to outliers
        metrics=['mae', 'mse', RootMeanSquaredError()]
    )

    return model

def prepare_data(data_list, scaler=None):
    """Prepare training data with advanced preprocessing."""
    try:
        # Combine all data
        combined_data = np.concatenate([d.values for d in data_list if d is not None], axis=0)
        
        # Use RobustScaler for better handling of outliers
        if scaler is None:
            scaler = RobustScaler()
            data_scaled = scaler.fit_transform(combined_data)
        else:
            data_scaled = scaler.transform(combined_data)

        # Create sequences with overlap
        X, y = [], []
        for i in range(WINDOW_SIZE, len(data_scaled)):
            X.append(data_scaled[i - WINDOW_SIZE:i])
            y.append(data_scaled[i, 0])  # Predicting 'Close' price

        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        
        # Shuffle data while maintaining temporal order
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        return X, y, scaler
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None, None, None

def train_or_load_model():
    """Enhanced training process with advanced features."""
    try:
        # Load and process data
        all_data = []
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.csv'):
                file_path = os.path.join(DATA_DIR, filename)
                df = pd.read_csv(file_path)
                df = compute_advanced_technical_indicators(df)
                if df is not None and len(df) > WINDOW_SIZE:
                    all_data.append(df)

        if not all_data:
            logger.error("No valid data found")
            return None, None, None

        # Prepare data
        X, y, scaler = prepare_data(all_data)
        if X is None or y is None:
            return None, None, None

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, val_idx = list(tscv.split(X))[-1]  # Use last split
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(buffer_size=1000)\
            .batch(BATCH_SIZE)\
            .prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
            .batch(BATCH_SIZE)\
            .prefetch(tf.data.AUTOTUNE)

        # Create or load model
        model = create_advanced_model((WINDOW_SIZE, X.shape[2]))

        # Callbacks
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
                filepath='best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            TensorBoard(
                log_dir=f'./logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1
            )
        ]

        # Train model
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )

        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

        # Evaluate model
        test_loss = model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation Loss: {test_loss[0]:.6f}")
        logger.info(f"Validation MAE: {test_loss[1]:.6f}")
        logger.info(f"Validation RMSE: {test_loss[2]:.6f}")

        # Save model and scaler
        model.save(MODEL_FILE)
        np.save(SCALER_FILE, {'min_': scaler.min_, 'scale_': scaler.scale_})
        logger.info("Model and scaler saved successfully")

        return model, scaler, all_data

    except Exception as e:
        logger.error(f"Error in training process: {e}")
        return None, None, None

def analyze_performance(stock_data):
    """Analyze stock performance based on indicators."""
    performance_reasons = []
    latest = stock_data.iloc[-1]

    if latest['RSI'] > 70:
        performance_reasons.append("RSI > 70: Potentially overbought")
    elif latest['RSI'] < 30:
        performance_reasons.append("RSI < 30: Potentially oversold")

    if latest['MACD'] > 0:
        performance_reasons.append("MACD > 0: Bullish trend")
    else:
        performance_reasons.append("MACD < 0: Bearish trend")

    return performance_reasons

if __name__ == "__main__":
    logger.info("Starting advanced training process...")
    model, scaler, all_data = train_or_load_model()
    if model and scaler and all_data:
        logger.info("Training completed successfully")
        last_data = all_data[-1]
        reasons = analyze_performance(last_data)
        logger.info(f"Performance reasons for last dataset: {reasons}")
    else:
        logger.error("Training failed")