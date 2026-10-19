from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import logging
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State definition for LangGraph
class StockAnalysisState(TypedDict):
    ticker: str
    stock_data: pd.DataFrame
    technical_indicators: Dict[str, Any]
    sentiment_score: float
    model_predictions: List[float]
    risk_assessment: Dict[str, Any]
    recommendation: str
    confidence_score: float
    market_conditions: str
    execution_path: List[str]
    errors: List[str]
    company_name: str

# Agent 1: Data Collection Agent
def data_collection_agent(state: StockAnalysisState) -> StockAnalysisState:
    """Collects stock data and basic market information."""
    try:
        logger.info(f"Data collection agent processing {state['ticker']}")
        
        # Add delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        # Fetch stock data with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(state['ticker'])
                hist = stock.history(period="2y")
                
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
                    company_name = info.get('longName', state['ticker'])
                except Exception as e:
                    logger.warning(f"Could not fetch company info: {e}")
                    company_name = state['ticker']
                
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

# Agent 2: Technical Analysis Agent
def technical_analysis_agent(state: StockAnalysisState) -> StockAnalysisState:
    """Performs comprehensive technical analysis."""
    try:
        logger.info(f"Technical analysis agent processing {state['ticker']}")
        
        data = state['stock_data']
        
        if data.empty:
            raise Exception("No stock data available")
        
        # Calculate technical indicators
        indicators = {}
        
        # Moving averages
        indicators['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
        indicators['SMA_50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()
        indicators['EMA_20'] = EMAIndicator(close=data['Close'], window=20).ema_indicator()
        
        # Momentum indicators
        indicators['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
        
        # Volatility indicators
        bb = BollingerBands(close=data['Close'])
        indicators['BB_Upper'] = bb.bollinger_hband()
        indicators['BB_Lower'] = bb.bollinger_lband()
        indicators['BB_Mid'] = bb.bollinger_mavg()
        
        # Trend indicators
        macd = MACD(close=data['Close'])
        indicators['MACD'] = macd.macd()
        indicators['MACD_Signal'] = macd.macd_signal()
        
        # Determine market conditions
        current_price = data['Close'].iloc[-1]
        sma_20 = indicators['SMA_20'].iloc[-1]
        sma_50 = indicators['SMA_50'].iloc[-1]
        
        if pd.isna(sma_20) or pd.isna(sma_50):
            market_condition = "neutral"
        elif current_price > sma_20 > sma_50:
            market_condition = "bullish"
        elif current_price < sma_20 < sma_50:
            market_condition = "bearish"
        else:
            market_condition = "neutral"
        
        state['technical_indicators'] = indicators
        state['market_conditions'] = market_condition
        state['execution_path'].append('technical_analysis')
        
        logger.info(f"Technical analysis completed. Market condition: {market_condition}")
        
    except Exception as e:
        logger.error(f"Technical analysis failed for {state['ticker']}: {e}")
        state['errors'].append(f"Technical analysis error: {str(e)}")
        state['technical_indicators'] = {}
        state['market_conditions'] = "neutral"
    
    return state

# Agent 3: Sentiment Analysis Agent
def sentiment_analysis_agent(state: StockAnalysisState) -> StockAnalysisState:
    """Analyzes market sentiment and news."""
    try:
        logger.info(f"Sentiment analysis agent processing {state['ticker']}")
        
        data = state['stock_data']
        
        if data.empty:
            raise Exception("No stock data available")
        
        # Calculate sentiment based on recent price movement
        recent_returns = data['Close'].pct_change().tail(10)
        
        # Calculate sentiment based on recent price movement
        avg_return = recent_returns.mean()
        sentiment_score = np.tanh(avg_return * 100)  # Normalize to [-1, 1]
        
        # Add some randomness for demo purposes
        sentiment_score += np.random.normal(0, 0.1)
        sentiment_score = np.clip(sentiment_score, -1, 1)
        
        state['sentiment_score'] = sentiment_score
        state['execution_path'].append('sentiment_analysis')
        
        logger.info(f"Sentiment analysis completed. Score: {sentiment_score:.3f}")
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {state['ticker']}: {e}")
        state['errors'].append(f"Sentiment analysis error: {str(e)}")
        state['sentiment_score'] = 0.0
    
    return state

# Agent 4: Model Prediction Agent
def model_prediction_agent(state: StockAnalysisState) -> StockAnalysisState:
    """Generates price predictions using trained models."""
    try:
        logger.info(f"Model prediction agent processing {state['ticker']}")
        
        data = state['stock_data']
        
        if data.empty:
            raise Exception("No stock data available")
        
        current_price = data['Close'].iloc[-1]
        
        # Get technical indicators
        indicators = state.get('technical_indicators', {})
        sentiment = state.get('sentiment_score', 0.0)
        
        # Generate realistic predictions based on technical indicators
        rsi = indicators.get('RSI', pd.Series([50])).iloc[-1] if 'RSI' in indicators else 50
        macd = indicators.get('MACD', pd.Series([0])).iloc[-1] if 'MACD' in indicators else 0
        
        # Simple prediction logic (replace with actual model)
        base_change = 0.02  # 2% base change
        rsi_factor = (rsi - 50) / 50 * 0.1  # RSI influence
        macd_factor = macd * 0.01  # MACD influence
        sentiment_factor = sentiment * 0.05  # Sentiment influence
        
        total_change = base_change + rsi_factor + macd_factor + sentiment_factor
        
        # Generate 30-day predictions
        predictions = []
        for day in range(1, 31):
            daily_change = total_change / 30 + np.random.normal(0, 0.01)
            predicted_price = current_price * (1 + daily_change * day)
            predictions.append(predicted_price)
        
        state['model_predictions'] = predictions
        state['execution_path'].append('model_prediction')
        
        logger.info(f"Model predictions generated. 30-day forecast: {predictions[-1]:.2f}")
        
    except Exception as e:
        logger.error(f"Model prediction failed for {state['ticker']}: {e}")
        state['errors'].append(f"Model prediction error: {str(e)}")
        state['model_predictions'] = []
    
    return state

# Agent 5: Risk Assessment Agent
def risk_assessment_agent(state: StockAnalysisState) -> StockAnalysisState:
    """Assesses risk based on multiple factors."""
    try:
        logger.info(f"Risk assessment agent processing {state['ticker']}")
        
        data = state['stock_data']
        indicators = state.get('technical_indicators', {})
        
        if data.empty:
            raise Exception("No stock data available")
        
        # Calculate volatility
        returns = data['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate risk metrics
        risk_assessment = {
            'volatility': volatility,
            'max_drawdown': (data['Close'] / data['Close'].cummax() - 1).min(),
            'rsi_risk': 'high' if indicators.get('RSI', pd.Series([50])).iloc[-1] > 70 else 'low' if indicators.get('RSI', pd.Series([50])).iloc[-1] < 30 else 'medium',
            'bb_position': 'upper' if data['Close'].iloc[-1] > indicators.get('BB_Upper', pd.Series([data['Close'].iloc[-1]])).iloc[-1] else 'lower' if data['Close'].iloc[-1] < indicators.get('BB_Lower', pd.Series([data['Close'].iloc[-1]])).iloc[-1] else 'middle',
            'overall_risk': 'high' if volatility > 0.3 else 'medium' if volatility > 0.15 else 'low'
        }
        
        state['risk_assessment'] = risk_assessment
        state['execution_path'].append('risk_assessment')
        
        logger.info(f"Risk assessment completed. Overall risk: {risk_assessment['overall_risk']}")
        
    except Exception as e:
        logger.error(f"Risk assessment failed for {state['ticker']}: {e}")
        state['errors'].append(f"Risk assessment error: {str(e)}")
        state['risk_assessment'] = {'overall_risk': 'unknown'}
    
    return state

# Agent 6: Recommendation Engine Agent
def recommendation_engine_agent(state: StockAnalysisState) -> StockAnalysisState:
    """Generates final recommendation based on all analysis."""
    try:
        logger.info(f"Recommendation engine agent processing {state['ticker']}")
        
        # Get all analysis results
        market_condition = state.get('market_conditions', 'neutral')
        sentiment = state.get('sentiment_score', 0.0)
        risk_level = state.get('risk_assessment', {}).get('overall_risk', 'medium')
        predictions = state.get('model_predictions', [])
        
        if not predictions:
            state['recommendation'] = "Hold"
            state['confidence_score'] = 0.0
            return state
        
        # Calculate predicted return
        current_price = state['stock_data']['Close'].iloc[-1]
        predicted_price = predictions[-1]
        predicted_return = (predicted_price - current_price) / current_price
        
        # Decision logic based on multiple factors
        buy_signals = 0
        sell_signals = 0
        
        # Market condition signals
        if market_condition == "bullish":
            buy_signals += 1
        elif market_condition == "bearish":
            sell_signals += 1
        
        # Sentiment signals
        if sentiment > 0.2:
            buy_signals += 1
        elif sentiment < -0.2:
            sell_signals += 1
        
        # Prediction signals
        if predicted_return > 0.05:  # 5% gain
            buy_signals += 2
        elif predicted_return < -0.05:  # 5% loss
            sell_signals += 2
        
        # Risk adjustment
        if risk_level == "high":
            buy_signals *= 0.5  # Reduce buy signals for high risk
            sell_signals *= 1.2  # Increase sell signals for high risk
        
        # Generate recommendation
        if buy_signals > sell_signals and buy_signals >= 2:
            recommendation = "Strong Buy"
            confidence = min(0.9, 0.5 + (buy_signals - sell_signals) * 0.1)
        elif buy_signals > sell_signals:
            recommendation = "Buy"
            confidence = 0.6 + (buy_signals - sell_signals) * 0.1
        elif sell_signals > buy_signals and sell_signals >= 2:
            recommendation = "Strong Sell"
            confidence = min(0.9, 0.5 + (sell_signals - buy_signals) * 0.1)
        elif sell_signals > buy_signals:
            recommendation = "Sell"
            confidence = 0.6 + (sell_signals - buy_signals) * 0.1
        else:
            recommendation = "Hold"
            confidence = 0.5
        
        state['recommendation'] = recommendation
        state['confidence_score'] = confidence
        state['execution_path'].append('recommendation_engine')
        
        logger.info(f"Recommendation generated: {recommendation} (confidence: {confidence:.2f})")
        
    except Exception as e:
        logger.error(f"Recommendation engine failed for {state['ticker']}: {e}")
        state['errors'].append(f"Recommendation engine error: {str(e)}")
        state['recommendation'] = "Hold"
        state['confidence_score'] = 0.0
    
    return state

# Conditional routing function
def should_continue_analysis(state: StockAnalysisState) -> str:
    """Determines if analysis should continue based on current state."""
    if len(state['errors']) > 2:
        return "end_analysis"
    elif state.get('market_conditions') == "high_volatility":
        return "enhanced_risk_analysis"
    else:
        return "continue_analysis"

# Enhanced risk analysis agent (conditional)
def enhanced_risk_analysis_agent(state: StockAnalysisState) -> StockAnalysisState:
    """Performs enhanced risk analysis for volatile markets."""
    try:
        logger.info(f"Enhanced risk analysis for {state['ticker']}")
        
        # Add additional risk metrics
        data = state['stock_data']
        
        # Calculate additional risk metrics
        var_95 = data['Close'].pct_change().quantile(0.05)
        max_loss = data['Close'].pct_change().min()
        
        state['risk_assessment'].update({
            'var_95': var_95,
            'max_loss': max_loss,
            'enhanced_analysis': True
        })
        
        state['execution_path'].append('enhanced_risk_analysis')
        
    except Exception as e:
        logger.error(f"Enhanced risk analysis failed: {e}")
        state['errors'].append(f"Enhanced risk analysis error: {str(e)}")
    
    return state

# Create the LangGraph workflow
def create_stock_analysis_workflow():
    """Creates the LangGraph workflow for stock analysis."""
    
    # Create the graph
    workflow = StateGraph(StockAnalysisState)
    
    # Add nodes
    workflow.add_node("data_collection", data_collection_agent)
    workflow.add_node("technical_analysis", technical_analysis_agent)
    workflow.add_node("sentiment_analysis", sentiment_analysis_agent)
    workflow.add_node("model_prediction", model_prediction_agent)
    workflow.add_node("risk_assessment", risk_assessment_agent)
    workflow.add_node("enhanced_risk_analysis", enhanced_risk_analysis_agent)
    workflow.add_node("recommendation_engine", recommendation_engine_agent)
    
    # Set entry point
    workflow.set_entry_point("data_collection")
    
    # Add edges
    workflow.add_edge("data_collection", "technical_analysis")
    workflow.add_edge("technical_analysis", "sentiment_analysis")
    workflow.add_edge("sentiment_analysis", "model_prediction")
    workflow.add_edge("model_prediction", "risk_assessment")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "risk_assessment",
        should_continue_analysis,
        {
            "enhanced_risk_analysis": "enhanced_risk_analysis",
            "continue_analysis": "recommendation_engine",
            "end_analysis": END
        }
    )
    
    workflow.add_edge("enhanced_risk_analysis", "recommendation_engine")
    workflow.add_edge("recommendation_engine", END)
    
    return workflow.compile()

# Example usage
def analyze_stock_with_langgraph(ticker: str) -> StockAnalysisState:
    """Analyze a stock using the LangGraph workflow."""
    
    # Initialize state
    initial_state = StockAnalysisState(
        ticker=ticker,
        stock_data=pd.DataFrame(),
        technical_indicators={},
        sentiment_score=0.0,
        model_predictions=[],
        risk_assessment={},
        recommendation="",
        confidence_score=0.0,
        market_conditions="",
        execution_path=[],
        errors=[],
        company_name=""
    )
    
    # Create and run workflow
    workflow = create_stock_analysis_workflow()
    result = workflow.invoke(initial_state)
    
    return result

# Example usage
if __name__ == "__main__":
    # Test the LangGraph workflow
    ticker = "AAPL"
    result = analyze_stock_with_langgraph(ticker)
    
    print(f"\n=== Stock Analysis Results for {ticker} ===")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Market Condition: {result['market_conditions']}")
    print(f"Risk Level: {result['risk_assessment'].get('overall_risk', 'unknown')}")
    print(f"Execution Path: {' -> '.join(result['execution_path'])}")
    
    if result['errors']:
        print(f"Errors: {result['errors']}") 