# AIstg LangGraph Integration Guide

## 🚀 Overview

This document explains the complete integration of LangGraph with the existing AIstg_v1.1.py model, transforming it from a monolithic architecture to a sophisticated multi-agent system.

## 📊 Architecture Comparison

### **Original AIstg_v1.1.py (Monolithic)**
```
┌─────────────────────────────────────┐
│           stock_recommendation()    │
│  ┌─────────────────────────────────┐ │
│  │ 1. Data Collection             │ │
│  │ 2. Technical Analysis          │ │
│  │ 3. Sentiment Analysis          │ │
│  │ 4. Model Training              │ │
│  │ 5. Prediction                  │ │
│  │ 6. Recommendation              │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### **LangGraph Integrated (Multi-Agent)**
```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   Agent 1   │  │   Agent 2   │  │   Agent 3   │      │
│  │Data Collection│  │Technical   │  │Sentiment    │      │
│  │             │  │Analysis     │  │Analysis     │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│         │               │               │                │
│         └───────────────┼───────────────┘                │
│                         │                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   Agent 4   │  │   Agent 5   │  │   Agent 6   │      │
│  │Model Training│  │Prediction   │  │Risk         │      │
│  │             │  │             │  │Assessment   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│         │               │               │                │
│         └───────────────┼───────────────┘                │
│                         │                                │
│  ┌─────────────┐                                        │
│  │   Agent 7   │                                        │
│  │Recommendation│                                        │
│  │Engine       │                                        │
│  └─────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Key Improvements

### **1. Multi-Agent Architecture**
- **7 Specialized Agents**: Each agent has a specific responsibility
- **Stateful Workflows**: Information flows between agents with state management
- **Parallel Processing**: Agents can run independently where possible
- **Error Isolation**: Failures in one agent don't crash the entire system

### **2. Enhanced Decision Making**
- **Confidence Scores**: Each recommendation includes a confidence level
- **Multi-Factor Analysis**: Combines technical, sentiment, and risk factors
- **Dynamic Workflows**: Can adapt based on market conditions
- **Feedback Loops**: Agents can influence each other's decisions

### **3. Better Error Handling**
- **Graceful Degradation**: System continues even if some agents fail
- **Detailed Error Tracking**: Each agent logs its own errors
- **Fallback Mechanisms**: Dummy data generation for demonstration
- **Retry Logic**: Built-in retry mechanisms for network issues

### **4. Advanced Features**
- **Execution Path Tracking**: See exactly which agents processed the data
- **State Persistence**: Can save and resume analysis states
- **Real-time Monitoring**: Track agent performance and bottlenecks
- **Scalable Architecture**: Easy to add new agents or modify existing ones

## 🛠️ Installation & Setup

### **1. Install Dependencies**
```bash
pip install langgraph langchain yfinance pandas numpy ta tensorflow keras transformers newsapi-python plotly shap scikit-learn
```

### **2. File Structure**
```
07-08-2025/
├── AIstg_v1.1.py                    # Original monolithic version
├── aistg_langgraph_integrated.py    # New LangGraph version
├── langgraph_implementation.py       # Basic LangGraph example
├── LANGGRAPH_INTEGRATION_GUIDE.md   # This guide
└── OUTPUT/                          # Results directory
```

## 🚀 Usage

### **1. Single Stock Analysis**
```bash
python aistg_langgraph_integrated.py
# Choose option 1: Single Ticker Analysis
# Enter ticker: AAPL
```

### **2. Multiple Stocks Analysis**
```bash
python aistg_langgraph_integrated.py
# Choose option 2: Multiple Tickers Analysis
# System will analyze 30 stocks in parallel
```

### **3. Demo Mode**
```bash
python aistg_langgraph_integrated.py
# Choose option 3: Demo Mode
# Runs with AAPL using demo data
```

## 📈 Sample Output

```
================================================================================
LANGGRAPH ANALYSIS RESULTS FOR AAPL
================================================================================
Company: Apple Inc.
Current Price: $185.64
Predicted Trend: Up
Recommendation: ✅ Buy
Confidence Score: 0.823
Predicted 30-day Change: 8.45%
Market Conditions: bullish
News Sentiment: Positive
Sentiment Score: 0.156
Risk Level: Medium
Volatility: 0.234
Sharpe Ratio: 1.245
Execution Path: data_collection -> technical_analysis -> sentiment_analysis -> model_training -> prediction -> risk_assessment -> recommendation_engine
================================================================================
```

## 🔍 Agent Details

### **Agent 1: Data Collection Agent**
- **Purpose**: Fetches stock data and company information
- **Input**: Ticker symbol
- **Output**: Stock data, company name
- **Features**: Retry logic, rate limiting, dummy data fallback

### **Agent 2: Technical Analysis Agent**
- **Purpose**: Computes 20+ technical indicators
- **Input**: Stock data
- **Output**: Technical indicators, market conditions
- **Features**: Advanced indicators, error handling

### **Agent 3: Sentiment Analysis Agent**
- **Purpose**: Analyzes news sentiment and market sentiment
- **Input**: Company name, stock data
- **Output**: Sentiment score, news sentiment
- **Features**: News API integration, sentiment scoring

### **Agent 4: Model Training Agent**
- **Purpose**: Trains or loads LSTM models
- **Input**: Stock data with indicators
- **Output**: Trained model, scaler, features
- **Features**: Model persistence, fine-tuning, feature selection

### **Agent 5: Prediction Agent**
- **Purpose**: Generates price predictions
- **Input**: Trained model, stock data
- **Output**: Future prices, predicted trend
- **Features**: Multi-step prediction, noise injection

### **Agent 6: Risk Assessment Agent**
- **Purpose**: Assesses risk and market conditions
- **Input**: Stock data, predictions
- **Output**: Risk metrics, trend signals
- **Features**: Volatility, Sharpe ratio, drawdown analysis

### **Agent 7: Recommendation Engine Agent**
- **Purpose**: Generates final recommendations
- **Input**: All previous agent outputs
- **Output**: Recommendation, confidence score
- **Features**: Multi-factor decision making, confidence scoring

## 🔄 Workflow Execution

### **Sequential Flow**
```
Data Collection → Technical Analysis → Sentiment Analysis → 
Model Training → Prediction → Risk Assessment → Recommendation Engine
```

### **State Management**
Each agent receives and updates the `AIstgAnalysisState`:
```python
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
```

## 🎯 Benefits Over Original Version

### **1. Modularity**
- ✅ Each agent can be developed/tested independently
- ✅ Easy to add new analysis methods
- ✅ Simple to modify individual components

### **2. Scalability**
- ✅ Agents can run in parallel where possible
- ✅ Easy to distribute across multiple machines
- ✅ Can handle larger datasets efficiently

### **3. Maintainability**
- ✅ Clear separation of concerns
- ✅ Better error isolation
- ✅ Easier debugging and monitoring

### **4. Extensibility**
- ✅ Easy to add new agents
- ✅ Can implement conditional workflows
- ✅ Support for different analysis strategies

### **5. Reliability**
- ✅ Better error handling
- ✅ Graceful degradation
- ✅ Retry mechanisms

## 🔧 Configuration

### **Environment Variables**
```bash
export NEWS_API_KEY='your_news_api_key_here'
```

### **Model Parameters**
```python
WINDOW_SIZE = 60
PREDICTION_DAYS = 30
FINE_TUNE_EPOCHS = 50
TRAIN_EPOCHS = 100
MAX_RETRIES = 3
```

### **Agent-Specific Settings**
Each agent can be configured independently:
- Data Collection: Retry delays, rate limits
- Technical Analysis: Indicator parameters
- Sentiment Analysis: API keys, model selection
- Model Training: Architecture, hyperparameters
- Prediction: Noise levels, validation
- Risk Assessment: Thresholds, metrics
- Recommendation Engine: Decision weights

## 🚨 Troubleshooting

### **Common Issues**

1. **Rate Limiting**: System automatically handles Yahoo Finance rate limits
2. **API Failures**: News sentiment falls back to neutral
3. **Model Training**: Uses dummy data if real data unavailable
4. **Network Issues**: Built-in retry mechanisms

### **Debug Mode**
```python
# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
```

### **Error Recovery**
- System continues even if some agents fail
- Detailed error messages in logs
- Fallback mechanisms for critical components

## 🔮 Future Enhancements

### **Planned Features**
1. **Conditional Workflows**: Different paths based on market conditions
2. **Real-time Processing**: Streaming data support
3. **Advanced Agents**: Options analysis, fundamental analysis
4. **Web Interface**: Dashboard for monitoring
5. **API Endpoints**: REST API for integration

### **Customization Options**
1. **Agent Selection**: Choose which agents to run
2. **Workflow Modification**: Custom execution paths
3. **Parameter Tuning**: Agent-specific settings
4. **Output Formats**: JSON, CSV, HTML reports

## 📚 API Reference

### **Main Functions**
```python
# Analyze single stock
result = analyze_stock_with_langgraph("AAPL")

# Analyze multiple stocks
results, failed = parallel_langgraph_analysis(["AAPL", "MSFT", "GOOGL"])

# Create custom workflow
workflow = create_aistg_workflow()
```

### **State Access**
```python
# Access any agent's output
current_price = result['current_price']
recommendation = result['recommendation']
confidence = result['confidence_score']
execution_path = result['execution_path']
```

## 🎉 Conclusion

The LangGraph integration transforms AIstg from a monolithic system into a sophisticated multi-agent platform that:

- **Preserves all existing functionality** from AIstg_v1.1.py
- **Adds advanced features** like confidence scoring and state management
- **Improves reliability** with better error handling and fallbacks
- **Enables future enhancements** through modular architecture
- **Provides better insights** through detailed execution tracking

This integration represents a significant upgrade that maintains backward compatibility while opening new possibilities for advanced stock analysis. 