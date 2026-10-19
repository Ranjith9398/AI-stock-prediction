# LangGraph Stock Analysis - Usage Guide

## 🚀 Quick Start

### 1. Installation
```bash
# Install LangGraph and dependencies
pip install langgraph langchain yfinance pandas numpy ta
```

### 2. Basic Usage

#### Interactive Mode
```bash
python run_langgraph_example.py
```
This will start an interactive menu where you can:
- Analyze single stocks
- Analyze multiple stocks
- Run demo with sample stocks

#### Command Line Mode

**Analyze a single stock:**
```bash
python run_langgraph_example.py --single AAPL
```

**Analyze multiple stocks:**
```bash
python run_langgraph_example.py --multiple AAPL MSFT GOOGL TSLA
```

**Run demo:**
```bash
python run_langgraph_example.py --demo
```

## 📊 Understanding the Output

### Sample Output
```
============================================================
ANALYZING STOCK: AAPL
============================================================

📊 ANALYSIS RESULTS FOR AAPL
========================================
🏢 Company: Apple Inc.
💡 Recommendation: Buy
🎯 Confidence: 0.75
📈 Market Condition: bullish
⚠️  Risk Level: medium

🔄 Execution Path:
   1. data_collection
   2. technical_analysis
   3. sentiment_analysis
   4. model_prediction
   5. risk_assessment
   6. recommendation_engine

📊 Price Predictions (Next 30 Days):
   Current Price: $150.25
   7-day: $152.30
   15-day: $154.80
   30-day: $158.45
   Predicted Return: +5.46%

⚠️  Risk Assessment:
   Volatility: 0.245
   Max Drawdown: -12.34%
   RSI Risk: medium
```

### Output Explanation

| Field | Description |
|-------|-------------|
| **Company** | Company name from Yahoo Finance |
| **Recommendation** | Buy/Sell/Hold/Strong Buy/Strong Sell |
| **Confidence** | 0.0-1.0 score indicating prediction confidence |
| **Market Condition** | bullish/bearish/neutral based on technical analysis |
| **Risk Level** | low/medium/high based on volatility and other factors |
| **Execution Path** | Shows which agents were executed in order |
| **Price Predictions** | 30-day price forecast with key milestones |
| **Risk Assessment** | Detailed risk metrics |

## 🔧 LangGraph Architecture

### Agent Workflow
```
Data Collection → Technical Analysis → Sentiment Analysis → 
Model Prediction → Risk Assessment → Recommendation Engine
```

### State Management
The system maintains state throughout the analysis:
```python
state = {
    "ticker": "AAPL",
    "stock_data": pd.DataFrame(...),
    "technical_indicators": {...},
    "sentiment_score": 0.25,
    "model_predictions": [150.25, 152.30, ...],
    "risk_assessment": {...},
    "recommendation": "Buy",
    "confidence_score": 0.75,
    "market_conditions": "bullish",
    "execution_path": ["data_collection", "technical_analysis", ...],
    "errors": []
}
```

## 🎯 Advanced Usage

### 1. Custom Analysis
```python
from langgraph_implementation import analyze_stock_with_langgraph

# Analyze a stock
result = analyze_stock_with_langgraph("AAPL")

# Access specific data
print(f"Current price: ${result['stock_data']['Close'].iloc[-1]:.2f}")
print(f"RSI: {result['technical_indicators']['RSI'].iloc[-1]:.2f}")
print(f"Predicted 30-day price: ${result['model_predictions'][-1]:.2f}")
```

### 2. Batch Analysis
```python
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
results = []

for ticker in tickers:
    result = analyze_stock_with_langgraph(ticker)
    if result:
        results.append(result)

# Sort by confidence
results.sort(key=lambda x: x['confidence_score'], reverse=True)
```

### 3. Risk-Based Filtering
```python
# Filter stocks by risk level
low_risk_stocks = [r for r in results if r['risk_assessment']['overall_risk'] == 'low']
high_confidence_stocks = [r for r in results if r['confidence_score'] > 0.8]
```

## 🔍 Understanding the Agents

### 1. Data Collection Agent
- Fetches stock data from Yahoo Finance
- Retrieves company information
- Handles data validation and cleaning

### 2. Technical Analysis Agent
- Calculates 20+ technical indicators
- Determines market conditions (bullish/bearish/neutral)
- Includes: RSI, MACD, Bollinger Bands, Moving Averages

### 3. Sentiment Analysis Agent
- Analyzes market sentiment
- Considers recent price movements
- Provides sentiment score (-1 to +1)

### 4. Model Prediction Agent
- Generates 30-day price predictions
- Uses technical indicators and sentiment
- Applies realistic constraints

### 5. Risk Assessment Agent
- Calculates volatility metrics
- Assesses maximum drawdown
- Determines overall risk level

### 6. Recommendation Engine Agent
- Combines all analysis results
- Generates final recommendation
- Calculates confidence score

## ⚙️ Configuration

### Modifying Analysis Parameters
Edit `langgraph_implementation.py` to customize:

```python
# Technical Analysis Parameters
WINDOW_SIZE = 60  # Days of historical data
PREDICTION_DAYS = 30  # Days to predict

# Risk Thresholds
HIGH_VOLATILITY_THRESHOLD = 0.3
LOW_CONFIDENCE_THRESHOLD = 0.7

# Sentiment Analysis
SENTIMENT_WEIGHT = 0.05
```

### Adding Custom Indicators
```python
def custom_technical_analysis_agent(state):
    # Add your custom indicators here
    data = state['stock_data']
    
    # Example: Custom momentum indicator
    data['Custom_Momentum'] = data['Close'].pct_change(5).rolling(10).mean()
    
    state['technical_indicators']['Custom_Momentum'] = data['Custom_Momentum']
    return state
```

## 🚨 Error Handling

### Common Issues

1. **Network Errors**
   - The system will retry data collection
   - Falls back to cached data if available

2. **Invalid Ticker**
   - Check ticker symbol format (e.g., "AAPL" not "apple")
   - Use correct exchange suffix (e.g., ".NS" for NSE stocks)

3. **Insufficient Data**
   - System requires at least 60 days of data
   - Will log warnings for insufficient data

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Tips

### 1. Parallel Processing
For multiple stocks, consider running agents in parallel:
```python
import concurrent.futures

def analyze_multiple_parallel(tickers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(analyze_stock_with_langgraph, ticker): ticker 
                  for ticker in tickers}
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    return results
```

### 2. Caching
The system includes basic caching for stock data:
```python
@lru_cache(maxsize=100)
def get_cached_stock_data(ticker):
    # Cached data retrieval
    pass
```

## 🔄 Integration with Existing AIstg

### Migration Path
1. **Phase 1**: Use LangGraph alongside existing system
2. **Phase 2**: Gradually migrate functions to agents
3. **Phase 3**: Full LangGraph implementation

### Comparison
| Feature | Current AIstg | LangGraph Version |
|---------|---------------|-------------------|
| Architecture | Monolithic | Multi-agent |
| State Management | Stateless | Stateful |
| Decision Logic | Fixed if-else | Dynamic routing |
| Error Handling | Basic | Graceful degradation |
| Learning | None | Feedback loops |
| Scalability | Limited | High |

## 📝 Example Scripts

### Simple Analysis
```python
#!/usr/bin/env python3
from langgraph_implementation import analyze_stock_with_langgraph

# Quick analysis
ticker = "AAPL"
result = analyze_stock_with_langgraph(ticker)

if result:
    print(f"{ticker}: {result['recommendation']} (confidence: {result['confidence_score']:.2f})")
else:
    print(f"Failed to analyze {ticker}")
```

### Portfolio Analysis
```python
#!/usr/bin/env python3
from langgraph_implementation import analyze_stock_with_langgraph

portfolio = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
results = []

for stock in portfolio:
    result = analyze_stock_with_langgraph(stock)
    if result:
        results.append(result)

# Sort by confidence
results.sort(key=lambda x: x['confidence_score'], reverse=True)

print("Portfolio Analysis Results:")
for result in results:
    print(f"{result['ticker']}: {result['recommendation']} ({result['confidence_score']:.2f})")
```

## 🆘 Troubleshooting

### Installation Issues
```bash
# If you get import errors
pip install --upgrade langgraph langchain

# If you get TA-Lib errors
pip install TA-Lib
```

### Runtime Issues
```bash
# Check Python version (3.8+ required)
python --version

# Check installed packages
pip list | grep lang
```

### Data Issues
- Ensure internet connection for Yahoo Finance data
- Check ticker symbols are correct
- Verify stock is actively traded

## 📞 Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify all dependencies are installed
3. Test with known working tickers (AAPL, MSFT, GOOGL)
4. Check network connectivity for data fetching

---

**Happy Trading! 🚀📈** 