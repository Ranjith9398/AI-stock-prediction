#!/usr/bin/env python3
"""
LangGraph Stock Analysis Example
This script demonstrates how to use the LangGraph implementation for stock analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add current directory to path to import our LangGraph implementation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our LangGraph implementation
from langgraph_implementation import (
    analyze_stock_with_langgraph,
    create_stock_analysis_workflow,
    StockAnalysisState
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_single_stock_analysis(ticker: str):
    """Run analysis for a single stock using LangGraph."""
    print(f"\n{'='*60}")
    print(f"ANALYZING STOCK: {ticker}")
    print(f"{'='*60}")
    
    try:
        # Run the LangGraph workflow
        result = analyze_stock_with_langgraph(ticker)
        
        # Display results
        print(f"\n📊 ANALYSIS RESULTS FOR {ticker}")
        print(f"{'='*40}")
        print(f"🏢 Company: {result.get('company_name', 'Unknown')}")
        print(f"💡 Recommendation: {result['recommendation']}")
        print(f"🎯 Confidence: {result['confidence_score']:.2f}")
        print(f"📈 Market Condition: {result['market_conditions']}")
        print(f"⚠️  Risk Level: {result['risk_assessment'].get('overall_risk', 'unknown')}")
        
        # Show execution path
        print(f"\n🔄 Execution Path:")
        for i, step in enumerate(result['execution_path'], 1):
            print(f"   {i}. {step}")
        
        # Show predictions if available
        if result['model_predictions']:
            print(f"\n📊 Price Predictions (Next 30 Days):")
            current_price = result['stock_data']['Close'].iloc[-1]
            print(f"   Current Price: ${current_price:.2f}")
            
            # Show key prediction points
            predictions = result['model_predictions']
            if len(predictions) >= 7:
                print(f"   7-day: ${predictions[6]:.2f}")
            if len(predictions) >= 15:
                print(f"   15-day: ${predictions[14]:.2f}")
            if len(predictions) >= 30:
                print(f"   30-day: ${predictions[29]:.2f}")
                
            # Calculate predicted return
            if len(predictions) >= 30:
                predicted_return = (predictions[29] - current_price) / current_price * 100
                print(f"   Predicted Return: {predicted_return:+.2f}%")
        
        # Show risk assessment details
        if result['risk_assessment']:
            print(f"\n⚠️  Risk Assessment:")
            risk = result['risk_assessment']
            if 'volatility' in risk:
                print(f"   Volatility: {risk['volatility']:.3f}")
            if 'max_drawdown' in risk:
                print(f"   Max Drawdown: {risk['max_drawdown']:.2%}")
            if 'rsi_risk' in risk:
                print(f"   RSI Risk: {risk['rsi_risk']}")
        
        # Show errors if any
        if result['errors']:
            print(f"\n❌ Errors Encountered:")
            for error in result['errors']:
                print(f"   - {error}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        print(f"❌ Failed to analyze {ticker}: {e}")
        return None

def run_multiple_stock_analysis(tickers: list):
    """Run analysis for multiple stocks using LangGraph."""
    print(f"\n{'='*60}")
    print(f"ANALYZING MULTIPLE STOCKS: {len(tickers)} stocks")
    print(f"{'='*60}")
    
    results = []
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Analyzing {ticker}...")
        
        result = run_single_stock_analysis(ticker)
        
        if result:
            results.append(result)
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(tickers)}")
    
    # Categorize recommendations
    if results:
        buy_stocks = [r for r in results if 'Buy' in r['recommendation']]
        sell_stocks = [r for r in results if 'Sell' in r['recommendation']]
        hold_stocks = [r for r in results if r['recommendation'] == 'Hold']
        
        print(f"\n📈 Recommendations:")
        print(f"   Buy: {len(buy_stocks)}")
        print(f"   Sell: {len(sell_stocks)}")
        print(f"   Hold: {len(hold_stocks)}")
        
        # Show top recommendations
        if buy_stocks:
            print(f"\n🎯 Top Buy Recommendations:")
            buy_stocks.sort(key=lambda x: x['confidence_score'], reverse=True)
            for stock in buy_stocks[:3]:
                print(f"   - {stock['ticker']}: {stock['recommendation']} (confidence: {stock['confidence_score']:.2f})")
        
        if sell_stocks:
            print(f"\n⚠️  Top Sell Recommendations:")
            sell_stocks.sort(key=lambda x: x['confidence_score'], reverse=True)
            for stock in sell_stocks[:3]:
                print(f"   - {stock['ticker']}: {stock['recommendation']} (confidence: {stock['confidence_score']:.2f})")
    
    return results

def interactive_mode():
    """Run interactive mode for stock analysis."""
    print(f"\n{'='*60}")
    print(f"LANGGRAPH STOCK ANALYSIS - INTERACTIVE MODE")
    print(f"{'='*60}")
    
    while True:
        print(f"\nOptions:")
        print(f"1. Analyze single stock")
        print(f"2. Analyze multiple stocks")
        print(f"3. Run demo with sample stocks")
        print(f"4. Exit")
        
        choice = input(f"\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
            if ticker:
                run_single_stock_analysis(ticker)
            else:
                print("❌ Invalid ticker")
        
        elif choice == '2':
            tickers_input = input("Enter stock tickers separated by commas (e.g., AAPL,MSFT,GOOGL): ").strip()
            if tickers_input:
                tickers = [t.strip().upper() for t in tickers_input.split(',')]
                run_multiple_stock_analysis(tickers)
            else:
                print("❌ Invalid input")
        
        elif choice == '3':
            # Demo with sample stocks
            demo_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
            print(f"Running demo with: {', '.join(demo_tickers)}")
            run_multiple_stock_analysis(demo_tickers)
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")

def main():
    """Main function to run the LangGraph stock analysis."""
    print(f"🚀 LangGraph Stock Analysis System")
    print(f"Version: 1.0")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            # Run demo
            demo_tickers = ["AAPL", "MSFT", "GOOGL"]
            run_multiple_stock_analysis(demo_tickers)
        elif sys.argv[1] == '--single':
            if len(sys.argv) > 2:
                ticker = sys.argv[2].upper()
                run_single_stock_analysis(ticker)
            else:
                print("❌ Please provide a ticker symbol: python run_langgraph_example.py --single AAPL")
        elif sys.argv[1] == '--multiple':
            if len(sys.argv) > 2:
                tickers = [t.upper() for t in sys.argv[2:]]
                run_multiple_stock_analysis(tickers)
            else:
                print("❌ Please provide ticker symbols: python run_langgraph_example.py --multiple AAPL MSFT GOOGL")
        else:
            print("❌ Invalid argument. Use --demo, --single, or --multiple")
    else:
        # Run interactive mode
        interactive_mode()

if __name__ == "__main__":
    main() 