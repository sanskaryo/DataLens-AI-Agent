import streamlit as st
from langchain_groq import ChatGroq
import os
import yfinance as yf
import pandas as pd
import numpy as np
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import ta  
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="📈 Stock Analyzer AI Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@tool
def get_stock_info(symbol, key):
    '''Enhanced stock info retrieval with support for Indian stocks'''
    data = yf.Ticker(symbol)
    stock_info = data.info
    
    key_mapping = {
        'current_price': 'regularMarketPrice',
        'price': 'regularMarketPrice',
        'market_price': 'regularMarketPrice',
        'stock_price': 'regularMarketPrice',
        'open': 'regularMarketOpen',
        'high': 'regularMarketDayHigh',
        'low': 'regularMarketDayLow',
        'volume': 'regularMarketVolume',
        'previous_close': 'regularMarketPreviousClose',
        'market_cap': 'marketCap',
        'pe_ratio': 'trailingPE',
        'eps': 'trailingEps',
        'dividend_yield': 'dividendYield',
        'beta': 'beta',
        '52_week_high': 'fiftyTwoWeekHigh',
        '52_week_low': 'fiftyTwoWeekLow',
        'company_name': 'longName',
        'sector': 'sector',
        'industry': 'industry'
    }
    
    key = key.lower()
    if key in key_mapping:
        key = key_mapping[key]
    
    try:
        value = stock_info[key]
  
        if isinstance(value, (int, float)):
            if key in ['marketCap']:
                return f"₹{value:,.0f}" if '.NS' in symbol else f"${value:,.0f}"
            elif key in ['regularMarketPrice', 'regularMarketOpen', 'regularMarketDayHigh', 'regularMarketDayLow']:
                return f"₹{value:,.2f}" if '.NS' in symbol else f"${value:,.2f}"
            elif key in ['dividendYield', 'beta']:
                return f"{value:.2f}"
        return value
    except KeyError:
        available_keys = list(key_mapping.keys())
        return f"Available info types: {', '.join(available_keys)}"

@tool
def get_historical_price(symbol, start_date, end_date):
    """Enhanced historical price data with technical indicators"""
    data = yf.Ticker(symbol)
    hist = data.history(start=start_date, end=end_date)
 
    hist['SMA_20'] = ta.trend.sma_indicator(hist['Close'], window=20)
    hist['SMA_50'] = ta.trend.sma_indicator(hist['Close'], window=50)
    hist['RSI'] = ta.momentum.rsi(hist['Close'])
    hist['MACD'] = ta.trend.macd_diff(hist['Close'])
    
    hist = hist.reset_index()
    hist[symbol] = hist['Close']
    
    return hist

@tool
def get_company_info(symbol):
    """Get detailed company information"""
    data = yf.Ticker(symbol)
    info = data.info
    
    return {
        'name': info.get('longName', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'description': info.get('longBusinessSummary', 'N/A'),
        'website': info.get('website', 'N/A'),
        'employees': info.get('fullTimeEmployees', 'N/A'),
        'country': info.get('country', 'N/A')
    }

def plot_price_over_time(historical_price_dfs):
    '''Enhanced price visualization with technical indicators'''
    df = historical_price_dfs[0]  
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    # Add SMAs
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['SMA_20'],
        name='SMA 20',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['SMA_50'],
        name='SMA 50',
        line=dict(color='blue')
    ))
    
    # Volume bars
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))
    
    fig.update_layout(
        title='Stock Price Analysis with Technical Indicators',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        xaxis_title='Date',
        template='plotly_white',
        height=800,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_technical_indicators(df):
    '''Plot additional technical indicators'''
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('RSI', 'MACD', 'Volume'))
    
    # RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD'), row=2, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'), row=3, col=1)
    
    fig.update_layout(height=800, title_text="Technical Indicators")
    return fig

def initialize_session_state():
    """Initialize session state with enhanced capabilities"""
    load_dotenv()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm" not in st.session_state:
        st.session_state.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model='llama3-70b-8192'
        )
        tools = [get_stock_info, get_historical_price, get_company_info]
        st.session_state.llm_with_tools = st.session_state.llm.bind_tools(tools)

def process_user_input(user_input):
    """Enhanced user input processing with more context"""
    system_prompt = f'''You are an expert stock market analyst assistant. Today is {date.today()}
    
    You can help with:
    1. Stock price information (current, historical, technical analysis)
    2. Company information and fundamentals
    3. Technical indicators (SMA, RSI, MACD)
    4. Market analysis and insights
    
    For Indian stocks, use the .NS suffix (e.g., TATAMOTORS.NS, RELIANCE.NS)
    
    Available tools and their capabilities:
    1. get_stock_info: Get current stock data and fundamentals
    2. get_historical_price: Get historical prices with technical indicators
    3. get_company_info: Get detailed company information
    
    Common queries you can handle:
    - Current price and market data
    - Historical price analysis
    - Technical analysis and indicators
    - Company fundamentals and information
    - Market trends and patterns
    '''
    
    messages = [SystemMessage(system_prompt), HumanMessage(user_input)]
    ai_msg = st.session_state.llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    historical_price_dfs = []
    company_infos = []
    
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"].lower()
        tool_output = {
            "get_stock_info": get_stock_info,
            "get_historical_price": get_historical_price,
            "get_company_info": get_company_info
        }[tool_name].invoke(tool_call["args"])
        
        if tool_name == 'get_historical_price':
            historical_price_dfs.append(tool_output)
        elif tool_name == 'get_company_info':
            company_infos.append(tool_output)
        
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    
    response = st.session_state.llm_with_tools.invoke(messages).content
    return response, historical_price_dfs, company_infos

def main():
    st.title("📈 Stock Analyzer AI Agent")
    
    initialize_session_state()
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("📊 Features & Capabilities")
        
        # Quick Access Buttons
        st.subheader("🔍 Quick Analysis")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 Fundamentals"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": "What are the key financial metrics for AAPL"
                })
        with col2:
            if st.button("📉 Technical Analysis"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": "Show me technical analysis for TATASTEEL.NS including RSI and MACD"
                })
        
        st.markdown("""
            <style>
            .main { padding: 2rem; }
            .stButton>button {
                width: 100%;
                border-radius: 5px;
                height: 3em;
                background-color: #1f77b4;
                color: white;
                border: none;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #155987;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .chat-message {
                padding: 1.5rem;
                border-radius: 0.8rem;
                margin: 1.2rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .user-message {
                background-color: #e3f2fd;
                border-left: 4px solid #1976d2;
            }
            .assistant-message {
                background-color: #f5f5f5;
                border-left: 4px solid #2e7d32;
            }
            .stock-metrics {
                background-color: #fff;
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            ### 🎯 Capabilities
            - Real-time stock prices
            - Technical analysis (SMA, RSI, MACD)
            - Company fundamentals
            - Historical price trends
            - Market insights
            
            ### 💡 Example Queries
            - "Show me key statistics for MSFT?"
            - "Show technical analysis for INFY.NS"
            - "Display volume profile with price action for AMZN"
            - "Give me company info for TCS.NS"
            - "What is the highest price of RELIANCE.NS in the last month?"
            
            ### 🎨 Chart Types
            - Candlestick charts
            - Technical indicators
            - Volume analysis
            - Price trends
        """)
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "chart" in message:
                st.plotly_chart(message["chart"], use_container_width=True, key=f"chart_{message['role']}_{idx}")
    
    # Enhanced chat input
    if prompt := st.chat_input("Ask about stocks (e.g., 'Analyze RELIANCE.NS performance')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing market data..."):
                response, historical_price_dfs, company_infos = process_user_input(prompt)
                st.markdown(response)
                
                if historical_price_dfs:
                    chart = plot_price_over_time(historical_price_dfs)
                    st.plotly_chart(chart, use_container_width=True, key=f"price_chart_{len(st.session_state.messages)}")
                    
                    # Show technical indicators in a separate chart
                    tech_chart = plot_technical_indicators(historical_price_dfs[0])
                    st.plotly_chart(tech_chart, use_container_width=True, key=f"tech_chart_{len(st.session_state.messages)}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "chart": chart
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

if __name__ == "__main__":
    main() 
    
    
    
    
    
    
    
    
    
#     Stocks (e.g., AAPL, MSFT, GOOGL)
# 1. Current Price Queries:
# "What is the current price of AAPL?"
# "Get the current stock price of MSFT."
# 2. Previous Closing Price:
# "What was the previous closing price of GOOGL?"
# "Tell me the last closing price of NVDA."
# 3. Historical Price Analysis:
# "Show me the historical prices of AMZN for the last 6 months."
# "What were the closing prices of TSLA over the last month?"
# 4. Comparative Analysis:
# "Compare the stock prices of AAPL and MSFT over the last year."
# "How did the stock prices of GOOGL and FB perform in the last quarter?"
# 5. Technical Analysis:
# "Show me the RSI and MACD for AAPL."
# "What is the 50-day SMA for MSFT?"
# Fundamental Analysis:
# "What are the key statistics for AMZN?"
# "Tell me about the company fundamentals of FB."
# Market Insights:
# "What is the trading volume of NFLX for the last week?"
# "What was the market cap of TSLA yesterday?"
# Chart Generation:
# "Plot the historical prices of AAPL for the last 3 months."
# "Generate a candlestick chart for MSFT over the last month."
# Queries for Indian Stocks (e.g., RELIANCE.NS, TCS.NS, HDFCBANK.NS)
# 1. Current Price Queries:
# "What is the current price of RELIANCE.NS?"
# "Get the current stock price of TCS.NS."
# 2. Previous Closing Price:
# "What was the previous closing price of HDFCBANK.NS?"
# "Tell me the last closing price of INFY.NS."
# 3. Historical Price Analysis:
# "Show me the historical prices of SBIN.NS for the last 6 months."
# "What were the closing prices of HINDUNILVR.NS over the last month?"
# 4. Comparative Analysis:
# "Compare the stock prices of HDFCBANK.NS and ICICIBANK.NS over the last year."
# "How did the stock prices of TCS.NS and WIPRO.NS perform in the last quarter?"
# 5. Technical Analysis:
# "Show me the RSI and MACD for RELIANCE.NS."
# "What is the 50-day SMA for HDFCBANK.NS?"
# Fundamental Analysis:
# "What are the key statistics for TATAMOTORS.NS?"
# "Tell me about the company fundamentals of HINDALCO.NS."
# 7. Market Insights:
# "What is the trading volume of KOTAKBANK.NS for the last week?"
# "What was the market cap of BAJFINANCE.NS yesterday?"
# Chart Generation:
# "Plot the historical prices of TCS.NS for the last 3 months."
# "Generate a candlestick chart for INFY.NS over the last month."
# General Queries
# "What are the available info types for AAPL?"
# "List the available info types for RELIANCE.NS."
# "What is the highest price of MSFT in the last month?"