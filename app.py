import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import db

# Initialize database on first app load
db.ensure_db_initialized()

st.set_page_config(page_title="Stock Dashboard", layout="wide")

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'tickers' not in st.session_state:
    st.session_state.tickers = []

# Redirect to login if not authenticated
if not st.session_state.user_id:
    st.title("📈 Stock Dashboard")
    st.info("Please login to access the dashboard.")
    
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("👉 Go to Login Page", type="primary", use_container_width=True):
            st.switch_page("pages/1_Login.py")
    st.stop()

# Main dashboard
st.title("📈 Stock Dashboard")
st.markdown(f"Welcome back, **{st.session_state.username}**! Track your favorite stocks with real-time data and interactive charts")

# Sidebar for navigation and logout
with st.sidebar:
    st.markdown(f"### Logged in as: {st.session_state.username}")
    st.markdown(f"📧 {st.session_state.email}")
    
    if st.button("📊 View Portfolio", use_container_width=True):
        st.switch_page("pages/2_Portfolio.py")
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.email = None
        st.session_state.tickers = []
        st.success("Logged out successfully!")
        st.rerun()

# Exchange rate: USD to INR
USD_TO_INR = 83.5

def is_indian_stock(ticker):
    """Check if a stock is from Indian market"""
    return ticker.endswith('.NS') or ticker.endswith('.BO')

def format_price(price, ticker):
    """Format price with correct currency symbol"""
    if is_indian_stock(ticker):
        return f"₹{price:.2f}"
    else:
        return f"₹{price * USD_TO_INR:.2f}"

def get_currency_label(ticker):
    """Get currency label for charts"""
    if is_indian_stock(ticker):
        return "Price (INR)"
    else:
        return "Price (INR - USD converted)"

def search_ticker(query):
    query = query.strip()
    if not query:
        return None
    
    query_upper = query.upper()
    
    try:
        test_stock = yf.Ticker(query_upper)
        test_info = test_stock.info
        
        if test_info and test_info.get('symbol'):
            return query_upper
    except:
        pass
    
    common_stocks = {
        'APPLE': 'AAPL',
        'MICROSOFT': 'MSFT',
        'GOOGLE': 'GOOGL',
        'ALPHABET': 'GOOGL',
        'AMAZON': 'AMZN',
        'TESLA': 'TSLA',
        'META': 'META',
        'FACEBOOK': 'META',
        'NVIDIA': 'NVDA',
        'NETFLIX': 'NFLX',
        'DISNEY': 'DIS',
        'WALMART': 'WMT',
        'COCA COLA': 'KO',
        'COCA-COLA': 'KO',
        'PEPSI': 'PEP',
        'MCDONALD': 'MCD',
        'MCDONALDS': 'MCD',
        'INTEL': 'INTC',
        'AMD': 'AMD',
        'ORACLE': 'ORCL',
        'IBM': 'IBM',
        'BOEING': 'BA',
        'NIKE': 'NKE',
        'STARBUCKS': 'SBUX',
        'PAYPAL': 'PYPL',
        'VISA': 'V',
        'MASTERCARD': 'MA',
        'JPMORGAN': 'JPM',
        'BANK OF AMERICA': 'BAC',
        'WELLS FARGO': 'WFC',
        'GOLDMAN SACHS': 'GS',
        'MORGAN STANLEY': 'MS',
        'EXXON': 'XOM',
        'CHEVRON': 'CVX',
        'PFIZER': 'PFE',
        'JOHNSON & JOHNSON': 'JNJ',
        'MODERNA': 'MRNA',
        'BERKSHIRE HATHAWAY': 'BRK.B',
        'FORD': 'F',
        'GENERAL MOTORS': 'GM',
        'GM': 'GM',
        'CISCO': 'CSCO',
        'ADOBE': 'ADBE',
        'SALESFORCE': 'CRM',
        'ZOOM': 'ZM',
        'SPOTIFY': 'SPOT',
        'UBER': 'UBER',
        'LYFT': 'LYFT',
        'AIRBNB': 'ABNB',
        'TWITTER': 'TWTR',
        'SNAPCHAT': 'SNAP',
        'SNAP': 'SNAP',
        'PINTEREST': 'PINS',
        'SQUARE': 'SQ',
        'BLOCK': 'SQ',
        'ROBINHOOD': 'HOOD',
        'COINBASE': 'COIN',
        'AT&T': 'T',
        'VERIZON': 'VZ',
        'T-MOBILE': 'TMUS',
        'COMCAST': 'CMCSA',
        'SONY': 'SONY',
        'SAMSUNG': '005930.KS',
        'TOYOTA': 'TM',
        'HONDA': 'HMC',
        'ALIBABA': 'BABA',
        'TENCENT': 'TCEHY',
        'BAIDU': 'BIDU',
        'TATA STEEL': 'TATASTEEL.NS',
        'TATASTEEL': 'TATASTEEL.NS',
        'TATA MOTORS': 'TATAMOTORS.NS',
        'TATAMOTORS': 'TATAMOTORS.NS',
        'RELIANCE': 'RELIANCE.NS',
        'RELIANCE INDUSTRIES': 'RELIANCE.NS',
        'INFOSYS': 'INFY.NS',
        'TCS': 'TCS.NS',
        'TATA CONSULTANCY': 'TCS.NS',
        'TATA CONSULTANCY SERVICES': 'TCS.NS',
        'WIPRO': 'WIPRO.NS',
        'HDFC BANK': 'HDFCBANK.NS',
        'HDFCBANK': 'HDFCBANK.NS',
        'ICICI BANK': 'ICICIBANK.NS',
        'ICICIBANK': 'ICICIBANK.NS',
        'SBI': 'SBIN.NS',
        'STATE BANK': 'SBIN.NS',
        'STATE BANK OF INDIA': 'SBIN.NS',
        'BHARTI AIRTEL': 'BHARTIARTL.NS',
        'AIRTEL': 'BHARTIARTL.NS',
        'ITC': 'ITC.NS',
        'ADANI': 'ADANIENT.NS',
        'ADANI ENTERPRISES': 'ADANIENT.NS',
        'MAHINDRA': 'M&M.NS',
        'MAHINDRA & MAHINDRA': 'M&M.NS',
        'ASIAN PAINTS': 'ASIANPAINT.NS',
        'ASIANPAINTS': 'ASIANPAINT.NS',
        'HUL': 'HINDUNILVR.NS',
        'HINDUSTAN UNILEVER': 'HINDUNILVR.NS',
        'BAJAJ': 'BAJFINANCE.NS',
        'BAJAJ FINANCE': 'BAJFINANCE.NS',
        'MARUTI': 'MARUTI.NS',
        'MARUTI SUZUKI': 'MARUTI.NS',
        'LARSEN': 'LT.NS',
        'LARSEN & TOUBRO': 'LT.NS',
        'L&T': 'LT.NS',
    }
    
    query_clean = query_upper.replace('.', '').replace(',', '').replace('&', '')
    
    if query_clean in common_stocks:
        return common_stocks[query_clean]
    
    for name, ticker in common_stocks.items():
        if query_clean in name or name in query_clean:
            return ticker
    
    if not '.' in query_upper:
        for suffix in ['.NS', '.BO']:
            try:
                test_ticker = query_upper + suffix
                test_stock = yf.Ticker(test_ticker)
                test_info = test_stock.info
                
                if test_info and test_info.get('symbol'):
                    return test_ticker
            except:
                continue
    
    return query_upper

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

def analyze_sentiment(hist):
    if len(hist) < 30:
        return "Neutral", "Insufficient data for analysis", {
            'RSI': 50.0,
            '7-Day SMA': 0.0,
            '30-Day SMA': 0.0,
            '7-Day Change': 0.0,
            '30-Day Change': 0.0
        }
    
    close_prices = hist['Close']
    
    sma_7 = close_prices.rolling(window=7).mean().iloc[-1]
    sma_30 = close_prices.rolling(window=30).mean().iloc[-1]
    current_price = close_prices.iloc[-1]
    
    rsi = calculate_rsi(close_prices)
    
    price_change_7d = ((current_price - close_prices.iloc[-7]) / close_prices.iloc[-7]) * 100
    price_change_30d = ((current_price - close_prices.iloc[-30]) / close_prices.iloc[-30]) * 100
    
    metrics = {
        'RSI': rsi,
        '7-Day SMA': sma_7,
        '30-Day SMA': sma_30,
        '7-Day Change': price_change_7d,
        '30-Day Change': price_change_30d
    }
    
    sentiment_score = 0
    
    if current_price > sma_7:
        sentiment_score += 1
    if sma_7 > sma_30:
        sentiment_score += 1
    if rsi > 50:
        sentiment_score += 1
    if price_change_7d > 0:
        sentiment_score += 1
    
    if rsi > 70:
        sentiment = "Overbought"
        description = f"The stock shows strong upward momentum but may be overbought (RSI: {rsi:.1f}). Price is {price_change_7d:.1f}% over the last 7 days."
    elif rsi < 30:
        sentiment = "Oversold"
        description = f"The stock may be oversold (RSI: {rsi:.1f}) and could see a rebound. Price is {price_change_7d:.1f}% over the last 7 days."
    elif sentiment_score >= 3:
        sentiment = "Bullish"
        description = f"Technical indicators suggest positive momentum. Price is above moving averages with {price_change_7d:.1f}% gain in 7 days."
    elif sentiment_score <= 1:
        sentiment = "Bearish"
        description = f"Technical indicators show weakness. Price has moved {price_change_7d:.1f}% in the last 7 days."
    else:
        sentiment = "Neutral"
        description = f"Mixed signals from technical indicators. Price change: {price_change_7d:.1f}% (7d), {price_change_30d:.1f}% (30d)."
    
    return sentiment, description, metrics

def predict_future_prices(hist, days=7):
    if len(hist) < 14:
        return None, None
    
    close_prices = hist['Close'].values
    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = close_prices
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.arange(len(close_prices), len(close_prices) + days).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    recent_errors = y[-7:] - model.predict(X[-7:])
    mae = np.abs(recent_errors).mean()
    
    close_series = pd.Series(close_prices)
    sma_7 = close_series.rolling(window=7).mean().iloc[-1]
    sma_30 = close_series.rolling(window=30).mean().iloc[-1] if len(close_prices) >= 30 else sma_7
    
    ema_span = 12
    ema = close_series.ewm(span=ema_span, adjust=False).mean().iloc[-1]
    
    trend_direction = "Upward" if predictions[-1] > close_prices[-1] else "Downward"
    
    return predictions, {
        'mae': mae,
        'trend': trend_direction,
        'sma_7': sma_7,
        'sma_30': sma_30,
        'ema': ema,
        'current': close_prices[-1]
    }

col1, col2 = st.columns([3, 1])

with col1:
    ticker_input = st.text_input(
        "Enter Stock Ticker or Company Name",
        placeholder="e.g., AAPL or Apple",
        key="ticker_input"
    )

with col2:
    st.write("")
    st.write("")
    if st.button("Add Stock", type="primary"):
        if ticker_input:
            ticker_symbol = search_ticker(ticker_input)
            if ticker_symbol and ticker_symbol not in st.session_state.tickers:
                st.session_state.tickers.append(ticker_symbol)
                st.rerun()
            elif ticker_symbol in st.session_state.tickers:
                st.warning(f"{ticker_symbol} is already in your watchlist")
            else:
                st.error(f"Could not find stock for '{ticker_input}'")

if st.session_state.tickers:
    st.markdown("---")
    st.subheader("Your Watchlist")
    
    for ticker in st.session_state.tickers:
        try:
            stock = yf.Ticker(ticker)
            
            info = stock.info
            hist = stock.history(period="1mo")
            
            if hist.empty:
                st.error(f"Unable to fetch data for {ticker}. Please check the ticker symbol.")
                if st.button(f"Remove {ticker}", key=f"remove_{ticker}"):
                    st.session_state.tickers.remove(ticker)
                    st.rerun()
                continue
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            with st.container():
                col_info, col_remove = st.columns([6, 1])
                
                with col_info:
                    st.markdown(f"### {ticker}")
                    company_name = info.get('longName', ticker)
                    st.markdown(f"**{company_name}**")
                
                with col_remove:
                    if st.button("🗑️ Remove", key=f"remove_{ticker}"):
                        st.session_state.tickers.remove(ticker)
                        st.rerun()
                
                col_price, col_change = st.columns(2)
                
                with col_price:
                    st.metric("Current Price", format_price(current_price, ticker))
                
                with col_change:
                    st.metric(
                        "Daily Change",
                        format_price(price_change, ticker),
                        f"{price_change_pct:.2f}%"
                    )
                
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name=ticker
                ))
                
                fig.update_layout(
                    title=f"{ticker} - Last 30 Days",
                    xaxis_title="Date",
                    yaxis_title=get_currency_label(ticker),
                    template="plotly_white",
                    height=400,
                    hovermode='x unified'
                )
                
                fig.update_xaxes(
                    rangeslider_visible=False,
                    rangebreaks=[
                        dict(bounds=["sat", "mon"])
                    ]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander(f"🎯 Sentiment & Momentum Analysis for {ticker}"):
                    try:
                        sentiment, description, metrics = analyze_sentiment(hist)
                        
                        sentiment_color = {
                            'Bullish': 'green',
                            'Bearish': 'red',
                            'Neutral': 'gray',
                            'Overbought': 'orange',
                            'Oversold': 'blue'
                        }.get(sentiment, 'gray')
                        
                        st.markdown(f"**Market Sentiment:** :{sentiment_color}[{sentiment}]")
                        st.info(description)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("RSI (14)", f"{metrics['RSI']:.1f}")
                        with col2:
                            st.metric("7-Day Change", f"{metrics['7-Day Change']:.2f}%")
                        with col3:
                            st.metric("30-Day Change", f"{metrics['30-Day Change']:.2f}%")
                        
                        st.caption("💡 **Note:** Sentiment is based on technical indicators (RSI, moving averages, price momentum). This is not financial advice.")
                    except Exception as e:
                        st.error(f"Error calculating sentiment: {str(e)}")
                
                with st.expander(f"🔮 Price Outlook & Predictions for {ticker}"):
                    try:
                        predictions, pred_info = predict_future_prices(hist, days=7)
                        
                        if predictions is not None and pred_info is not None:
                            st.markdown(f"**Trend Direction:** {pred_info['trend']}")
                            
                            st.markdown("**7-Day Price Scenario:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Price", format_price(pred_info['current'], ticker))
                            with col2:
                                st.metric("7-Day SMA", format_price(pred_info['sma_7'], ticker))
                            with col3:
                                st.metric("30-Day SMA", format_price(pred_info['sma_30'], ticker))
                            
                            forecast_df = pd.DataFrame({
                                'Day': [f'Day +{i+1}' for i in range(len(predictions))],
                                'Projected Price': [format_price(p, ticker) for p in predictions]
                            })
                            st.dataframe(forecast_df, use_container_width=True)
                            
                            fig_pred = go.Figure()
                            
                            fig_pred.add_trace(go.Scatter(
                                x=list(range(len(hist))),
                                y=hist['Close'].values,
                                mode='lines',
                                name='Historical',
                                line=dict(color='blue')
                            ))
                            
                            future_days = list(range(len(hist), len(hist) + len(predictions)))
                            fig_pred.add_trace(go.Scatter(
                                x=future_days,
                                y=predictions,
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='orange', dash='dash')
                            ))
                            
                            fig_pred.update_layout(
                                title="Historical Prices + 7-Day Forecast",
                                xaxis_title="Days",
                                yaxis_title=get_currency_label(ticker),
                                template="plotly_white",
                                height=300
                            )
                            
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            st.caption(f"⚠️ **Disclaimer:** These are statistical projections based on historical patterns (recent average error: {format_price(pred_info['mae'], ticker)}). Stock prices are influenced by many unpredictable factors. This is not financial advice and should not be used as the sole basis for investment decisions.")
                        else:
                            st.warning("Insufficient data for reliable predictions. Need at least 14 days of historical data.")
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
                
                with st.expander(f"📊 View Daily Price Data for {ticker}"):
                    recent_data = hist.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    recent_data.index = recent_data.index.to_series().dt.strftime('%Y-%m-%d')
                    recent_data = recent_data.round(2)
                    st.dataframe(recent_data, use_container_width=True)
                
                st.markdown("---")
                
        except Exception as e:
            st.error(f"Error loading data for {ticker}: {str(e)}")
            if st.button(f"Remove {ticker}", key=f"remove_error_{ticker}"):
                st.session_state.tickers.remove(ticker)
                st.rerun()
            st.markdown("---")
else:
    st.info("👆 Enter a stock ticker above to get started!")
    st.markdown("""
    ### Popular Stock Tickers:
    - **AAPL** - Apple Inc.
    - **GOOGL** - Alphabet Inc. (Google)
    - **MSFT** - Microsoft Corporation
    - **AMZN** - Amazon.com Inc.
    - **TSLA** - Tesla Inc.
    - **META** - Meta Platforms Inc. (Facebook)
    - **NVDA** - NVIDIA Corporation
    
    ### Indian Stocks:
    - **TATASTEEL.NS** - Tata Steel Ltd
    - **RELIANCE.NS** - Reliance Industries
    - **INFY.NS** - Infosys Ltd
    - **TCS.NS** - Tata Consultancy Services
    """)
