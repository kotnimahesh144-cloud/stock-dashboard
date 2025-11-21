import streamlit as st
import db
import yfinance as yf
import pandas as pd

# Initialize database on first page load
db.ensure_db_initialized()

st.set_page_config(page_title="My Portfolio - Stock Dashboard", layout="wide")

# Check authentication
if 'user_id' not in st.session_state or not st.session_state.user_id:
    st.error("Please login first!")
    st.stop()

st.title("📊 My Portfolio")
st.write(f"Welcome, **{st.session_state.username}**!")

# Get user's portfolio from database
portfolio = db.get_user_portfolio(st.session_state.user_id)

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Add Stock to Portfolio")
    ticker_input = st.text_input("Enter stock ticker (e.g., AAPL, TATASTEEL.NS)")
    quantity = st.number_input("Quantity", min_value=1, value=1)
    buy_price = st.number_input("Buy Price (optional)", min_value=0.0, value=0.0)
    
    if st.button("Add to Portfolio", type="primary"):
        if ticker_input:
            try:
                # Validate ticker
                stock = yf.Ticker(ticker_input.upper())
                info = stock.info
                if info and info.get('symbol'):
                    if db.add_to_portfolio(st.session_state.user_id, ticker_input.upper(), quantity, buy_price):
                        st.success(f"✅ {ticker_input.upper()} added to portfolio!")
                        st.rerun()
                    else:
                        st.error("Failed to add stock to portfolio")
                else:
                    st.error("Invalid ticker symbol")
            except:
                st.error("Could not validate ticker. Please check the symbol.")
        else:
            st.warning("Please enter a ticker symbol")

with col2:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh Portfolio"):
        st.rerun()

st.markdown("---")

if portfolio:
    st.subheader("Your Holdings")
    
    # Display portfolio
    portfolio_data = []
    total_value = 0
    total_cost = 0
    
    for ticker, quantity, buy_price in portfolio:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                
                # Check if Indian stock
                is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
                if not is_indian:
                    current_price *= 83.5
                    if buy_price > 0:
                        buy_price *= 83.5
                
                current_value = current_price * quantity
                cost_basis = buy_price * quantity if buy_price > 0 else 0
                gain_loss = current_value - cost_basis
                gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
                
                currency = "₹"
                
                portfolio_data.append({
                    'Ticker': ticker,
                    'Quantity': quantity,
                    'Current Price': f"{currency}{current_price:.2f}",
                    'Current Value': f"{currency}{current_value:.2f}",
                    'Buy Price': f"{currency}{buy_price:.2f}" if buy_price > 0 else "N/A",
                    'Gain/Loss': f"{currency}{gain_loss:.2f}" if buy_price > 0 else "N/A",
                    'Gain/Loss %': f"{gain_loss_pct:.2f}%" if buy_price > 0 else "N/A"
                })
                
                total_value += current_value
                if cost_basis > 0:
                    total_cost += cost_basis
        except:
            pass
    
    if portfolio_data:
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True)
        
        # Summary
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Value", f"₹{total_value:.2f}")
        with col2:
            if total_cost > 0:
                st.metric("Total Cost", f"₹{total_cost:.2f}")
        with col3:
            if total_cost > 0:
                total_gain_loss = total_value - total_cost
                total_gain_loss_pct = (total_gain_loss / total_cost * 100)
                st.metric("Total Gain/Loss", f"₹{total_gain_loss:.2f}", f"{total_gain_loss_pct:.2f}%")
        
        # Remove stock option
        st.markdown("---")
        st.subheader("Remove Stock")
        ticker_to_remove = st.selectbox("Select stock to remove", [p[0] for p in portfolio])
        if st.button("Remove Selected Stock", type="secondary"):
            if db.remove_from_portfolio(st.session_state.user_id, ticker_to_remove):
                st.success(f"✅ {ticker_to_remove} removed from portfolio")
                st.rerun()
            else:
                st.error("Failed to remove stock")
    else:
        st.info("No stocks in your portfolio yet. Add one above!")
else:
    st.info("Your portfolio is empty. Add your first stock above!")

st.markdown("---")
if st.button("Logout"):
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.email = None
    st.session_state.tickers = []
    st.success("Logged out successfully!")
    st.rerun()
