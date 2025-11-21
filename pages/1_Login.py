import streamlit as st
import db

# Initialize database on first page load
db.ensure_db_initialized()

st.set_page_config(page_title="Login - Stock Dashboard", layout="centered")

st.title("📈 Stock Dashboard Login")

# Check if user is already logged in
if 'user_id' in st.session_state and st.session_state.user_id:
    st.success("You're already logged in!")
    st.info("Go to the Dashboard to view your stocks.")
    st.stop()

# Create tabs for login and registration
tab1, tab2 = st.tabs(["Login", "Register"])

with tab1:
    st.subheader("Login to Your Account")
    
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login", type="primary"):
        if username and password:
            success, user_data = db.login_user(username, password)
            if success:
                st.session_state.user_id = user_data[0]
                st.session_state.username = user_data[1]
                st.session_state.email = user_data[2]
                st.session_state.tickers = []
                st.success("Login successful! 🎉")
                st.balloons()
                st.rerun()
            else:
                st.error("Invalid username or password")
        else:
            st.warning("Please enter both username and password")

with tab2:
    st.subheader("Create a New Account")
    
    reg_username = st.text_input("Choose a username", key="reg_username")
    reg_email = st.text_input("Email address", key="reg_email")
    reg_password = st.text_input("Password", type="password", key="reg_password")
    reg_password_confirm = st.text_input("Confirm password", type="password", key="reg_password_confirm")
    
    if st.button("Register", type="primary"):
        if not reg_username:
            st.warning("Please enter a username")
        elif not reg_email:
            st.warning("Please enter an email")
        elif not reg_password:
            st.warning("Please enter a password")
        elif reg_password != reg_password_confirm:
            st.error("Passwords do not match")
        elif len(reg_password) < 6:
            st.error("Password must be at least 6 characters long")
        else:
            success, message = db.register_user(reg_username, reg_email, reg_password)
            if success:
                st.success(message)
                st.info("Now you can login with your credentials!")
            else:
                st.error(f"Registration failed: {message}")

st.markdown("---")
st.markdown("### Demo Credentials")
st.info("""
Try these demo credentials or create your own account:
- **Username:** demo
- **Password:** demo123

(First time? Register a new account above!)
""")
