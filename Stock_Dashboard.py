import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import time

# --- CONFIG ---
st.set_page_config(page_title="📈 Stock Dashboard", layout="wide")

# --- AUTO REFRESH SETUP ---
REFRESH_INTERVAL = 60 # seconds
now = time.time()

# --- SESSION STATE SETUP ---
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = now

# --- REFRESH BUTTON ---
refresh_clicked = st.button("🔄 Refresh Data", key="manual_refresh")

if refresh_clicked or (now - st.session_state.last_refresh > REFRESH_INTERVAL):
    st.session_state.last_refresh = now
    st.rerun()

# --- GOOGLE SHEETS CONNECTION ---
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "Credentials.json"

creds = Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
client = gspread.authorize(creds)

# Open Spreadsheet and Worksheet
spreadsheet = client.open("Finacial Dashboard")
worksheet = spreadsheet.worksheet("finance")

# Fetch data
data = worksheet.get_all_records()
df = pd.DataFrame(data)

# --- STREAMLIT DASHBOARD ---
st.title("📈 Stock Dashboard")
st.markdown("### Company Basics")

def table_company_basics():
    st.title("Company Basics")

    # Extract relevant fields
    basics = df[["Company Name", "Previous Day Price", "Last Price", "Change", "Change Pct"]].copy()
    basics.rename(columns={
        "Previous Day Price": "Previous Price",
        "Last Price": "Current Price",
        "Change": "Price Change",
        "Change Pct": "Change (%)"
    }, inplace=True)

    # Format the Change (%) as a percentage string
    basics["Change (%)"] = basics["Change (%)"].map(lambda x: f"{x:.2f}%")

    st.dataframe(basics)


def table_trading_activity():
    st.subheader("📈 Trading Activity")
    try:
        activity = df[[
            "Ticker", "Volume", "Volume Avg", "Day High", "Last Price", "Day Low", "Dividend Yield"
        ]].copy()
    except KeyError:
        # Fallback if 'Dividend Yield' is not available
        st.warning("⚠️ 'Dividend Yield' column not found. Omitting from display.")
        activity = df[[
            "Ticker", "Volume", "Volume Avg", "Day High", "Last Price", "Day Low"
        ]].copy()

    activity.rename(columns={
        "Last Price": "Current Price",
        "Volume Avg": "Average Volume"
    }, inplace=True)
    st.dataframe(activity)


def table_company_fiance():
    st.subheader("💰 Company Financials")

    def format_market_cap(value):
        try:
            value = float(value)
            if value >= 1_000_000_000_000:
                return f"${value / 1_000_000_000_000:.2f}T"
            elif value >= 1_000_000_000:
                return f"${value / 1_000_000_000:.2f}B"
            elif value >= 1_000_000:
                return f"${value / 1_000_000:.2f}M"
            else:
                return f"${value:,.0f}"
        except:
            return "N/A"

    try:
        finance = df[["Ticker", "Market Cap", "P/E Ratio", "EPS"]].copy()
        finance["Market Cap"] = finance["Market Cap"].apply(format_market_cap)
        st.dataframe(finance)
    except KeyError as e:
        st.error(f"Missing expected column: {e}")

# Streamlit app execution
if __name__ == "__main__":
    table_company_basics()
    table_trading_activity()
    table_company_fiance()
    
