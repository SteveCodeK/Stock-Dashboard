import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import requests
from transformers import pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import time
from openai import OpenAI
from twilio.rest import Client
from google.oauth2 import service_account


#d202c30a899a4e1e9c731012e9b3987c

NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", 
          "https://www.googleapis.com/auth/drive"]

creds = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=SCOPES
)
client = gspread.authorize(creds)
# Open Spreadsheet and Worksheet
spreadsheet = client.open("Finacial Dashboard")
worksheet = spreadsheet.worksheet("SNOW")

# Fetch data
data = worksheet.get_all_records()
df = pd.DataFrame(data)
df.replace({'#N/A': None, 'N/A': None, 'na': None}, inplace=True)

def ai_detect_sell_buy(df):
    st.subheader("🧠 AI Buy/Sell Detector (SMA Crossover)")

    # Ensure data is sorted and numeric
    df = df.sort_values("Date")
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    df["SMA 20"] = df["Close"].rolling(window=20).mean()
    df["SMA 50"] = df["Close"].rolling(window=50).mean()

    # Session state to track last action
    if "last_action" not in st.session_state:
        st.session_state.last_action = "none"

    # Detect crossover (latest signal)
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    signal = "hold"
    if prev_row["SMA 20"] < prev_row["SMA 50"] and last_row["SMA 20"] > last_row["SMA 50"]:
        signal = "buy"
    elif prev_row["SMA 20"] > prev_row["SMA 50"] and last_row["SMA 20"] < last_row["SMA 50"]:
        signal = "sell"

    # Display signal
    st.write(f"📌 Current AI Signal: **{signal.upper()}**")

    # Action buttons
    st.markdown("### ➕ Update Your Position")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ I Bought"):
            st.session_state.last_action = "buy"
            st.success("Position updated to BUY.")
    with col2:
        if st.button("❌ I Sold"):
            st.session_state.last_action = "sell"
            st.success("Position updated to SELL.")
    with col3:
        if st.button("🔁 Reset"):
            st.session_state.last_action = "none"
            st.info("Position reset.")

    # Notification based on AI signal and user state
    if st.session_state.last_action == "buy" and signal == "sell":
        st.warning("🔔 AI recommends SELL based on your BUY position.")
    elif st.session_state.last_action == "sell" and signal == "buy":
        st.success("🔔 AI recommends BUY based on your SELL position.")
    elif st.session_state.last_action == "none":
        st.info("ℹ️ No position set. Use the buttons above to track your status.")


# Initialize sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Replace with your News API key

def ai_news_sentiment():
    st.subheader("📰 AI News Sentiment Analysis")

    company_name = "Snowflake"  # Or allow user input

    # Fetch news articles
    url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])[:5]  # Limit to 5 articles

    if not articles:
        st.warning("No news articles found.")
        return

    for article in articles:
        title = article["title"]
        description = article["description"] or ""
        content = f"{title}. {description}"
        sentiment = sentiment_analyzer(content[:512])[0]  # Truncate if needed

        sentiment_label = sentiment["label"]
        confidence = sentiment["score"]

        if sentiment_label == "POSITIVE":
            st.success(f"🟢 **{title}**")
        elif sentiment_label == "NEGATIVE":
            st.error(f"🔴 **{title}**")
        else:
            st.info(f"⚪ **{title}**")

        st.write(f"**Sentiment:** {sentiment_label} ({confidence:.2f})")
        st.caption(description)
        st.divider()


def ai_price_prediction():
    st.subheader("📈 AI Price Prediction (Next 5 Days)")

    # Load the latest dataframe
    df_pred = df.copy()
    df_pred = df_pred.sort_values("Date")

    # Prepare data
    df_pred["Close"] = pd.to_numeric(df_pred["Close"], errors="coerce")
    df_pred = df_pred.dropna(subset=["Close"])

    df_pred["Days"] = np.arange(len(df_pred))  # Treat index as time (simplified)

    # Train model
    model = LinearRegression()
    model.fit(df_pred[["Days"]], df_pred["Close"])

    # Predict next 5 days
    future_days = np.arange(len(df_pred), len(df_pred) + 5).reshape(-1, 1)
    predicted_prices = model.predict(future_days)

    # Display predictions
    for i, price in enumerate(predicted_prices, start=1):
        st.write(f"🔮 Day {i}: **${price:.2f}**")

    # Summary
    st.markdown("### 📋 Summary")
    st.info("This 5-day price prediction is based on a linear trend using historical closing prices.")

    # Influencing factors (static list for now)
    st.markdown("### 📌 Factors That May Affect Price")
    st.write("""
    - Recent news and sentiment (see News section)
    - Overall market trend (e.g., NASDAQ/S&P500)
    - Earnings reports and financial statements
    - Analyst upgrades/downgrades
    - Macroeconomic indicators (interest rates, inflation, etc.)
    """)

# Initialize OpenAI client (at the top of your script or in __main__)

def ai_chatbot():
    st.subheader("💬 Ask AI About SNOW Stock")

    user_input = st.text_input("Ask me anything about SNOW (e.g., 'Is SNOW a good buy now?')")

    if st.button("Ask AI") and user_input:
        # Fetch recent news (optional)
        news_summary = ""
        try:
            url = f"https://newsapi.org/v2/everything?q=Snowflake&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
            response = requests.get(url)
            articles = response.json().get("articles", [])[:3]
            news_summary = "\n".join([f"- {a['title']}" for a in articles])
        except:
            news_summary = "No news available or API key missing."

        # Build prompt
        prompt = f"""
        You are a stock analyst assistant. The user has asked a question about the company Snowflake (ticker: SNOW).
        Here are the 3 most recent news headlines:
        {news_summary}

        Question: {user_input}
        Answer in a clear, friendly way.
        """

        try:
            with st.spinner("🤖 Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content
                st.success("📡 AI Response:")
                st.write(answer)

                if st.checkbox("📲 Send this answer to WhatsApp"):
                    send_whatsapp_message(answer)

        except Exception as e:
            st.error(f"❌ AI Error: {e}")

# Format and calculate KPIs
def display_summary_kpis(df):
    st.title("📊 SNOW Summary Statistics")

    # Convert values
    highest_close = df["Close"].max()
    lowest_close = df["Close"].min()
    avg_volume = df["Volume"].mean()
    start_price = df.iloc[0]["Close"]
    end_price = df.iloc[-1]["Close"]
    pct_change = ((end_price - start_price) / start_price) * 100

    st.metric("🔼 Highest Close", f"${highest_close:.2f}")
    st.metric("🔽 Lowest Close", f"${lowest_close:.2f}")
    st.metric("📊 Avg Volume", f"{avg_volume:,.0f}")
    st.metric("📈 % Change", f"{pct_change:.2f}%")

def time_line_graph(df):
    st.subheader("📅 SNOW Stock Timeline")

    # Convert 'Date' to datetime for accurate plotting
    
    # Line chart for Open, Close, High, Low
    fig_price = px.line(df, x="Date", y=["Open", "Close", "High", "Low"],
                        title="Open, Close, High, and Low Prices Over Time",
                        labels={"value": "Price ($)", "variable": "Price Type"})
    fig_price.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
    
    # Bar chart for Volume Over Time
    fig_volume = px.bar(df, x="Date", y="Volume", 
                        title="Trading Volume Over Time",
                        labels={"Volume": "Volume", "Date": "Date"})
    fig_volume.update_layout(xaxis_title="Date", yaxis_title="Volume")
    
    # Display graphs
    st.plotly_chart(fig_price)
    st.plotly_chart(fig_volume)


def candlestick_chart(df):
    st.subheader("📉 Candlestick Chart")

    # Convert Date to datetime and ensure numeric columns
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create candlestick chart using Plotly
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color='green',
        decreasing_line_color='red',
        name="Candlestick"
    )])

    fig.update_layout(
        title="SNOW Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig)
def moving_average(df):
    st.subheader("📈 Moving Averages")

    # Ensure proper types

    # Sort by date
    df = df.sort_values("Date")

    # Calculate moving averages
    df["SMA 20"] = df["Close"].rolling(window=20).mean()
    df["SMA 50"] = df["Close"].rolling(window=50).mean()

    # Plot with Plotly
    fig = px.line(df, x="Date", y=["Close", "SMA 20", "SMA 50"],
                  labels={"value": "Price ($)", "variable": "Legend"},
                  title="Closing Price with 20 & 50-Day Moving Averages")
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig)

def volatility_indicator(df):
    st.subheader("🌪️ Volatility Indicator (Daily Range)")

    # Ensure proper types

    # Calculate daily range
    df["Daily Range"] = df["High"] - df["Low"]

    # Plot bar chart of daily range
    fig = px.bar(df, x="Date", y="Daily Range",
                 title="Daily Price Range (High - Low)",
                 labels={"Daily Range": "Price Range ($)"})
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Price Range ($)")
    st.plotly_chart(fig)
    
def interactive_filters(df):
    st.sidebar.header("🔧 Filters")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')

    # Default date range: current year
    current_year = datetime.now().year
    default_start = pd.Timestamp(f"{current_year}-01-01")
    default_end = pd.Timestamp(f"{current_year}-12-31")
    
    # Clip to available data range
    min_date = df["Date"].min()
    max_date = df["Date"].max()

    default_start = max(default_start, min_date)
    default_end = min(default_end, max_date)

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [default_start, default_end],
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

    # Volume filter
    volume_threshold = st.sidebar.number_input("Minimum Daily Volume", min_value=0, value=0, step=100000)
    df = df[df["Volume"] >= volume_threshold]

    # Price series selector
    series_options = ["Open", "Close", "High", "Low"]
    selected_series = st.sidebar.multiselect("Price Series to Show", series_options, default=series_options)

    return df, selected_series



if __name__ == "__main__":
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Fetch raw data
    df = pd.DataFrame(data)

    # Apply filters
    df_filtered, selected_series = interactive_filters(df)

    # --- Layout: 3 Columns ---
    col1, col2, col3 = st.columns(3)

    # Column 1: Summary KPIs (vertically)
    with col1:
        with st.expander("📊 SNOW Summary Statistics", expanded=True):
            display_summary_kpis(df_filtered)
            

    # Column 2: Timeline charts (line + volume)
    with col2:
        with st.expander("📅 Stock Timeline", expanded=True):
            time_line_graph(df_filtered)
        

    # Column 3: Candlestick + Moving Averages
    with col3:
        with st.expander("📉 Candlestick Chart", expanded=True):
            candlestick_chart(df_filtered)

        with st.expander("📈 Moving Averages"):
            moving_average(df_filtered)

    # Below columns: Volatility
    with st.expander("🌪️ Volatility Indicator (Daily Range)"):
        volatility_indicator(df_filtered)


    with st.expander("🧠 AI Detection (Buy/Sell Signals)", expanded=True):
        ai_detect_sell_buy(df_filtered)

    with st.expander("📰 AI News Sentiment Analysis", expanded=True):
        ai_news_sentiment()

    with st.expander("📈 AI Price Prediction", expanded=True):
        ai_price_prediction()

    with st.expander("💬 Ask AI About the Stock", expanded=True):
        ai_chatbot()
