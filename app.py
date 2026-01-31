import streamlit as st
import pandas as pd
import plotly.express as px
from train_model import run_pipeline
import plotly.graph_objects as go

if "model_ran" not in st.session_state:
    st.session_state.model_ran = False


st.set_page_config(
    page_title="Stock Predictor",
    layout="wide"
)

st.title("Stock Return Prediction Dashboard")
stock_categories = {
    "US Stocks": [
        "MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"
    ],
    "Canadian Stocks": [
        "SHOP", "RY", "TD", "BNS", "BMO", "ENB", "CNQ", "CP", "CNR", "SU"
    ],
    "Indian Stocks": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS", "BHARTIARTL.NS"
    ],
    "Canadian ETFs": [
        "XIU.TO", "VCN.TO", "VFV.TO", "XIC.TO", "ZCN.TO"
    ]
}


category = st.selectbox(
    "Select a market",
    list(stock_categories.keys())
)
stock_symbol = st.selectbox(
    "Select a stock or ETF",
    stock_categories[category],
    on_change=lambda: st.session_state.update({"model_ran": False})
)




if st.button("Run Model"):
    with st.spinner("Running model"):
        run_pipeline(stock_symbol)
    st.session_state.model_ran = True
    st.success("Model finished")


def load_data(symbol):
    feature_df = pd.read_json(f"outputs/feature_importance_{symbol}.json")
    metrics_df = pd.read_json(f"outputs/metrics_{symbol}.json")
    stock_info = pd.read_json(
        f"outputs/stock_info_{symbol}.json"
    ).iloc[0]
    return feature_df, metrics_df, stock_info


def load_price_data(symbol):
    df = pd.read_csv(f"data/{symbol}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

if st.session_state.model_ran:
    try:
        feature_df, metrics_df, stock_info = load_data(stock_symbol)

        # ===============================
        # STOCK OVERVIEW
        # ===============================
        st.subheader("Stock Overview")
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Company", stock_info["company"])
        c2.metric("Symbol", stock_info["symbol"])
        c3.metric("Last Close", f"{stock_info['last_close']:.2f}")
        c4.metric("Last Trading Day", stock_info["last_date"])

        st.divider()

        # ===============================
        # PRICE TREND
        # ===============================
        st.subheader("Price Trend with Peaks and Drops")

        price_df = load_price_data(stock_symbol)

        price_df = load_price_data(stock_symbol)

        if price_df is None or price_df.empty:
            st.info("Price data is not available yet.")
        else:
            price_df["Close"] = pd.to_numeric(price_df["Close"], errors="coerce")
            price_df.dropna(subset=["Close"], inplace=True)

            price_df = price_df.sort_values("Date")

            price_df["price_change"] = price_df["Close"].diff()

            price_df["peak"] = (
                (price_df["price_change"] > 0) &
                (price_df["price_change"].shift(-1) < 0)
            )

            price_df["drop"] = (
                (price_df["price_change"] < 0) &
                (price_df["price_change"].shift(-1) > 0)
            )

            fig_price = px.line(
                price_df,
                x="Date",
                y="Close",
                title="Closing Price Over Time"
            )

            fig_price.add_scatter(
                x=price_df.loc[price_df["peak"], "Date"],
                y=price_df.loc[price_df["peak"], "Close"],
                mode="markers",
                name="Peaks"
            )

            fig_price.add_scatter(
                x=price_df.loc[price_df["drop"], "Date"],
                y=price_df.loc[price_df["drop"], "Close"],
                mode="markers",
                name="Drops"
            )

            fig_price.update_layout(
                xaxis_title="Date",
                yaxis_title="Closing Price",
                height=500
            )

            st.plotly_chart(fig_price, use_container_width=True)


            st.divider()

            # ===============================
            # MODEL PERFORMANCE
            # ===============================
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)

            col1.metric("MAE", round(metrics_df["mae"][0], 4))
            col2.metric("R2 Score", round(metrics_df["r2"][0], 4))
            col3.metric(
                "Directional Accuracy",
                f"{metrics_df['directional_accuracy'][0] * 100:.2f} percent"
            )
            recent_return = price_df["Close"].pct_change().iloc[-1]
            volatility = price_df["Close"].pct_change().rolling(5).std().iloc[-1]
            directional_acc = metrics_df["directional_accuracy"][0]

            if recent_return > 0 and directional_acc > 0.5:
                outlook = "bullish"
            elif recent_return < 0 and directional_acc > 0.5:
                outlook = "bearish"
            else:
                outlook = "uncertain"

            st.divider()
        # ===============================
        # CONFIDENCE SCORE
        # ===============================
        base_confidence = directional_acc * 100

        # Reduce confidence if volatility is high
        if volatility > 0.03:
            base_confidence -= 15
        elif volatility > 0.02:
            base_confidence -= 10
        elif volatility > 0.01:
            base_confidence -= 5

        # Reduce confidence if outlook is uncertain
        if outlook == "uncertain":
            base_confidence -= 15

        # Clamp between 0 and 100
        confidence_score = max(0, min(100, round(base_confidence)))
        st.subheader("Confidence Meter")

        fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence_score,
                number={"suffix": " percent"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 40], "color": "#8b0000"},
                        {"range": [40, 70], "color": "#f4a261"},
                        {"range": [70, 100], "color": "#2a9d8f"}
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": confidence_score
                    }
                }
            ))

        fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=10)
            )

        st.plotly_chart(fig_gauge, use_container_width=True)
        if confidence_score >= 70:
            st.caption("High confidence in the short term signal")
        elif confidence_score >= 50:
            st.caption("Moderate confidence in the short term signal")
        else:
            st.caption("Low confidence due to uncertainty or volatility")




        st.subheader("My Model Interpretation")

        if outlook == "bullish":
                st.success(
                    "My view: The short term trend looks slightly positive.\n\n"
        "What I am seeing is that the stock has moved up recently, and the model has "
        "been reasonably good at predicting direction in similar situations. "
        "Because both of these signals line up, I would expect some continued upward "
        "movement in the near term.\n\n"
        "However, the market is still volatile. This is not a guarantee or a price target, "
        "just a data driven signal based on recent behaviour."
                )

        elif outlook == "bearish":
                st.warning(
                    "My view: The short term trend looks slightly negative.\n\n"
        "The stock has dropped recently, and the model has shown decent accuracy when "
        "identifying downward movements in the past. Based on this combination, "
        "there is a higher chance of continued weakness in the near term.\n\n"
        "That said, this is not a prediction of a major decline. It is simply a cautious "
        "signal given recent price action and current volatility."
                )

        else:
                st.info(
                    "My view: The outlook is unclear right now.\n\n"
        "The recent price movement and the model’s past accuracy do not strongly support "
        "either an upward or downward trend. Because of this, I do not have high confidence "
        "in a short term directional move.\n\n"
        "In my view, this suggests the stock may move sideways or remain unpredictable "
        "until clearer signals appear."
                )

        st.caption("This interpretation reflects short term signals and should not be taken as financial advice.")

            # ===============================
            # FEATURE IMPORTANCE
            # ===============================
        st.subheader("Feature Importance")

        fig = px.bar(
                feature_df.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues"
            )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading results: {e}")

else:
    st.info("Click Run Model to generate results")
