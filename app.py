# app.py     ──  Streamlit version ──  Walk-forward / no look-ahead bias

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from hmmlearn import hmm
from datetime import datetime

st.set_page_config(page_title="HMM Gamma Strategy (No Leakage)", layout="wide")

st.title("Nifty Gamma + HMM Strategy – Walk-Forward (No Look-Ahead Bias)")
st.markdown("Fits model using **only past data**. Bull/bear labels also based on past only.")

# ────────────────────────────────────────────────
# Upload
# ────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload gamma_values-old.csv", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Loading data..."):
        df = pd.read_csv(uploaded_file)
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%m/%d/%Y', errors='coerce')
        df = df.sort_values('TIMESTAMP').dropna(subset=['TIMESTAMP', 'NiftyClose', 'GEX'])

        df['Returns']    = df['NiftyClose'].pct_change().fillna(0)
        df['Volatility'] = df['Returns'].rolling(5).std().fillna(0)
        # GEX_norm computed in walk-forward loop

    st.success(f"Loaded {len(df)} rows  •  {df['TIMESTAMP'].min().date()} → {df['TIMESTAMP'].max().date()}")

    # ────────────────────────────────────────────────
    # Controls
    # ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        n_states = st.selectbox("Number of hidden states", [2, 3, 4], index=1)

    with col2:
        feature_preset = st.selectbox(
            "Features",
            [
                "Only Returns",
                "Only Volatility",
                "Only GEX_norm",
                "Returns + Volatility",
                "Returns + GEX_norm",
                "Volatility + GEX_norm",
                "All three"
            ],
            index=4
        )

        feature_map = {
            "Only Returns": ["Returns"],
            "Only Volatility": ["Volatility"],
            "Only GEX_norm": ["GEX_norm"],
            "Returns + Volatility": ["Returns", "Volatility"],
            "Returns + GEX_norm": ["Returns", "GEX_norm"],
            "Volatility + GEX_norm": ["Volatility", "GEX_norm"],
            "All three": ["Returns", "Volatility", "GEX_norm"]
        }
        selected_features = feature_map[feature_preset]

    with col3:
        warmup_days = st.number_input("Warm-up period (days)", min_value=50, max_value=400, value=200, step=50)

    with col4:
        run_button = st.button("Run Walk-Forward Backtest", type="primary", use_container_width=True)

    if run_button:
        with st.spinner(f"Running walk-forward backtest ({n_states} states, {len(selected_features)} features)..."):
            df_result = df.copy()
            df_result['State'] = np.nan
            df_result['Strategy_Return'] = 0.0
            df_result['Cum_BH'] = (1 + df_result['Returns']).cumprod()
            df_result['Cum_Strategy'] = 1.0

            has_gex = 'GEX_norm' in selected_features

            for t in range(warmup_days, len(df_result)):
                past = df_result.iloc[:t].copy()

                # Normalize GEX using only past data
                if has_gex:
                    past_mean = past['GEX'].mean()
                    past_std  = past['GEX'].std() or 1.0
                    past['GEX_norm'] = (past['GEX'] - past_mean) / past_std
                    today_gex_norm = (df_result.iloc[t]['GEX'] - past_mean) / past_std

                past_obs = past[selected_features].values

                # Fit model on past only
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=80,
                    init_params="stmc",
                    verbose=False
                )

                try:
                    model.fit(past_obs)
                except:
                    continue  # skip if fit fails

                # Get past states to compute historical means
                past_states = model.predict(past_obs)
                past['State'] = past_states

                # Decide bull/bear using PAST returns only
                mean_ret = past.groupby('State')['Returns'].mean()

                if len(mean_ret) < 2:
                    strategy_ret_today = 0.0
                else:
                    bull_state = mean_ret.idxmax()
                    bear_state = mean_ret.idxmin()

                    # Today's observation
                    today_row = df_result.iloc[t:t+1].copy()
                    if has_gex:
                        today_row['GEX_norm'] = today_gex_norm
                    today_obs = today_row[selected_features].values

                    # Predict today's state
                    today_state = model.predict(today_obs)[0]

                    if today_state == bull_state:
                        strategy_ret_today = df_result.iloc[t]['Returns']
                    elif today_state == bear_state:
                        strategy_ret_today = -df_result.iloc[t]['Returns']
                    else:
                        strategy_ret_today = 0.0

                    df_result.iloc[t, df_result.columns.get_loc('State')] = today_state

                df_result.iloc[t, df_result.columns.get_loc('Strategy_Return')] = strategy_ret_today

                # Update cumulative
                prev = df_result.iloc[t-1]['Cum_Strategy']
                df_result.iloc[t, df_result.columns.get_loc('Cum_Strategy')] = prev * (1 + strategy_ret_today)

        # ────────────────────────────────────────────────
        # Results
        # ────────────────────────────────────────────────
        oos = df_result.iloc[warmup_days:]

        bh_total  = oos['Cum_BH'].iloc[-1] / oos['Cum_BH'].iloc[0] - 1
        strat_total = oos['Cum_Strategy'].iloc[-1] / oos['Cum_Strategy'].iloc[0] - 1

        bh_sharpe    = oos['Returns'].mean() / oos['Returns'].std() * np.sqrt(252) if oos['Returns'].std() else 0
        strat_sharpe = oos['Strategy_Return'].mean() / oos['Strategy_Return'].std() * np.sqrt(252) if oos['Strategy_Return'].std() else 0

        trades = (oos['State'].diff() != 0).sum()

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Strategy Return (OOS)", f"{strat_total:.2%}")
        colB.metric("Buy & Hold Return (OOS)", f"{bh_total:.2%}")
        colC.metric("Strategy Sharpe", f"{strat_sharpe:.2f}")
        colD.metric("Trades", int(trades))

        # Equity curve
        st.subheader("Equity Curve (after warm-up)")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=oos['TIMESTAMP'], y=oos['Cum_BH'],
            name="Buy & Hold", line=dict(color='gray')
        ))

        fig.add_trace(go.Scatter(
            x=oos['TIMESTAMP'], y=oos['Cum_Strategy'],
            name="Strategy", line=dict(color='#00CC96')
        ))

        fig.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Regime plot
        st.subheader("Detected States (after warm-up)")

        fig_reg = px.line(
            oos, x='TIMESTAMP', y='NiftyClose', color='State',
            title=f"{n_states}-state regimes (walk-forward)"
        )
        fig_reg.update_traces(line_width=2.2)
        st.plotly_chart(fig_reg, use_container_width=True)

        # Table preview
        with st.expander("Last 200 rows of results"):
            st.dataframe(
                oos[['TIMESTAMP', 'NiftyClose', 'Returns', 'State', 'Strategy_Return']]
                .tail(200)
                .style.format({
                    'Returns': '{:.4f}',
                    'Strategy_Return': '{:.4f}'
                })
            )

else:
    st.info("Please upload your gamma_values-old.csv file.")
