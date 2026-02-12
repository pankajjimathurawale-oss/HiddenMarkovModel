# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from hmmlearn import hmm
from datetime import datetime

# ────────────────────────────────────────────────
#  Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty Gamma + HMM Strategy Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("Nifty Gamma Exposure + HMM Trading Strategy Backtest")
st.markdown("Upload your `gamma_values-old.csv` file and test different HMM configurations.")

# ────────────────────────────────────────────────
#  File uploader
# ────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload gamma_values-old.csv", type=["csv"])

if uploaded_file is not None:
    # ────────────────────────────────────────────────
    #  Load & prepare data
    # ────────────────────────────────────────────────
    with st.spinner("Loading and preparing data..."):
        df = pd.read_csv(uploaded_file)
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%m/%d/%Y', errors='coerce')
        df = df.sort_values('TIMESTAMP').dropna(subset=['TIMESTAMP', 'NiftyClose', 'GEX'])

        df['Returns']     = df['NiftyClose'].pct_change().fillna(0)
        df['Volatility']  = df['Returns'].rolling(window=5).std().fillna(0)
        df['GEX_norm']    = (df['GEX'] - df['GEX'].mean()) / df['GEX'].std()

    st.success(f"Data loaded — {len(df)} rows from {df['TIMESTAMP'].min().date()} to {df['TIMESTAMP'].max().date()}")

    # ────────────────────────────────────────────────
    #  User controls
    # ────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2,2,1])

    with col1:
        n_states = st.selectbox(
            "Number of Hidden States",
            options=[2, 3, 4],
            index=1,
            help="More states = potentially better regime detection but higher risk of overfitting"
        )

    with col2:
        feature_options = {
            "Only Returns":          ["Returns"],
            "Only Volatility":       ["Volatility"],
            "Only Normalized GEX":   ["GEX_norm"],
            "Returns + Volatility":  ["Returns", "Volatility"],
            "Returns + GEX_norm":    ["Returns", "GEX_norm"],
            "Volatility + GEX_norm": ["Volatility", "GEX_norm"],
            "All three":             ["Returns", "Volatility", "GEX_norm"]
        }

        selected_preset = st.selectbox(
            "Observation Features",
            options=list(feature_options.keys()),
            index=4  # default = Returns + GEX_norm
        )

        selected_features = feature_options[selected_preset]

    with col3:
        run_button = st.button("Run HMM Backtest", type="primary", use_container_width=True)

    # ────────────────────────────────────────────────
    #  Run button logic
    # ────────────────────────────────────────────────
    if run_button:
        with st.spinner(f"Fitting {n_states}-state Gaussian HMM using {selected_features} ..."):
            # Prepare observation matrix
            obs = df[selected_features].values

            # Fit HMM with hmmlearn (much faster and more stable)
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=100,               # usually enough
                init_params="stmc",
                verbose=False
            )

            try:
                model.fit(obs)
                states = model.predict(obs)
            except Exception as e:
                st.error(f"HMM fitting failed: {str(e)}")
                st.stop()

        df_result = df.copy()
        df_result['State'] = states

        # ────────────────────────────────────────────────
        #  Build strategy
        # ────────────────────────────────────────────────
        df_result['Strategy_Return'] = 0.0

        mean_ret_by_state = df_result.groupby('State')['Returns'].mean()
        if len(mean_ret_by_state) < 2:
            st.warning("Could not identify distinct bull/bear states.")
            bull_state = bear_state = mean_ret_by_state.index[0]
        else:
            bull_state  = mean_ret_by_state.idxmax()
            bear_state  = mean_ret_by_state.idxmin()

        df_result.loc[df_result['State'] == bull_state, 'Strategy_Return'] = df_result['Returns']
        df_result.loc[df_result['State'] == bear_state, 'Strategy_Return'] = -df_result['Returns']

        if n_states > 2:
            neutral = set(range(n_states)) - {bull_state, bear_state}
            for ns in neutral:
                df_result.loc[df_result['State'] == ns, 'Strategy_Return'] = 0.0

        # Cumulative returns
        df_result['Cum_BH']       = (1 + df_result['Returns']).cumprod()
        df_result['Cum_Strategy'] = (1 + df_result['Strategy_Return']).cumprod()

        # Performance metrics
        bh_total    = df_result['Cum_BH'].iloc[-1] - 1
        strat_total = df_result['Cum_Strategy'].iloc[-1] - 1

        bh_mean     = df_result['Returns'].mean()
        bh_std      = df_result['Returns'].std()
        bh_sharpe   = bh_mean / bh_std * np.sqrt(252) if bh_std != 0 else 0

        strat_mean  = df_result['Strategy_Return'].mean()
        strat_std   = df_result['Strategy_Return'].std()
        strat_sharpe = strat_mean / strat_std * np.sqrt(252) if strat_std != 0 else 0

        trades = (df_result['State'].diff() != 0).sum()

        # ────────────────────────────────────────────────
        #  Display results
        # ────────────────────────────────────────────────
        st.subheader("Performance Summary")

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Strategy Total Return", f"{strat_total:.2%}")
        colB.metric("Buy & Hold Total Return", f"{bh_total:.2%}")
        colC.metric("Strategy Sharpe", f"{strat_sharpe:.2f}")
        colD.metric("Regime Changes (Trades)", trades)

        # ────────────────────────────────────────────────
        #  Equity curve
        # ────────────────────────────────────────────────
        st.subheader("Equity Curve")

        fig_curve = go.Figure()

        fig_curve.add_trace(go.Scatter(
            x=df_result['TIMESTAMP'],
            y=df_result['Cum_BH'],
            name="Buy & Hold",
            line=dict(color='gray')
        ))

        fig_curve.add_trace(go.Scatter(
            x=df_result['TIMESTAMP'],
            y=df_result['Cum_Strategy'],
            name="HMM Strategy",
            line=dict(color='#00CC96')
        ))

        fig_curve.update_layout(
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_curve, use_container_width=True)

        # ────────────────────────────────────────────────
        #  Regime timeline
        # ────────────────────────────────────────────────
        st.subheader("Detected Market Regimes")

        color_map = {bull_state: "#00CC96", bear_state: "#FF4B4B"}
        if n_states > 2:
            neutral_colors = ["#FFD700", "#FFA500", "#9370DB"]
            for i, ns in enumerate(neutral):
                color_map[ns] = neutral_colors[i % len(neutral_colors)]

        fig_regime = px.line(
            df_result,
            x='TIMESTAMP',
            y='NiftyClose',
            color='State',
            color_discrete_map=color_map,
            title=f"{n_states}-state HMM Regimes • Bull = {bull_state} • Bear = {bear_state}"
        )

        fig_regime.update_traces(line=dict(width=2.5))
        fig_regime.update_layout(height=450)

        st.plotly_chart(fig_regime, use_container_width=True)

        # ────────────────────────────────────────────────
        #  Raw table (collapsible)
        # ────────────────────────────────────────────────
        with st.expander("View full backtest table (last 300 rows)", expanded=False):
            st.dataframe(
                df_result[['TIMESTAMP', 'NiftyClose', 'GEX', 'Returns', 'State', 'Strategy_Return']]
                .tail(300)
                .style.format({
                    'Returns': '{:.4f}',
                    'Strategy_Return': '{:.4f}'
                })
            )

else:
    st.info("Please upload your gamma_values-old.csv file to begin.")
