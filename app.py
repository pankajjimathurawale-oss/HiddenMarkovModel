import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import streamlit as st
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
# DQN Network
# ────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_dim=6, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)

# ────────────────────────────────────────────────
# Replay Buffer
# ────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

# ────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────
st.title("DQN RL Trading Simulator – Sharpe Reward Shaping")

uploaded_file = st.file_uploader("Upload gamma_values-old.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['TIMESTAMP'], date_format='%m/%d/%Y')
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)

    # ────────────────────────────────────────────────
    # Feature engineering – lagged to prevent look-ahead bias
    # ────────────────────────────────────────────────
    df['prev_return'] = df['NiftyClose'].pct_change().fillna(0)
    df['vol_5d'] = df['prev_return'].rolling(5).std().fillna(0)

    df['lag_GEX'] = df['GEX'].shift(1)
    df['lag_Gamma_Flip'] = df['Gamma Flip'].shift(1)
    df['lag_NiftyClose'] = df['NiftyClose'].shift(1)
    df['lag_Max_SuperGamma'] = df['Max SuperGamma'].shift(1)
    df['lag_DTE'] = df['DTE'].shift(1)

    df['GEX_norm'] = df['lag_GEX'] / 1e6
    df['flip_prox'] = (df['lag_NiftyClose'] - df['lag_Gamma_Flip']) / df['lag_NiftyClose']
    df['superg_spread'] = (df['lag_Max_SuperGamma'] - df['lag_Gamma_Flip']) / 100
    df['dte_norm'] = df['lag_DTE'] / 10

    state_cols = ['GEX_norm', 'flip_prox', 'superg_spread', 'dte_norm', 'prev_return', 'vol_5d']

    df[state_cols] = df[state_cols].fillna(0)
    df[state_cols] = (df[state_cols] - df[state_cols].mean()) / df[state_cols].std()

    # Intraday return & market benchmark
    df['intraday_return'] = (df['NiftyClose'] - df['NiftyOpen']) / df['NiftyOpen']
    df['intraday_return'] = df['intraday_return'].fillna(0)
    df['market_intraday'] = df['intraday_return']

    # Split
    train_df = df[df['TIMESTAMP'] < '2025-07-01'].copy()
    test_df = df[df['TIMESTAMP'] >= '2025-07-01'].copy()

    st.write(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    # ────────────────────────────────────────────────
    # DQN Parameters
    # ────────────────────────────────────────────────
    state_dim = len(state_cols)
    n_actions = 3
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.02
    epsilon_decay = 0.998
    batch_size = 64
    target_update_freq = 500
    lr = 5e-5
    replay_capacity = 20000
    n_episodes = 150
    position_size = 0.1
    transaction_cost = 0.0002
    sharpe_window = 20
    sharpe_scale = 10.0
    epsilon_div = 1e-6

    if st.button("Train DQN Model"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy_net = DQN(state_dim, n_actions).to(device)
        target_net = DQN(state_dim, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        memory = ReplayBuffer(replay_capacity)

        epsilon = epsilon_start
        steps = 0
        episode_rewards = []

        progress_bar = st.progress(0)

        for episode in range(n_episodes):
            state = train_df[state_cols].iloc[0].values
            total_reward = 0
            position = 0

            for t in range(len(train_df) - 1):
                steps += 1

                if random.random() < epsilon:
                    action = random.randint(0, n_actions - 1)
                else:
                    with torch.no_grad():
                        q_values = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                        action = q_values.argmax().item()

                new_position = 0 if action == 0 else (position_size if action == 1 else -position_size)
                pos_change = abs(new_position - position)

                # Sharpe-shaped reward
                window_start = max(0, t - sharpe_window + 1)
                window_excess = new_position * (train_df['intraday_return'].iloc[window_start:t+1] - train_df['market_intraday'].iloc[window_start:t+1])
                mean_excess = np.mean(window_excess)
                std_excess = np.std(window_excess) + epsilon_div
                sharpe_reward = mean_excess / std_excess
                reward = sharpe_reward * sharpe_scale - transaction_cost * pos_change
                reward = np.clip(reward, -0.5, 0.5)

                next_state = train_df[state_cols].iloc[t + 1].values
                done = (t == len(train_df) - 2)

                memory.push(state, action, reward, next_state, done)

                state = next_state
                position = new_position
                total_reward += reward

                if len(memory) >= batch_size:
                    states, actions, rewards_b, next_states, dones = memory.sample(batch_size)

                    states_t = torch.FloatTensor(states).to(device)
                    next_states_t = torch.FloatTensor(next_states).to(device)
                    actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
                    rewards_t = torch.FloatTensor(rewards_b).to(device)
                    dones_t = torch.FloatTensor(dones).to(device)

                    q_values = policy_net(states_t).gather(1, actions_t).squeeze()

                    with torch.no_grad():
                        next_actions = policy_net(next_states_t).argmax(1, keepdim=True)
                        next_q = target_net(next_states_t).gather(1, next_actions).squeeze()
                        target = rewards_t + gamma * next_q * (1 - dones_t)

                    loss = nn.MSELoss()(q_values, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(total_reward)

            if episode % 10 == 0:
                st.write(f"Episode {episode}: Total Sharpe-shaped reward: {total_reward:.4f}")

            progress_bar.progress((episode + 1) / n_episodes)

        # Training plot
        fig, ax = plt.subplots()
        ax.plot(episode_rewards)
        ax.set_title("Training Total Sharpe-shaped Rewards per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.grid(True)
        st.pyplot(fig)

        # ────────────────────────────────────────────────
        # Test evaluation (greedy policy)
        # ────────────────────────────────────────────────
        test_rewards = []
        position = 0
        state = test_df[state_cols].iloc[0].values
        test_actions = []

        with torch.no_grad():
            for t in range(len(test_df) - 1):
                q_values = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                action = q_values.argmax().item()
                test_actions.append(action)

                new_position = 0 if action == 0 else (position_size if action == 1 else -position_size)
                pos_change = abs(new_position - position)

                window_start = max(0, t - sharpe_window + 1)
                window_excess = new_position * (test_df['intraday_return'].iloc[window_start:t+1] - test_df['market_intraday'].iloc[window_start:t+1])
                mean_excess = np.mean(window_excess)
                std_excess = np.std(window_excess) + epsilon_div
                sharpe_reward = mean_excess / std_excess
                reward = sharpe_reward * sharpe_scale - transaction_cost * pos_change
                reward = np.clip(reward, -0.5, 0.5)

                test_rewards.append(reward)
                state = test_df[state_cols].iloc[t + 1].values
                position = new_position

        test_cum_return = (1 + np.array(test_rewards)).cumprod()[-1] - 1
        test_sharpe = np.mean(test_rewards) / (np.std(test_rewards) + 1e-8) * np.sqrt(252)

        st.success(f"Test Cumulative Return: {test_cum_return:.2%}")
        st.success(f"Annualized Sharpe: {test_sharpe:.2f}")

        # Equity curve
        test_equity = np.cumprod(1 + np.array(test_rewards)) - 1
        fig_test, ax_test = plt.subplots()
        ax_test.plot(test_equity, label='DQN Strategy')
        bh_cum = np.cumprod(1 + test_df['intraday_return'].values) - 1
        ax_test.plot(bh_cum, label='Buy & Hold', linestyle='--')
        ax_test.set_title("Test Equity Curve")
        ax_test.set_xlabel("Test Days")
        ax_test.set_ylabel("Cumulative Return")
        ax_test.legend()
        ax_test.grid(True)
        st.pyplot(fig_test)

        # Diagnostics
        st.subheader("Diagnostics")
        counts = np.bincount(test_actions, minlength=3)
        st.write("Test action counts [Hold, Long, Short]:", counts)
        probs = counts / len(test_actions) if len(test_actions) > 0 else [0,0,0]
        st.write("Test action probs [Hold, Long, Short]:", probs.round(4))
        st.write("Train intraday mean/std:", train_df['intraday_return'].mean().round(6), train_df['intraday_return'].std().round(6))
        st.write("Test intraday mean/std:", test_df['intraday_return'].mean().round(6), test_df['intraday_return'].std().round(6))
