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
st.title("DQN RL Trading Simulator")

uploaded_file = st.file_uploader("Upload gamma_values-old.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['TIMESTAMP'], date_format='%m/%d/%Y')
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)

    # ────────────────────────────────────────────────
    # Feature engineering – lagged to prevent look-ahead
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

    # Intraday reward
    df['intraday_return'] = (df['NiftyClose'] - df['NiftyOpen']) / df['NiftyOpen']
    df['intraday_return'] = df['intraday_return'].fillna(0)

    # Split
    train_df = df[df['TIMESTAMP'] < '2025-07-01'].copy()
    test_df = df[df['TIMESTAMP'] >= '2025-07-01'].copy()

    # Parameters
    state_dim = len(state_cols)
    n_actions = 3
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    batch_size = 64
    target_update_freq = 500
    learning_rate = 1e-4
    replay_capacity = 20000
    n_episodes = 30  # episodes = full train sequence replays

    if st.button("Train DQN Model"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy_net = DQN(state_dim, n_actions).to(device)
        target_net = DQN(state_dim, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        memory = ReplayBuffer(replay_capacity)

        epsilon = epsilon_start
        steps = 0
        episode_rewards = []

        with st.spinner("Training DQN..."):
            for episode in range(n_episodes):
                state = train_df[state_cols].iloc[0].values
                total_reward = 0
                position = 0  # start flat

                for t in range(len(train_df) - 1):
                    steps += 1

                    # Epsilon-greedy action
                    if random.random() < epsilon:
                        action = random.randint(0, n_actions - 1)
                    else:
                        with torch.no_grad():
                            q_values = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                            action = q_values.argmax().item()

                    # Execute action
                    new_position = 0 if action == 0 else (1 if action == 1 else -1)
                    pos_change = abs(new_position - position)
                    reward = new_position * train_df['intraday_return'].iloc[t] - 0.0005 * pos_change
                    reward = np.clip(reward, -0.03, 0.03)  # prevent explosion

                    next_state = train_df[state_cols].iloc[t + 1].values
                    done = (t == len(train_df) - 2)

                    memory.push(state, action, reward, next_state, done)

                    state = next_state
                    position = new_position
                    total_reward += reward

                    # Training step
                    if len(memory) >= batch_size:
                        states, actions, rewards_b, next_states, dones = memory.sample(batch_size)

                        states_t = torch.FloatTensor(states).to(device)
                        next_states_t = torch.FloatTensor(next_states).to(device)
                        actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
                        rewards_t = torch.FloatTensor(rewards_b).to(device)
                        dones_t = torch.FloatTensor(dones).to(device)

                        q_values = policy_net(states_t).gather(1, actions_t).squeeze()
                        with torch.no_grad():
                            next_q = target_net(next_states_t).max(1)[0]
                            target = rewards_t + gamma * next_q * (1 - dones_t)

                        loss = nn.MSELoss()(q_values, target)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Update target network
                    if steps % target_update_freq == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                episode_rewards.append(total_reward)

                if episode % 5 == 0:
                    st.write(f"Episode {episode}: Total reward: {total_reward:.4f}")

        # Plot training progress
        fig, ax = plt.subplots()
        ax.plot(episode_rewards)
        ax.set_title("Training Total Rewards per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        st.pyplot(fig)

        # ────────────────────────────────────────────────
        # Test evaluation (greedy policy)
        # ────────────────────────────────────────────────
        test_rewards = []
        position = 0
        state = test_df[state_cols].iloc[0].values

        with torch.no_grad():
            for t in range(len(test_df) - 1):
                q_values = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                action = q_values.argmax().item()

                new_position = 0 if action == 0 else (1 if action == 1 else -1)
                pos_change = abs(new_position - position)
                reward = new_position * test_df['intraday_return'].iloc[t] - 0.0005 * pos_change
                reward = np.clip(reward, -0.03, 0.03)

                test_rewards.append(reward)
                state = test_df[state_cols].iloc[t + 1].values
                position = new_position

        test_cum_return = (1 + np.array(test_rewards)).cumprod()[-1] - 1
        test_sharpe = np.mean(test_rewards) / np.std(test_rewards) * np.sqrt(252) if np.std(test_rewards) > 0 else 0

        st.success(f"Test Cumulative Return: {test_cum_return:.2%}")
        st.success(f"Annualized Sharpe: {test_sharpe:.2f}")

        # Diagnostics
        st.subheader("Diagnostics")
        test_actions = []  # collect for stats
        # Re-run test to collect actions (simple re-loop)
        position = 0
        for t in range(len(test_df) - 1):
            q_values = policy_net(torch.FloatTensor(test_df[state_cols].iloc[t].values).unsqueeze(0).to(device))
            action = q_values.argmax().item()
            test_actions.append(action)
            new_position = 0 if action == 0 else (1 if action == 1 else -1)
            position = new_position

        counts = np.bincount(test_actions, minlength=3)
        st.write("Test action counts [Hold, Long, Short]:", counts)

        probs_approx = counts / len(test_actions)
        st.write("Approx test action probs [Hold, Long, Short]:", probs_approx.round(4))

        st.write("Test daily intraday mean / std:", test_df['intraday_return'].mean().round(6), test_df['intraday_return'].std().round(6))
