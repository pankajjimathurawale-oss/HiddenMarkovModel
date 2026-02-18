import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import streamlit as st
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
# Networks
# ────────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, state_dim=6, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions), nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# ────────────────────────────────────────────────
# PPO core functions
# ────────────────────────────────────────────────
def collect_rollout(actor, states):
    """ Deterministic replay → one long episode """
    states_t  = torch.FloatTensor(states.values)
    probs     = actor(states_t)
    dist      = Categorical(probs)
    actions   = dist.sample()
    log_probs = dist.log_prob(actions)
    return actions.numpy(), log_probs.detach().numpy(), dist

def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    advantages = np.zeros(len(rewards))
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns

def ppo_update(actor, critic, optimizer_a, optimizer_c, states, actions, old_log_probs,
               advantages, returns, clip_eps=0.2, entropy_coeff=0.03, epochs=10, batch_size=64):
    states_t = torch.FloatTensor(states.values)
    actions_t = torch.LongTensor(actions)
    old_log_probs_t = torch.FloatTensor(old_log_probs)
    adv_t = torch.FloatTensor(advantages)
    ret_t = torch.FloatTensor(returns)
    
    for _ in range(epochs):
        perm = torch.randperm(len(states))
        for idx in range(0, len(states), batch_size):
            end = idx + batch_size
            s_idx = perm[idx:end]
            s = states_t[s_idx]
            a = actions_t[s_idx]
            old_lp = old_log_probs_t[s_idx]
            adv = adv_t[s_idx]
            ret = ret_t[s_idx]
            
            # Actor
            probs = actor(s)
            dist = Categorical(probs)
            new_lp = dist.log_prob(a)
            ratio = torch.exp(new_lp - old_lp)
            
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coeff * dist.entropy().mean()
            
            # Critic
            value = critic(s).squeeze()
            critic_loss = (value - ret).pow(2).mean()
            
            optimizer_a.zero_grad()
            actor_loss.backward()
            optimizer_a.step()
            
            optimizer_c.zero_grad()
            critic_loss.backward()
            optimizer_c.step()

# ────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────
st.title("PPO RL Trading Simulator")

uploaded_file = st.file_uploader("Upload gamma_values-old.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['TIMESTAMP'], date_format='%m/%d/%Y')
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)

    # ────────────────────────────────────────────────
    # Feature engineering – lagged to prevent look-ahead bias
    # ────────────────────────────────────────────────
    df['prev_return'] = df['NiftyClose'].pct_change().fillna(0)
    df['vol_5d']      = df['prev_return'].rolling(5).std().fillna(0)

    # Lag gamma & price features (decision at open of t uses data up to t-1)
    df['lag_GEX']            = df['GEX'].shift(1)
    df['lag_Gamma_Flip']     = df['Gamma Flip'].shift(1)
    df['lag_NiftyClose']     = df['NiftyClose'].shift(1)
    df['lag_Max_SuperGamma'] = df['Max SuperGamma'].shift(1)
    df['lag_DTE']            = df['DTE'].shift(1)

    df['GEX_norm']      = df['lag_GEX'] / 1e6
    df['flip_prox']     = (df['lag_NiftyClose'] - df['lag_Gamma_Flip']) / df['lag_NiftyClose']
    df['superg_spread'] = (df['lag_Max_SuperGamma'] - df['lag_Gamma_Flip']) / 100
    df['dte_norm']      = df['lag_DTE'] / 10

    state_cols = ['GEX_norm', 'flip_prox', 'superg_spread', 'dte_norm', 'prev_return', 'vol_5d']

    # Fill NaNs (first rows) and normalize
    df[state_cols] = df[state_cols].fillna(0)
    df[state_cols] = (df[state_cols] - df[state_cols].mean()) / df[state_cols].std()

    # ────────────────────────────────────────────────
    # INTRADAY reward (open-to-close, no overnight)
    # ────────────────────────────────────────────────
    df['intraday_return'] = (df['NiftyClose'] - df['NiftyOpen']) / df['NiftyOpen']
    df['intraday_return'] = df['intraday_return'].fillna(0)   # last day or missing

    # Split train/test
    train_df = df[df['TIMESTAMP'] < '2025-07-01'].copy()
    test_df = df[df['TIMESTAMP'] >= '2025-07-01'].copy()

    if st.button("Train PPO Model"):
        actor = Actor()
        critic = Critic()
        opt_a = optim.Adam(actor.parameters(), lr=3e-4)
        opt_c = optim.Adam(critic.parameters(), lr=1e-3)

        n_updates = 50  # Reduced to prevent overfitting
        train_returns = []

        with st.spinner("Training..."):
            for update in range(n_updates):
                actions, old_log_probs, dist = collect_rollout(actor, train_df[state_cols])
                
                positions = np.where(actions == 0, 0, np.where(actions == 1, 0.3, -0.3))  # Reduced position size
                
                pos_diff = np.abs(np.diff(positions, prepend=0))
                rewards = positions * train_df['intraday_return'].values - 0.0005 * pos_diff
                
                # Amplify rewards and clip extreme
                rewards = rewards * 5.0
                rewards = np.clip(rewards, -0.10, 0.10)
                
                with torch.no_grad():
                    values = critic(torch.FloatTensor(train_df[state_cols].values)).squeeze().numpy()
                    next_val = critic(torch.FloatTensor(train_df[state_cols].iloc[-1:].values)).item()
                
                adv, rets = compute_gae(rewards, values, next_val)
                
                # Normalize and clip advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                adv = np.clip(adv, -5, 5)
                
                ppo_update(actor, critic, opt_a, opt_c, train_df[state_cols], actions, old_log_probs, adv, rets)
                
                cum_ret = (1 + rewards).cumprod()[-1] - 1
                train_returns.append(cum_ret)
                
                if update % 10 == 0:
                    st.write(f"Update {update}: Train cum return: {cum_ret:.3%}")

                # Decay learning rate
                for param_group in opt_a.param_groups:
                    param_group['lr'] *= 0.995
                for param_group in opt_c.param_groups:
                    param_group['lr'] *= 0.995

        # Plot training progress
        fig, ax = plt.subplots()
        ax.plot(train_returns)
        ax.set_title("Training Cumulative Returns")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Cumulative Return")
        st.pyplot(fig)

        # Test evaluation
        with torch.no_grad():
            test_actions, _, _ = collect_rollout(actor, test_df[state_cols])
        
        test_positions = np.where(test_actions == 0, 0, np.where(test_actions == 1, 0.3, -0.3))
        test_pos_diff = np.abs(np.diff(test_positions, prepend=0))
        test_rewards = test_positions * test_df['intraday_return'].values - 0.0005 * test_pos_diff
        
        # Amplify and clip test rewards
        test_rewards = test_rewards * 5.0
        test_rewards = np.clip(test_rewards, -0.10, 0.10)
        
        test_cum_return = (1 + test_rewards).cumprod()[-1] - 1
        test_sharpe = test_rewards.mean() / test_rewards.std() * np.sqrt(252) if test_rewards.std() > 0 else 0
        
        st.success(f"Test Cumulative Return: {test_cum_return:.2%}")
        st.success(f"Annualized Sharpe: {test_sharpe:.2f}")

        # Diagnostics
        st.subheader("Diagnostics")

        # Action distribution on test
        test_probs = actor(torch.FloatTensor(test_df[state_cols].values)).detach().numpy().mean(axis=0)
        st.write("Average test action probabilities [Hold, Long, Short]:", test_probs.round(4))

        # Daily reward stats
        st.write("Train daily intraday return mean / std:", train_df['intraday_return'].mean().round(6), train_df['intraday_return'].std().round(6))
        st.write("Test daily intraday return mean / std:", test_df['intraday_return'].mean().round(6), test_df['intraday_return'].std().round(6))

        # Count how often each action is taken in test
        counts = np.bincount(test_actions, minlength=3)
        st.write("Test action counts [Hold, Long, Short]:", counts)
