import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
import os


# 4. Вспомогательные функции PPO (GAE и расчет потерь)
def compute_gae(next_value_scalar, rewards_1d, dones_1d, values_1d, gamma, gae_lambda):
    advantages = np.zeros_like(rewards_1d, dtype=np.float32)
    last_gae = 0.0
    next_val = next_value_scalar 
    for t in reversed(range(len(rewards_1d))):
        delta = rewards_1d[t] + gamma * next_val * (1 - dones_1d[t]) - values_1d[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones_1d[t]) * last_gae
        advantages[t] = last_gae
        next_val = values_1d[t]
    returns = advantages + values_1d
    return advantages, returns 


def ppo_update(agent, optimizer, batch_data, PPO_EPOCHS, MINIBATCH_SIZE, CLIP_EPS):
    obs, actions, logprobs_old, returns, advantages = (
        batch_data['obs'], batch_data['actions'], batch_data['logprobs'], 
        batch_data['returns'], batch_data['advantages']
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        indices = np.arange(len(obs))
        np.random.shuffle(indices)
        for start in range(0, len(obs), MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mini_batch_indices = indices[start:end]
            mb_obs, mb_actions, mb_logprobs_old, mb_returns, mb_advantages = (
                obs[mini_batch_indices], actions[mini_batch_indices], 
                logprobs_old[mini_batch_indices], returns[mini_batch_indices], 
                advantages[mini_batch_indices]
            )

            _, logprobs_new, entropy, values_new = agent.get_action_and_value(mb_obs, mb_actions)
            ratio = torch.exp(logprobs_new - mb_logprobs_old)
            clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
            loss_policy = -torch.min(ratio * mb_advantages, clip_adv).mean()
            loss_value = 0.5 * ((mb_returns - values_new.squeeze()) ** 2).mean()
            loss_entropy = entropy.mean()
            total_loss = loss_policy + loss_value * 0.9 - loss_entropy * 0.01

            total_loss.backward()
