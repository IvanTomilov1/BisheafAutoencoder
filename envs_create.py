import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
import gfootball.env as football_env
from typing import List
import os


# 2. Настройка среды (с SMM-представлением)
def make_gfootball_env(NUM_AGENTS):
    env = football_env.create_environment(
        env_name= '5_vs_5_easy',#'academy_3_vs_1_with_keeper',
        stacked=False, # SMM уже 3D, поэтому не будем удваивать
        representation='extracted', # <-- Используем SMM
        rewards='scoring', #checkpoints,
        render=False,
        number_of_left_players_agent_controls=NUM_AGENTS,
    )
    return env