import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
from typing import List
import os


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

    
# Стандартный агент для PPO
class AgentNN(nn.Module):
    def __init__(self, observation_shape, action_dim):
        super(AgentNN, self).__init__()
        # Входные данные SMM: (72, 96, 4) по умолчанию
        in_channels = observation_shape[2] # 4 канала
        height = observation_shape[0]      # 72
        width = observation_shape[1]       # 96
        
        self.cnn_trunk = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            ResNet(
                torch.nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding = 1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding = 1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(32),
                )
            ),
            ResNet(
                torch.nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding = 1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding = 1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(32),
                )
            ),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            ResNet(
                torch.nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding = 1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(64),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding = 1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(64),
                )
            ),
            ResNet(
                torch.nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding = 1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(64),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding = 1),
                    nn.LeakyReLU(0.01),
                    nn.BatchNorm2d(64),
                )
            ),
            nn.Flatten()
        )
        
        # Размерность выхода CNN, чтобы создать полносвязные слои
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, height, width)
            n_features = self.cnn_trunk(dummy_input).numel()

        self.actor_head = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, action_dim)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # Входной формат SMM от gfootball: (Высота, Ширина, Каналы) 
        # Транспонируем в формат PyTorch: (Каналы, Высота, Ширина)
        x = x.permute(0, 3, 1, 2)
        cnn_output = self.cnn_trunk(x)
        policy_logits = self.actor_head(cnn_output)
        value = self.critic_head(cnn_output)
        return policy_logits, value
    
    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

    def get_value(self, x):
        x = x.permute(0, 3, 1, 2)
        cnn_output = self.cnn_trunk(x)
        return self.critic_head(cnn_output)


