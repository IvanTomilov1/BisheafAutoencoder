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



class Autoencoder(nn.Module):
    def __init__(self, in_channels, height, width):
        super(Autoencoder, self).__init__()
        
        self.in_channels = in_channels
        self.height = height
        self.width = width
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, 16, kernel_size=3, stride=2, padding=1), # Output: H/2, W/2
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: H/4, W/4
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: H/8, W/8
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# Output: H/16, W/16
            nn.LeakyReLU(0.01),
            nn.Flatten()
        )

        # Вычисление размерности эмбеддинга
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels * 2, height, width)
            self.embedding_dim = self.encoder(dummy_input).numel()
        
        self.final_enc_c = 128
        self.final_enc_h = height // 16 + (1 if height % 16 != 0 else 0)
        self.final_enc_w = width // 16 + (1 if width % 16 != 0 else 0)

        self.decoder_reshape = nn.Linear(self.embedding_dim, self.final_enc_c * self.final_enc_h * self.final_enc_w)

        self.decoder = nn.Sequential(
            # Upsample 1: 128 -> 64
            nn.ConvTranspose2d(self.final_enc_c, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            # Upsample 2: 64 -> 32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            # Upsample 3: 32 -> 16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            # Upsample 4: 16 -> in_channels
            # Финальный слой должен давать количество каналов исходного изображения
            nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1),
            # Используем Sigmoid для выходного изображения, если оно нормализовано в [0, 1]
            nn.Sigmoid() 
        )

    def forward(self, x):
        # Метод forward не использует декодер, только энкодер
        embedding = self.encoder(x)
        return embedding

    def decode(self, embedding):
        # Метод decode используется только для реконструкции
        # Сначала преобразуем плоский вектор обратно в объемный тензор
        x = self.decoder_reshape(embedding)
        x = x.view(-1, self.final_enc_c, self.final_enc_h, self.final_enc_w)
        # Затем прогоняем через декодер
        decoded = self.decoder(x)
        return decoded

# Обновленный класс AgentNN
class AgentNN_mod(nn.Module):
    def __init__(self, observation_shape, action_dim):
        super(AgentNN_mod, self).__init__()
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

        # Инициализация автокодировщика
        self.autoencoder = Autoencoder(in_channels, height, width)

        # Размерность выхода CNN_trunk
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, height, width)
            cnn_features_dim = self.cnn_trunk(dummy_input).numel()

        # Общая размерность признаков = размерность CNN_trunk + размерность эмбеддинга автокодировщика
        total_features_dim = cnn_features_dim + self.autoencoder.embedding_dim

        self.actor_head = nn.Sequential(
            nn.Linear(total_features_dim, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, action_dim)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(total_features_dim, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x - текущее наблюдение. Формат SMM (B, H, W, C)
        
        # 1. Обработка через cnn_trunk
        # Транспонируем в формат PyTorch: (B, C, H, W)
        x_permuted = x.permute(0, 3, 1, 2)
        cnn_output = self.cnn_trunk(x_permuted)

        # 2. Получение эмбеддинга от автокодировщика (только энкодер)
        # Для forward_pass автокодировщик принимает (x, гауссов шум размера x)
        noise = torch.randn_like(x_permuted)
        # Конкатенируем вдоль канала (dim=1)
        ae_input = torch.cat([x_permuted, noise], dim=1)
        ae_embedding = self.autoencoder(ae_input)

        # 3. Конкатенация выходов
        combined_features = torch.cat([cnn_output, ae_embedding], dim=1)

        # 4. Проход через головы
        policy_logits = self.actor_head(combined_features)
        value = self.critic_head(combined_features)
        return policy_logits, value
    
    def autoencode_pair(self, img1, img2):
        # Метод для обучения автокодировщика
        # Принимает два изображения, конкатенирует их и прогоняет через AE
        # img1 и img2 ожидаются в формате (B, H, W, C) как в forward()

        # Транспонируем в формат PyTorch: (B, C, H, W)
        img1_p = img1.permute(0, 3, 1, 2)
        # img2_p = img2.permute(0, 3, 1, 2) # img2 не используется для цели реконструкции

        # Конкатенируем вдоль канала для входа в энкодер
        ae_input = torch.cat([img1_p, img2.permute(0, 3, 1, 2)], dim=1)
        
        # Получаем эмбеддинг
        embedding = self.autoencoder.encoder(ae_input)
        
        # Декодируем для получения выходного изображения (реконструкции)
        # В качестве цели реконструкции используется img1_p
        reconstructed_img = self.autoencoder.decode(embedding)
        
        # Возвращаем реконструкцию и исходное изображение для расчета лосса
        return reconstructed_img, img1_p

    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

    def get_value(self, x):
        # Использует тот же путь, что и forward, но возвращает только value
        x_permuted = x.permute(0, 3, 1, 2)
        cnn_output = self.cnn_trunk(x_permuted)

        noise = torch.randn_like(x_permuted)
        ae_input = torch.cat([x_permuted, noise], dim=1)
        ae_embedding = self.autoencoder(ae_input)

        combined_features = torch.cat([cnn_output, ae_embedding], dim=1)
        return self.critic_head(combined_features)
