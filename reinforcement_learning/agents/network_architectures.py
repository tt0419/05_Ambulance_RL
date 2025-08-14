"""
network_architectures.py
Actor-Criticネットワークの定義
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

class ActorNetwork(nn.Module):
    """
    方策ネットワーク（Actor）
    状態から行動確率分布を出力
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim  # 192台の救急車
        
        # ネットワーク構造の設定
        hidden_layers = config.get('network', {}).get('actor', {}).get('hidden_layers', [256, 128, 64])
        activation = config.get('network', {}).get('actor', {}).get('activation', 'relu')
        dropout_rate = config.get('network', {}).get('actor', {}).get('dropout', 0.1)
        
        # 活性化関数の選択
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # ネットワーク層の構築
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # 正規化層
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 最終層（行動確率出力）
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 重み初期化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """重みの初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            state: 状態テンソル [batch_size, state_dim]
            
        Returns:
            action_probs: 行動確率 [batch_size, action_dim]
        """
        # ネットワークを通す
        logits = self.network(state)
        
        # Softmaxで確率分布に変換
        action_probs = F.softmax(logits, dim=-1)
        
        # 数値安定性のため小さな値を加える
        action_probs = action_probs + 1e-8
        
        return action_probs


class CriticNetwork(nn.Module):
    """
    価値ネットワーク（Critic）
    状態から状態価値を出力
    """
    
    def __init__(self, state_dim: int, config: Dict):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # ネットワーク構造の設定
        hidden_layers = config.get('network', {}).get('critic', {}).get('hidden_layers', [256, 128])
        activation = config.get('network', {}).get('critic', {}).get('activation', 'relu')
        dropout_rate = config.get('network', {}).get('critic', {}).get('dropout', 0.1)
        
        # 活性化関数の選択
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # ネットワーク層の構築
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 最終層（状態価値出力）
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みの初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            state: 状態テンソル [batch_size, state_dim]
            
        Returns:
            value: 状態価値 [batch_size, 1]
        """
        value = self.network(state)
        return value


class AttentionActorNetwork(nn.Module):
    """
    注意機構を持つActorネットワーク（発展版）
    救急車と事案の関係を学習
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        super(AttentionActorNetwork, self).__init__()
        
        self.action_dim = action_dim  # 192
        self.num_ambulances = 192
        self.ambulance_features = 4
        self.incident_features = 10
        
        # 救急車エンコーダ
        self.ambulance_encoder = nn.Sequential(
            nn.Linear(self.ambulance_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 事案エンコーダ
        self.incident_encoder = nn.Sequential(
            nn.Linear(self.incident_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # 注意機構
        self.attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )
        
        # 最終層
        self.output_layer = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        注意機構を使った順伝播
        """
        batch_size = state.shape[0]
        
        # 状態を分解
        # 救急車情報: [batch_size, num_ambulances, ambulance_features]
        ambulance_states = state[:, :self.num_ambulances * self.ambulance_features]
        ambulance_states = ambulance_states.view(batch_size, self.num_ambulances, self.ambulance_features)
        
        # 事案情報: [batch_size, incident_features]
        incident_state = state[:, self.num_ambulances * self.ambulance_features:
                               self.num_ambulances * self.ambulance_features + self.incident_features]
        
        # エンコード
        ambulance_encoded = self.ambulance_encoder(ambulance_states)  # [batch, 192, 32]
        incident_encoded = self.incident_encoder(incident_state)  # [batch, 32]
        incident_encoded = incident_encoded.unsqueeze(1)  # [batch, 1, 32]
        
        # 注意機構（事案を基準に救急車を評価）
        attended, _ = self.attention(
            query=incident_encoded,
            key=ambulance_encoded,
            value=ambulance_encoded
        )  # [batch, 1, 32]
        
        # 各救急車に対する相対スコア
        scores = self.output_layer(ambulance_encoded)  # [batch, 192, 1]
        scores = scores.squeeze(-1)  # [batch, 192]
        
        # Softmaxで確率分布に
        action_probs = F.softmax(scores, dim=-1)
        
        return action_probs