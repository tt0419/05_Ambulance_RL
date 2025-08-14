"""
buffer.py
経験リプレイバッファの実装
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple

class RolloutBuffer:
    """
    PPO用のロールアウトバッファ
    エピソード中の経験を保存
    """
    
    def __init__(self, 
                 buffer_size: int,
                 state_dim: int,
                 device: torch.device):
        """
        Args:
            buffer_size: バッファサイズ
            state_dim: 状態空間の次元
            device: 計算デバイス
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.device = device
        
        # バッファの初期化
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, 192), dtype=bool)  # 192台の救急車
        
        self.pos = 0
        self.full = False
        
    def add(self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
            log_prob: float,
            value: float,
            action_mask: Optional[np.ndarray] = None):
        """
        経験を追加
        """
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        
        if action_mask is not None:
            self.action_masks[self.pos] = action_mask
        else:
            self.action_masks[self.pos] = np.ones(192, dtype=bool)
        
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """
        全データを取得
        """
        if self.full:
            indices = np.arange(self.buffer_size)
        else:
            indices = np.arange(self.pos)
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.BoolTensor(self.dones[indices]).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs[indices]).to(self.device),
            'values': torch.FloatTensor(self.values[indices]).to(self.device),
            'action_masks': torch.BoolTensor(self.action_masks[indices]).to(self.device)
        }
        
        return batch
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        ランダムサンプリング
        """
        max_idx = self.buffer_size if self.full else self.pos
        indices = np.random.choice(max_idx, batch_size, replace=False)
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.BoolTensor(self.dones[indices]).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs[indices]).to(self.device),
            'values': torch.FloatTensor(self.values[indices]).to(self.device),
            'action_masks': torch.BoolTensor(self.action_masks[indices]).to(self.device)
        }
        
        return batch
    
    def clear(self):
        """バッファをクリア"""
        self.pos = 0
        self.full = False
    
    def __len__(self):
        """バッファ内のデータ数"""
        return self.buffer_size if self.full else self.pos