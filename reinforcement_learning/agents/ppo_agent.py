"""
ppo_agent.py
PPO (Proximal Policy Optimization) エージェントの実装
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque

from .network_architectures import ActorNetwork, CriticNetwork
from .buffer import RolloutBuffer

class PPOAgent:
    """
    PPOアルゴリズムによる救急車ディスパッチエージェント
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: Dict,
                 device: str = None):
        """
        Args:
            state_dim: 状態空間の次元
            action_dim: 行動空間の次元（救急車数）
            config: PPO設定
            device: 計算デバイス
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # デバイス設定
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"PPOエージェント初期化")
        print(f"  状態次元: {state_dim}")
        print(f"  行動次元: {action_dim}")
        print(f"  デバイス: {self.device}")
        
        # ネットワークの初期化
        self.actor = ActorNetwork(state_dim, action_dim, config).to(self.device)
        self.critic = CriticNetwork(state_dim, config).to(self.device)
        
        # オプティマイザ
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config['learning_rate']['actor']
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config['learning_rate']['critic']
        )
        
        # PPOハイパーパラメータ
        self.clip_epsilon = config['clip_epsilon']
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.n_epochs = config['n_epochs']
        self.batch_size = config['batch_size']
        self.entropy_coef = config['entropy_coef']
        
        # 経験バッファ
        self.buffer = RolloutBuffer(
            buffer_size=2048,
            state_dim=state_dim,
            device=self.device
        )
        
        # 統計情報
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100)
        }
        
    def select_action(self, 
                     state: np.ndarray,
                     action_mask: Optional[np.ndarray] = None,
                     deterministic: bool = False) -> Tuple[int, float, float]:
        """
        行動を選択
        
        Args:
            state: 状態ベクトル
            action_mask: 利用可能な行動のマスク
            deterministic: 決定的選択フラグ
            
        Returns:
            action: 選択された行動
            log_prob: 対数確率
            value: 状態価値
        """
        # テンソルに変換
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if action_mask is not None:
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
        else:
            mask_tensor = torch.ones(1, self.action_dim, dtype=torch.bool).to(self.device)
        
        with torch.no_grad():
            # Actor出力（行動確率）
            action_probs = self.actor(state_tensor)
            
            # マスキング
            masked_probs = action_probs.clone()
            masked_probs[~mask_tensor] = 0.0
            
            # 正規化
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True)
            else:
                # 全て使用不可の場合は一様分布
                masked_probs = mask_tensor.float() / mask_tensor.sum()
            
            # 行動選択
            if deterministic:
                action = masked_probs.argmax(dim=1).item()
                log_prob = torch.log(masked_probs[0, action])
            else:
                dist = Categorical(masked_probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action))
            
            # Critic出力（状態価値）
            value = self.critic(state_tensor).squeeze().item()
        
        return action, log_prob.item(), value
    
    def update(self) -> Dict[str, float]:
        """
        PPOアルゴリズムによるパラメータ更新
        
        Returns:
            更新統計
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # バッファからデータ取得
        batch = self.buffer.get_all()
        
        # GAEによるAdvantage計算
        advantages = self._compute_gae(
            batch['rewards'],
            batch['values'],
            batch['dones']
        )
        
        # リターンの計算
        returns = advantages + batch['values']
        
        # 正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 複数エポックで更新
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        update_count = 0
        
        for epoch in range(self.n_epochs):
            # ミニバッチ処理
            indices = np.random.permutation(len(batch['states']))
            
            for start_idx in range(0, len(indices), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # バッチデータ
                states = batch['states'][batch_indices]
                actions = batch['actions'][batch_indices]
                old_log_probs = batch['log_probs'][batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                masks = batch['action_masks'][batch_indices]
                
                # 現在の方策での評価
                action_probs = self.actor(states)
                values = self.critic(states).squeeze()
                
                # マスキング適用
                masked_probs = action_probs.clone()
                masked_probs[~masks] = 0.0
                masked_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-8)
                
                dist = Categorical(masked_probs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy()
                
                # PPO損失計算
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, batch_returns)
                
                # エントロピー正則化
                entropy_loss = -self.entropy_coef * entropy.mean()
                
                # 総損失
                total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
                
                # 勾配更新
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # 統計更新
                with torch.no_grad():
                    kl = (old_log_probs - log_probs).mean()
                    total_kl += kl.item()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                update_count += 1
                
                # KL divergenceが大きすぎる場合は早期終了
                if kl > 0.02:
                    break
        
        # バッファクリア
        self.buffer.clear()
        
        # 統計記録
        avg_actor_loss = total_actor_loss / update_count
        avg_critic_loss = total_critic_loss / update_count
        avg_entropy = total_entropy / update_count
        avg_kl = total_kl / update_count
        
        self.training_stats['actor_loss'].append(avg_actor_loss)
        self.training_stats['critic_loss'].append(avg_critic_loss)
        self.training_stats['entropy'].append(avg_entropy)
        self.training_stats['kl_divergence'].append(avg_kl)
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy,
            'kl_divergence': avg_kl
        }
    
    def _compute_gae(self, 
                    rewards: torch.Tensor,
                    values: torch.Tensor,
                    dones: torch.Tensor) -> torch.Tensor:
        """
        Generalized Advantage Estimation (GAE) の計算
        
        Args:
            rewards: 報酬系列
            values: 価値系列
            dones: 終了フラグ系列
            
        Returns:
            advantages: Advantage系列
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                next_value = 0
                last_advantage = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
            advantages[t] = last_advantage
        
        return advantages
    
    def store_transition(self,
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool,
                        log_prob: float,
                        value: float,
                        action_mask: Optional[np.ndarray] = None):
        """
        経験をバッファに保存
        """
        self.buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value,
            action_mask=action_mask
        )
    
    def save(self, path: str):
        """モデルを保存"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config
        }, path)
        print(f"モデル保存完了: {path}")
    
    def load(self, path: str):
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"モデル読み込み完了: {path}")