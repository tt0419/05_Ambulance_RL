"""
trainer.py
PPO学習のトレーナークラス
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import time
from datetime import datetime
from tqdm import tqdm
import wandb

from ..environment.ems_environment import EMSEnvironment
from ..agents.ppo_agent import PPOAgent

class PPOTrainer:
    """
    PPO学習を管理するトレーナー
    """
    
    def __init__(self,
                 agent: PPOAgent,
                 env: EMSEnvironment,
                 config: Dict,
                 output_dir: Path):
        """
        Args:
            agent: PPOエージェント
            env: 環境
            config: 設定
            output_dir: 出力ディレクトリ
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.output_dir = Path(output_dir)
        
        # ディレクトリ作成
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 学習設定
        self.n_episodes = config['ppo']['n_episodes']
        self.checkpoint_interval = config['training']['checkpoint_interval']
        self.eval_interval = config['evaluation']['interval']
        self.n_eval_episodes = config['evaluation']['n_eval_episodes']
        
        # 統計情報
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = []
        self.eval_stats = []
        
        # 早期終了
        self.early_stopping = config['training']['early_stopping']['enabled']
        self.patience = config['training']['early_stopping']['patience']
        self.min_delta = config['training']['early_stopping']['min_delta']
        self.best_eval_reward = -float('inf')
        self.patience_counter = 0
        
        # ログ設定
        self.use_tensorboard = config['training']['logging']['tensorboard']
        self.use_wandb = config['training']['logging']['wandb']
        
        if self.use_wandb:
            try:
                print("WandB初期化中...")
                wandb.init(
                    project="ems-ppo",
                    name=output_dir.name,
                    config=config,
                    settings=wandb.Settings(init_timeout=180)  # タイムアウトを180秒に延長
                )
                print("✓ WandB初期化完了")
            except Exception as e:
                print(f"⚠️ WandB初期化に失敗しました: {e}")
                print("WandBを無効にして学習を続行します...")
                self.use_wandb = False
        
        print(f"PPOトレーナー初期化完了")
        print(f"  総エピソード数: {self.n_episodes}")
        print(f"  チェックポイント間隔: {self.checkpoint_interval}")
        print(f"  評価間隔: {self.eval_interval}")
        
    def train(self):
        """
        学習のメインループ
        """
        print("\n" + "=" * 60)
        print("PPO学習開始")
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in range(1, self.n_episodes + 1):
            # エピソード実行
            episode_reward, episode_length, episode_stats = self._run_episode(training=True)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # PPO更新
            if len(self.agent.buffer) >= self.agent.batch_size:
                update_stats = self.agent.update()
                self.training_stats.append(update_stats)
            
            # ログ出力（毎回表示）
            self._log_training_progress(episode, episode_reward, episode_length, episode_stats)
            
            # 評価
            if episode % self.eval_interval == 0:
                eval_reward = self._evaluate()
                
                # 早期終了チェック
                if self.early_stopping:
                    if eval_reward > self.best_eval_reward + self.min_delta:
                        self.best_eval_reward = eval_reward
                        self.patience_counter = 0
                        # 最良モデルの保存
                        self._save_checkpoint(episode, is_best=True)
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            print(f"\n早期終了: {self.patience}エピソード改善なし")
                            break
            
            # チェックポイント保存
            if episode % self.checkpoint_interval == 0:
                self._save_checkpoint(episode)
        
        # 学習完了
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("学習完了")
        print(f"  総時間: {elapsed_time/3600:.2f}時間")
        print(f"  最良評価報酬: {self.best_eval_reward:.2f}")
        print("=" * 60)
        
        # 最終モデルの保存
        self.agent.save(self.output_dir / "final_model.pth")
        self._save_training_stats()
        
    def _run_episode(self, training: bool = True, use_teacher: bool = True) -> Tuple[float, int, Dict]:
        """
        1エピソードを実行（教師あり学習オプション付き）
        
        Args:
            training: 学習モードフラグ
            use_teacher: 教師あり学習を使用するか
            
        Returns:
            episode_reward: エピソード報酬
            episode_length: エピソード長
            episode_stats: エピソード統計
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # 学習の進行に応じて教師の使用率を減らす
        if use_teacher and training:
            # エピソード数に応じて教師確率を減衰
            current_episode = len(self.episode_rewards)
            # 最初は80%、1000エピソードで20%まで減衰
            teacher_prob = max(0.2, 0.8 - (current_episode / 1000) * 0.6)
        else:
            teacher_prob = 0.0
        
        while True:
            # 行動選択
            action_mask = self.env.get_action_mask()
            
            if training and use_teacher:
                # 最適行動を取得
                optimal_action = self.env.get_optimal_action()
                
                # 教師あり混合選択
                action, log_prob, value = self.agent.select_action_with_teacher(
                    state, 
                    action_mask,
                    optimal_action,
                    teacher_prob,
                    deterministic=not training
                )
            else:
                # 通常のPPO選択
                action, log_prob, value = self.agent.select_action(
                    state, 
                    action_mask,
                    deterministic=not training
                )
            
            # 環境ステップ
            step_result = self.env.step(action)
            next_state = step_result.observation
            reward = step_result.reward
            done = step_result.done
            
            episode_reward += reward
            episode_length += 1
            
            # 経験を保存（学習時のみ）
            if training:
                self.agent.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                    action_mask=action_mask
                )
            
            state = next_state
            
            if done:
                break
        
        # エピソード統計
        episode_stats = self.env.episode_stats
        
        return episode_reward, episode_length, episode_stats
    
    def _evaluate(self) -> float:
        """
        エージェントを評価
        
        Returns:
            平均評価報酬
        """
        print("\n評価中...")
        eval_rewards = []
        eval_stats = {
            'response_times': [],
            'achieved_6min': 0,
            'achieved_13min': 0,
            'critical_6min': 0,
            'critical_total': 0
        }
        
        for _ in range(self.n_eval_episodes):
            episode_reward, _, episode_stats = self._run_episode(training=False)
            eval_rewards.append(episode_reward)
            
            # 統計集計
            if episode_stats['response_times']:
                eval_stats['response_times'].extend(episode_stats['response_times'])
            eval_stats['achieved_6min'] += episode_stats['achieved_6min']
            eval_stats['achieved_13min'] += episode_stats['achieved_13min']
            eval_stats['critical_6min'] += episode_stats['critical_6min']
            eval_stats['critical_total'] += episode_stats['critical_total']
        
        mean_reward = np.mean(eval_rewards)
        
        # 評価結果の表示
        print(f"  平均報酬: {mean_reward:.2f}")
        if eval_stats['response_times']:
            avg_rt = np.mean(eval_stats['response_times'])
            print(f"  平均応答時間: {avg_rt:.2f}分")
        
        if eval_stats['achieved_6min'] > 0:
            total_calls = eval_stats['achieved_6min'] + eval_stats['achieved_13min']
            rate_6min = eval_stats['achieved_6min'] / total_calls * 100
            print(f"  6分達成率: {rate_6min:.1f}%")
        
        if eval_stats['critical_total'] > 0:
            critical_rate = eval_stats['critical_6min'] / eval_stats['critical_total'] * 100
            print(f"  重症系6分達成率: {critical_rate:.1f}%")
        
        self.eval_stats.append({
            'mean_reward': mean_reward,
            'stats': eval_stats
        })
        
        return mean_reward
    
    def _log_training_progress(self, episode: int, reward: float, length: int, stats: Dict):
        """学習進捗のログ出力"""
        # 直近の平均
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        avg_reward = np.mean(recent_rewards)
        
        # 毎エピソード表示（簡潔版）
        print(f"Episode {episode}/{self.n_episodes}")
        print(f"  報酬: {reward:.2f} (平均: {avg_reward:.2f})")
        print(f"  長さ: {length}")
        
        if stats['response_times']:
            avg_rt = np.mean(stats['response_times'])
            print(f"  平均応答時間: {avg_rt:.2f}分")
        
        # 詳細なログは10エピソードごとに表示
        if episode % 10 == 0 and self.training_stats:
            latest_stats = self.training_stats[-1]
            print(f"  Actor損失: {latest_stats.get('actor_loss', 0):.4f}")
            print(f"  Critic損失: {latest_stats.get('critic_loss', 0):.4f}")
        
        # WandBログ
        if self.use_wandb:
            wandb.log({
                'episode': episode,
                'reward': reward,
                'avg_reward': avg_reward,
                'episode_length': length
            })
    
    def _save_checkpoint(self, episode: int, is_best: bool = False):
        """チェックポイントの保存"""
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_ep{episode}.pth"
        
        self.agent.save(path)
        
        # 古いチェックポイントの削除
        if not is_best:
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.pth"))
            if len(checkpoints) > 5:  # 最新5つを保持
                for old_checkpoint in checkpoints[:-5]:
                    old_checkpoint.unlink()
    
    def _save_training_stats(self):
        """学習統計の保存"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_stats': self.training_stats,
            'eval_stats': self.eval_stats,
            'config': self.config
        }
        
        with open(self.log_dir / "training_stats.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイントから再開"""
        self.agent.load(checkpoint_path)
        print(f"チェックポイント読み込み完了: {checkpoint_path}")