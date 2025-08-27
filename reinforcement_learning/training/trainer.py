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
import sys
import os

# 統一された傷病度定数をインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from constants import get_severity_english

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
        
        # ★★★ wandb初期化の修正 ★★★
        if self.use_wandb:
            try:
                print("WandB初期化中...")
                
                # wandb設定の準備
                wandb_config = {
                    "project": "ems-dispatch-optimization",
                    "entity": None,  # 自動検出
                    "name": f"{config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "tags": ["ppo", "emergency-dispatch", "tokyo"],
                    "notes": f"PPO training with {config['ppo']['n_episodes']} episodes",
                    "config": config,
                    "settings": wandb.Settings(
                        init_timeout=300,  # タイムアウトを5分に延長
                        _disable_stats=True,  # システム統計を無効化（軽量化）
                        _disable_meta=True,   # メタデータを無効化（軽量化）
                    )
                }
                
                # 初期化実行
                wandb.init(**wandb_config)
                print("✓ WandB初期化成功")
                
            except Exception as e:
                print(f"⚠️ WandB初期化に失敗しました: {e}")
                print("WandBを無効にして学習を続行します...")
                self.use_wandb = False
        
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
            # ★★★【修正点②】★★★
            # configファイルから教師あり学習の設定を読み込む
            teacher_config = self.config.get('teacher', {})
            if teacher_config.get('enabled', False):
                current_episode = len(self.episode_rewards)
                
                # 設定値の取得 (デフォルト値も設定)
                initial_prob = teacher_config.get('initial_prob', 0.8)
                final_prob = teacher_config.get('final_prob', 0.2)
                decay_episodes = teacher_config.get('decay_episodes', 1000)
                
                if current_episode >= decay_episodes:
                    teacher_prob = final_prob
                else:
                    # 線形に減衰
                    decay_rate = (initial_prob - final_prob) / decay_episodes
                    teacher_prob = initial_prob - decay_rate * current_episode
            else:
                teacher_prob = 0.0
            # ★★★【修正ここまで】★★★
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
        
        # エピソード統計（拡張版統計を取得）
        episode_stats = self.env.get_episode_statistics() if hasattr(self.env, 'get_episode_statistics') else self.env.episode_stats
        
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
        """学習進捗のログ出力（拡張版）"""
        # 直近の平均
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        avg_reward = np.mean(recent_rewards)
        
        # 基本メトリクス（毎エピソード表示）
        print(f"Episode {episode}/{self.n_episodes}")
        print(f"  報酬: {reward:.2f} (平均: {avg_reward:.2f})")
        print(f"  長さ: {length}")
        
        # 救急ディスパッチ特有のメトリクス
        if stats and 'response_times' in stats and stats['response_times']:
            response_times = stats['response_times']
            avg_rt = np.mean(response_times)
            median_rt = np.median(response_times)
            
            print(f"  平均応答時間: {avg_rt:.2f}分")
            
            # 目標達成率
            under_6min = sum(1 for rt in response_times if rt <= 6) / len(response_times) * 100
            under_13min = sum(1 for rt in response_times if rt <= 13) / len(response_times) * 100
            print(f"  6分達成率: {under_6min:.1f}%")
            
            # 詳細統計は10エピソードごとに表示
            if episode % 10 == 0:
                rt_95th = np.percentile(response_times, 95)
                print(f"  応答時間詳細: 中央値={median_rt:.2f}分, 95%ile={rt_95th:.2f}分")
        
        # 傷病度別性能（10エピソードごと）
        if episode % 10 == 0 and stats and 'response_times_by_severity' in stats:
            severity_stats = stats['response_times_by_severity']
            for severity, times in severity_stats.items():
                if times:
                    avg_time = np.mean(times)
                    under_6 = sum(1 for t in times if t <= 6) / len(times) * 100
                    print(f"    {severity}: 平均={avg_time:.2f}分, 6分以内={under_6:.1f}%")
        
        # 学習統計（10エピソードごと）
        if episode % 10 == 0 and self.training_stats:
            latest_stats = self.training_stats[-1]
            print(f"  Actor損失: {latest_stats.get('actor_loss', 0):.4f}")
            print(f"  Critic損失: {latest_stats.get('critic_loss', 0):.4f}")
        
        # WandBログ（拡張版）
        if self.use_wandb:
            log_data = {
                # 基本メトリクス
                'episode': episode,
                'reward/episode': reward,
                'reward/avg_100': avg_reward,
                'episode_length': length,
            }
            
            # 応答時間メトリクス
            if stats and 'response_times' in stats and stats['response_times']:
                response_times = stats['response_times']
                log_data.update({
                    'performance/mean_response_time': np.mean(response_times),
                    'performance/median_response_time': np.median(response_times),
                    'performance/95th_response_time': np.percentile(response_times, 95),
                    'performance/6min_achievement_rate': sum(1 for rt in response_times if rt <= 6) / len(response_times),
                    'performance/13min_achievement_rate': sum(1 for rt in response_times if rt <= 13) / len(response_times),
                    'performance/total_calls': len(response_times)
                })
            
            # 傷病度別メトリクス
            if stats and 'response_times_by_severity' in stats:
                for severity, times in stats['response_times_by_severity'].items():
                    if times:
                        # 統一された傷病度英語マッピングを使用
                        severity_key = get_severity_english(severity)
                        log_data[f'severity/{severity_key}_mean_time'] = np.mean(times)
                        log_data[f'severity/{severity_key}_6min_rate'] = sum(1 for t in times if t <= 6) / len(times)
                        log_data[f'severity/{severity_key}_count'] = len(times)
            
            # 学習統計
            if self.training_stats:
                latest_stats = self.training_stats[-1]
                log_data.update({
                    'train/actor_loss': latest_stats.get('actor_loss', 0),
                    'train/critic_loss': latest_stats.get('critic_loss', 0),
                    'train/kl_divergence': latest_stats.get('kl_div', 0),
                    'train/entropy': latest_stats.get('entropy', 0),
                })
            
            wandb.log(log_data)
        
        # 追加のログ機能
        if episode % 25 == 0:
            self._log_baseline_comparison(episode)
            self._log_curriculum_progress(episode)
    
    def _log_baseline_comparison(self, episode: int):
        """ベースライン手法との比較ログ"""
        try:
            # 現在のPPO性能（直近10エピソード平均）
            recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
            if not recent_rewards:
                return
            
            ppo_performance = np.mean(recent_rewards)
            
            # ベースライン性能（推定値 - 実際のベースライン実験結果に基づく）
            baseline_performance = {
                'closest': -3200,  # 直近隊運用の推定性能
                'severity_based': -2800,  # 傷病度考慮運用の推定性能
            }
            
            # 改善率計算
            improvements = {}
            for baseline_name, baseline_score in baseline_performance.items():
                improvement = (baseline_score - ppo_performance) / abs(baseline_score) * 100
                improvements[baseline_name] = improvement
            
            if self.use_wandb:
                comparison_data = {
                    f'comparison/vs_{name}': improvement 
                    for name, improvement in improvements.items()
                }
                comparison_data['comparison/episode'] = episode
                wandb.log(comparison_data)
            
            print(f"\n=== ベースライン比較 (Episode {episode}) ===")
            for name, improvement in improvements.items():
                print(f"  vs {name}: {improvement:+.1f}% {'向上' if improvement > 0 else '悪化'}")
                
        except Exception as e:
            print(f"ベースライン比較ログでエラー: {e}")
    
    def _log_curriculum_progress(self, episode: int):
        """カリキュラム学習の進捗ログ"""
        try:
            # 教師あり学習の設定確認
            if 'teacher' in self.config:
                teacher_config = self.config['teacher']
                if teacher_config.get('enabled', False):
                    # 現在の教師使用率を計算
                    decay_episodes = teacher_config.get('decay_episodes', 100)
                    initial_prob = teacher_config.get('initial_prob', 0.8)
                    final_prob = teacher_config.get('final_prob', 0.2)
                    
                    # エピソード数に応じた確率計算
                    if episode <= decay_episodes:
                        progress = episode / decay_episodes
                        current_prob = initial_prob - (initial_prob - final_prob) * progress
                    else:
                        current_prob = final_prob
                    
                    if self.use_wandb:
                        wandb.log({
                            'curriculum/teacher_probability': current_prob,
                            'curriculum/progress': min(episode / decay_episodes, 1.0),
                            'curriculum/episode': episode
                        })
                    
                    if episode % 50 == 0:  # 50エピソードごとに表示
                        print(f"  教師確率: {current_prob:.2f}")
                        
        except Exception as e:
            print(f"カリキュラム学習ログでエラー: {e}")
    
    def _save_checkpoint(self, episode: int, is_best: bool = False):
        """チェックポイントの保存"""
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_ep{episode}.pth"
        
        self.agent.save(path)
        print(f"モデル保存完了: {path}")
        
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