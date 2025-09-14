#!/usr/bin/env python3
"""
PPO学習スクリプト（ハイブリッドモード対応版）
"""

import os
import sys
import yaml
import argparse
import numpy as np
from datetime import datetime
import wandb

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from reinforcement_learning.environment.ems_environment import EMSEnvironment
from reinforcement_learning.agents.ppo_agent import PPOAgent
from reinforcement_learning.training.trainer import PPOTrainer

def load_config(config_path):
    """設定ファイルを読み込み、継承を処理"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # inherits指定がある場合、ベース設定を読み込んで統合
    if 'inherits' in config:
        base_path = config['inherits']
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        
        with open(base_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # ベース設定に現在の設定を上書き
        merged_config = deep_merge(base_config, config)
        return merged_config
    
    return config

def deep_merge(base_dict, override_dict):
    """辞書を再帰的にマージ"""
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def main():
    """メイン実行関数"""
    
    # コマンドライン引数
    parser = argparse.ArgumentParser(description='PPO学習（ハイブリッドモード対応）')
    parser.add_argument(
        '--config', 
        type=str, 
        default='experiments/config_tokyo23_hybrid.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['normal', 'hybrid'],
        default='normal',
        help='学習モード（normal: 通常, hybrid: ハイブリッド）'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='W&Bプロジェクト名（設定ファイルの値を上書き）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='ランダムシード（設定ファイルの値を上書き）'
    )
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    print(f"設定ファイルを読み込み中: {args.config}")
    config = load_config(args.config)
    
    # コマンドライン引数で設定を上書き
    if args.seed is not None:
        config['experiment']['seed'] = args.seed
    
    # ハイブリッドモードの確認と設定
    is_hybrid = config.get('hybrid_mode', {}).get('enabled', False)
    if args.mode == 'hybrid' and not is_hybrid:
        print("警告: 設定ファイルでハイブリッドモードが無効になっています。有効化します。")
        if 'hybrid_mode' not in config:
            config['hybrid_mode'] = {}
        config['hybrid_mode']['enabled'] = True
        is_hybrid = True
    
    # 実験情報の表示
    print("\n" + "=" * 60)
    print("実験設定:")
    print(f"- 実験名: {config.get('experiment', {}).get('name', 'unnamed')}")
    print(f"- モード: {'ハイブリッド' if is_hybrid else '通常'}")
    if is_hybrid:
        print("  - 重症系（重症・重篤・死亡）: 直近隊運用")
        print("  - 軽症系（軽症・中等症）: PPO学習")
        print(f"  - 報酬バランス: RT {config['hybrid_mode']['reward_weights']['response_time']:.0%}, "
              f"カバレッジ {config['hybrid_mode']['reward_weights']['coverage']:.0%}, "
              f"稼働 {config['hybrid_mode']['reward_weights']['workload_balance']:.0%}")
    print(f"- エピソード数: {config['ppo']['n_episodes']}")
    print(f"- デバイス: {config['experiment'].get('device', 'cpu')}")
    print("=" * 60 + "\n")
    
    # W&B初期化
    if config.get('training', {}).get('logging', {}).get('wandb', False):
        wandb_project = args.wandb_project or config.get('training', {}).get('logging', {}).get('wandb_project', 'ems_ppo')
        
        wandb.init(
            project=wandb_project,
            name=f"{config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            tags=['hybrid'] if is_hybrid else ['normal']
        )
        print(f"W&B初期化完了: {wandb_project}")
    
    # 環境とエージェントの初期化
    print("\n環境を初期化中...")
    env = EMSEnvironment(config)
    
    print("エージェントを初期化中...")
    state_dim = config['data']['area_restriction'].get('state_dim', env.state_dim)
    action_dim = config['data']['area_restriction'].get('action_dim', env.action_dim)
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config['ppo']
    )
    
    # トレーナーの初期化と学習
    print("トレーナーを初期化中...")
    trainer = PPOTrainer(config, env, agent)
    
    print("\n学習を開始します...")
    print("-" * 60)
    
    try:
        best_reward = trainer.train()
        
        # モデル保存
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{'hybrid' if is_hybrid else 'normal'}_ppo_{timestamp}.pth"
        model_path = os.path.join(model_dir, model_name)
        
        agent.save(model_path)
        print(f"\nモデルを保存しました: {model_path}")
        
        # 最終統計の表示
        if is_hybrid and hasattr(trainer, 'hybrid_stats'):
            print("\n" + "=" * 60)
            print("ハイブリッドモード学習結果:")
            
            stats = trainer.hybrid_stats
            if stats.get('severe_rt_history'):
                print(f"- 重症系平均RT: {np.mean(stats['severe_rt_history']):.1f}秒 "
                      f"({np.mean(stats['severe_rt_history'])/60:.1f}分)")
            
            if stats.get('mild_rt_history'):
                print(f"- 軽症系平均RT: {np.mean(stats['mild_rt_history']):.1f}秒 "
                      f"({np.mean(stats['mild_rt_history'])/60:.1f}分)")
            
            if stats.get('coverage_history'):
                print(f"- 平均カバレッジ: {np.mean(stats['coverage_history']):.2%}")
            
            if 'episodes_with_warning' in stats:
                total_episodes = config['ppo']['n_episodes']
                warning_rate = stats['episodes_with_warning'] / total_episodes
                print(f"- 20分超過エピソード: {stats['episodes_with_warning']}回 ({warning_rate:.1%})")
            
            print("=" * 60)
        
        print(f"\n最終報酬: {best_reward:.2f}")
        
    except KeyboardInterrupt:
        print("\n\n学習を中断しました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if config.get('training', {}).get('logging', {}).get('wandb', False):
            wandb.finish()

if __name__ == "__main__":
    main()