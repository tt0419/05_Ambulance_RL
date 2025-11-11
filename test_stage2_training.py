"""
test_stage2_training.py
Stage 2（イベント駆動版）の学習動作確認スクリプト

目的：
- trainer.pyとの統合に問題がないか確認
- イベント駆動ループが学習中に正常に動作するか確認
- 短期間（5エピソード）で動作確認
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reinforcement_learning.environment.ems_environment_v2 import EMSEnvironment
from reinforcement_learning.training.trainer import PPOTrainer
from reinforcement_learning.config_utils import load_config_with_inheritance
import numpy as np

def test_training_integration(num_episodes=5):
    """学習との統合テスト"""
    print("=" * 60)
    print(f"Stage 2 学習統合テスト（{num_episodes}エピソード）")
    print("=" * 60)
    
    # 設定読み込み
    config = load_config_with_inheritance("reinforcement_learning/experiments/config_continuous.yaml")
    
    # 短期確認用に設定を上書き
    config['training']['max_episodes'] = num_episodes
    config['training']['save_interval'] = num_episodes + 1  # 保存しない
    config['training']['eval_interval'] = num_episodes + 1  # 評価しない
    
    print(f"\n設定:")
    print(f"  エピソード数: {num_episodes}")
    print(f"  状態空間: {config['environment']['state_dim']}次元")
    print(f"  行動空間: {config['environment']['action_dim']}次元")
    
    # 環境作成
    print(f"\n環境作成中...")
    env = EMSEnvironment(
        config_path="reinforcement_learning/experiments/config_continuous.yaml",
        mode="train"
    )
    print(f"✓ 環境作成完了")
    
    # トレーナー作成
    print(f"\nトレーナー作成中...")
    trainer = PPOTrainer(env, config)
    print(f"✓ トレーナー作成完了")
    
    # 学習実行
    print(f"\n{num_episodes}エピソードの学習を開始...")
    print("-" * 60)
    
    try:
        trainer.train()
        print("-" * 60)
        print(f"✓ {num_episodes}エピソードの学習が正常に完了しました")
        
        # 統計情報の表示
        print("\n統計情報:")
        if hasattr(trainer, 'episode_rewards') and len(trainer.episode_rewards) > 0:
            rewards = trainer.episode_rewards[-num_episodes:]
            print(f"  平均報酬: {np.mean(rewards):.2f}")
            print(f"  最大報酬: {np.max(rewards):.2f}")
            print(f"  最小報酬: {np.min(rewards):.2f}")
        
        print("\n" + "=" * 60)
        print("✅ Stage 2学習統合テスト合格")
        print("=" * 60)
        print("\n結論:")
        print("  - trainer.pyとの統合: 正常")
        print("  - イベント駆動ループ: 正常")
        print("  - 学習プロセス: 正常")
        print("\n次のステップ:")
        print("  → Stage 3の実装に進むことを推奨します")
        print("  → Stage 3完了後に本格的な学習を実行してください")
        
        return True
        
    except Exception as e:
        print("-" * 60)
        print(f"❌ 学習中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("❌ Stage 2学習統合テスト失敗")
        print("=" * 60)
        print("\n問題:")
        print(f"  {str(e)}")
        print("\n対応:")
        print("  → エラー内容を確認してください")
        print("  → Stage 3に進む前に修正が必要です")
        
        return False

def quick_sanity_check():
    """超短期の動作確認（1エピソード）"""
    print("\n" + "=" * 60)
    print("超短期動作確認（1エピソード）")
    print("=" * 60)
    
    env = EMSEnvironment(
        config_path="reinforcement_learning/experiments/config_continuous.yaml",
        mode="train"
    )
    
    obs = env.reset()
    print(f"✓ reset()完了: observation shape = {obs.shape}")
    
    steps = 0
    total_reward = 0.0
    
    while not env._is_episode_done() and steps < 10:  # 最初の10ステップのみ
        mask = env.get_action_mask()
        valid_actions = np.where(mask)[0]
        
        if len(valid_actions) == 0:
            action = 0
        else:
            action = valid_actions[0]
        
        result = env.step(action)
        total_reward += result.reward
        steps += 1
    
    print(f"✓ {steps}ステップ実行完了")
    print(f"✓ 累積報酬: {total_reward:.2f}")
    print(f"✓ 時間推移: {env.current_time:.2f}秒")
    
    print("\n✅ 超短期動作確認OK")
    return True

def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 2学習統合テスト')
    parser.add_argument('--episodes', type=int, default=5,
                       help='テスト実行するエピソード数（デフォルト: 5）')
    parser.add_argument('--quick', action='store_true',
                       help='超短期確認のみ（trainer不要）')
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            # 超短期確認のみ
            success = quick_sanity_check()
        else:
            # 完全な学習統合テスト
            success = test_training_integration(args.episodes)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()




