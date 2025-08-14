"""
test_ppo.py
PPO実装のテスト実行スクリプト
"""

import sys
import os
import traceback
import yaml
from pathlib import Path
import torch
import numpy as np

# パス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """必要なモジュールのインポートテスト"""
    print("=" * 60)
    print("1. インポートテスト")
    print("=" * 60)
    
    try:
        from reinforcement_learning.environment.ems_environment import EMSEnvironment
        print("✓ EMSEnvironment")
    except Exception as e:
        print(f"✗ EMSEnvironment: {e}")
        return False
    
    try:
        from reinforcement_learning.environment.state_encoder import StateEncoder
        print("✓ StateEncoder")
    except Exception as e:
        print(f"✗ StateEncoder: {e}")
        return False
    
    try:
        from reinforcement_learning.environment.reward_designer import RewardDesigner
        print("✓ RewardDesigner")
    except Exception as e:
        print(f"✗ RewardDesigner: {e}")
        return False
    
    try:
        from reinforcement_learning.agents.ppo_agent import PPOAgent
        print("✓ PPOAgent")
    except Exception as e:
        print(f"✗ PPOAgent: {e}")
        return False
    
    try:
        from reinforcement_learning.agents.network_architectures import ActorNetwork, CriticNetwork
        print("✓ Networks")
    except Exception as e:
        print(f"✗ Networks: {e}")
        return False
    
    try:
        from reinforcement_learning.agents.buffer import RolloutBuffer
        print("✓ RolloutBuffer")
    except Exception as e:
        print(f"✗ RolloutBuffer: {e}")
        return False
    
    try:
        from reinforcement_learning.training.trainer import PPOTrainer
        print("✓ PPOTrainer")
    except Exception as e:
        print(f"✗ PPOTrainer: {e}")
        return False
    
    print("全モジュールのインポート成功！\n")
    return True

def test_environment():
    """環境の初期化テスト"""
    print("=" * 60)
    print("2. 環境初期化テスト")
    print("=" * 60)
    
    try:
        from reinforcement_learning.environment.ems_environment import EMSEnvironment
        
        # テスト設定ファイルのパス
        config_path = "reinforcement_learning/experiments/config_test.yaml"
        if not os.path.exists(config_path):
            print(f"設定ファイルが見つかりません: {config_path}")
            print("reinforcement_learning/experiments/config_test.yamlを作成してください")
            return False
        
        # 環境の初期化
        print("環境を初期化中...")
        env = EMSEnvironment(config_path, mode="train")
        print("✓ 環境初期化成功")
        
        # リセットテスト
        print("\n環境リセット中...")
        initial_state = env.reset()
        print(f"✓ 初期状態の次元: {initial_state.shape}")
        
        # 行動マスク取得
        action_mask = env.get_action_mask()
        print(f"✓ 利用可能な救急車: {np.sum(action_mask)}/192台")
        
        # 1ステップ実行
        print("\n1ステップ実行中...")
        available_actions = np.where(action_mask)[0]
        if len(available_actions) > 0:
            action = available_actions[0]
            step_result = env.step(action)
            print(f"✓ ステップ実行成功")
            print(f"  報酬: {step_result.reward:.2f}")
            print(f"  終了: {step_result.done}")
        
        print("\n環境テスト成功！\n")
        return True
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        traceback.print_exc()
        return False

def test_agent():
    """エージェントの初期化テスト"""
    print("=" * 60)
    print("3. PPOエージェント初期化テスト")
    print("=" * 60)
    
    try:
        from reinforcement_learning.agents.ppo_agent import PPOAgent
        
        # 設定読み込み
        config_path = "reinforcement_learning/experiments/config_test.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 環境から状態次元を取得
        from reinforcement_learning.environment.ems_environment import EMSEnvironment
        env = EMSEnvironment(config_path, mode="train")
        initial_state = env.reset()
        state_dim = len(initial_state)
        action_dim = 192
        
        # デバイス選択
        device = get_device()
        
        # エージェント初期化
        print("PPOエージェントを初期化中...")
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config['ppo'],
            device=device
        )
        print("✓ エージェント初期化成功")
        
        # ネットワーク確認
        print(f"✓ Actorパラメータ数: {sum(p.numel() for p in agent.actor.parameters()):,}")
        print(f"✓ Criticパラメータ数: {sum(p.numel() for p in agent.critic.parameters()):,}")
        
        # 行動選択テスト
        print("\n行動選択テスト...")
        dummy_state = np.random.randn(state_dim).astype(np.float32)
        dummy_mask = np.ones(action_dim, dtype=bool)
        dummy_mask[100:] = False  # 一部を利用不可に
        
        action, log_prob, value = agent.select_action(dummy_state, dummy_mask)
        print(f"✓ 選択された行動: {action}")
        print(f"✓ 対数確率: {log_prob:.4f}")
        print(f"✓ 状態価値: {value:.4f}")
        
        print("\nエージェントテスト成功！\n")
        return True
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        traceback.print_exc()
        return False

def test_mini_training():
    """ミニ学習テスト"""
    print("=" * 60)
    print("4. ミニ学習テスト（5エピソード）")
    print("=" * 60)
    
    try:
        from reinforcement_learning.environment.ems_environment import EMSEnvironment
        from reinforcement_learning.agents.ppo_agent import PPOAgent
        
        # 設定読み込み
        config_path = "reinforcement_learning/experiments/config_test.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 環境とエージェントの初期化
        print("環境とエージェントを初期化中...")
        env = EMSEnvironment(config_path, mode="train")
        
        # 状態次元を環境から取得
        initial_state = env.reset()
        state_dim = len(initial_state)
        action_dim = 192
        
        # デバイス選択
        device = get_device()
        
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config['ppo'],
            device=device
        )
        print("✓ 初期化完了")
        
        # 5エピソード実行
        print("\n学習開始...")
        for episode in range(1, 6):
            state = env.reset()
            episode_reward = 0
            steps = 0
            
            # 最大100ステップまで
            for _ in range(100):
                action_mask = env.get_action_mask()
                action, log_prob, value = agent.select_action(state, action_mask)
                
                step_result = env.step(action)
                next_state = step_result.observation
                reward = step_result.reward
                done = step_result.done
                
                # バッファに保存
                agent.store_transition(
                    state, action, reward, next_state, 
                    done, log_prob, value, action_mask
                )
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            print(f"Episode {episode}: 報酬={episode_reward:.2f}, ステップ数={steps}")
            
            # PPO更新（バッファが十分な場合）
            if len(agent.buffer) >= agent.batch_size:
                update_stats = agent.update()
                if update_stats:
                    print(f"  更新: Actor損失={update_stats['actor_loss']:.4f}, "
                          f"Critic損失={update_stats['critic_loss']:.4f}")
        
        print("\nミニ学習テスト成功！\n")
        return True
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        traceback.print_exc()
        return False

def get_device():
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU利用可能: {torch.cuda.get_device_name(0)}")
        print(f"GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        return device
    else:
        print("GPUが利用できません。CPUを使用します。")
        return torch.device("cpu")

def main():
    """メインテスト実行"""
    print("\n" + "=" * 60)
    print("PPO実装テスト開始")
    print("=" * 60 + "\n")
    
    # デバイス選択
    device = get_device()
    print()
    
    # 各テストを実行
    tests = [
        ("インポート", test_imports),
        ("環境", test_environment),
        ("エージェント", test_agent),
        ("ミニ学習", test_mini_training)
    ]
    
    results = []
    for name, test_func in tests:
        success = test_func()
        results.append((name, success))
        if not success:
            print(f"\n{name}テストで失敗しました。")
            break
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失敗"
        print(f"{name}テスト: {status}")
    
    if all(success for _, success in results):
        print("\n全テスト成功！本格的な学習を開始できます。")
        print("\n次のステップ:")
        print("1. config.yamlを作成（本番用設定）")
        print("2. python train_ppo.py --config config.yaml")
    else:
        print("\nテストに失敗しました。エラーを確認してください。")

if __name__ == "__main__":
    main()