"""
test_reward_fix.py
修正後の報酬計算をテスト

実行方法:
python test_reward_fix.py
"""

import numpy as np
from reinforcement_learning.environment.ems_environment import EMSEnvironment

def test_reward_calculation():
    """報酬計算のクイックテスト"""
    
    print("=" * 80)
    print("報酬計算テスト（修正後）")
    print("=" * 80)
    
    config_path = "reinforcement_learning/experiments/config_tokyo23_simple.yaml"
    
    print(f"\n設定ファイル: {config_path}")
    print("環境を初期化中...")
    
    # 環境初期化
    env = EMSEnvironment(config_path)
    
    print(f"\n✅ 環境初期化完了")
    print(f"   mode: {env.mode}")
    print(f"   hybrid_mode: {env.hybrid_mode}")
    print(f"   reward_mode: {env.reward_designer.mode}")
    
    # 10ステップ実行
    print("\n" + "-" * 80)
    print("10ステップのテスト実行")
    print("-" * 80)
    
    obs = env.reset()
    total_reward = 0
    non_zero_count = 0
    
    for step in range(10):
        # 利用可能な救急車からランダムに選択
        available_mask = env.get_available_ambulances_mask()
        if available_mask.any():
            available_actions = np.where(available_mask)[0]
            action = np.random.choice(available_actions)
        else:
            action = 0
        
        # ステップ実行
        result = env.step(action)
        reward = result.reward
        total_reward += reward
        
        if abs(reward) > 0.001:
            non_zero_count += 1
        
        severity = env.pending_call.get('severity', 'N/A') if env.pending_call else 'N/A'
        print(f"  Step {step+1}: 報酬={reward:+7.2f}, 傷病度={severity}")
    
    print("\n" + "=" * 80)
    print("テスト結果")
    print("=" * 80)
    print(f"  総ステップ数: 10")
    print(f"  総報酬: {total_reward:.2f}")
    print(f"  非ゼロ報酬の回数: {non_zero_count}/10")
    print(f"  平均報酬: {total_reward/10:.2f}")
    
    if non_zero_count >= 8:  # 80%以上
        print("\n✅ 成功: 報酬が正しく計算されています！")
        print("   → 再学習を実行してください")
        return True
    elif non_zero_count > 0:
        print("\n⚠️  警告: 一部の報酬が0です")
        print(f"   → hybrid_mode が影響している可能性")
        return False
    else:
        print("\n❌ 失敗: 全ての報酬が0です")
        print("   → さらなる調査が必要")
        return False

if __name__ == "__main__":
    success = test_reward_calculation()
    
    if success:
        print("\n" + "=" * 80)
        print("次のステップ: 再学習を実行")
        print("=" * 80)
        print("\nコマンド:")
        print("  python train_ppo.py --config reinforcement_learning/experiments/config_tokyo23_simple.yaml")
        print()

