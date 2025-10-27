"""
debug_hybrid_logic_direct.py
ハイブリッドロジックが実際に機能しているか直接確認

目的:
1. PPO戦略のハイブリッドモード設定を確認
2. 各傷病度でどのメソッドが呼ばれるかテスト
3. 条件分岐が正しく動作するか確認
"""

import sys
from dispatch_strategies import PPOStrategy, EmergencyRequest, AmbulanceInfo, DispatchContext, DispatchPriority

def test_hybrid_logic():
    """ハイブリッドロジックの動作を直接テスト"""
    print("=" * 80)
    print("ハイブリッドロジック 直接テスト")
    print("=" * 80)
    
    # PPOStrategyの初期化
    print("\n1. PPO戦略初期化中...")
    ppo_config = {
        'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/final_model.pth',
        'config_path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/configs/config.json',
        'hybrid_mode': True,  # ←ハイブリッドモード有効
        'severe_conditions': ['重症', '重篤', '死亡'],
        'mild_conditions': ['軽症', '中等症']
    }
    
    strategy = PPOStrategy()
    strategy.initialize(ppo_config)
    
    print(f"\n✅ 初期化完了")
    print(f"   hybrid_mode: {strategy.hybrid_mode}")
    print(f"   severe_conditions: {strategy.severe_conditions}")
    print(f"   mild_conditions: {strategy.mild_conditions}")
    
    # 2. 各傷病度でテスト
    print("\n" + "=" * 80)
    print("2. 傷病度別の動作確認")
    print("=" * 80)
    
    test_severities = ['軽症', '中等症', '重症', '重篤', '死亡']
    
    for severity in test_severities:
        print(f"\n--- {severity} ---")
        
        # EmergencyRequestを作成
        request = EmergencyRequest(
            id=f"test_{severity}",
            h3_index="892f5a3269bffff",
            severity=severity,
            time=0.0,
            priority=strategy.get_severity_priority(severity)
        )
        
        # 条件判定をテスト
        is_hybrid = strategy.hybrid_mode
        is_severe = request.severity in strategy.severe_conditions
        
        print(f"  hybrid_mode: {is_hybrid}")
        print(f"  severity in severe_conditions: {is_severe}")
        print(f"  → 使用されるロジック: ", end="")
        
        if is_hybrid and is_severe:
            print("✅ 直近隊（_select_closest）")
        else:
            print("🔵 PPO（_select_with_ppo）")
    
    # 3. 文字列マッチングの詳細確認
    print("\n" + "=" * 80)
    print("3. 文字列マッチングの詳細確認")
    print("=" * 80)
    
    for severity in test_severities:
        print(f"\n{severity}:")
        print(f"  repr: {repr(severity)}")
        print(f"  len: {len(severity)}")
        print(f"  in severe_conditions: {severity in strategy.severe_conditions}")
        print(f"  in mild_conditions: {severity in strategy.mild_conditions}")
        
        # 個別に比較
        for severe in strategy.severe_conditions:
            match = (severity == severe)
            print(f"  == '{severe}': {match}")
    
    print("\n" + "=" * 80)
    print("診断完了")
    print("=" * 80)

if __name__ == "__main__":
    test_hybrid_logic()

