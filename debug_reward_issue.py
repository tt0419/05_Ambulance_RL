"""
debug_reward_issue.py
報酬が全て0になる問題を診断

確認項目:
1. EMSEnvironmentのhybrid_mode設定
2. RewardDesignerのmode設定
3. 実際の報酬計算
4. 事案の傷病度分布
"""

import sys
import yaml
import json
from pathlib import Path
from reinforcement_learning.environment.ems_environment import EMSEnvironment
from reinforcement_learning.environment.reward_designer import RewardDesigner

def diagnose_reward_issue():
    """報酬計算の問題を診断"""
    
    print("=" * 80)
    print("報酬計算問題の診断")
    print("=" * 80)
    
    # 1. 設定ファイルの確認
    print("\n【Step 1】設定ファイルの確認")
    print("-" * 80)
    
    config_path = "reinforcement_learning/experiments/config_tokyo23_simple.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"設定ファイル: {config_path}")
    print(f"  reward.core.mode: {config.get('reward', {}).get('core', {}).get('mode', 'N/A')}")
    print(f"  hybrid_mode.enabled: {config.get('hybrid_mode', {}).get('enabled', 'N/A')}")
    
    # simple_params を確認
    simple_params = config.get('reward', {}).get('core', {}).get('simple_params', {})
    print(f"\n  simple_params:")
    for key, value in simple_params.items():
        print(f"    {key}: {value}")
    
    # 2. EMSEnvironment の初期化と設定確認
    print("\n【Step 2】EMSEnvironment の設定確認")
    print("-" * 80)
    
    try:
        env = EMSEnvironment(config_path)
        print(f"✅ EMSEnvironment初期化成功")
        print(f"  hybrid_mode: {env.hybrid_mode}")
        if env.hybrid_mode:
            print(f"  severe_conditions: {env.severe_conditions}")
            print(f"  mild_conditions: {env.mild_conditions}")
        
        # RewardDesigner の設定確認
        print(f"\n  RewardDesigner:")
        print(f"    mode: {env.reward_designer.mode}")
        print(f"    hybrid_enabled: {env.reward_designer.hybrid_enabled}")
        
        if hasattr(env.reward_designer, 'simple_params'):
            print(f"    simple_params loaded: Yes")
            print(f"      time_penalty_per_minute: {env.reward_designer.simple_params.get('time_penalty_per_minute', 'N/A')}")
        else:
            print(f"    simple_params loaded: No")
        
    except Exception as e:
        print(f"❌ EMSEnvironment初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 報酬計算のテスト
    print("\n【Step 3】報酬計算のテスト")
    print("-" * 80)
    
    test_cases = [
        {'severity': '軽症', 'response_time': 8.0},
        {'severity': '軽症', 'response_time': 15.0},
        {'severity': '中等症', 'response_time': 10.0},
        {'severity': '重症', 'response_time': 5.0},
    ]
    
    for case in test_cases:
        severity = case['severity']
        response_time = case['response_time']
        
        # RewardDesignerで直接計算
        reward = env.reward_designer.calculate_reward(
            severity=severity,
            response_time_minutes=response_time,
            coverage_before=0.7,
            coverage_after=0.7,
            coverage_impact=0.0,
            additional_info={}
        )
        
        print(f"\n  事案: 傷病度={severity}, 応答時間={response_time}分")
        print(f"    計算された報酬: {reward:.2f}")
        
        # 期待される報酬を手動計算
        if env.reward_designer.mode == 'simple':
            expected = simple_params['time_penalty_per_minute'] * response_time
            if response_time <= 13 and severity in ['軽症', '中等症']:
                expected += simple_params['mild_under_13min_bonus']
            if response_time > 13:
                expected += simple_params['over_13min_penalty']
            if response_time > 20:
                expected += simple_params['over_20min_penalty']
            
            print(f"    期待される報酬: {expected:.2f}")
            
            if abs(reward - expected) < 0.01:
                print(f"    ✅ 報酬計算が正しい")
            else:
                print(f"    ❌ 報酬計算に誤差あり")
    
    # 4. 1エピソードのシミュレーション
    print("\n【Step 4】1エピソードのテスト実行")
    print("-" * 80)
    
    obs = env.reset()
    total_reward = 0
    non_zero_rewards = 0
    steps = 0
    max_steps = 50  # 最初の50ステップのみ
    
    print("シミュレーション開始...")
    
    while not env.episode_done and steps < max_steps:
        # ランダムな行動
        import numpy as np
        available_mask = env.get_available_ambulances_mask()
        if available_mask.any():
            available_actions = np.where(available_mask)[0]
            action = np.random.choice(available_actions)
        else:
            action = 0
        
        # ステップ実行
        result = env.step(action)
        obs = result.observation
        reward = result.reward
        
        total_reward += reward
        if abs(reward) > 0.001:
            non_zero_rewards += 1
            if non_zero_rewards <= 5:  # 最初の5件のみ表示
                incident_severity = env.pending_call.get('severity', 'N/A') if env.pending_call else 'N/A'
                print(f"  Step {steps}: 報酬={reward:.2f}, 傷病度={incident_severity}")
        
        steps += 1
    
    print(f"\nテスト完了:")
    print(f"  総ステップ数: {steps}")
    print(f"  総報酬: {total_reward:.2f}")
    print(f"  非ゼロ報酬の回数: {non_zero_rewards}/{steps}")
    print(f"  平均報酬: {total_reward/steps:.2f}")
    
    # 診断結果
    print("\n" + "=" * 80)
    print("診断結果")
    print("=" * 80)
    
    if non_zero_rewards == 0:
        print("❌ 全ての報酬が0です")
        print("\n考えられる原因:")
        if env.hybrid_mode:
            print("  1. hybrid_mode が有効になっています")
            print("     → 全事案が重症系として処理され、報酬が0になっている可能性")
        else:
            print("  1. EMSEnvironmentのstep()で報酬が正しく計算されていない")
            print("  2. RewardDesignerの実装にバグがある")
    elif non_zero_rewards < steps * 0.5:
        print("⚠️  約半分の報酬が0です")
        print("\n考えられる原因:")
        print("  1. hybrid_modeで重症系が直近隊運用されている（正常）")
        print("  2. 一部の事案で報酬計算がスキップされている")
    else:
        print("✅ 報酬は正常に計算されています")
        print("\n学習が進まない場合は:")
        print("  1. 報酬のスケールが小さすぎる可能性")
        print("  2. 学習率が高すぎる/低すぎる可能性")
        print("  3. バッチサイズやエポック数の問題")

if __name__ == "__main__":
    diagnose_reward_issue()

