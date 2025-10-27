"""
compare_ppo_vs_nearest.py
PPOと直近隊の選択を直接比較

軽症事案において：
- PPOが選択する救急車
- 直近隊が選択する救急車
- 両者の応答時間の差
を分析
"""

import sys
import numpy as np
from pathlib import Path
from dispatch_strategies import PPOStrategy, NearestAmbulanceStrategy, EmergencyRequest, AmbulanceInfo, DispatchContext, DispatchPriority
from validation_simulation import ValidationSimulator, get_emergency_data_cache
import pandas as pd

def compare_selections():
    """PPOと直近隊の選択を比較"""
    
    print("=" * 80)
    print("PPO vs 直近隊 選択比較分析")
    print("=" * 80)
    
    # 戦略の初期化
    print("\n1. PPO戦略初期化中...")
    ppo_strategy = PPOStrategy()
    ppo_config = {
        'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/final_model.pth',
        'config_path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/configs/config.json',
        'hybrid_mode': False,  # ハイブリッド無効で全てPPO選択
        'severe_conditions': ['重症', '重篤', '死亡'],
        'mild_conditions': ['軽症', '中等症']
    }
    ppo_strategy.initialize(ppo_config)
    
    print("\n2. 直近隊戦略初期化中...")
    nearest_strategy = NearestAmbulanceStrategy()
    nearest_strategy.initialize({})
    
    print("\n3. テストデータ準備中...")
    # 簡易的なテストケースを作成
    test_cases = []
    
    # 軽症の事案を10件作成
    severities = ['軽症'] * 7 + ['中等症'] * 3
    h3_indices = [
        '892f5a3269bffff',
        '892f5a32693ffff', 
        '892f5a32c4bffff',
        '892f5a36343ffff',
        '892f5aade27ffff',
        '892f5aadecbffff',
        '892f5a329b3ffff',
        '892f5aad893ffff',
        '892f5aaca53ffff',
        '892f5aad287ffff'
    ]
    
    for i, (severity, h3) in enumerate(zip(severities, h3_indices)):
        request = EmergencyRequest(
            id=f"test_{i}",
            h3_index=h3,
            severity=severity,
            time=0.0,
            priority=DispatchPriority.LOW if severity == '軽症' else DispatchPriority.MEDIUM
        )
        test_cases.append(request)
    
    # ダミーの救急車リストを作成（実際のシミュレータから取得するのが理想）
    print("\n⚠️ 注意: このスクリプトは簡易版です")
    print("完全な比較には、実際のシミュレーション環境が必要です")
    print()
    print("次のステップ:")
    print("  → debug_ppo_io_trace.py を実行してPPOの入出力を確認")
    print("  → PPOの状態エンコーディングを検証")
    print("  → PPOモデルの再学習が必要かどうか判断")
    
    return

if __name__ == "__main__":
    compare_selections()

