"""
test_hybrid_mode_simple.py
ハイブリッドモードが正しく動作しているか簡単なテストを実行

実行方法:
python test_hybrid_mode_simple.py
"""

import sys
from pathlib import Path
from validation_simulation import run_validation_simulation

def test_hybrid_mode():
    """ハイブリッドモードのテスト実行"""
    
    print("=" * 80)
    print("ハイブリッドモード 簡易テスト")
    print("=" * 80)
    
    # PPO設定
    ppo_config = {
        'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/final_model.pth',
        'config_path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/configs/config.json',
        'hybrid_mode': True,
        'severe_conditions': ['重症', '重篤', '死亡'],
        'mild_conditions': ['軽症', '中等症']
    }
    
    print("\n設定:")
    print(f"  hybrid_mode: {ppo_config['hybrid_mode']}")
    print(f"  severe_conditions: {ppo_config['severe_conditions']}")
    print(f"  mild_conditions: {ppo_config['mild_conditions']}")
    print()
    
    # 1時間だけ実行（デバッグ用）
    target_date = '20230615'  # YYYYMMDD形式
    output_dir = 'debug_output/hybrid_test'
    
    print(f"実行日: {target_date}")
    print(f"実行時間: 1時間")
    print(f"出力先: {output_dir}")
    print()
    print("シミュレーション開始...")
    print("（デバッグ出力が表示されます）")
    print("=" * 80)
    
    # シミュレーション実行
    run_validation_simulation(
        target_date_str=target_date,
        output_dir=output_dir,
        simulation_duration_hours=1.0,  # 1時間のみ
        random_seed=42,
        verbose_logging=False,
        enable_visualization=False,
        enable_detailed_reports=True,
        dispatch_strategy='ppo_agent',
        strategy_config=ppo_config
    )
    
    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)
    print("\n💡 デバッグ出力を確認してください:")
    print("  - [HYBRID-DEBUG]が表示されていれば、ハイブリッドロジックが動作しています")
    print("  - 直近隊選択とPPO選択の回数が表示されているはずです")
    print()

if __name__ == "__main__":
    test_hybrid_mode()

