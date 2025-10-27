"""
analyze_ppo_selections.py
PPOの救急車選択を詳細分析

目的:
1. PPOが選択した救急車と直近隊の比較
2. 応答時間の差を分析
3. PPOの選択パターンを特定
"""

import sys
from pathlib import Path
from validation_simulation import run_validation_simulation
import json
import pandas as pd
import numpy as np

def run_detailed_ppo_analysis():
    """PPOの選択を詳細分析"""
    
    print("=" * 80)
    print("PPO選択の詳細分析")
    print("=" * 80)
    
    # PPO設定
    ppo_config = {
        'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/final_model.pth',
        'config_path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/configs/config.json',
        'hybrid_mode': True,
        'severe_conditions': ['重症', '重篤', '死亡'],
        'mild_conditions': ['軽症', '中等症']
    }
    
    target_date = '20230615'
    output_dir = 'debug_output/ppo_analysis_full'
    
    print(f"\n実行日: {target_date}")
    print(f"実行時間: 24時間")
    print(f"出力先: {output_dir}")
    print()
    print("シミュレーション開始...")
    print("=" * 80)
    
    # 24時間フル実行
    run_validation_simulation(
        target_date_str=target_date,
        output_dir=output_dir,
        simulation_duration_hours=24.0,
        random_seed=42,
        verbose_logging=False,
        enable_visualization=False,
        enable_detailed_reports=True,
        dispatch_strategy='ppo_agent',
        strategy_config=ppo_config
    )
    
    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)
    
    # レポート読み込み
    report_path = Path(output_dir) / "simulation_report.json"
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        print("\n結果サマリー:")
        print(f"  総事案数: {report.get('total_calls', 0)}件")
        print(f"  平均応答時間: {report.get('overall_avg_response_time_minutes', 0):.2f}分")
        
        severity_stats = report.get('severity_breakdown', {})
        print("\n傷病度別:")
        for severity, stats in severity_stats.items():
            print(f"  {severity}: {stats.get('avg_response_time_minutes', 0):.2f}分 (n={stats.get('count', 0)})")

if __name__ == "__main__":
    run_detailed_ppo_analysis()

