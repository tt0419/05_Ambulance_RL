"""
test_new_model.py
新しいモデルの性能を検証

比較対象:
1. 旧モデル (ppo_20251017_113908)
2. 新モデル (ppo_20251017_160958)
3. 直近隊ベースライン
"""

import sys
from pathlib import Path
from validation_simulation import run_validation_simulation

def test_model(model_name, model_path, config_path):
    """指定されたモデルをテスト"""
    
    print("\n" + "=" * 80)
    print(f"モデルテスト: {model_name}")
    print("=" * 80)
    
    ppo_config = {
        'model_path': model_path,
        'config_path': config_path,
        'hybrid_mode': False,  # 全事案でPPOを使用（公平な比較のため）
        'severe_conditions': ['重症', '重篤', '死亡'],
        'mild_conditions': ['軽症', '中等症']
    }
    
    target_date = '20230615'
    output_dir = f'debug_output/model_test_{model_name}'
    
    print(f"\n実行日: {target_date}")
    print(f"実行時間: 24時間")
    print(f"出力先: {output_dir}")
    print(f"ハイブリッドモード: {ppo_config['hybrid_mode']}")
    print()
    print("シミュレーション開始...")
    print("=" * 80)
    
    # シミュレーション実行
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
    
    # 結果を読み込んで返す
    import json
    report_path = Path(output_dir) / "simulation_report.json"
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        return report
    return None

def main():
    """メインテスト"""
    
    print("=" * 80)
    print("PPO新旧モデル比較実験")
    print("=" * 80)
    
    # テスト対象
    models = [
        {
            'name': '旧モデル',
            'path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/final_model.pth',
            'config': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/configs/config.json'
        },
        {
            'name': '新モデル',
            'path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_160958/final_model.pth',
            'config': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_160958/configs/config.json'
        }
    ]
    
    results = {}
    
    for model in models:
        result = test_model(model['name'], model['path'], model['config'])
        if result:
            results[model['name']] = result
    
    # 結果比較
    print("\n" + "=" * 80)
    print("結果比較")
    print("=" * 80)
    
    for model_name, result in results.items():
        print(f"\n【{model_name}】")
        print(f"  全体平均応答時間: {result.get('overall_avg_response_time_minutes', 'N/A'):.2f}分")
        
        severity_stats = result.get('severity_breakdown', {})
        for severity in ['軽症', '中等症', '重症', '重篤', '死亡']:
            if severity in severity_stats:
                stats = severity_stats[severity]
                print(f"  {severity}: {stats.get('avg_response_time_minutes', 0):.2f}分 (n={stats.get('count', 0)})")
    
    # 差分計算
    if '旧モデル' in results and '新モデル' in results:
        old_mild = results['旧モデル'].get('severity_breakdown', {}).get('軽症', {}).get('avg_response_time_minutes', 0)
        new_mild = results['新モデル'].get('severity_breakdown', {}).get('軽症', {}).get('avg_response_time_minutes', 0)
        
        print("\n" + "=" * 80)
        print("改善度")
        print("=" * 80)
        print(f"軽症系応答時間:")
        print(f"  旧モデル: {old_mild:.2f}分")
        print(f"  新モデル: {new_mild:.2f}分")
        print(f"  差分: {new_mild - old_mild:+.2f}分 ({((new_mild - old_mild) / old_mild * 100):+.1f}%)")
        print(f"  目標: 7.33分（直近隊）")
        print(f"  直近隊との差: {new_mild - 7.33:+.2f}分")

if __name__ == "__main__":
    main()

