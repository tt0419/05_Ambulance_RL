"""
debug_single_episode_full.py
1エピソードの完全デバッグ（ユーザー提示版）

目的:
1. 6月15日データで1エピソード実行
2. 各事案で詳細な配車情報を記録
3. PPO vs 直近隊の使用率と性能を比較
4. 問題箇所（フォールバック、busyな隊の選択など）を特定
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# baseline_comparison.pyから必要な設定をインポート
from baseline_comparison import EXPERIMENT_CONFIG
from validation_simulation import run_validation_simulation
from dispatch_strategies import StrategyFactory

def run_single_episode_debug(target_date: str = "20230615", 
                             duration_hours: int = 24,
                             random_seed: int = 42):
    """
    1エピソードを完全デバッグモードで実行
    
    Args:
        target_date: 対象日（YYYYMMDD形式）
        duration_hours: シミュレーション時間
        random_seed: 乱数シード
    """
    print("=" * 80)
    print("1エピソード完全デバッグ実行")
    print("=" * 80)
    print(f"対象日: {target_date}")
    print(f"時間: {duration_hours}時間")
    print(f"乱数シード: {random_seed}")
    print("=" * 80)
    
    # PPOハイブリッド戦略の設定
    ppo_config = {
        'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/final_model.pth',
        'config_path': 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/configs/config.json',
        'hybrid_mode': True,
        'severe_conditions': ['重症', '重篤', '死亡'],
        'mild_conditions': ['軽症', '中等症']
    }
    
    # 出力ディレクトリ
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    
    print("\nPPOハイブリッド戦略初期化中...")
    
    # カスタムディスパッチロガーを設定（詳細ログ用）
    dispatch_log = []
    
    # バッチディスパッチロギング用クラス
    class DebugDispatchLogger:
        def __init__(self):
            self.logs = []
            self.incident_count = 0
        
        def log_dispatch(self, incident, ambulance, method, response_time, 
                        ambulance_busy, distance, optimal_ambulance=None, 
                        optimal_time=None):
            """配車ログを記録"""
            self.incident_count += 1
            
            log_entry = {
                'incident_id': self.incident_count,
                'severity': incident.get('severity', 'unknown'),
                'method': method,
                'ambulance_id': ambulance.get('id') if ambulance else None,
                'ambulance_name': ambulance.get('name') if ambulance else None,
                'response_time': response_time,
                'ambulance_was_busy': ambulance_busy,
                'distance': distance,
                'optimal_ambulance_id': optimal_ambulance.get('id') if optimal_ambulance else None,
                'optimal_response_time': optimal_time
            }
            
            self.logs.append(log_entry)
            
            # 最初の10件は詳細表示
            if self.incident_count <= 10:
                print(f"\nIncident {self.incident_count}:")
                print(f"  Severity: {log_entry['severity']}")
                print(f"  Method: {log_entry['method']}")
                print(f"  Response Time: {response_time:.2f}分")
                print(f"  Ambulance: {log_entry['ambulance_name']} (ID: {log_entry['ambulance_id']})")
                print(f"  Ambulance Busy?: {ambulance_busy}")
                if optimal_ambulance:
                    match_str = "✅ 一致" if ambulance.get('id') == optimal_ambulance.get('id') else "❌ 不一致"
                    print(f"  Optimal: {optimal_ambulance.get('name')} ({optimal_time:.2f}分) {match_str}")
        
        def get_summary(self):
            """統計サマリーを生成"""
            ppo_dispatches = [d for d in self.logs if d['method'] == 'PPO']
            nearest_dispatches = [d for d in self.logs if d['method'] == 'NEAREST']
            fallback_dispatches = [d for d in self.logs if d['method'] == 'FALLBACK_NEAREST']
            
            summary = {
                'total_incidents': len(self.logs),
                'ppo_count': len(ppo_dispatches),
                'nearest_count': len(nearest_dispatches),
                'fallback_count': len(fallback_dispatches),
            }
            
            if ppo_dispatches:
                ppo_times = [d['response_time'] for d in ppo_dispatches if d['response_time']]
                summary['ppo_avg_time'] = np.mean(ppo_times) if ppo_times else 0
                summary['ppo_std_time'] = np.std(ppo_times) if ppo_times else 0
            
            if nearest_dispatches:
                nearest_times = [d['response_time'] for d in nearest_dispatches if d['response_time']]
                summary['nearest_avg_time'] = np.mean(nearest_times) if nearest_times else 0
                summary['nearest_std_time'] = np.std(nearest_times) if nearest_times else 0
            
            if fallback_dispatches:
                fallback_times = [d['response_time'] for d in fallback_dispatches if d['response_time']]
                summary['fallback_avg_time'] = np.mean(fallback_times) if fallback_times else 0
            
            return summary
    
    logger = DebugDispatchLogger()
    
    # 実際にはvalidation_simulation.pyを直接呼び出すのではなく、
    # カスタムシミュレータを作成して詳細ログを取得する必要がある
    # ここでは簡易版として、run_validation_simulationを使用
    
    print("\nシミュレーション実行中...")
    print("（注: 詳細ログは実装中）")
    
    # シミュレーション実行
    run_validation_simulation(
        target_date_str=target_date,
        output_dir=str(output_dir),
        simulation_duration_hours=duration_hours,
        random_seed=random_seed,
        verbose_logging=True,  # 詳細ログを有効化
        enable_visualization=False,
        enable_detailed_reports=True,
        dispatch_strategy='ppo_agent',
        strategy_config=ppo_config
    )
    
    # レポートを読み込んで分析
    report_path = output_dir / "simulation_report.json"
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        print("\n" + "=" * 80)
        print("シミュレーション結果サマリー")
        print("=" * 80)
        
        # 基本統計
        total_calls = report.get('total_calls', 0)
        print(f"\n総事案数: {total_calls}件")
        
        # 応答時間統計
        rt_stats = report.get('response_times', {})
        overall_rt = rt_stats.get('overall', {})
        print(f"\n全体平均応答時間: {overall_rt.get('mean', 0):.2f} ± {overall_rt.get('std', 0):.2f}分")
        
        # 傷病度別
        by_severity = rt_stats.get('by_severity', {})
        print("\n傷病度別平均応答時間:")
        for severity, stats in by_severity.items():
            print(f"  {severity}: {stats.get('mean', 0):.2f} ± {stats.get('std', 0):.2f}分 "
                  f"(n={stats.get('count', 0)})")
        
        # 閾値達成率
        threshold_perf = report.get('threshold_performance', {})
        print(f"\n6分以内達成率: {threshold_perf.get('6_minutes', {}).get('rate', 0):.1f}%")
        print(f"13分以内達成率: {threshold_perf.get('13_minutes', {}).get('rate', 0):.1f}%")
        
        # ハイブリッドモード統計（もしあれば）
        if 'hybrid_stats' in report:
            hybrid = report['hybrid_stats']
            print("\n" + "-" * 80)
            print("ハイブリッドモード統計")
            print("-" * 80)
            print(f"直近隊運用: {hybrid.get('direct_count', 0)}回")
            print(f"PPO運用: {hybrid.get('ppo_count', 0)}回")
            print(f"直近隊比率: {hybrid.get('direct_ratio', 0):.1%}")
    else:
        print(f"\n⚠️  レポートファイルが見つかりません: {report_path}")
    
    # 詳細ログファイルの確認
    log_file = output_dir / "dispatch_log.csv"
    if log_file.exists():
        print(f"\n✅ 詳細ログファイル: {log_file}")
        analyze_dispatch_log(log_file)
    else:
        print(f"\n⚠️  詳細ログファイルが見つかりません: {log_file}")
    
    print("\n" + "=" * 80)
    print("デバッグ完了")
    print("=" * 80)
    
    return report_path

def analyze_dispatch_log(log_file: Path):
    """配車ログファイルを分析"""
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("配車ログ詳細分析")
    print("=" * 80)
    
    try:
        df = pd.read_csv(log_file)
        
        print(f"\nログエントリ数: {len(df)}件")
        
        # 列の確認
        print(f"\n利用可能な列: {', '.join(df.columns)}")
        
        # 傷病度分布
        if 'severity' in df.columns:
            print("\n傷病度分布:")
            severity_counts = df['severity'].value_counts()
            for severity, count in severity_counts.items():
                print(f"  {severity}: {count}件")
        
        # 応答時間統計
        if 'response_time_minutes' in df.columns:
            print(f"\n応答時間統計:")
            print(f"  平均: {df['response_time_minutes'].mean():.2f}分")
            print(f"  中央値: {df['response_time_minutes'].median():.2f}分")
            print(f"  標準偏差: {df['response_time_minutes'].std():.2f}分")
            print(f"  最小: {df['response_time_minutes'].min():.2f}分")
            print(f"  最大: {df['response_time_minutes'].max():.2f}分")
        
        # 教師一致率（もしあれば）
        if 'teacher_match' in df.columns:
            match_rate = df['teacher_match'].mean() * 100
            print(f"\n教師一致率: {match_rate:.1f}%")
            print(f"  一致: {df['teacher_match'].sum()}件")
            print(f"  不一致: {(~df['teacher_match']).sum()}件")
        
        # 利用可能救急車数の推移
        if 'available_count' in df.columns and 'total_count' in df.columns:
            avg_available = df['available_count'].mean()
            avg_utilization = (1 - df['available_count'] / df['total_count']).mean() * 100
            print(f"\n救急車稼働率:")
            print(f"  平均利用可能数: {avg_available:.1f}台")
            print(f"  平均稼働率: {avg_utilization:.1f}%")
        
        # 仮想救急車の使用率
        if 'is_virtual' in df.columns:
            virtual_count = df['is_virtual'].sum()
            virtual_rate = df['is_virtual'].mean() * 100
            print(f"\n仮想救急車使用:")
            print(f"  使用回数: {virtual_count}回")
            print(f"  使用率: {virtual_rate:.1f}%")
        
        # 最悪事案のリスト（応答時間上位10件）
        if 'response_time_minutes' in df.columns:
            print("\n⚠️  応答時間が長い事案（上位10件）:")
            worst_cases = df.nlargest(10, 'response_time_minutes')
            for idx, row in worst_cases.iterrows():
                severity = row.get('severity', 'unknown')
                rt = row.get('response_time_minutes', 0)
                amb_name = row.get('ambulance_name', 'unknown')
                print(f"  {severity}: {rt:.2f}分 (救急車: {amb_name})")
        
    except Exception as e:
        print(f"❌ ログ分析エラー: {e}")
        import traceback
        traceback.print_exc()

def compare_with_baseline(ppo_report_path: Path):
    """ベースライン（直近隊）と比較"""
    print("\n" + "=" * 80)
    print("ベースライン比較")
    print("=" * 80)
    
    # 直近隊戦略で同じ条件で実行
    output_dir = Path("debug_output")
    
    print("\n直近隊戦略で同一条件実行中...")
    run_validation_simulation(
        target_date_str="20230615",
        output_dir=str(output_dir / "baseline"),
        simulation_duration_hours=24,
        random_seed=42,
        verbose_logging=False,
        enable_visualization=False,
        enable_detailed_reports=True,
        dispatch_strategy='closest',
        strategy_config={}
    )
    
    baseline_report_path = output_dir / "baseline" / "simulation_report.json"
    
    if not baseline_report_path.exists():
        print("⚠️  ベースラインレポートが見つかりません")
        return
    
    # 両方のレポートを読み込んで比較
    with open(ppo_report_path, 'r', encoding='utf-8') as f:
        ppo_report = json.load(f)
    
    with open(baseline_report_path, 'r', encoding='utf-8') as f:
        baseline_report = json.load(f)
    
    print("\n比較結果:")
    print("-" * 80)
    
    # 全体平均応答時間
    ppo_mean = ppo_report.get('response_times', {}).get('overall', {}).get('mean', 0)
    baseline_mean = baseline_report.get('response_times', {}).get('overall', {}).get('mean', 0)
    
    print(f"\n全体平均応答時間:")
    print(f"  PPOハイブリッド: {ppo_mean:.2f}分")
    print(f"  直近隊: {baseline_mean:.2f}分")
    print(f"  差分: {ppo_mean - baseline_mean:+.2f}分 ({(ppo_mean - baseline_mean) / baseline_mean * 100:+.1f}%)")
    
    # 傷病度別比較
    print(f"\n傷病度別比較:")
    ppo_by_severity = ppo_report.get('response_times', {}).get('by_severity', {})
    baseline_by_severity = baseline_report.get('response_times', {}).get('by_severity', {})
    
    for severity in ['軽症', '中等症', '重症', '重篤', '死亡']:
        if severity in ppo_by_severity and severity in baseline_by_severity:
            ppo_time = ppo_by_severity[severity].get('mean', 0)
            baseline_time = baseline_by_severity[severity].get('mean', 0)
            diff = ppo_time - baseline_time
            diff_pct = (diff / baseline_time * 100) if baseline_time > 0 else 0
            
            symbol = "❌" if diff > 0 else "✅"
            print(f"  {severity}: PPO={ppo_time:.2f}分 vs 直近隊={baseline_time:.2f}分 "
                  f"({diff:+.2f}分, {diff_pct:+.1f}%) {symbol}")
    
    # 閾値達成率比較
    print(f"\n閾値達成率比較:")
    ppo_6min = ppo_report.get('threshold_performance', {}).get('6_minutes', {}).get('rate', 0)
    baseline_6min = baseline_report.get('threshold_performance', {}).get('6_minutes', {}).get('rate', 0)
    
    ppo_13min = ppo_report.get('threshold_performance', {}).get('13_minutes', {}).get('rate', 0)
    baseline_13min = baseline_report.get('threshold_performance', {}).get('13_minutes', {}).get('rate', 0)
    
    print(f"  6分以内: PPO={ppo_6min:.1f}% vs 直近隊={baseline_6min:.1f}% "
          f"({ppo_6min - baseline_6min:+.1f}%)")
    print(f"  13分以内: PPO={ppo_13min:.1f}% vs 直近隊={baseline_13min:.1f}% "
          f"({ppo_13min - baseline_13min:+.1f}%)")

def main():
    """メイン実行"""
    print("=" * 80)
    print("PPO戦略 1エピソード完全デバッグツール")
    print("=" * 80)
    print()
    
    # Phase 1: 1エピソード実行
    ppo_report_path = run_single_episode_debug(
        target_date="20230615",
        duration_hours=24,
        random_seed=42
    )
    
    # Phase 2: ベースライン比較
    if ppo_report_path and Path(ppo_report_path).exists():
        compare_with_baseline(Path(ppo_report_path))
    
    print("\n" + "=" * 80)
    print("診断完了")
    print("=" * 80)
    print("\n次のステップ:")
    print("  1. 上記の結果から問題箇所を特定")
    print("  2. 必要に応じて debug_ppo_io_trace.py で詳細調査")
    print("  3. ID対応表やハイブリッドロジックの修正")
    print("=" * 80)

if __name__ == "__main__":
    main()

