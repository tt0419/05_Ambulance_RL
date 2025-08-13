"""
baseline_comparison.py
直近隊運用 vs 傷病度考慮運用の比較実験
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# 日本語フォント設定
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['font.size'] = 12

def run_comparison_experiment(
    target_date: str,
    duration_hours: int = 168,
    num_runs: int = 5,
    output_base_dir: str = 'data/tokyo/experiments'
):
    """
    両戦略の比較実験を実行
    
    Args:
        target_date: シミュレーション開始日（YYYYMMDD形式）
        duration_hours: シミュレーション期間（時間）
        num_runs: 各戦略の実行回数
        output_base_dir: 結果出力ディレクトリ
    """
    
    # validation_simulation.pyのrun_validation_simulation関数をインポート
    from validation_simulation import run_validation_simulation
    
    # 実験結果格納用
    results = {
        'closest': [],
        'severity_based': []
    }
    
    strategies = ['closest', 'severity_based']
    
    print("=" * 60)
    print("ディスパッチ戦略比較実験")
    print(f"対象期間: {target_date} から {duration_hours}時間")
    print(f"実行回数: 各戦略 {num_runs}回")
    print("=" * 60)
    
    for strategy in strategies:
        print(f"\n戦略: {strategy}")
        print("-" * 40)
        
        for run_idx in range(num_runs):
            print(f"  実行 {run_idx + 1}/{num_runs}...")
            
            # 出力ディレクトリの設定
            output_dir = os.path.join(
                output_base_dir,
                f"{strategy}_{target_date}_{duration_hours}h_run{run_idx + 1}"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # シミュレーション実行
            run_validation_simulation(
                target_date_str=target_date,
                output_dir=output_dir,
                simulation_duration_hours=duration_hours,
                random_seed=42 + run_idx,  # 各実行で異なるシード
                verbose_logging=False,
                dispatch_strategy=strategy,
                strategy_config={} if strategy == 'closest' else {
                    'coverage_radius_km': 5.0,
                    'severe_conditions': ['重症', '重篤', '死亡'],
                    'mild_conditions': ['軽症', '中等症']
                }
            )
            
            # 結果の読み込み
            report_path = os.path.join(output_dir, 'simulation_report.json')
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
                results[strategy].append(report)
    
    # 結果の分析と比較
    print("\n" + "=" * 60)
    print("実験結果の分析")
    print("=" * 60)
    
    analysis_results = analyze_results(results)
    
    # 結果の可視化
    visualize_comparison(analysis_results, output_base_dir)
    
    # サマリーレポートの作成
    create_summary_report(analysis_results, output_base_dir)
    
    return analysis_results

def analyze_results(results: Dict[str, List]) -> Dict:
    """
    実験結果を分析
    
    Returns:
        分析結果の辞書
    """
    analysis = {}
    
    for strategy in results.keys():
        strategy_results = results[strategy]
        
        # 応答時間の統計
        response_times_all = []
        response_times_severe = []
        response_times_mild = []
        
        # 閾値達成率
        threshold_6min_rates = []
        threshold_13min_rates = []
        threshold_6min_severe_rates = []
        
        for report in strategy_results:
            # 全体の応答時間
            if 'response_times' in report and 'overall' in report['response_times']:
                rt_mean = report['response_times']['overall']['mean']
                response_times_all.append(rt_mean)
            
            # 傷病度別応答時間
            if 'by_severity' in report['response_times']:
                # 重症系
                for sev in ['重症', '重篤', '死亡']:
                    if sev in report['response_times']['by_severity']:
                        response_times_severe.append(
                            report['response_times']['by_severity'][sev]['mean']
                        )
                # 軽症系
                for sev in ['軽症', '中等症']:
                    if sev in report['response_times']['by_severity']:
                        response_times_mild.append(
                            report['response_times']['by_severity'][sev]['mean']
                        )
            
            # 閾値達成率
            if 'threshold_performance' in report:
                threshold_6min_rates.append(
                    report['threshold_performance']['6_minutes']['rate']
                )
                threshold_13min_rates.append(
                    report['threshold_performance']['13_minutes']['rate']
                )
                
                # 重症系の6分達成率
                if 'by_severity' in report['threshold_performance']:
                    severe_6min_rates = []
                    for sev in ['重症', '重篤']:
                        if sev in report['threshold_performance']['by_severity']['6_minutes']:
                            severe_6min_rates.append(
                                report['threshold_performance']['by_severity']['6_minutes'][sev]['rate']
                            )
                    if severe_6min_rates:
                        threshold_6min_severe_rates.append(np.mean(severe_6min_rates))
        
        # 統計値の計算
        analysis[strategy] = {
            'response_time_overall': {
                'mean': np.mean(response_times_all),
                'std': np.std(response_times_all),
                'values': response_times_all
            },
            'response_time_severe': {
                'mean': np.mean(response_times_severe) if response_times_severe else 0,
                'std': np.std(response_times_severe) if response_times_severe else 0,
                'values': response_times_severe
            },
            'response_time_mild': {
                'mean': np.mean(response_times_mild) if response_times_mild else 0,
                'std': np.std(response_times_mild) if response_times_mild else 0,
                'values': response_times_mild
            },
            'threshold_6min': {
                'mean': np.mean(threshold_6min_rates),
                'std': np.std(threshold_6min_rates),
                'values': threshold_6min_rates
            },
            'threshold_13min': {
                'mean': np.mean(threshold_13min_rates),
                'std': np.std(threshold_13min_rates),
                'values': threshold_13min_rates
            },
            'threshold_6min_severe': {
                'mean': np.mean(threshold_6min_severe_rates) if threshold_6min_severe_rates else 0,
                'std': np.std(threshold_6min_severe_rates) if threshold_6min_severe_rates else 0,
                'values': threshold_6min_severe_rates
            }
        }
    
    return analysis

def visualize_comparison(analysis: Dict, output_dir: str):
    """
    比較結果の可視化
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ディスパッチ戦略比較: 直近隊 vs 傷病度考慮', fontsize=16)
    
    strategies = ['closest', 'severity_based']
    strategy_labels = {'closest': '直近隊', 'severity_based': '傷病度考慮'}
    colors = {'closest': '#3498db', 'severity_based': '#e74c3c'}
    
    # 1. 全体平均応答時間
    ax = axes[0, 0]
    means = [analysis[s]['response_time_overall']['mean'] for s in strategies]
    stds = [analysis[s]['response_time_overall']['std'] for s in strategies]
    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                   color=[colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies])
    ax.set_ylabel('平均応答時間（分）')
    ax.set_title('全体平均応答時間')
    ax.grid(True, alpha=0.3)
    
    # 数値表示
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.1, f'{mean:.2f}±{std:.2f}', 
                ha='center', va='bottom')
    
    # 2. 重症系の平均応答時間
    ax = axes[0, 1]
    means_severe = [analysis[s]['response_time_severe']['mean'] for s in strategies]
    stds_severe = [analysis[s]['response_time_severe']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_severe, yerr=stds_severe, capsize=5,
                   color=[colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies])
    ax.set_ylabel('平均応答時間（分）')
    ax.set_title('重症・重篤・死亡の平均応答時間')
    ax.grid(True, alpha=0.3)
    
    # 改善率を計算して表示
    if means_severe[0] > 0:
        improvement = (means_severe[0] - means_severe[1]) / means_severe[0] * 100
        ax.text(0.5, max(means_severe) * 0.9, 
                f'改善率: {improvement:.1f}%',
                ha='center', transform=ax.transData,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 3. 軽症系の平均応答時間
    ax = axes[0, 2]
    means_mild = [analysis[s]['response_time_mild']['mean'] for s in strategies]
    stds_mild = [analysis[s]['response_time_mild']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_mild, yerr=stds_mild, capsize=5,
                   color=[colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies])
    ax.set_ylabel('平均応答時間（分）')
    ax.set_title('軽症・中等症の平均応答時間')
    ax.grid(True, alpha=0.3)
    
    # 4. 6分以内達成率（全体）
    ax = axes[1, 0]
    means_6min = [analysis[s]['threshold_6min']['mean'] for s in strategies]
    stds_6min = [analysis[s]['threshold_6min']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_6min, yerr=stds_6min, capsize=5,
                   color=[colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies])
    ax.set_ylabel('達成率（%）')
    ax.set_title('6分以内達成率（全体）')
    ax.set_ylim(0, 100)
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='目標90%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 13分以内達成率（全体）
    ax = axes[1, 1]
    means_13min = [analysis[s]['threshold_13min']['mean'] for s in strategies]
    stds_13min = [analysis[s]['threshold_13min']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_13min, yerr=stds_13min, capsize=5,
                   color=[colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies])
    ax.set_ylabel('達成率（%）')
    ax.set_title('13分以内達成率（全体）')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # 6. 重症系の6分以内達成率
    ax = axes[1, 2]
    means_6min_severe = [analysis[s]['threshold_6min_severe']['mean'] for s in strategies]
    stds_6min_severe = [analysis[s]['threshold_6min_severe']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_6min_severe, yerr=stds_6min_severe, capsize=5,
                   color=[colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies])
    ax.set_ylabel('達成率（%）')
    ax.set_title('6分以内達成率（重症・重篤）')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n比較グラフを保存: {os.path.join(output_dir, 'strategy_comparison.png')}")

def create_summary_report(analysis: Dict, output_dir: str):
    """
    サマリーレポートの作成
    """
    report_path = os.path.join(output_dir, 'comparison_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ディスパッチ戦略比較実験 サマリーレポート\n")
        f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # 戦略別の結果
        for strategy in ['closest', 'severity_based']:
            strategy_label = '直近隊運用' if strategy == 'closest' else '傷病度考慮運用'
            f.write(f"【{strategy_label}】\n")
            f.write("-" * 40 + "\n")
            
            data = analysis[strategy]
            
            f.write(f"1. 平均応答時間\n")
            f.write(f"   全体: {data['response_time_overall']['mean']:.2f} ± {data['response_time_overall']['std']:.2f} 分\n")
            f.write(f"   重症系: {data['response_time_severe']['mean']:.2f} ± {data['response_time_severe']['std']:.2f} 分\n")
            f.write(f"   軽症系: {data['response_time_mild']['mean']:.2f} ± {data['response_time_mild']['std']:.2f} 分\n\n")
            
            f.write(f"2. 閾値達成率\n")
            f.write(f"   6分以内（全体）: {data['threshold_6min']['mean']:.1f} ± {data['threshold_6min']['std']:.1f} %\n")
            f.write(f"   13分以内（全体）: {data['threshold_13min']['mean']:.1f} ± {data['threshold_13min']['std']:.1f} %\n")
            f.write(f"   6分以内（重症系）: {data['threshold_6min_severe']['mean']:.1f} ± {data['threshold_6min_severe']['std']:.1f} %\n\n")
        
        # 比較結果
        f.write("=" * 60 + "\n")
        f.write("【比較結果】\n")
        f.write("-" * 40 + "\n")
        
        # 重症系の改善率
        rt_severe_closest = analysis['closest']['response_time_severe']['mean']
        rt_severe_severity = analysis['severity_based']['response_time_severe']['mean']
        if rt_severe_closest > 0:
            improvement_severe = (rt_severe_closest - rt_severe_severity) / rt_severe_closest * 100
            f.write(f"重症系応答時間の改善: {improvement_severe:+.1f}%\n")
            f.write(f"  直近隊: {rt_severe_closest:.2f}分 → 傷病度考慮: {rt_severe_severity:.2f}分\n\n")
        
        # 6分達成率の改善
        th6_severe_closest = analysis['closest']['threshold_6min_severe']['mean']
        th6_severe_severity = analysis['severity_based']['threshold_6min_severe']['mean']
        improvement_th6 = th6_severe_severity - th6_severe_closest
        f.write(f"重症系6分達成率の改善: {improvement_th6:+.1f}ポイント\n")
        f.write(f"  直近隊: {th6_severe_closest:.1f}% → 傷病度考慮: {th6_severe_severity:.1f}%\n\n")
        
        # 統計的有意性の簡易チェック（t検定）
        from scipy import stats
        
        if (len(analysis['closest']['response_time_severe']['values']) > 1 and 
            len(analysis['severity_based']['response_time_severe']['values']) > 1):
            
            t_stat, p_value = stats.ttest_ind(
                analysis['closest']['response_time_severe']['values'],
                analysis['severity_based']['response_time_severe']['values']
            )
            
            f.write(f"統計的有意性（重症系応答時間）:\n")
            f.write(f"  t統計量: {t_stat:.3f}\n")
            f.write(f"  p値: {p_value:.4f}\n")
            f.write(f"  結果: {'有意差あり' if p_value < 0.05 else '有意差なし'} (α=0.05)\n")
    
    print(f"サマリーレポートを保存: {report_path}")

if __name__ == "__main__":
    # 実験パラメータ
    target_date = "20230401"  # 開始日
    duration_hours = 720       # 1週間
    num_runs = 5              # 各戦略5回実行
    
    # 実験実行
    results = run_comparison_experiment(
        target_date=target_date,
        duration_hours=duration_hours,
        num_runs=num_runs
    )
    
    print("\n実験完了！")