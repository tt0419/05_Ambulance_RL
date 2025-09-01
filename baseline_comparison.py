"""
baseline_comparison.py
複数ディスパッチ戦略の比較実験システム

【設定変更ガイド】
1. 戦略の追加・削除: EXPERIMENT_CONFIG の strategies リストを編集
2. 戦略設定の変更: strategy_configs 辞書を編集
3. 戦略ラベルの変更: strategy_labels 辞書を編集
4. 色の変更: strategy_colors 辞書を編集

【PPO戦略の使用方法】
1. PPO戦略を含む比較実験を実行する場合:
   - EXPERIMENT_CONFIG['strategies']で'ppo_agent'のコメントアウトを外す
   - または、main_with_ppo()関数を呼び出す
2. PPO戦略の設定:
   - strategy_configs['ppo_agent']でmodel_pathとconfig_pathを指定
   - 学習済みモデル(.pth)と設定ファイル(.yaml)のパスを正しく設定
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime  # ★★★ datetimeを直接インポート
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from scipy import stats

# ★★★ 変更点: wandbをインポート ★★★
import wandb

# ★★★ 修正: matplotlibバックエンドを非インタラクティブに設定 ★★★
import matplotlib
matplotlib.use('Agg')  # ファイル出力専用バックエンド

# 日本語フォント設定
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['font.size'] = 12

# ディスパッチ戦略のインポート
from dispatch_strategies import STRATEGY_CONFIGS

# 統一された傷病度定数をインポート
from constants import SEVERITY_GROUPS

# ============================================================
# 【設定変更箇所1】実験設定
# ============================================================
EXPERIMENT_CONFIG = {
    # 比較する戦略のリスト（ここで戦略を追加・削除）
    'strategies': [#'closest', 'severity_based',
                   # 'advanced_severity',
                   'ppo_agent'  # ★★★ PPO戦略をコメントアウトで追加 ★★★
                   ],
    
    # 各戦略の日本語表示名
    'strategy_labels': {
        'closest': '直近隊運用',
        'severity_based': '傷病度考慮運用',
        'advanced_severity': '高度傷病度考慮運用',
        'ppo_agent': 'PPOエージェント運用'  # ★★★ PPO戦略のラベルを追加 ★★★
    },
    
    # 各戦略の色設定
    'strategy_colors': {
        'closest': '#3498db',        # 青
        'severity_based': '#e74c3c',  # 赤
        'advanced_severity': '#2ecc71', # 緑
        'ppo_agent': '#9b59b6'       # 紫 ★★★ PPO戦略の色を追加 ★★★
    },
    
    # 各戦略の設定（STRATEGY_CONFIGSから選択またはカスタム設定）
    'strategy_configs': {
        'closest': {},  # デフォルト設定
        'severity_based': {
            'coverage_radius_km': 5.0,
            'severe_conditions': SEVERITY_GROUPS['severe_conditions'],  # 統一された定数を使用
            'mild_conditions': SEVERITY_GROUPS['mild_conditions'],      # 統一された定数を使用
            
            # ★★★ 新規追加: パラメータ設定 ★★★
            'time_score_weight': 0.8,            # 応答時間の重みを20%に
            'coverage_loss_weight': 0.2,         # カバレッジ損失の重みを80%に
            'mild_time_limit_sec': 1080,         # 軽症の許容時間を18分(1080秒)に
            'moderate_time_limit_sec': 900       # 中等症の許容時間を15分(900秒)に
        },
        'advanced_severity': STRATEGY_CONFIGS['aggressive'],  # 推奨設定
        'ppo_agent': {  # ★★★ PPO戦略の設定を追加 ★★★
            'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_20250829_154139/final_model.pth',  # 学習済みモデルのパス
            'config_path': 'reinforcement_learning/experiments/config_area1.yaml',  # 第1方面限定の学習設定（学習時と同じ）
            # ★★★ 地域制限の設定を明示的に指定 ★★★
            'area_restriction': {
                'enabled': True,
                'area_name': '第一方面',
                'section_code': 1,
                'districts': ['千代田区', '中央区', '港区']
            }
        }
    }
}

# ★★★ 変更点: レポートの辞書をフラット化するヘルパー関数を追加 ★★★
def flatten_dict(d, parent_key='', sep='.'):
    """ネストした辞書をフラットな辞書に変換する"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def run_comparison_experiment(
    target_date: str,
    duration_hours: int = 168,
    num_runs: int = 5,
    output_base_dir: str = 'data/tokyo/experiments',
    # ★★★ 変更点: wandbのプロジェクト名を受け取る引数を追加 ★★★
    wandb_project: str = "ambulance-dispatch-simulation"
):
    
    # ★★★ 修正: wandbの初期設定 ★★★
    try:
        wandb.login()
        print(f"wandbログイン成功: プロジェクト '{wandb_project}' に接続します")
    except Exception as e:
        print(f"警告: wandbログインに失敗しました。ローカルモードで実行します: {e}")
        wandb.init(mode="disabled")
    """
    複数戦略の比較実験を実行 (wandb連携版)
    
    Args:
        target_date: シミュレーション開始日（YYYYMMDD形式）
        duration_hours: シミュレーション期間（時間）
        num_runs: 各戦略の実行回数
        output_base_dir: 結果出力ディレクトリ
        wandb_project: wandbのプロジェクト名
    """
    
    # validation_simulation.pyのrun_validation_simulation関数をインポート
    from validation_simulation import run_validation_simulation
    
    # 設定から戦略リストを取得
    strategies = EXPERIMENT_CONFIG['strategies']
    strategy_configs = EXPERIMENT_CONFIG['strategy_configs']
    
    # 実験結果格納用（動的に初期化）
    results = {strategy: [] for strategy in strategies}
    
    print("=" * 60)
    print("ディスパッチ戦略比較実験 (wandb連携)")
    print(f"W&B Project: {wandb_project}")
    print(f"対象期間: {target_date} から {duration_hours}時間")
    print(f"実行回数: 各戦略 {num_runs}回")
    print(f"比較戦略: {', '.join([EXPERIMENT_CONFIG['strategy_labels'][s] for s in strategies])}")
    print("=" * 60)
    
    for strategy in strategies:
        print(f"\n戦略: {EXPERIMENT_CONFIG['strategy_labels'][strategy]} ({strategy})")
        print("-" * 40)
        
        for run_idx in range(num_runs):
            print(f"  実行 {run_idx + 1}/{num_runs}...")
            
            # ★★★ 変更点: 実行日時を取得して、wandbの実行名を生成 ★★★
            run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            run_name = f"{strategy}-{run_timestamp}"
            
            # ★★★ 修正: matplotlibの状態をリセット ★★★
            plt.close('all')  # 全てのプロットを閉じる
            
            # --- wandb連携のための設定 ---
            
            # ★★★ 変更点: wandbに渡すコンフィグ情報を作成 ★★★
            current_strategy_config = strategy_configs.get(strategy, {})
            config_for_wandb = {
                "target_date": target_date,
                "duration_hours": duration_hours,
                "num_runs": num_runs,
                "run_index": run_idx + 1,
                "random_seed": 42 + run_idx,
                "dispatch_strategy": strategy,
                **flatten_dict(current_strategy_config, parent_key='strategy_params')
            }
            
            # ★★★ 修正: wandbの実行をwith文で初期化 ★★★
            try:
                with wandb.init(
                    project=wandb_project,
                    config=config_for_wandb,
                    group=f"{strategy}-{target_date}", # 同じ戦略の実行をグループ化
                    # ★★★ 変更点: 生成した実行名を設定 ★★★
                    name=run_name,
                    job_type="simulation",
                    tags=["baseline", strategy],
                    reinit=True # ループ内で複数回initを呼ぶために必要
                ) as run:
                    
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
                        strategy_config=current_strategy_config
                    )
                    
                    # 結果の読み込みとwandbへの記録
                    report_path = os.path.join(output_dir, 'simulation_report.json')
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            report = json.load(f)
                            results[strategy].append(report)
                            
                            # ★★★ 変更点: PPOのメトリクス名にマッピングしてログを記録 ★★★
                            # 1. PPOと共通の主要メトリクスを抽出・名前変更
                            unified_metrics = {}
                            rt_stats = report.get('response_times', {})
                            
                            # 全体平均RT
                            unified_metrics['charts/response_time_mean'] = rt_stats.get('overall', {}).get('mean', 0)
                            
                            # 傷病度別平均RT
                            rt_by_severity = rt_stats.get('by_severity', {})
                            unified_metrics['charts/response_time_mild_mean'] = rt_by_severity.get('軽症', {}).get('mean', 0)
                            unified_metrics['charts/response_time_moderate_mean'] = rt_by_severity.get('中等症', {}).get('mean', 0)
                            unified_metrics['charts/response_time_severe_mean'] = rt_by_severity.get('重症', {}).get('mean', 0)
                            unified_metrics['charts/response_time_critical_mean'] = rt_by_severity.get('重篤', {}).get('mean', 0)
                            
                            # 重症6分以内到着率
                            th_by_severity = report.get('threshold_performance', {}).get('by_severity', {}).get('6_minutes', {})
                            unified_metrics['charts/response_time_severe_under_6min_rate'] = th_by_severity.get('重症', {}).get('rate', 0)
                            
                            # 2. 統一されたメトリクスをwandbに記録
                            wandb.log(unified_metrics)
                            
                            # 3. (オプション) 元の詳細なレポートも別途記録
                            wandb.log({"full_report": report})

                            print(f"  - wandbに統一されたメトリクスを記録しました。 (Run Name: {run_name})")
                    
                    except FileNotFoundError:
                        print(f"  - エラー: レポートファイルが見つかりません: {report_path}")
                        # ★★★ 変更点: wandbにエラー情報を記録することも可能 ★★★
                        wandb.log({"error": "report_not_found"})

            except Exception as e:
                print(f"  - wandb連携エラー: {e}")
                # wandbのエラーが発生してもシミュレーション結果は保存
                # 結果をローカルに保存
                try:
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report = json.load(f)
                        results[strategy].append(report)
                        print(f"  - ローカルに結果を保存しました")
                except:
                    print(f"  - 警告: 結果の保存に失敗しました")
                continue
    
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
    実験結果を分析（複数戦略対応）
    
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
    比較結果の可視化（複数戦略対応）
    """
    strategies = EXPERIMENT_CONFIG['strategies']
    strategy_labels = EXPERIMENT_CONFIG['strategy_labels']
    strategy_colors = EXPERIMENT_CONFIG['strategy_colors']
    
    # 戦略数に応じてレイアウトを調整
    num_strategies = len(strategies)
    
    if num_strategies <= 3:
        # 3つ以下の場合：2行3列のレイアウト（基本6個のグラフ）
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        max_plots = 6  # 最大6個のサブプロット
    else:
        # 4つ以上の場合：3行3列のレイアウト
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()
        max_plots = 9  # 最大9個のサブプロット
    
    fig.suptitle(f'ディスパッチ戦略比較: {len(strategies)}戦略', fontsize=16)
    
    x_pos = np.arange(len(strategies))
    
    # 1. 全体平均応答時間
    ax = axes[0]
    means = [analysis[s]['response_time_overall']['mean'] for s in strategies]
    stds = [analysis[s]['response_time_overall']['std'] for s in strategies]
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('平均応答時間（分）')
    ax.set_title('全体平均応答時間')
    ax.grid(True, alpha=0.3)
    
    # 数値表示
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.1, f'{mean:.2f}±{std:.2f}', 
                ha='center', va='bottom', fontsize=10)
    
    # 2. 重症系の平均応答時間
    ax = axes[1]
    means_severe = [analysis[s]['response_time_severe']['mean'] for s in strategies]
    stds_severe = [analysis[s]['response_time_severe']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_severe, yerr=stds_severe, capsize=5,
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('平均応答時間（分）')
    ax.set_title('重症・重篤・死亡の平均応答時間')
    ax.grid(True, alpha=0.3)
    
    # 3. 軽症系の平均応答時間
    ax = axes[2]
    means_mild = [analysis[s]['response_time_mild']['mean'] for s in strategies]
    stds_mild = [analysis[s]['response_time_mild']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_mild, yerr=stds_mild, capsize=5,
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('平均応答時間（分）')
    ax.set_title('軽症・中等症の平均応答時間')
    ax.grid(True, alpha=0.3)
    
    # 4. 6分以内達成率（全体）
    ax = axes[3]
    means_6min = [analysis[s]['threshold_6min']['mean'] for s in strategies]
    stds_6min = [analysis[s]['threshold_6min']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_6min, yerr=stds_6min, capsize=5,
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('達成率（%）')
    ax.set_title('6分以内達成率（全体）')
    ax.set_ylim(0, 100)
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='目標90%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 13分以内達成率（全体）
    ax = axes[4]
    means_13min = [analysis[s]['threshold_13min']['mean'] for s in strategies]
    stds_13min = [analysis[s]['threshold_13min']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_13min, yerr=stds_13min, capsize=5,
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('達成率（%）')
    ax.set_title('13分以内達成率（全体）')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # 6. 重症系の6分以内達成率
    ax = axes[5]
    means_6min_severe = [analysis[s]['threshold_6min_severe']['mean'] for s in strategies]
    stds_6min_severe = [analysis[s]['threshold_6min_severe']['std'] for s in strategies]
    bars = ax.bar(x_pos, means_6min_severe, yerr=stds_6min_severe, capsize=5,
                   color=[strategy_colors[s] for s in strategies], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel('達成率（%）')
    ax.set_title('6分以内達成率（重症・重篤）')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # 7. 統計的有意性のヒートマップ（4つ以上の戦略がある場合）
    if num_strategies >= 4 and 6 < max_plots:
        ax = axes[6]
        create_significance_heatmap(analysis, strategies, strategy_labels, ax)
    
    # 8. 改善率の比較（ベースライン戦略との比較）
    if num_strategies >= 2:
        # 3つの戦略の場合：6番目のサブプロットに改善率比較を表示
        # 4つ以上の戦略の場合：7番目のサブプロットに改善率比較を表示
        plot_index = 6 if num_strategies == 3 else (7 if 7 < max_plots else 6)
        if plot_index < max_plots:
            ax = axes[plot_index]
            create_improvement_comparison(analysis, strategies, strategy_labels, strategy_colors, ax)
    
    # 未使用のサブプロットを非表示
    for i in range(len(axes)):
        if i >= max_plots:
            axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n比較グラフを保存: {os.path.join(output_dir, 'strategy_comparison.png')}")

def create_significance_heatmap(analysis: Dict, strategies: List[str], strategy_labels: Dict, ax):
    """統計的有意性のヒートマップを作成"""
    num_strategies = len(strategies)
    p_values = np.zeros((num_strategies, num_strategies))
    
    # 重症系応答時間での統計的有意性を計算
    for i, strategy1 in enumerate(strategies):
        for j, strategy2 in enumerate(strategies):
            if i == j:
                p_values[i, j] = 1.0
            else:
                values1 = analysis[strategy1]['response_time_severe']['values']
                values2 = analysis[strategy2]['response_time_severe']['values']
                if len(values1) > 1 and len(values2) > 1:
                    _, p_val = stats.ttest_ind(values1, values2)
                    p_values[i, j] = p_val
                else:
                    p_values[i, j] = 1.0
    
    # ヒートマップの作成
    im = ax.imshow(p_values, cmap='RdYlBu_r', vmin=0, vmax=0.1)
    ax.set_xticks(range(num_strategies))
    ax.set_yticks(range(num_strategies))
    ax.set_xticklabels([strategy_labels[s] for s in strategies], rotation=45, ha='right')
    ax.set_yticklabels([strategy_labels[s] for s in strategies])
    ax.set_title('統計的有意性（重症系応答時間）\np値 < 0.05で有意差あり')
    
    # 数値表示
    for i in range(num_strategies):
        for j in range(num_strategies):
            text = ax.text(j, i, f'{p_values[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='p値')

def create_improvement_comparison(analysis: Dict, strategies: List[str], strategy_labels: Dict, strategy_colors: Dict, ax):
    """ベースライン戦略との改善率比較"""
    baseline = strategies[0]  # 最初の戦略をベースラインとする
    improvements = []
    labels = []
    
    for strategy in strategies[1:]:
        baseline_mean = analysis[baseline]['response_time_severe']['mean']
        strategy_mean = analysis[strategy]['response_time_severe']['mean']
        
        if baseline_mean > 0:
            improvement = (baseline_mean - strategy_mean) / baseline_mean * 100
            improvements.append(improvement)
            labels.append(strategy_labels[strategy])
    
    if improvements:
        bars = ax.bar(range(len(improvements)), improvements, 
                     color=[strategy_colors[s] for s in strategies[1:]])
        ax.set_xticks(range(len(improvements)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('改善率（%）')
        ax.set_title(f'ベースライン（{strategy_labels[baseline]}）との比較')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # 数値表示
        for i, (bar, improvement) in enumerate(zip(bars, improvements)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{improvement:+.1f}%', ha='center', va='bottom')

def create_summary_report(analysis: Dict, output_dir: str):
    """
    サマリーレポートの作成（複数戦略対応）
    """
    report_path = os.path.join(output_dir, 'comparison_summary.txt')
    strategies = EXPERIMENT_CONFIG['strategies']
    strategy_labels = EXPERIMENT_CONFIG['strategy_labels']
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ディスパッチ戦略比較実験 サマリーレポート\n")
        f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"比較戦略数: {len(strategies)}\n")
        f.write("=" * 60 + "\n\n")
        
        # 戦略別の結果
        for strategy in strategies:
            strategy_label = strategy_labels[strategy]
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
        
        # 統計的比較結果
        f.write("=" * 60 + "\n")
        f.write("【統計的比較結果】\n")
        f.write("-" * 40 + "\n")
        
        if len(strategies) >= 3:
            # 3つ以上の戦略：ANOVA
            f.write("3群以上の比較（ANOVA）:\n")
            severe_times = [analysis[s]['response_time_severe']['values'] for s in strategies]
            severe_times = [times for times in severe_times if len(times) > 1]
            
            if len(severe_times) >= 3:
                f_stat, p_value = stats.f_oneway(*severe_times)
                f.write(f"  重症系応答時間: F={f_stat:.3f}, p={p_value:.4f}\n")
                f.write(f"  結果: {'有意差あり' if p_value < 0.05 else '有意差なし'} (α=0.05)\n\n")
        
        # ペアワイズ比較
        f.write("ペアワイズ比較（t検定）:\n")
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies[i+1:], i+1):
                values1 = analysis[strategy1]['response_time_severe']['values']
                values2 = analysis[strategy2]['response_time_severe']['values']
                
                if len(values1) > 1 and len(values2) > 1:
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    f.write(f"  {strategy_labels[strategy1]} vs {strategy_labels[strategy2]}: ")
                    f.write(f"t={t_stat:.3f}, p={p_value:.4f} ")
                    f.write(f"({'有意差あり' if p_value < 0.05 else '有意差なし'})\n")
        
        # 改善率の比較
        f.write("\n改善率の比較（ベースライン: 直近隊運用）:\n")
        baseline = strategies[0]
        baseline_mean = analysis[baseline]['response_time_severe']['mean']
        
        for strategy in strategies[1:]:
            strategy_mean = analysis[strategy]['response_time_severe']['mean']
            if baseline_mean > 0:
                improvement = (baseline_mean - strategy_mean) / baseline_mean * 100
                f.write(f"  {strategy_labels[strategy]}: {improvement:+.1f}%\n")
    
    print(f"サマリーレポートを保存: {report_path}")

if __name__ == "__main__":
    # ============================================================
    # 【設定変更箇所2】実験パラメータ
    # ============================================================
    EXPERIMENT_PARAMS = {
        'target_date': "20240401",  # 開始日
        'duration_hours': 720,       # 30日間
        'num_runs': 5,              # 各戦略5回実行
        'output_base_dir': 'data/tokyo/experiments',
        # ★★★ 変更点: wandbのプロジェクト名を指定 ★★★
        'wandb_project': 'ems-dispatch-optimization'
    }
    
    # 実験実行
    results = run_comparison_experiment(
        target_date=EXPERIMENT_PARAMS['target_date'],
        duration_hours=EXPERIMENT_PARAMS['duration_hours'],
        num_runs=EXPERIMENT_PARAMS['num_runs'],
        output_base_dir=EXPERIMENT_PARAMS['output_base_dir'],
        wandb_project=EXPERIMENT_PARAMS['wandb_project']
    )
    
    print("\n実験完了！")

# ★★★【追加】PPO戦略の比較実験を実行する関数 ★★★
def main_with_ppo():
    """
    PPO戦略を含む比較実験を実行する関数
    使用方法: baseline_comparison.pyのEXPERIMENT_CONFIGで'ppo_agent'のコメントアウトを外してから実行
    """
    print("PPO戦略を含むベースライン戦略の性能評価を開始します。")
    
    # 実験パラメータ
    EXPERIMENT_PARAMS = {
        'target_date': '20240101',  # シミュレーション対象日
        'duration_hours': 24,       # シミュレーション期間（時間）
        'num_runs': 3,              # 各戦略の実行回数（PPOは重いので少なめ）
        'output_base_dir': 'data/tokyo/experiments',
        'wandb_project': 'ems-dispatch-optimization-with-ppo'
    }
    
    # PPO戦略を有効にするために、一時的に戦略リストを変更
    original_strategies = EXPERIMENT_CONFIG['strategies'].copy()
    EXPERIMENT_CONFIG['strategies'] = ['closest', 'severity_based', 'ppo_agent']
    
    try:
        # 実験実行
        results = run_comparison_experiment(
            target_date=EXPERIMENT_PARAMS['target_date'],
            duration_hours=EXPERIMENT_PARAMS['duration_hours'],
            num_runs=EXPERIMENT_PARAMS['num_runs'],
            output_base_dir=EXPERIMENT_PARAMS['output_base_dir'],
            wandb_project=EXPERIMENT_PARAMS['wandb_project']
        )
        
        print("\nPPO戦略を含む実験完了！")
        
    finally:
        # 元の設定に戻す
        EXPERIMENT_CONFIG['strategies'] = original_strategies