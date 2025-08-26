#!/usr/bin/env python3
"""
test_wandb_integration.py
wandb連携のテスト用スクリプト

このスクリプトは軽量なテスト実行を行い、wandb連携が正常に動作することを確認します。
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 非インタラクティブバックエンド
import matplotlib.pyplot as plt
import wandb

# 日本語フォント設定
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['font.size'] = 12

def create_test_data():
    """テスト用のダミーデータを作成"""
    return {
        "response_times": {
            "overall": {"mean": 8.5, "std": 2.1},
            "by_severity": {
                "重症": {"mean": 6.2, "std": 1.8},
                "軽症": {"mean": 10.3, "std": 2.5}
            }
        },
        "threshold_performance": {
            "6_minutes": {"rate": 85.6},
            "13_minutes": {"rate": 96.2}
        },
        "utilization_rate": 0.78,
        "total_calls_handled": 1000
    }

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

def test_wandb_integration():
    """wandb連携のテスト"""
    print("=" * 60)
    print("wandb連携テスト開始")
    print("=" * 60)
    
    # wandbログイン
    try:
        wandb.login()
        print("✓ wandbログイン成功")
    except Exception as e:
        print(f"✗ wandbログイン失敗: {e}")
        print("ローカルモードで実行します")
        wandb.init(mode="disabled")
        return
    
    # テスト実行
    strategies = ['closest', 'severity_based']
    
    for strategy in strategies:
        print(f"\n戦略: {strategy}")
        print("-" * 30)
        
        for run_idx in range(2):  # 軽量なので2回のみ
            print(f"  実行 {run_idx + 1}/2...")
            
            # matplotlibの状態をリセット
            plt.close('all')
            
            # wandb初期化
            try:
                with wandb.init(
                    project="ems-dispatch-test",
                    config={
                        "test_mode": True,
                        "strategy": strategy,
                        "run_index": run_idx + 1,
                        "timestamp": datetime.now().isoformat()
                    },
                    group=f"{strategy}-test",
                    name=f"test-run-{run_idx + 1}",
                    tags=["test", strategy],
                    reinit=True
                ) as run:
                    
                    # テストデータ生成
                    test_data = create_test_data()
                    
                    # 簡単なグラフ生成テスト
                    fig, ax = plt.subplots(figsize=(8, 6))
                    categories = ['重症', '軽症']
                    values = [
                        test_data['response_times']['by_severity']['重症']['mean'],
                        test_data['response_times']['by_severity']['軽症']['mean']
                    ]
                    ax.bar(categories, values)
                    ax.set_title(f'{strategy} - 応答時間比較')
                    ax.set_ylabel('平均応答時間（分）')
                    
                    # グラフをファイルに保存
                    test_output_dir = 'test_output'
                    os.makedirs(test_output_dir, exist_ok=True)
                    graph_path = os.path.join(test_output_dir, f'{strategy}_run{run_idx + 1}.png')
                    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # wandbに結果を記録
                    flat_data = flatten_dict(test_data)
                    wandb.log(flat_data)
                    wandb.log({"test_graph": wandb.Image(graph_path)})
                    
                    print(f"  ✓ 実行完了: グラフ保存 {graph_path}")
                    print(f"  ✓ wandbに結果を記録しました")
                    
            except Exception as e:
                print(f"  ✗ エラー: {e}")
                continue
    
    # 結果まとめ
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
    print("生成されたファイル:")
    for file in os.listdir('test_output'):
        if file.endswith('.png'):
            print(f"  - test_output/{file}")
    
    print("\nwandbダッシュボードで結果を確認してください:")
    print("  https://wandb.ai/[your-username]/ems-dispatch-test")

if __name__ == "__main__":
    test_wandb_integration()
