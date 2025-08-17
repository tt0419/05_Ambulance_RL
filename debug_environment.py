"""
debug_environment.py
環境の動作を詳細に確認するデバッグスクリプト
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reinforcement_learning.environment.ems_environment import EMSEnvironment

def debug_environment():
    """環境の詳細デバッグ"""
    print("=" * 60)
    print("環境デバッグ開始")
    print("=" * 60)
    
    # 設定ファイル
    config_path = "reinforcement_learning/experiments/config_quick.yaml"
    
    # 環境初期化
    print("\n1. 環境初期化")
    env = EMSEnvironment(config_path, mode="train")
    
    # リセット
    print("\n2. 環境リセット")
    state = env.reset()
    print(f"初期状態の形状: {state.shape}")
    print(f"初期状態の統計: min={state.min():.3f}, max={state.max():.3f}, mean={state.mean():.3f}")
    
    # 事案の確認
    print("\n3. 現在の事案確認")
    if env.pending_call:
        print(f"事案ID: {env.pending_call['id']}")
        print(f"傷病度: {env.pending_call['severity']}")
        print(f"位置: {env.pending_call['h3_index']}")
        print(f"発生時刻: {env.pending_call['datetime']}")
    else:
        print("❌ 事案がありません！")
        return
    
    # 救急車の状態確認
    print("\n4. 救急車の状態")
    available_count = 0
    for amb_id, amb_state in list(env.ambulance_states.items())[:5]:  # 最初の5台を表示
        print(f"救急車 {amb_id}: {amb_state['status']}")
        if amb_state['status'] == 'available':
            available_count += 1
    
    total_available = sum(1 for a in env.ambulance_states.values() if a['status'] == 'available')
    print(f"利用可能な救急車: {total_available}/{len(env.ambulance_states)}台")
    
    # 行動マスクの確認
    print("\n5. 行動マスク確認")
    action_mask = env.get_action_mask()
    print(f"マスクの形状: {action_mask.shape}")
    print(f"利用可能な行動数: {np.sum(action_mask)}")
    
    if np.sum(action_mask) == 0:
        print("❌ 利用可能な救急車がありません！")
        return
    
    # 6. テスト配車実行
    print("\n6. テスト配車実行")
    available_actions = np.where(action_mask)[0]
    print(f"利用可能な救急車ID: {available_actions[:10]}...")
    
    # ❌ 現在の問題のあるコード
    # action = available_actions[0]  # 常に最初の救急車
    
    # ✅ 修正案1: ランダム選択（PPOの初期状態を模擬）
    action = np.random.choice(available_actions)
    
    # ✅ 修正案2: 最近接選択（validation_simulation.pyと同じ）
    best_action = None
    min_time = float('inf')
    for amb_id in available_actions:
        travel_time = env._calculate_travel_time(
            env.ambulance_states[amb_id]['current_h3'],
            env.pending_call['h3_index']
        )
        if travel_time < min_time:
            min_time = travel_time
            best_action = amb_id
    action = best_action if best_action is not None else available_actions[0]
    
    print(f"選択した救急車: {action} (移動時間: {min_time/60:.1f}分)")
    
    # ステップ実行
    print("\n7. ステップ実行")
    step_result = env.step(action)
    
    print(f"報酬: {step_result.reward}")
    print(f"終了フラグ: {step_result.done}")
    print(f"追加情報: {step_result.info}")
    
    # dispatch_resultの詳細確認
    if 'dispatch_result' in step_result.info:
        dr = step_result.info['dispatch_result']
        print(f"\n配車結果詳細:")
        print(f"  成功: {dr.get('success', False)}")
        if not dr.get('success'):
            print(f"  失敗理由: {dr.get('reason', '不明')}")
        else:
            print(f"  応答時間: {dr.get('response_time', 0):.1f}秒")
            print(f"  応答時間（分）: {dr.get('response_time_minutes', 0):.1f}分")
    
    # エピソード統計
    print("\n8. エピソード統計")
    stats = env.episode_stats
    print(f"総配車数: {stats['total_dispatches']}")
    print(f"失敗配車数: {stats['failed_dispatches']}")
    if stats['response_times']:
        print(f"平均応答時間: {np.mean(stats['response_times']):.1f}分")
    
    # 複数ステップ実行
    print("\n9. 追加ステップ実行（最大10ステップ）")
    for i in range(10):
        if step_result.done:
            print(f"ステップ {i+1}: エピソード終了")
            break
        
        action_mask = env.get_action_mask()
        if np.sum(action_mask) == 0:
            print(f"ステップ {i+1}: 利用可能な救急車なし")
            break
        
        available_actions = np.where(action_mask)[0]
        action = available_actions[0]
        step_result = env.step(action)
        print(f"ステップ {i+1}: 報酬={step_result.reward:.2f}, 終了={step_result.done}")

def check_data_loading():
    """データ読み込みの確認"""
    print("\n" + "=" * 60)
    print("データ読み込み確認")
    print("=" * 60)
    
    # 救急事案データの確認
    calls_path = "C:/Users/hp/OneDrive - Yokohama City University/30_データカタログ/tfd_data/hanso_special_wards.csv"
    
    if not os.path.exists(calls_path):
        print(f"❌ データファイルが見つかりません: {calls_path}")
        return
    
    print(f"✓ データファイル存在確認")
    
    # データ読み込み
    calls_df = pd.read_csv(calls_path, encoding='utf-8')
    print(f"総レコード数: {len(calls_df):,}")
    
    # 日付変換
    calls_df['出場年月日時分'] = pd.to_datetime(calls_df['出場年月日時分'], errors='coerce')
    
    # 2023年4月1日のデータ確認
    target_date = pd.to_datetime('2023-04-01')
    mask = (calls_df['出場年月日時分'].dt.date == target_date.date())
    day_data = calls_df[mask]
    
    print(f"\n2023年4月1日のデータ:")
    print(f"  事案数: {len(day_data)}")
    
    if len(day_data) > 0:
        # 傷病度の分布
        severity_counts = day_data['収容所見程度'].value_counts()
        print(f"\n傷病度分布:")
        for severity, count in severity_counts.items():
            print(f"  {severity}: {count}")
        
        # 時間帯分布
        day_data['hour'] = day_data['出場年月日時分'].dt.hour
        hour_counts = day_data['hour'].value_counts().sort_index()
        print(f"\n時間帯分布:")
        for hour in range(0, 24, 6):
            count = hour_counts[hour:hour+6].sum() if hour in hour_counts.index else 0
            print(f"  {hour:02d}:00-{hour+6:02d}:00: {count}件")
    else:
        print("❌ 該当日のデータがありません")

def test_small_episode():
    """小規模エピソードのテスト"""
    print("\n" + "=" * 60)
    print("小規模エピソードテスト")
    print("=" * 60)
    
    config_path = "reinforcement_learning/experiments/config_quick.yaml"
    
    # 環境初期化
    env = EMSEnvironment(config_path, mode="train")
    
    # エピソード実行
    state = env.reset()
    total_reward = 0
    steps = 0
    
    print("エピソード開始")
    
    for step in range(100):  # 最大100ステップ
        # 行動選択（ランダム）
        action_mask = env.get_action_mask()
        
        if np.sum(action_mask) == 0:
            print(f"ステップ {step}: 利用可能な救急車なし")
            break
        
        available_actions = np.where(action_mask)[0]
        action = np.random.choice(available_actions)
        
        # ステップ実行
        step_result = env.step(action)
        
        total_reward += step_result.reward
        steps += 1
        
        if step % 10 == 0:
            print(f"ステップ {step}: 累積報酬={total_reward:.2f}")
        
        if step_result.done:
            print(f"エピソード終了（ステップ {step}）")
            break
    
    print(f"\n結果:")
    print(f"  総ステップ数: {steps}")
    print(f"  総報酬: {total_reward:.2f}")
    print(f"  平均報酬: {total_reward/steps if steps > 0 else 0:.2f}")
    
    # 最終統計
    stats = env.episode_stats
    print(f"\n最終統計:")
    print(f"  総配車数: {stats['total_dispatches']}")
    print(f"  失敗配車数: {stats['failed_dispatches']}")
    if stats['response_times']:
        print(f"  平均応答時間: {np.mean(stats['response_times']):.1f}分")
        print(f"  6分達成率: {stats['achieved_6min']/stats['total_dispatches']*100:.1f}%")
        print(f"  13分達成率: {stats['achieved_13min']/stats['total_dispatches']*100:.1f}%")

def main():
    """メインデバッグ実行"""
    print("\n環境デバッグツール")
    print("=" * 60)
    
    # 1. データ確認
    check_data_loading()
    
    # 2. 環境詳細デバッグ
    debug_environment()
    
    # 3. エピソードテスト
    test_small_episode()
    
    print("\n" + "=" * 60)
    print("デバッグ完了")
    print("=" * 60)

if __name__ == "__main__":
    main()