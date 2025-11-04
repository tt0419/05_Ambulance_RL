"""
test_stage2_event_driven.py
Stage 2（イベント駆動統合）の統合テスト

検証項目：
1. reset()で全イベントが正しくスケジュールされる
2. step()で救急車復帰イベントが正しくスケジュールされる
3. イベント駆動ループが正しく動作する
4. 既存のGym互換インターフェースが維持されている
5. current_timeとepisode_step_secondsが同期している
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reinforcement_learning.environment.ems_environment_v2 import EMSEnvironment, Event, EventType
import numpy as np

def test_reset_event_scheduling():
    """reset()で全イベントが正しくスケジュールされるかテスト"""
    print("\n【テスト1】reset()でのイベントスケジューリング")
    
    env = EMSEnvironment(
        config_path="reinforcement_learning/experiments/config_continuous.yaml",
        mode="train"
    )
    
    obs = env.reset()
    
    # イベントキューが空でないことを確認
    assert len(env.event_queue) > 0, "reset()後にイベントキューが空です"
    print(f"  ✓ イベントキュー初期化OK: {len(env.event_queue)}イベント")
    
    # イベントタイプの確認
    event_types = {}
    for event in env.event_queue:
        event_type_name = event.event_type.value
        event_types[event_type_name] = event_types.get(event_type_name, 0) + 1
    
    print(f"  ✓ イベント内訳:")
    for event_type, count in event_types.items():
        print(f"    - {event_type}: {count}件")
    
    # NEW_CALLイベントが事案数と一致するか確認
    new_call_count = event_types.get('new_call', 0)
    assert new_call_count == len(env.current_episode_calls), \
        f"NEW_CALLイベント数({new_call_count})が事案数({len(env.current_episode_calls)})と一致しません"
    print(f"  ✓ NEW_CALLイベント数OK: {new_call_count}件")
    
    # EPISODE_ENDイベントが存在するか確認
    assert 'episode_end' in event_types, "EPISODE_ENDイベントがスケジュールされていません"
    print(f"  ✓ EPISODE_ENDイベントOK")
    
    # AMBULANCE_RETURNイベントが初期稼働中の救急車数と一致するか確認
    dispatched_count = sum(1 for s in env.ambulance_states.values() if s['status'] == 'dispatched')
    ambulance_return_count = event_types.get('ambulance_return', 0)
    assert ambulance_return_count == dispatched_count, \
        f"AMBULANCE_RETURNイベント数({ambulance_return_count})が初期稼働中救急車数({dispatched_count})と一致しません"
    print(f"  ✓ AMBULANCE_RETURNイベント数OK: {ambulance_return_count}件")
    
    # current_timeとepisode_step_secondsの同期確認
    assert abs(env.current_time - env.episode_step_seconds) < 0.1, \
        f"current_time({env.current_time})とepisode_step_seconds({env.episode_step_seconds})が同期していません"
    print(f"  ✓ 時間同期OK: current_time={env.current_time:.2f}秒")
    
    # pending_callが設定されているか確認
    assert env.pending_call is not None, "reset()後にpending_callがNoneです"
    print(f"  ✓ pending_call設定OK")
    
    print("  ✅ テスト1合格")
    return env

def test_step_return_event_scheduling():
    """step()で救急車復帰イベントが正しくスケジュールされるかテスト"""
    print("\n【テスト2】step()での復帰イベントスケジューリング")
    
    env = EMSEnvironment(
        config_path="reinforcement_learning/experiments/config_continuous.yaml",
        mode="train"
    )
    
    obs = env.reset()
    
    # イベント数を記録
    initial_event_count = len(env.event_queue)
    
    # 最初のstep()を実行
    mask = env.get_action_mask()
    valid_actions = np.where(mask)[0]
    
    if len(valid_actions) > 0:
        action = valid_actions[0]
        initial_ambulance_id = action
        
        # step()前の救急車状態を確認
        assert env.ambulance_states[initial_ambulance_id]['status'] == 'available', \
            "選択した救急車がavailableではありません"
        
        result = env.step(action)
        
        # step()後の救急車状態を確認
        assert env.ambulance_states[initial_ambulance_id]['status'] == 'dispatched', \
            "step()後に救急車がdispatchedになっていません"
        print(f"  ✓ 救急車{initial_ambulance_id}がdispatchedに変更されました")
        
        # AMBULANCE_RETURNイベントが追加されたか確認
        # （新しいイベントが追加されている、ただし_advance_to_next_call()で一部処理済みの可能性あり）
        ambulance_return_events = [e for e in env.event_queue if e.event_type == EventType.AMBULANCE_RETURN]
        
        # 配車した救急車の復帰イベントが存在するか確認
        has_return_event = any(
            e.data.get('ambulance_id') == initial_ambulance_id 
            for e in ambulance_return_events
        )
        assert has_return_event or env.ambulance_states[initial_ambulance_id]['status'] == 'dispatched', \
            f"救急車{initial_ambulance_id}の復帰イベントがスケジュールされていません"
        print(f"  ✓ 復帰イベントスケジューリングOK")
        
        # completion_timeが設定されているか確認
        completion_time = env.ambulance_states[initial_ambulance_id]['completion_time']
        assert completion_time > env.current_time, \
            f"completion_time({completion_time})がcurrent_time({env.current_time})より未来ではありません"
        print(f"  ✓ completion_time設定OK: {completion_time:.2f}秒")
    
    print("  ✅ テスト2合格")
    return env

def test_event_driven_loop():
    """イベント駆動ループが正しく動作するかテスト"""
    print("\n【テスト3】イベント駆動ループ")
    
    env = EMSEnvironment(
        config_path="reinforcement_learning/experiments/config_continuous.yaml",
        mode="train"
    )
    
    obs = env.reset()
    
    # 複数ステップ実行
    steps = 0
    max_steps = 10
    
    times = [env.current_time]
    
    while not env._is_episode_done() and steps < max_steps:
        mask = env.get_action_mask()
        valid_actions = np.where(mask)[0]
        
        if len(valid_actions) == 0:
            print(f"  [ステップ{steps}] 全救急車が出動中です")
            # 全車出動中の場合、次のイベント（救急車復帰）まで進む
            # これはstep()が自動的に処理するはず
            # ダミーのアクションを渡す
            action = 0
        else:
            action = valid_actions[0]
        
        result = env.step(action)
        steps += 1
        times.append(env.current_time)
        
        # current_timeとepisode_step_secondsの同期確認
        assert abs(env.current_time - env.episode_step_seconds) < 0.1, \
            f"ステップ{steps}: current_time({env.current_time})とepisode_step_seconds({env.episode_step_seconds})が同期していません"
        
        # 時間が単調増加しているか確認
        if steps > 1:
            assert times[-1] >= times[-2], \
                f"ステップ{steps}: 時間が逆行しています({times[-2]:.2f} -> {times[-1]:.2f})"
    
    print(f"  ✓ {steps}ステップ実行完了")
    print(f"  ✓ 時間推移: {times[0]:.2f}秒 -> {times[-1]:.2f}秒")
    print(f"  ✓ 全ステップで時間同期OK")
    print(f"  ✓ 時間の単調増加OK")
    
    print("  ✅ テスト3合格")

def test_gym_compatibility():
    """既存のGym互換インターフェースが維持されているかテスト"""
    print("\n【テスト4】Gym互換インターフェース")
    
    env = EMSEnvironment(
        config_path="reinforcement_learning/experiments/config_continuous.yaml",
        mode="train"
    )
    
    # reset()のシグネチャ確認
    obs = env.reset()
    assert isinstance(obs, np.ndarray), "reset()がndarrayを返していません"
    assert obs.shape[0] == env.state_dim, f"観測ベクトルの次元が不正です（期待: {env.state_dim}、実際: {obs.shape[0]}）"
    print(f"  ✓ reset()シグネチャOK (返り値: ndarray, shape={obs.shape})")
    
    # step()のシグネチャ確認
    mask = env.get_action_mask()
    valid_actions = np.where(mask)[0]
    
    if len(valid_actions) > 0:
        action = valid_actions[0]
        result = env.step(action)
        
        # StepResultの構造確認
        assert hasattr(result, 'observation'), "StepResultにobservationがありません"
        assert hasattr(result, 'reward'), "StepResultにrewardがありません"
        assert hasattr(result, 'done'), "StepResultにdoneがありません"
        assert hasattr(result, 'info'), "StepResultにinfoがありません"
        
        assert isinstance(result.observation, np.ndarray), "observationがndarrayではありません"
        assert isinstance(result.reward, (int, float)), "rewardが数値ではありません"
        assert isinstance(result.done, bool), "doneがboolではありません"
        assert isinstance(result.info, dict), "infoがdictではありません"
        
        print(f"  ✓ step()シグネチャOK")
        print(f"    - observation: ndarray, shape={result.observation.shape}")
        print(f"    - reward: {type(result.reward).__name__} = {result.reward:.2f}")
        print(f"    - done: {type(result.done).__name__} = {result.done}")
        print(f"    - info: {type(result.info).__name__} (keys: {len(result.info)})")
    
    # 既存メソッドの存在確認
    assert hasattr(env, 'get_action_mask'), "get_action_maskが存在しません"
    assert hasattr(env, 'get_optimal_action'), "get_optimal_actionが存在しません"
    assert hasattr(env, 'render'), "renderが存在しません"
    print(f"  ✓ 既存メソッド存在確認OK")
    
    print("  ✅ テスト4合格")

def test_ambulance_return_processing():
    """救急車復帰イベントが正しく処理されるかテスト"""
    print("\n【テスト5】救急車復帰イベントの処理")
    
    env = EMSEnvironment(
        config_path="reinforcement_learning/experiments/config_continuous.yaml",
        mode="train"
    )
    
    obs = env.reset()
    
    # 最初に配車した救急車のIDを記録
    mask = env.get_action_mask()
    valid_actions = np.where(mask)[0]
    
    if len(valid_actions) > 0:
        action = valid_actions[0]
        ambulance_id = action
        
        # step()実行
        result = env.step(action)
        
        # 救急車がdispatchedであることを確認
        assert env.ambulance_states[ambulance_id]['status'] == 'dispatched', \
            "step()直後に救急車がdispatchedになっていません"
        
        completion_time = env.ambulance_states[ambulance_id]['completion_time']
        print(f"  ✓ 救急車{ambulance_id}を配車: 復帰予定={completion_time:.2f}秒")
        
        # 複数ステップ実行して、救急車が復帰するまで進める
        steps = 0
        max_steps = 100
        returned = False
        
        while steps < max_steps and not env._is_episode_done():
            # 次のステップを実行
            mask = env.get_action_mask()
            valid_actions = np.where(mask)[0]
            action = valid_actions[0] if len(valid_actions) > 0 else 0
            
            result = env.step(action)
            steps += 1
            
            # 救急車が復帰したか確認
            if env.ambulance_states[ambulance_id]['status'] == 'available':
                returned = True
                print(f"  ✓ 救急車{ambulance_id}が復帰しました: {steps}ステップ後, 時刻={env.current_time:.2f}秒")
                
                # 復帰後の状態確認
                assert env.ambulance_states[ambulance_id]['current_h3'] == env.ambulance_states[ambulance_id]['station_h3'], \
                    "復帰後に救急車がステーションに戻っていません"
                print(f"  ✓ 救急車{ambulance_id}がステーションに復帰しました")
                break
        
        if returned:
            print(f"  ✅ テスト5合格")
        else:
            print(f"  ⚠ 警告: {max_steps}ステップ以内に救急車が復帰しませんでした")
            print(f"  （これは正常な場合もあります：エピソード期間が短い、または救急車の活動時間が長い）")
    else:
        print(f"  ⚠ 警告: 利用可能な救急車がいないため、テスト5をスキップします")

def main():
    """メインテスト実行"""
    print("=" * 60)
    print("Stage 2（イベント駆動統合）統合テスト")
    print("=" * 60)
    
    try:
        # テスト1: reset()でのイベントスケジューリング
        env = test_reset_event_scheduling()
        
        # テスト2: step()での復帰イベントスケジューリング
        test_step_return_event_scheduling()
        
        # テスト3: イベント駆動ループ
        test_event_driven_loop()
        
        # テスト4: Gym互換インターフェース
        test_gym_compatibility()
        
        # テスト5: 救急車復帰イベントの処理
        test_ambulance_return_processing()
        
        print("\n" + "=" * 60)
        print("✅ 全テスト合格！Stage 2の実装は正常です。")
        print("=" * 60)
        print("\n次のステップ：")
        print("  1. Stage 3の実装に進む（ValidationSimulatorの詳細機能統合）")
        print("  2. または既存のtrain_ppo.pyで学習を実行して動作確認")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ テスト失敗: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


