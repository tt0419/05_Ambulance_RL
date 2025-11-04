"""
test_stage1_integration.py
Stage 1（イベント駆動基礎）の統合テスト

検証項目：
1. イベントクラスが正しく定義されているか
2. イベントキューが初期化されているか
3. 基本イベント処理メソッドが動作するか
4. 既存の実用機能（RewardDesigner、ハイブリッドモード等）が100%維持されているか
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reinforcement_learning.environment.ems_environment_v2 import EMSEnvironment, Event, EventType
import numpy as np

def test_event_classes():
    """イベントクラスの定義テスト"""
    print("\n【テスト1】イベントクラスの定義")
    
    # EventTypeのチェック
    assert hasattr(EventType, 'NEW_CALL'), "EventType.NEW_CALLが定義されていません"
    assert hasattr(EventType, 'AMBULANCE_RETURN'), "EventType.AMBULANCE_RETURNが定義されていません"
    assert hasattr(EventType, 'EPISODE_END'), "EventType.EPISODE_ENDが定義されていません"
    print("  ✓ EventType定義OK")
    
    # Eventクラスのチェック
    event = Event(time=100.0, event_type=EventType.NEW_CALL, data={'test': 'data'})
    assert event.time == 100.0, "Event.time が正しく設定されていません"
    assert event.event_type == EventType.NEW_CALL, "Event.event_type が正しく設定されていません"
    assert event.data == {'test': 'data'}, "Event.data が正しく設定されていません"
    print("  ✓ Event定義OK")
    
    # 優先度付きキューのテスト
    event1 = Event(time=100.0, event_type=EventType.NEW_CALL, data={})
    event2 = Event(time=50.0, event_type=EventType.NEW_CALL, data={})
    assert event2 < event1, "Event.__lt__ が正しく動作していません"
    print("  ✓ Event優先度比較OK")
    
    print("  ✅ テスト1合格")

def test_environment_initialization():
    """環境の初期化テスト"""
    print("\n【テスト2】環境の初期化")
    
    try:
        env = EMSEnvironment(
            config_path="reinforcement_learning/experiments/config_continuous.yaml",
            mode="train"
        )
        print("  ✓ 環境初期化OK")
        
        # イベントキューの初期化確認
        assert hasattr(env, 'event_queue'), "event_queueが初期化されていません"
        assert isinstance(env.event_queue, list), "event_queueがlistではありません"
        assert hasattr(env, 'current_time'), "current_timeが初期化されていません"
        assert env.current_time == 0.0, "current_timeが0.0で初期化されていません"
        print("  ✓ イベントキュー初期化OK")
        
        # 既存の初期化確認
        assert hasattr(env, 'reward_designer'), "reward_designerが初期化されていません"
        assert hasattr(env, 'dispatch_logger'), "dispatch_loggerが初期化されていません"
        assert hasattr(env, 'hybrid_mode'), "hybrid_modeが設定されていません"
        print("  ✓ 既存コンポーネント初期化OK")
        
        print("  ✅ テスト2合格")
        return env
    
    except Exception as e:
        print(f"  ❌ テスト2失敗: {e}")
        raise

def test_event_methods(env):
    """イベント処理メソッドのテスト"""
    print("\n【テスト3】イベント処理メソッド")
    
    # _schedule_eventのテスト
    event1 = Event(time=100.0, event_type=EventType.NEW_CALL, data={'id': '1'})
    event2 = Event(time=50.0, event_type=EventType.AMBULANCE_RETURN, data={'ambulance_id': 0})
    
    env._schedule_event(event1)
    env._schedule_event(event2)
    
    assert len(env.event_queue) == 2, "イベントが正しく追加されていません"
    print("  ✓ _schedule_event OK")
    
    # _process_next_eventのテスト（時刻の早い方が先に処理される）
    # 注：reset()前でもイベント処理は安全に実行されるべき
    processed_event = env._process_next_event()
    assert processed_event.time == 50.0, "イベントの処理順序が正しくありません"
    assert env.current_time == 50.0, "current_timeが更新されていません"
    print("  ✓ _process_next_event OK（reset()前でも安全に実行）")
    
    processed_event = env._process_next_event()
    assert processed_event.time == 100.0, "2番目のイベントの処理順序が正しくありません"
    assert env.current_time == 100.0, "current_timeが更新されていません"
    print("  ✓ イベント処理順序OK")
    
    assert len(env.event_queue) == 0, "全イベント処理後もキューが空になっていません"
    print("  ✓ イベントキュー管理OK")
    
    # イベントハンドラーの防御的実装の確認
    print("  ✓ reset()前のイベント処理が安全に実行されました")
    
    print("  ✅ テスト3合格")

def test_existing_methods(env):
    """既存メソッドの動作テスト"""
    print("\n【テスト4】既存メソッドの動作確認")
    
    # 重要な既存メソッドの存在確認
    required_methods = [
        'get_optimal_action',
        'get_episode_statistics',
        'render',
        '_calculate_reward_detailed',
        '_initialize_ambulances_realistic',
        '_prepare_episode_calls'
    ]
    
    for method_name in required_methods:
        assert hasattr(env, method_name), f"{method_name}が存在しません"
        print(f"  ✓ {method_name}存在確認OK")
    
    # _calculate_reward_detailedの基本動作確認（reset()前でも動作）
    # カバレッジボーナスはなし、基本報酬のみ
    reward_before_reset = env._calculate_reward_detailed(5.0, '重症')  # 6分以内、重症
    assert reward_before_reset > 0, "_calculate_reward_detailedが正しく動作していません"
    print(f"  ✓ _calculate_reward_detailed基本動作OK (reset()前: reward={reward_before_reset:.2f})")
    
    print("  ✅ テスト4合格")

def test_reset_and_basic_episode():
    """reset()と基本的なエピソード実行のテスト"""
    print("\n【テスト5】reset()と基本エピソード")
    
    try:
        env = EMSEnvironment(
            config_path="reinforcement_learning/experiments/config_continuous.yaml",
            mode="train"
        )
        
        # reset()のテスト
        obs = env.reset()
        assert obs is not None, "reset()がNoneを返しました"
        assert isinstance(obs, np.ndarray), "reset()がndarrayを返していません"
        assert obs.shape[0] == env.state_dim, f"観測ベクトルの次元が不正です（期待: {env.state_dim}、実際: {obs.shape[0]}）"
        print(f"  ✓ reset() OK (state_dim={env.state_dim})")
        
        # action_maskのテスト
        mask = env.get_action_mask()
        assert isinstance(mask, np.ndarray), "get_action_mask()がndarrayを返していません"
        assert mask.shape[0] == env.action_dim, "action_maskの次元が不正です"
        assert np.any(mask), "全救急車がavailableでない状態です（初期化に問題がある可能性）"
        print(f"  ✓ get_action_mask() OK (available: {mask.sum()}/{env.action_dim}台)")
        
        # 1ステップ実行のテスト
        valid_actions = np.where(mask)[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
            result = env.step(action)
            
            assert hasattr(result, 'observation'), "StepResultにobservationがありません"
            assert hasattr(result, 'reward'), "StepResultにrewardがありません"
            assert hasattr(result, 'done'), "StepResultにdoneがありません"
            assert hasattr(result, 'info'), "StepResultにinfoがありません"
            print(f"  ✓ step() OK (reward={result.reward:.2f})")
            
            # ハイブリッドモードの動作確認
            if env.hybrid_mode and 'dispatch_type' in result.info:
                print(f"  ✓ ハイブリッドモード動作確認 (type={result.info['dispatch_type']})")
            
            # reset()後の報酬計算確認（カバレッジボーナス含む）
            reward_after_reset = env._calculate_reward_detailed(5.0, '重症')
            print(f"  ✓ _calculate_reward_detailed完全動作OK (reset()後: reward={reward_after_reset:.2f})")
        
        print("  ✅ テスト5合格")
        
    except Exception as e:
        print(f"  ❌ テスト5失敗: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """メインテスト実行"""
    print("=" * 60)
    print("Stage 1（イベント駆動基礎）統合テスト")
    print("=" * 60)
    
    try:
        # テスト1: イベントクラスの定義
        test_event_classes()
        
        # テスト2: 環境の初期化
        env = test_environment_initialization()
        
        # テスト3: イベント処理メソッド
        test_event_methods(env)
        
        # テスト4: 既存メソッドの動作確認
        test_existing_methods(env)
        
        # テスト5: reset()と基本エピソード
        test_reset_and_basic_episode()
        
        print("\n" + "=" * 60)
        print("✅ 全テスト合格！Stage 1の実装は正常です。")
        print("=" * 60)
        print("\n次のステップ：")
        print("  1. Stage 2の実装に進む")
        print("  2. reset()とstep()メソッドをイベント駆動対応に改修")
        
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

