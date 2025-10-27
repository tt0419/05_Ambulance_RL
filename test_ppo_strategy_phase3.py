"""
test_ppo_strategy_phase3.py
PPOStrategy Phase 3修正版の動作確認スクリプト
"""

import sys
from pathlib import Path

def test_imports():
    """必要なモジュールのインポートを確認"""
    print("=" * 60)
    print("1. インポートテスト")
    print("=" * 60)
    
    try:
        from dispatch_strategies import PPOStrategy
        print("✓ PPOStrategyのインポート成功")
    except ImportError as e:
        print(f"✗ PPOStrategyのインポート失敗: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandasのインポート成功")
    except ImportError as e:
        print(f"✗ pandasのインポート失敗: {e}")
        return False
    
    try:
        import h3
        print("✓ h3のインポート成功")
    except ImportError as e:
        print(f"✗ h3のインポート失敗: {e}")
        return False
    
    return True


def test_initialization():
    """PPOStrategyの初期化を確認"""
    print("\n" + "=" * 60)
    print("2. 初期化テスト")
    print("=" * 60)
    
    try:
        from dispatch_strategies import PPOStrategy
        strategy = PPOStrategy()
        print("✓ PPOStrategyのインスタンス化成功")
        
        # 属性の確認
        assert hasattr(strategy, 'ambulance_static_info'), "ambulance_static_info属性がありません"
        print("✓ ambulance_static_info属性が存在")
        
        assert hasattr(strategy, 'validation_id_to_action'), "validation_id_to_action属性がありません"
        print("✓ validation_id_to_action属性が存在")
        
        assert hasattr(strategy, 'action_to_validation_id'), "action_to_validation_id属性がありません"
        print("✓ action_to_validation_id属性が存在")
        
        return True
        
    except Exception as e:
        print(f"✗ 初期化テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_static_info_loading():
    """静的情報の読み込みを確認"""
    print("\n" + "=" * 60)
    print("3. 静的情報読み込みテスト")
    print("=" * 60)
    
    try:
        from dispatch_strategies import PPOStrategy
        strategy = PPOStrategy()
        
        # ID対応表の読み込み
        print("\n[ID対応表の読み込み]")
        strategy._load_id_mapping()
        
        if strategy.id_mapping_loaded:
            print(f"✓ ID対応表読み込み成功: {len(strategy.validation_id_to_action)}件")
            
            # サンプル表示
            if strategy.validation_id_to_action:
                sample = list(strategy.validation_id_to_action.items())[:3]
                print(f"\n  サンプル:")
                for val_id, action in sample:
                    print(f"    '{val_id}' → アクション{action}")
        else:
            print("⚠️ ID対応表が読み込まれませんでした（フォールバックモードで動作）")
        
        # 静的情報の読み込み
        print("\n[静的情報の読み込み]")
        strategy._load_ambulance_static_info()
        
        if strategy.ambulance_static_info:
            print(f"✓ 静的情報読み込み成功: {len(strategy.ambulance_static_info)}台")
            
            # 192台あるか確認
            if len(strategy.ambulance_static_info) == 192:
                print("✓ 192台分の静的情報が存在")
            else:
                print(f"⚠️ 静的情報が192台ではありません: {len(strategy.ambulance_static_info)}台")
            
            # サンプル表示
            print(f"\n  静的情報サンプル:")
            for action_idx in range(min(3, len(strategy.ambulance_static_info))):
                if action_idx in strategy.ambulance_static_info:
                    info = strategy.ambulance_static_info[action_idx]
                    print(f"    Action {action_idx}: {info['validation_id']}")
                    print(f"      チーム名: {info['team_name']}")
                    print(f"      所属署H3: {info['station_h3'][:15]}...")
        else:
            print("✗ 静的情報が読み込まれませんでした")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 静的情報読み込みテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_dict_building():
    """状態辞書構築のテスト"""
    print("\n" + "=" * 60)
    print("4. 状態辞書構築テスト")
    print("=" * 60)
    
    try:
        from dispatch_strategies import PPOStrategy, EmergencyRequest, AmbulanceInfo, DispatchContext, DispatchPriority
        
        strategy = PPOStrategy()
        strategy._load_id_mapping()
        strategy._load_ambulance_static_info()
        
        # モックデータの作成
        request = EmergencyRequest(
            id="test_001",
            h3_index="89283082837ffff",
            severity="重症",
            time=0.0,
            priority=DispatchPriority.HIGH
        )
        
        available_ambulances = []
        
        context = DispatchContext()
        context.current_time = 3600.0  # 1時間
        context.hour_of_day = 12
        context.all_ambulances = {}  # 空の辞書（不完全な情報をシミュレート）
        
        print("\n[空の救急車情報で状態辞書を構築]")
        state_dict = strategy._build_state_dict(request, available_ambulances, context)
        
        # 検証
        assert 'ambulances' in state_dict, "ambulancesキーが存在しません"
        assert len(state_dict['ambulances']) == 192, f"救急車数が不正: {len(state_dict['ambulances'])}"
        print(f"✓ 192台分の状態辞書を構築: {len(state_dict['ambulances'])}台")
        
        # 全ての救急車がunavailableか確認
        unavailable_count = sum(1 for amb in state_dict['ambulances'].values() if amb['status'] == 'unavailable')
        print(f"  - unavailable: {unavailable_count}台")
        
        # 事案情報の確認
        assert 'pending_call' in state_dict, "pending_callキーが存在しません"
        print(f"✓ 事案情報が含まれています")
        print(f"  - H3: {state_dict['pending_call']['h3_index']}")
        print(f"  - 重症度: {state_dict['pending_call']['severity']}")
        
        # 時間情報の確認
        assert 'episode_step' in state_dict, "episode_stepキーが存在しません"
        assert 'time_of_day' in state_dict, "time_of_dayキーが存在しません"
        print(f"✓ 時間情報が含まれています")
        print(f"  - エピソードステップ: {state_dict['episode_step']}")
        print(f"  - 時刻: {state_dict['time_of_day']}時")
        
        return True
        
    except Exception as e:
        print(f"✗ 状態辞書構築テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_mask():
    """アクションマスクのテスト"""
    print("\n" + "=" * 60)
    print("5. アクションマスクテスト")
    print("=" * 60)
    
    try:
        from dispatch_strategies import PPOStrategy, AmbulanceInfo
        import numpy as np
        
        strategy = PPOStrategy()
        strategy.action_dim = 192
        strategy._load_id_mapping()
        
        # 空の救急車リスト
        print("\n[空の救急車リストでマスクを作成]")
        available_ambulances = []
        mask = strategy._create_action_mask(available_ambulances)
        
        assert mask.shape == (192,), f"マスクのサイズが不正: {mask.shape}"
        print(f"✓ マスクサイズ: {mask.shape}")
        
        # フォールバックが働いているか確認
        if mask.any():
            print(f"✓ フォールバック機能が動作（{mask.sum()}台が利用可能）")
        else:
            print(f"⚠️ 利用可能な救急車が0台")
        
        return True
        
    except Exception as e:
        print(f"✗ アクションマスクテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("\n" + "=" * 60)
    print("PPOStrategy Phase 3 動作確認テスト")
    print("=" * 60)
    
    results = []
    
    # 各テストを実行
    results.append(("インポートテスト", test_imports()))
    results.append(("初期化テスト", test_initialization()))
    results.append(("静的情報読み込みテスト", test_static_info_loading()))
    results.append(("状態辞書構築テスト", test_state_dict_building()))
    results.append(("アクションマスクテスト", test_action_mask()))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print(f"✓ {test_name}: 成功")
            passed += 1
        else:
            print(f"✗ {test_name}: 失敗")
            failed += 1
    
    print("\n" + "-" * 60)
    print(f"合計: {passed + failed}件")
    print(f"成功: {passed}件")
    print(f"失敗: {failed}件")
    print("-" * 60)
    
    if failed == 0:
        print("\n✓ 全てのテストが成功しました！")
        return 0
    else:
        print(f"\n⚠️ {failed}件のテストが失敗しました。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

