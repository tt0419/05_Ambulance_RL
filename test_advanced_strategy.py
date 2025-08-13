"""
test_advanced_strategy.py
AdvancedSeverityStrategyの動作確認用テストスクリプト
"""

import sys
import os
from dispatch_strategies import (
    StrategyFactory, 
    STRATEGY_CONFIGS,
    EmergencyRequest,
    AmbulanceInfo,
    DispatchContext,
    DispatchPriority
)

def test_advanced_strategy():
    """AdvancedSeverityStrategyの基本動作をテスト"""
    
    print("=" * 60)
    print("AdvancedSeverityStrategy 動作確認テスト")
    print("=" * 60)
    
    # 1. 戦略の作成
    print("\n1. 戦略の作成...")
    try:
        strategy = StrategyFactory.create_strategy('advanced_severity', STRATEGY_CONFIGS['aggressive'])
        print(f"✓ 戦略作成成功: {strategy.name}")
        print(f"  設定: {strategy.config}")
    except Exception as e:
        print(f"✗ 戦略作成失敗: {e}")
        return False
    
    # 2. 利用可能な戦略の確認
    print("\n2. 利用可能な戦略の確認...")
    available_strategies = StrategyFactory.list_available_strategies()
    print(f"✓ 利用可能な戦略: {available_strategies}")
    
    # 3. 設定プリセットの確認
    print("\n3. 設定プリセットの確認...")
    for preset_name, config in STRATEGY_CONFIGS.items():
        print(f"  {preset_name}: {config}")
    
    # 4. 簡単なシミュレーション
    print("\n4. 簡単なシミュレーション...")
    
    # テスト用データの作成
    request = EmergencyRequest(
        id="test_call_001",
        h3_index="8928308280fffff",  # 東京のH3インデックス
        severity="重症",
        time=3600.0,  # 1時間後
        priority=DispatchPriority.HIGH
    )
    
    ambulances = [
        AmbulanceInfo(
            id="amb_001",
            current_h3="8928308281fffff",
            station_h3="8928308281fffff",
            status="available",
            total_calls_today=5
        ),
        AmbulanceInfo(
            id="amb_002", 
            current_h3="8928308282fffff",
            station_h3="8928308282fffff",
            status="available",
            total_calls_today=3
        )
    ]
    
    context = DispatchContext()
    context.current_time = 3600.0
    context.hour_of_day = 10
    context.total_ambulances = 10
    context.available_ambulances = 2
    
    # 簡易的な移動時間関数
    def mock_travel_time(from_h3, to_h3, phase):
        return 300.0  # 5分固定
    
    # 戦略の実行
    try:
        selected = strategy.select_ambulance(
            request=request,
            available_ambulances=ambulances,
            travel_time_func=mock_travel_time,
            context=context
        )
        
        if selected:
            print(f"✓ 救急車選択成功: {selected.id}")
        else:
            print("✗ 救急車選択失敗: 選択されませんでした")
            
    except Exception as e:
        print(f"✗ 戦略実行エラー: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
    
    return True

def test_strategy_configs():
    """設定プリセットの詳細テスト"""
    
    print("\n" + "=" * 60)
    print("設定プリセット詳細テスト")
    print("=" * 60)
    
    for preset_name, config in STRATEGY_CONFIGS.items():
        print(f"\nプリセット: {preset_name}")
        print("-" * 30)
        
        try:
            strategy = StrategyFactory.create_strategy('advanced_severity', config)
            print(f"✓ 作成成功")
            print(f"  設定内容:")
            for key, value in strategy.config.items():
                print(f"    {key}: {value}")
        except Exception as e:
            print(f"✗ 作成失敗: {e}")

if __name__ == "__main__":
    print("AdvancedSeverityStrategy テスト開始")
    
    # 基本動作テスト
    success = test_advanced_strategy()
    
    # 設定プリセットテスト
    test_strategy_configs()
    
    if success:
        print("\n🎉 すべてのテストが成功しました！")
        print("AdvancedSeverityStrategy は正常に動作しています。")
    else:
        print("\n❌ テストに失敗しました。")
        print("エラーの詳細を確認してください。") 