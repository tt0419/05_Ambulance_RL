#!/usr/bin/env python3
"""
test_area3_environment.py
第3方面限定環境のテストスクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from reinforcement_learning.environment.ems_environment import EMSEnvironment
from data_cache import get_emergency_data_cache
import logging

def test_area3_environment():
    """第3方面限定環境の動作確認"""
    print("=" * 80)
    print("第3方面限定環境テスト開始")
    print("=" * 80)
    
    # ログレベル設定
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 1. データキャッシュのテスト
        print("\n1. データキャッシュのテスト")
        print("-" * 40)
        cache = get_emergency_data_cache()
        
        # 第3方面のエリアフィルタ
        area3_districts = ["目黒区", "渋谷区", "世田谷区"]
        
        # テスト期間のデータ取得
        test_period_data = cache.get_period_data(
            "20230401", "20230407", 
            area_filter=area3_districts
        )
        print(f"第3方面の1週間データ: {len(test_period_data)}件")
        
        if len(test_period_data) > 0:
            print("出場先区市の分布:")
            if '出場先区市' in test_period_data.columns:
                district_counts = test_period_data['出場先区市'].value_counts()
                for district, count in district_counts.items():
                    print(f"  {district}: {count}件")
            
            print("\n傷病度の分布:")
            severity_counts = test_period_data['収容所見程度'].value_counts()
            for severity, count in severity_counts.head().items():
                print(f"  {severity}: {count}件")
        
        # 2. EMS環境の初期化テスト
        print("\n2. EMS環境の初期化テスト")
        print("-" * 40)
        
        config_path = "reinforcement_learning/experiments/config_area3.yaml"
        env = EMSEnvironment(config_path=config_path, mode="train")
        
        print(f"救急車数: {env.action_dim}台")
        print(f"状態空間次元: {env.state_dim}")
        
        # 救急車データの確認
        if hasattr(env, 'ambulance_data') and len(env.ambulance_data) > 0:
            print("\n救急車分布:")
            print(f"  第3方面の救急車: {len(env.ambulance_data)}台")
            
            # sectionカラムが存在する場合の確認
            if 'section' in env.ambulance_data.columns:
                section_counts = env.ambulance_data['section'].value_counts()
                print("  方面別分布:")
                for section, count in section_counts.items():
                    print(f"    第{section}方面: {count}台")
        
        # 3. 環境のリセットテスト
        print("\n3. 環境のリセットテスト")
        print("-" * 40)
        
        initial_obs = env.reset()
        print(f"初期観測の形状: {initial_obs.shape}")
        print(f"初期観測の範囲: [{initial_obs.min():.3f}, {initial_obs.max():.3f}]")
        
        # エピソードの事案数確認
        if hasattr(env, 'current_episode_calls'):
            print(f"エピソード内事案数: {len(env.current_episode_calls)}件")
            
            if len(env.current_episode_calls) > 0:
                first_call = env.current_episode_calls[0]
                print(f"最初の事案: {first_call.get('severity', 'Unknown')} at {first_call.get('datetime', 'Unknown')}")
        
        # 4. 1ステップの実行テスト
        print("\n4. 1ステップの実行テスト")
        print("-" * 40)
        
        # 利用可能な行動をチェック
        action_mask = env.get_action_mask()
        available_actions = [i for i, available in enumerate(action_mask) if available]
        print(f"利用可能な救急車: {len(available_actions)}台")
        
        if len(available_actions) > 0:
            # 最初の利用可能な救急車を選択
            test_action = available_actions[0]
            
            # 最適な行動を取得
            optimal_action = env.get_optimal_action()
            print(f"テスト行動: 救急車{test_action}")
            print(f"最適行動: 救急車{optimal_action}")
            
            # 1ステップ実行
            result = env.step(test_action)
            if result:
                print(f"報酬: {result.reward:.3f}")
                print(f"終了フラグ: {result.done}")
                print(f"次の観測形状: {result.observation.shape}")
        
        print("\n5. 設定値の確認")
        print("-" * 40)
        
        config = env.config
        area_restriction = config.get('data', {}).get('area_restriction', {})
        print(f"エリア制限有効: {area_restriction.get('enabled', False)}")
        print(f"対象方面: {area_restriction.get('section_code', 'N/A')}")
        print(f"対象区市: {', '.join(area_restriction.get('districts', []))}")
        
        ppo_config = config.get('ppo', {})
        print(f"エピソード数: {ppo_config.get('n_episodes', 'N/A')}")
        print(f"バッチサイズ: {ppo_config.get('batch_size', 'N/A')}")
        print(f"学習率(Actor): {ppo_config.get('learning_rate', {}).get('actor', 'N/A')}")
        
        print("\n✅ 第3方面限定環境のテスト完了")
        return True
        
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_scale_comparison():
    """データ規模の比較"""
    print("\n" + "=" * 80)
    print("データ規模比較")
    print("=" * 80)
    
    cache = get_emergency_data_cache()
    
    # 全体データ
    all_data = cache.get_period_data("20230401", "20230407")
    print(f"全23区の1週間データ: {len(all_data)}件")
    
    # 第3方面データ
    area3_data = cache.get_period_data(
        "20230401", "20230407", 
        area_filter=["目黒区", "渋谷区", "世田谷区"]
    )
    print(f"第3方面の1週間データ: {len(area3_data)}件")
    
    if len(all_data) > 0:
        reduction_ratio = len(area3_data) / len(all_data) * 100
        print(f"データ削減率: {reduction_ratio:.1f}% (約{100/reduction_ratio:.1f}分の1)")

if __name__ == "__main__":
    success = test_area3_environment()
    test_data_scale_comparison()
    
    if success:
        print("\n🎉 全てのテストが正常に完了しました！")
        print("第3方面限定環境で学習を開始できます。")
        print("\n次のコマンドで学習を実行してください:")
        print("python train_ppo.py --config reinforcement_learning/experiments/config_area3.yaml")
    else:
        print("\n⚠️ テストでエラーが発生しました。設定を確認してください。")
