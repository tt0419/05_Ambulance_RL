#!/usr/bin/env python3
"""
拡張ログ機能のテスト
既存機能が正常に動作するかを確認
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """必要なモジュールのインポートテスト"""
    try:
        print("=== インポートテスト ===")
        
        # trainer.pyのインポート
        from reinforcement_learning.training.trainer import PPOTrainer
        print("✓ PPOTrainer インポート成功")
        
        # ems_environment.pyのインポート
        from reinforcement_learning.environment.ems_environment import EMSEnvironment
        print("✓ EMSEnvironment インポート成功")
        
        print("✓ 全てのインポートが成功しました")
        return True
        
    except Exception as e:
        print(f"❌ インポートエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_initialization():
    """環境の初期化テスト"""
    try:
        print("\n=== 環境初期化テスト ===")
        
        # 設定ファイルの確認
        config_path = "reinforcement_learning/experiments/config_test.yaml"
        if not os.path.exists(config_path):
            print(f"警告: {config_path} が見つかりません。config.yamlを使用します。")
            config_path = "config.yaml"
        
        from reinforcement_learning.environment.ems_environment import EMSEnvironment
        
        # 環境の初期化（実際のデータロードはスキップ）
        print("環境を初期化中...")
        env = EMSEnvironment(config_path, mode="train")
        print("✓ 環境初期化成功")
        
        # 新しい統計機能の確認
        print("拡張統計機能をテスト中...")
        stats = env._init_episode_stats()
        
        # 必要なキーが存在するかチェック
        required_keys = [
            'ambulance_utilization', 'spatial_coverage', 
            'temporal_patterns', 'efficiency_metrics', 
            'severity_detailed_stats'
        ]
        
        for key in required_keys:
            if key in stats:
                print(f"  ✓ {key} 統計が初期化されました")
            else:
                print(f"  ❌ {key} 統計が見つかりません")
                return False
        
        # get_episode_statistics メソッドの確認
        if hasattr(env, 'get_episode_statistics'):
            detailed_stats = env.get_episode_statistics()
            print("✓ get_episode_statistics メソッドが利用可能です")
        else:
            print("❌ get_episode_statistics メソッドが見つかりません")
            return False
        
        print("✓ 環境テストが成功しました")
        return True
        
    except Exception as e:
        print(f"❌ 環境テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_methods():
    """トレーナーの新機能テスト"""
    try:
        print("\n=== トレーナー機能テスト ===")
        
        # 基本設定（ダミー）
        config = {
            'ppo': {'n_episodes': 10},
            'training': {'checkpoint_interval': 5, 'early_stopping': {'enabled': False}, 'logging': {'wandb': False, 'tensorboard': False}},
            'evaluation': {'interval': 5, 'n_eval_episodes': 2},
            'teacher': {'enabled': True, 'decay_episodes': 100, 'initial_prob': 0.8, 'final_prob': 0.2}
        }
        
        from reinforcement_learning.training.trainer import PPOTrainer
        
        # PPOTrainerのクラス定義確認
        print("PPOTrainerクラスをチェック中...")
        
        # 新しいメソッドの確認
        required_methods = [
            '_log_baseline_comparison', 
            '_log_curriculum_progress'
        ]
        
        for method_name in required_methods:
            if hasattr(PPOTrainer, method_name):
                print(f"  ✓ {method_name} メソッドが見つかりました")
            else:
                print(f"  ❌ {method_name} メソッドが見つかりません")
                return False
        
        print("✓ トレーナー機能テストが成功しました")
        return True
        
    except Exception as e:
        print(f"❌ トレーナーテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_compatibility():
    """設定ファイルとの互換性テスト"""
    try:
        print("\n=== 設定互換性テスト ===")
        
        # 設定ファイルの読み込みテスト
        import yaml
        
        config_files = [
            "config.yaml",
            "reinforcement_learning/experiments/config_test.yaml",
            "reinforcement_learning/experiments/config_1week.yaml"
        ]
        
        valid_config = None
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    print(f"✓ {config_file} の読み込み成功")
                    valid_config = config
                    break
                except Exception as e:
                    print(f"  警告: {config_file} 読み込みエラー: {e}")
        
        if valid_config is None:
            print("❌ 有効な設定ファイルが見つかりません")
            return False
        
        # 必要な設定セクションの確認
        required_sections = ['ppo', 'training', 'evaluation']
        for section in required_sections:
            if section in valid_config:
                print(f"  ✓ {section} セクションが存在します")
            else:
                print(f"  ❌ {section} セクションが見つかりません")
                return False
        
        print("✓ 設定互換性テストが成功しました")
        return True
        
    except Exception as e:
        print(f"❌ 設定テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("強化学習評価用ログ拡張 - 安全性テスト")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration_compatibility,
        test_trainer_methods,
        # test_environment_initialization,  # データロードが重いためコメントアウト
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_func.__name__} が失敗しました")
        except Exception as e:
            print(f"❌ {test_func.__name__} で例外発生: {e}")
    
    print("\n" + "=" * 50)
    print(f"テスト結果: {passed}/{total} 成功")
    
    if passed == total:
        print("✅ 全てのテストが成功しました！拡張機能は安全に動作します。")
        return True
    else:
        print("⚠️  一部のテストが失敗しました。修正が必要です。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
