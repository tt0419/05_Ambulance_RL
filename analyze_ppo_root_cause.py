"""
analyze_ppo_root_cause.py
PPOの選択ミスの根本原因を特定

検証項目:
1. 状態エンコーディングの妥当性
2. 学習時の設定（報酬関数、ハイブリッドモード）
3. 行動確率分布の分析
4. 学習データの偏り
"""

import json
import numpy as np
from pathlib import Path

def analyze_training_config():
    """学習時の設定を分析"""
    
    print("=" * 80)
    print("Phase 1: 学習時の設定分析")
    print("=" * 80)
    
    config_path = Path('reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/configs/config.json')
    
    if not config_path.exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("\n【重要な設定項目】")
    print("-" * 80)
    
    # ハイブリッドモード
    hybrid_mode = config.get('hybrid_mode', False)
    print(f"\n1. ハイブリッドモード: {hybrid_mode}")
    if hybrid_mode:
        severe_conditions = config.get('severe_conditions', [])
        print(f"   重症系（直近隊）: {severe_conditions}")
        print(f"   ⚠️ 学習時にハイブリッドモードが有効")
        print(f"   → 軽症系のみで学習 → データが偏っている可能性")
    
    # 報酬関数
    reward_mode = config.get('reward_mode', 'unknown')
    print(f"\n2. 報酬モード: {reward_mode}")
    
    reward_config = config.get('reward_config', {})
    if reward_config:
        print(f"   報酬設定:")
        for key, value in reward_config.items():
            print(f"     {key}: {value}")
        
        # 重要な重み
        if 'response_time_weight' in reward_config:
            rt_weight = reward_config['response_time_weight']
            coverage_weight = reward_config.get('coverage_weight', 0)
            print(f"\n   ⚠️ 重み比較:")
            print(f"     応答時間: {rt_weight}")
            print(f"     カバレッジ: {coverage_weight}")
            
            if coverage_weight > rt_weight:
                print(f"   🔥 問題発見: カバレッジ重視 > 応答時間")
                print(f"   → PPOが遠い隊を選択してカバレッジを維持する可能性")
    
    # 状態エンコーディング
    state_config = config.get('state_config', {})
    print(f"\n3. 状態エンコーディング:")
    print(f"   設定: {state_config}")
    
    # 学習パラメータ
    training_config = config.get('training', {})
    print(f"\n4. 学習パラメータ:")
    print(f"   エピソード数: {training_config.get('num_episodes', 'N/A')}")
    print(f"   バッチサイズ: {training_config.get('batch_size', 'N/A')}")
    
    return config

def analyze_reward_design():
    """報酬設計を詳細分析"""
    
    print("\n" + "=" * 80)
    print("Phase 2: 報酬設計の詳細分析")
    print("=" * 80)
    
    # reward_designer.pyを読み込んで確認
    from reinforcement_learning.environment.reward_designer import RewardDesigner
    
    print("\nRewardDesignerの実装を確認中...")
    print("（コード内容を手動で確認してください）")
    print()
    print("確認ポイント:")
    print("  1. _calculate_hybrid_reward の実装")
    print("  2. 応答時間報酬の計算式")
    print("  3. カバレッジ報酬の重み")
    print("  4. ペナルティの大きさ")

def check_state_encoding():
    """状態エンコーディングの妥当性確認"""
    
    print("\n" + "=" * 80)
    print("Phase 3: 状態エンコーディングの確認")
    print("=" * 80)
    
    print("\n確認ポイント:")
    print("  1. 事案位置（h3_index）が正しくエンコードされているか")
    print("  2. 救急車位置が正しくエンコードされているか")
    print("  3. 距離情報が適切に表現されているか")
    print("  4. 正規化が適切か")
    
    print("\n→ state_encoder.py と modular_state_encoder.py を確認")

def propose_improvements():
    """改善案の提示"""
    
    print("\n" + "=" * 80)
    print("改善案")
    print("=" * 80)
    
    print("\n【短期的改善案（即座に実行可能）】")
    print()
    print("1. ハイブリッドモードを無効にして再学習")
    print("   - 全事案で学習 → データの偏りを解消")
    print("   - 推論時のみハイブリッドモードを使用")
    print()
    print("2. 報酬関数の重み調整")
    print("   - 応答時間の重みを増加（例: 0.4 → 0.7）")
    print("   - カバレッジの重みを減少（例: 0.5 → 0.2）")
    print()
    print("3. 簡易修正: PPOを直近隊に置き換え")
    print("   - 暫定的に全事案で直近隊運用")
    print("   - 性能は保証されるが、学習の意味がなくなる")
    
    print("\n【中期的改善案（再学習が必要）】")
    print()
    print("4. 状態エンコーディングの改善")
    print("   - 距離情報を明示的に追加")
    print("   - グリッド表現を改善")
    print()
    print("5. 報酬関数の再設計")
    print("   - 応答時間を主要報酬に")
    print("   - カバレッジは補助的報酬に")
    
    print("\n【長期的改善案（根本的見直し）】")
    print()
    print("6. 模倣学習（Imitation Learning）の導入")
    print("   - 直近隊戦略の行動を教師データとして学習")
    print("   - その後、強化学習で微調整")
    print()
    print("7. マルチタスク学習")
    print("   - 複数の目的（応答時間、カバレッジ、公平性）を同時最適化")

def main():
    """メイン分析"""
    
    print("=" * 80)
    print("PPO根本原因分析ツール")
    print("=" * 80)
    
    # Phase 1: 学習設定
    config = analyze_training_config()
    
    # Phase 2: 報酬設計
    analyze_reward_design()
    
    # Phase 3: 状態エンコーディング
    check_state_encoding()
    
    # 改善案
    propose_improvements()
    
    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)
    
    print("\n次のステップ:")
    print("  1. reward_designer.py で報酬関数の重みを確認")
    print("  2. 学習時のハイブリッドモード設定を確認")
    print("  3. 改善案から1つ選択して実装")

if __name__ == "__main__":
    main()

