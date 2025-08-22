#!/usr/bin/env python3
"""
train_area3_ppo.py
第3方面限定でPPO学習を実行するスクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """第3方面限定PPO学習のメイン関数"""
    print("=" * 80)
    print("第3方面限定PPO学習開始")
    print("=" * 80)
    
    # 設定ファイルの確認
    config_path = "reinforcement_learning/experiments/config_area3.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return
    
    print(f"📋 設定ファイル: {config_path}")
    print("📍 対象エリア: 第3方面（目黒区、渋谷区、世田谷区）")
    print("🚑 対象救急車: section=3の救急隊のみ")
    print("🏥 対象病院: 全ての病院（搬送選択肢は変更なし）")
    
    try:
        # train_ppoをインポートして実行
        from train_ppo import main as train_main
        
        # sys.argvを設定して実行
        original_argv = sys.argv
        sys.argv = ['train_ppo.py', '--config', config_path]
        
        print("\n🚀 PPO学習を開始します...")
        train_main()
        
    except ImportError as e:
        print(f"❌ train_ppo.pyのインポートに失敗: {e}")
        print("\n代替案: 以下のコマンドを手動で実行してください:")
        print(f"python train_ppo.py --config {config_path}")
        
    except Exception as e:
        print(f"❌ 学習中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # sys.argvを復元
        if 'original_argv' in locals():
            sys.argv = original_argv

if __name__ == "__main__":
    # 事前チェック
    print("事前チェックを実行しています...")
    
    try:
        from test_area3_environment import test_area3_environment
        if test_area3_environment():
            print("✅ 事前チェック完了\n")
            main()
        else:
            print("❌ 事前チェックで問題が発生しました。")
    except Exception as e:
        print(f"⚠️ 事前チェックでエラー: {e}")
        print("学習を続行しますが、問題が発生する可能性があります。\n")
        main()
