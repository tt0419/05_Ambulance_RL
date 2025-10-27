"""
debug_id_mapping.py
ID対応表とValidationSimulatorの救急車IDの整合性を検証

目的:
1. id_mapping_proposal.jsonの存在と内容確認
2. ValidationSimulatorの実際の救急車ID一覧取得
3. マッピング率の計算
4. 未マッピングのIDリスト表示
"""

import json
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict

# 必要なモジュールのインポート
from validation_simulation import ValidationSimulator

def check_id_mapping_file():
    """ID対応表ファイルの存在と内容を確認"""
    print("=" * 80)
    print("Phase 1: ID対応表ファイルの確認")
    print("=" * 80)
    
    mapping_file = Path("id_mapping_proposal.json")
    
    if not mapping_file.exists():
        print("❌ エラー: id_mapping_proposal.json が見つかりません")
        print("   このファイルはPPO戦略が救急車IDをマッピングするために必要です")
        print("   phase1_id_validation.py を実行してファイルを生成してください")
        return None, None
    
    print("✅ id_mapping_proposal.json が見つかりました")
    
    # ファイルの読み込み
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        string_to_int = mapping_data.get('string_to_int', {})
        int_to_string_raw = mapping_data.get('int_to_string', {})
        int_to_string = {int(k): v for k, v in int_to_string_raw.items()}
        
        print(f"\n📊 マッピングデータ:")
        print(f"   string_to_int: {len(string_to_int)}件")
        print(f"   int_to_string: {len(int_to_string)}件")
        
        # サンプル表示
        print(f"\n📋 マッピングサンプル（最初の10件）:")
        for i, (val_id, action) in enumerate(list(string_to_int.items())[:10]):
            print(f"   {i+1}. '{val_id}' → アクション{action}")
        
        return string_to_int, int_to_string
        
    except Exception as e:
        print(f"❌ エラー: ファイルの読み込みに失敗: {e}")
        return None, None

def get_validation_simulator_ids():
    """ValidationSimulatorから実際の救急車IDを取得（救急車データファイルから直接読み込み）"""
    print("\n" + "=" * 80)
    print("Phase 2: 救急車ID取得（ValidationSimulatorと同じ方法）")
    print("=" * 80)
    
    try:
        # ValidationSimulatorと同じ方法で救急車データを読み込み
        print("救急車データ読み込み中...")
        
        firestation_path = "data/tokyo/import/amb_place_master.csv"
        ambulance_data = pd.read_csv(firestation_path, encoding='utf-8')
        
        # special_flag == 1 でフィルタリング
        ambulance_data = ambulance_data[ambulance_data['special_flag'] == 1]
        print(f"  special_flag=1フィルタ後: {len(ambulance_data)}件")
        
        # 「救急隊なし」を除外（ValidationSimulatorと同じ）
        before_no_team = len(ambulance_data)
        ambulance_data = ambulance_data[ambulance_data['team_name'] != '救急隊なし']
        excluded_no_team = before_no_team - len(ambulance_data)
        print(f"  「救急隊なし」除外: {before_no_team}件 → {len(ambulance_data)}件 (除外: {excluded_no_team}件)")
        
        # 「デイタイム救急」を除外（ValidationSimulatorのデフォルト動作）
        before_daytime = len(ambulance_data)
        ambulance_data = ambulance_data[~ambulance_data['team_name'].str.contains('デイタイム', na=False)]
        excluded_daytime = before_daytime - len(ambulance_data)
        print(f"  「デイタイム救急」除外: {before_daytime}件 → {len(ambulance_data)}件 (除外: {excluded_daytime}件)")
        
        print(f"✅ 最終救急車数: {len(ambulance_data)}台")
        
        # 救急車ID一覧を作成（ValidationSimulatorと同じ命名規則）
        ambulance_ids = []
        ambulance_details = []
        
        for idx, row in ambulance_data.iterrows():
            # ValidationSimulatorと同じロジック
            team_name = row.get('team_name', f'Station_{idx}')
            if not team_name:
                team_name = f'Station_{idx}'
            
            # amb カラムから救急車数を取得
            num_ambulances = 0
            if 'amb' in row and pd.notna(row['amb']):
                try:
                    amb_value = int(float(str(row['amb'])))
                    if amb_value > 0:
                        num_ambulances = 1  # ValidationSimulatorと同じロジック（1台に統一）
                except ValueError:
                    pass
            
            # ValidationSimulatorのAmbulanceクラスと同じID生成ロジック
            # 各署に1台ずつ（num_ambulances_to_create = 1）
            if num_ambulances > 0:
                for i in range(num_ambulances):
                    amb_id = f"{team_name}_{i}"
                    
                    ambulance_ids.append(amb_id)
                    ambulance_details.append({
                        'id': amb_id,
                        'name': team_name,
                        'station_name': row.get('name', 'unknown'),
                        'section': row.get('section', 0)
                    })
        
        # サンプル表示
        print(f"\n📋 救急車IDサンプル（最初の10件）:")
        for i, detail in enumerate(ambulance_details[:10]):
            print(f"   {i+1}. ID='{detail['id']}', 名前={detail['name']}, "
                  f"署={detail['station_name']}, 方面={detail['section']}")
        
        return ambulance_ids, ambulance_details
        
    except Exception as e:
        print(f"❌ エラー: 救急車データ読み込み失敗: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def analyze_mapping_coverage(string_to_int, int_to_string, validator_ids):
    """マッピングのカバレッジを分析"""
    print("\n" + "=" * 80)
    print("Phase 3: マッピングカバレッジ分析")
    print("=" * 80)
    
    if not string_to_int or not validator_ids:
        print("❌ データ不足のため分析できません")
        return
    
    # ValidationSimulatorのIDセット
    validator_id_set = set(validator_ids)
    
    # マッピングファイルのIDセット
    mapping_id_set = set(string_to_int.keys())
    
    # 一致・不一致分析
    matched_ids = validator_id_set & mapping_id_set
    missing_in_mapping = validator_id_set - mapping_id_set
    extra_in_mapping = mapping_id_set - validator_id_set
    
    # 統計
    total_validator = len(validator_id_set)
    total_matched = len(matched_ids)
    coverage_rate = (total_matched / total_validator * 100) if total_validator > 0 else 0
    
    print(f"\n📊 マッピング統計:")
    print(f"   ValidationSimulator救急車数: {total_validator}台")
    print(f"   マッピング済み: {total_matched}台")
    print(f"   マッピング率: {coverage_rate:.1f}%")
    print(f"   未マッピング: {len(missing_in_mapping)}台")
    print(f"   余剰マッピング: {len(extra_in_mapping)}台")
    
    # 問題判定
    print(f"\n🔍 診断結果:")
    if coverage_rate == 100.0:
        print("   ✅ 完璧: すべての救急車がマッピングされています")
    elif coverage_rate >= 95.0:
        print("   ⚠️  警告: わずかに未マッピングの救急車があります")
    elif coverage_rate >= 80.0:
        print("   ⚠️  注意: 一部の救急車がマッピングされていません")
    else:
        print("   ❌ 重大: 多数の救急車が未マッピングです")
        print("   この状態ではPPO戦略の精度が大幅に低下します")
    
    # 未マッピングIDの詳細表示
    if missing_in_mapping:
        print(f"\n⚠️  未マッピングの救急車ID（最大20件表示）:")
        for i, missing_id in enumerate(sorted(missing_in_mapping)[:20]):
            print(f"   {i+1}. '{missing_id}'")
        
        if len(missing_in_mapping) > 20:
            print(f"   ... 他 {len(missing_in_mapping) - 20}件")
    
    # 余剰マッピングIDの詳細表示
    if extra_in_mapping:
        print(f"\n💡 余剰マッピング（ValidationSimulatorにない）:")
        for i, extra_id in enumerate(sorted(extra_in_mapping)[:20]):
            action = string_to_int.get(extra_id)
            print(f"   {i+1}. '{extra_id}' → アクション{action}")
        
        if len(extra_in_mapping) > 20:
            print(f"   ... 他 {len(extra_in_mapping) - 20}件")
    
    return {
        'coverage_rate': coverage_rate,
        'matched_count': total_matched,
        'missing_count': len(missing_in_mapping),
        'extra_count': len(extra_in_mapping),
        'missing_ids': list(missing_in_mapping),
        'extra_ids': list(extra_in_mapping)
    }

def check_action_dimension_consistency(int_to_string, validator_ids):
    """行動次元の一貫性確認"""
    print("\n" + "=" * 80)
    print("Phase 4: 行動次元の一貫性確認")
    print("=" * 80)
    
    if not int_to_string or not validator_ids:
        print("❌ データ不足のため確認できません")
        return
    
    # 行動インデックスの範囲
    max_action = max(int_to_string.keys()) if int_to_string else 0
    min_action = min(int_to_string.keys()) if int_to_string else 0
    
    print(f"📊 行動空間:")
    print(f"   最小アクション: {min_action}")
    print(f"   最大アクション: {max_action}")
    print(f"   行動次元: {max_action + 1}")
    print(f"   ValidationSimulator救急車数: {len(validator_ids)}")
    
    # PPOモデルの期待次元（192台想定）
    expected_dim = 192
    print(f"   PPOモデル期待次元: {expected_dim}")
    
    # 一貫性チェック
    actual_dim = max_action + 1
    if actual_dim == len(validator_ids):
        print(f"   ✅ 一致: 行動次元とValidationSimulator救急車数が一致")
    else:
        print(f"   ⚠️  不一致: 行動次元({actual_dim}) ≠ 救急車数({len(validator_ids)})")
    
    if actual_dim == expected_dim:
        print(f"   ✅ 一致: PPOモデル期待次元と一致")
    else:
        print(f"   ⚠️  不一致: 行動次元({actual_dim}) ≠ PPOモデル期待({expected_dim})")
        print(f"   この不一致が問題を引き起こしている可能性があります")
    
    # 連続性チェック（欠番がないか）
    all_actions = set(int_to_string.keys())
    expected_actions = set(range(0, max_action + 1))
    missing_actions = expected_actions - all_actions
    
    if missing_actions:
        print(f"\n   ⚠️  欠番のあるアクション: {len(missing_actions)}個")
        if len(missing_actions) <= 10:
            print(f"   欠番リスト: {sorted(missing_actions)}")
        else:
            print(f"   欠番サンプル: {sorted(missing_actions)[:10]} ...")
    else:
        print(f"   ✅ 連続: アクション番号に欠番なし")

def generate_fix_suggestions(analysis_result):
    """修正提案を生成"""
    print("\n" + "=" * 80)
    print("Phase 5: 修正提案")
    print("=" * 80)
    
    if not analysis_result:
        print("分析結果がないため提案できません")
        return
    
    coverage_rate = analysis_result['coverage_rate']
    missing_count = analysis_result['missing_count']
    
    print("🔧 推奨アクション:")
    
    if coverage_rate == 100.0:
        print("\n✅ 問題なし")
        print("   ID対応表は完璧です。他の問題を調査してください。")
    
    elif coverage_rate >= 95.0:
        print("\n⚠️  軽微な問題")
        print("   1. phase1_id_validation.py を再実行してマッピングを更新")
        print("   2. または、未マッピングIDを手動で追加")
    
    elif coverage_rate >= 80.0:
        print("\n⚠️  中程度の問題")
        print("   1. phase1_id_validation.py を再実行（推奨）")
        print("   2. ValidationSimulatorとEMSEnvironmentで同じフィルタリングを使用")
        print("      - 「救急隊なし」の除外")
        print("      - 「デイタイム救急」の除外")
        print("      - エリア制限の設定")
    
    else:
        print("\n❌ 重大な問題")
        print("   ID対応表が大幅に不足しています。以下を実行してください：")
        print("\n   1. phase1_id_validation.py を実行:")
        print("      python phase1_id_validation.py")
        print("\n   2. EMSEnvironmentとValidationSimulatorの救急車フィルタリングを統一:")
        print("      - ems_environment.py: _load_base_data()の処理")
        print("      - validation_simulation.py: 救急車読み込み処理")
        print("      両方で同じ除外条件を適用")
        print("\n   3. 設定ファイル（config.yaml）のエリア制限を確認:")
        print("      data:")
        print("        area_restriction:")
        print("          enabled: true/false")
        print("          section_code: 1-10 または null")
    
    # フォールバックモードの影響推定
    if coverage_rate < 100.0:
        print(f"\n📉 性能への影響推定:")
        print(f"   未マッピング率: {100 - coverage_rate:.1f}%")
        print(f"   影響を受ける事案: 約{100 - coverage_rate:.1f}%の事案でフォールバックモードが使用")
        print(f"   フォールバックモードでは行動選択の精度が低下します")
        print(f"   これがPPO戦略の性能が「直近隊」より悪い原因の可能性が高いです")

def main():
    """メイン実行"""
    print("=" * 80)
    print("PPO戦略 ID対応表デバッグツール")
    print("=" * 80)
    print()
    
    # Phase 1: ID対応表ファイル確認
    string_to_int, int_to_string = check_id_mapping_file()
    
    if string_to_int is None:
        print("\n⚠️  ID対応表ファイルが見つからないため、これ以上の診断はできません")
        print("まず phase1_id_validation.py を実行してください")
        return
    
    # Phase 2: ValidationSimulatorのID取得
    validator_ids, validator_details = get_validation_simulator_ids()
    
    if not validator_ids:
        print("\n⚠️  ValidationSimulatorの初期化に失敗したため、これ以上の診断はできません")
        return
    
    # Phase 3: マッピングカバレッジ分析
    analysis_result = analyze_mapping_coverage(string_to_int, int_to_string, validator_ids)
    
    # Phase 4: 行動次元の一貫性確認
    check_action_dimension_consistency(int_to_string, validator_ids)
    
    # Phase 5: 修正提案
    generate_fix_suggestions(analysis_result)
    
    print("\n" + "=" * 80)
    print("診断完了")
    print("=" * 80)
    print("\n次のステップ:")
    print("  1. 上記の修正提案に従って問題を解決")
    print("  2. debug_single_episode_full.py を実行して動作確認")
    print("=" * 80)

if __name__ == "__main__":
    main()

