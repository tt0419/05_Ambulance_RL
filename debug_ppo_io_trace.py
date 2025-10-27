"""
debug_ppo_io_trace.py
PPOモデルの入出力を詳細にトレース

目的:
1. 状態ベクトルの内容確認
2. 行動選択プロセスの可視化
3. 選択された救急車の妥当性確認
4. 直近隊との応答時間比較
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# 必要なモジュールのインポート
from dispatch_strategies import PPOStrategy, DispatchContext, EmergencyRequest, AmbulanceInfo
from validation_simulation import ValidationSimulator

class PPOIOTracer:
    """PPOの入出力を詳細にトレースするクラス"""
    
    def __init__(self, model_path: str, config_path: str):
        """
        Args:
            model_path: PPOモデルファイルパス
            config_path: 設定ファイルパス
        """
        print("=" * 80)
        print("PPO入出力トレーサー初期化")
        print("=" * 80)
        
        # PPO戦略の初期化
        ppo_config = {
            'model_path': model_path,
            'config_path': config_path,
            'hybrid_mode': True,
            'severe_conditions': ['重症', '重篤', '死亡'],
            'mild_conditions': ['軽症', '中等症']
        }
        
        print("\nPPO戦略初期化中...")
        self.ppo_strategy = PPOStrategy()
        self.ppo_strategy.initialize(ppo_config)
        
        print("\n✅ PPO戦略初期化完了")
        print(f"   モデル: {model_path}")
        print(f"   行動次元: {self.ppo_strategy.action_dim}")
        print(f"   状態次元: {self.ppo_strategy.state_dim}")
        print(f"   ID対応表: {'ロード済み' if self.ppo_strategy.id_mapping_loaded else 'フォールバック'}")
        
        self.trace_logs = []
    
    def trace_single_dispatch(self, 
                             incident: Dict,
                             available_ambulances: List[Dict],
                             simulator: ValidationSimulator,
                             incident_idx: int) -> Dict:
        """
        1件の配車をトレース
        
        Args:
            incident: 事案情報
            available_ambulances: 利用可能な救急車リスト
            simulator: シミュレータ
            incident_idx: 事案番号
            
        Returns:
            トレース結果
        """
        print("\n" + "=" * 80)
        print(f"事案 #{incident_idx}: {incident['severity']}")
        print("=" * 80)
        
        # ハイブリッドモードチェック
        is_severe = incident['severity'] in self.ppo_strategy.severe_conditions
        
        if is_severe:
            print(f"⚠️  重症系事案 → 直近隊運用（PPOスキップ）")
            return {
                'incident_idx': incident_idx,
                'severity': incident['severity'],
                'method': 'direct_closest',
                'ppo_used': False
            }
        
        print(f"✅ 軽症系事案 → PPO学習対象")
        
        # === Step 1: 状態ベクトルの構築 ===
        print("\n--- Step 1: 状態ベクトル構築 ---")
        
        # EmergencyRequestオブジェクトの作成
        request = EmergencyRequest(
            id=incident['id'],
            h3_index=incident['h3_index'],
            severity=incident['severity'],
            time=0.0,
            priority=self.ppo_strategy.get_severity_priority(incident['severity'])
        )
        
        # AmbulanceInfoオブジェクトのリスト作成
        amb_info_list = []
        for amb_dict in available_ambulances:
            amb_info = AmbulanceInfo(
                id=amb_dict['id'],
                current_h3=amb_dict['current_h3'],
                station_h3=amb_dict['station_h3'],
                status=amb_dict['status']
            )
            amb_info_list.append(amb_info)
        
        # DispatchContextの作成
        context = DispatchContext()
        context.current_time = 0.0
        context.hour_of_day = 12
        context.available_ambulances = len(available_ambulances)
        context.total_ambulances = len(simulator.ambulances)
        context.grid_mapping = self.ppo_strategy.grid_mapping
        
        # 全救急車の状態情報を追加
        context.all_ambulances = {}
        for amb_id, amb_obj in simulator.ambulances.items():
            context.all_ambulances[amb_id] = amb_obj
        
        # 状態辞書を構築
        state_dict = self.ppo_strategy._build_state_dict(
            request, amb_info_list, context
        )
        
        print(f"状態辞書:")
        print(f"  救急車数: {len(state_dict['ambulances'])}台")
        print(f"  事案: {state_dict['pending_call']['severity']} @ {state_dict['pending_call']['h3_index']}")
        print(f"  時刻: ステップ{state_dict['episode_step']}, {state_dict['time_of_day']}時")
        
        # StateEncoderで状態ベクトルに変換
        state_vector = self.ppo_strategy.state_encoder.encode_state(state_dict)
        print(f"\n状態ベクトル:")
        print(f"  形状: {state_vector.shape}")
        print(f"  最小値: {state_vector.min():.3f}")
        print(f"  最大値: {state_vector.max():.3f}")
        print(f"  平均値: {state_vector.mean():.3f}")
        print(f"  非ゼロ要素数: {(state_vector != 0).sum()}/{len(state_vector)}")
        
        # === Step 2: 行動マスクの作成 ===
        print("\n--- Step 2: 行動マスク作成 ---")
        
        action_mask = self.ppo_strategy._create_action_mask(amb_info_list)
        print(f"行動マスク:")
        print(f"  True数: {action_mask.sum()}/{len(action_mask)}")
        print(f"  利用可能な行動率: {action_mask.sum() / len(action_mask) * 100:.1f}%")
        
        if action_mask.sum() == 0:
            print("  ❌ エラー: 利用可能な行動が0個")
            return {
                'incident_idx': incident_idx,
                'severity': incident['severity'],
                'method': 'ppo_failed',
                'ppo_used': True,
                'error': 'no_valid_actions'
            }
        
        # Trueの行動インデックスを表示
        valid_actions = np.where(action_mask)[0]
        print(f"  有効な行動インデックス（最初の10個）: {valid_actions[:10].tolist()}")
        
        # === Step 3: PPOで行動選択 ===
        print("\n--- Step 3: PPO行動選択 ---")
        
        try:
            with torch.no_grad():
                action, log_prob, value = self.ppo_strategy.agent.select_action(
                    state_vector,
                    action_mask,
                    deterministic=True
                )
            
            print(f"PPO出力:")
            print(f"  選択された行動: {action}")
            print(f"  対数確率: {log_prob:.4f}")
            print(f"  価値推定: {value:.4f}")
            print(f"  行動は有効?: {'✅' if action_mask[action] else '❌'}")
            
        except Exception as e:
            print(f"❌ PPO選択エラー: {e}")
            import traceback
            traceback.print_exc()
            return {
                'incident_idx': incident_idx,
                'severity': incident['severity'],
                'method': 'ppo_error',
                'ppo_used': True,
                'error': str(e)
            }
        
        # === Step 4: 行動→救急車マッピング ===
        print("\n--- Step 4: 行動→救急車マッピング ---")
        
        selected_ambulance = self.ppo_strategy._map_action_to_ambulance(
            action, amb_info_list
        )
        
        if selected_ambulance:
            print(f"✅ マッピング成功:")
            print(f"  救急車ID: {selected_ambulance.id}")
            print(f"  位置: {selected_ambulance.current_h3}")
            print(f"  状態: {selected_ambulance.status}")
        else:
            print(f"❌ マッピング失敗: 行動{action}に対応する救急車が見つからない")
            print(f"   フォールバックモード使用の可能性")
            return {
                'incident_idx': incident_idx,
                'severity': incident['severity'],
                'method': 'ppo_mapping_failed',
                'ppo_used': True,
                'action': action,
                'error': 'mapping_failed'
            }
        
        # === Step 5: 応答時間計算 ===
        print("\n--- Step 5: 応答時間計算 ---")
        
        # PPO選択の応答時間
        ppo_response_time = simulator.get_travel_time(
            selected_ambulance.current_h3,
            incident['h3_index'],
            'response'
        ) / 60.0
        
        print(f"PPO選択:")
        print(f"  救急車: {selected_ambulance.id}")
        print(f"  応答時間: {ppo_response_time:.2f}分")
        
        # === Step 6: 直近隊との比較 ===
        print("\n--- Step 6: 直近隊との比較 ---")
        
        # 直近隊を探索
        best_amb_id = None
        best_time = float('inf')
        
        for amb_info in amb_info_list:
            travel_time = simulator.get_travel_time(
                amb_info.current_h3,
                incident['h3_index'],
                'response'
            ) / 60.0
            
            if travel_time < best_time:
                best_time = travel_time
                best_amb_id = amb_info.id
        
        print(f"直近隊:")
        print(f"  救急車: {best_amb_id}")
        print(f"  応答時間: {best_time:.2f}分")
        
        # 比較
        time_diff = ppo_response_time - best_time
        time_diff_pct = (time_diff / best_time * 100) if best_time > 0 else 0
        
        is_optimal = (selected_ambulance.id == best_amb_id)
        
        if is_optimal:
            print(f"\n✅ PPOが最適解を選択！")
        else:
            print(f"\n⚠️  PPOは直近隊ではない救急車を選択")
            print(f"   差分: {time_diff:+.2f}分 ({time_diff_pct:+.1f}%)")
        
        # トレース結果を記録
        trace_result = {
            'incident_idx': incident_idx,
            'severity': incident['severity'],
            'method': 'ppo',
            'ppo_used': True,
            'action': int(action),
            'log_prob': float(log_prob),
            'value': float(value),
            'selected_ambulance_id': selected_ambulance.id,
            'ppo_response_time': ppo_response_time,
            'optimal_ambulance_id': best_amb_id,
            'optimal_response_time': best_time,
            'is_optimal': is_optimal,
            'time_diff': time_diff,
            'available_count': len(amb_info_list),
            'valid_actions_count': int(action_mask.sum())
        }
        
        self.trace_logs.append(trace_result)
        
        return trace_result
    
    def save_trace_logs(self, output_file: str):
        """トレースログをJSON形式で保存"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.trace_logs, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ トレースログを保存: {output_path}")
    
    def print_summary(self):
        """トレース結果のサマリーを表示"""
        print("\n" + "=" * 80)
        print("トレースサマリー")
        print("=" * 80)
        
        if not self.trace_logs:
            print("トレースログがありません")
            return
        
        ppo_logs = [log for log in self.trace_logs if log.get('ppo_used')]
        
        print(f"\nPPO使用事案: {len(ppo_logs)}件")
        
        if not ppo_logs:
            return
        
        # 最適解一致率
        optimal_count = sum(1 for log in ppo_logs if log.get('is_optimal'))
        optimal_rate = optimal_count / len(ppo_logs) * 100
        
        print(f"\n最適解一致率: {optimal_rate:.1f}% ({optimal_count}/{len(ppo_logs)}件)")
        
        # 応答時間統計
        ppo_times = [log['ppo_response_time'] for log in ppo_logs if 'ppo_response_time' in log]
        optimal_times = [log['optimal_response_time'] for log in ppo_logs if 'optimal_response_time' in log]
        
        if ppo_times and optimal_times:
            print(f"\n応答時間:")
            print(f"  PPO平均: {np.mean(ppo_times):.2f}分")
            print(f"  直近隊平均: {np.mean(optimal_times):.2f}分")
            print(f"  差分: {np.mean(ppo_times) - np.mean(optimal_times):+.2f}分")
        
        # 最悪ケース
        if len(ppo_logs) > 0:
            worst_cases = sorted(ppo_logs, key=lambda x: x.get('time_diff', 0), reverse=True)[:5]
            
            print(f"\n⚠️  最悪ケース（上位5件）:")
            for i, case in enumerate(worst_cases, 1):
                print(f"  {i}. 事案#{case['incident_idx']}: "
                      f"{case['severity']}, "
                      f"差分={case.get('time_diff', 0):+.2f}分 "
                      f"(PPO: {case.get('selected_ambulance_id')}, "
                      f"最適: {case.get('optimal_ambulance_id')})")

def main():
    """メイン実行"""
    print("=" * 80)
    print("PPO入出力詳細トレースツール")
    print("=" * 80)
    print()
    
    # PPOモデルのパス
    model_path = 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/final_model.pth'
    config_path = 'reinforcement_learning/experiments/ppo_training/ppo_20251017_113908/configs/config.json'
    
    # トレーサー初期化
    tracer = PPOIOTracer(model_path, config_path)
    
    # ValidationSimulator初期化
    print("\n" + "=" * 80)
    print("ValidationSimulator初期化")
    print("=" * 80)
    
    simulator = ValidationSimulator(
        target_date_str="20230615",
        simulation_duration_hours=24,
        random_seed=42,
        verbose_logging=False
    )
    
    print(f"✅ シミュレータ初期化完了")
    print(f"   救急車数: {len(simulator.ambulances)}台")
    print(f"   事案数: {len(simulator.sorted_calls)}件")
    
    # 最初の10件の軽症系事案をトレース
    print("\n" + "=" * 80)
    print("軽症系事案のトレース（最大10件）")
    print("=" * 80)
    
    mild_conditions = ['軽症', '中等症']
    traced_count = 0
    max_trace = 10
    
    for idx, call in enumerate(simulator.sorted_calls):
        if traced_count >= max_trace:
            break
        
        if call.severity not in mild_conditions:
            continue
        
        # 利用可能な救急車を取得
        available_ambulances = []
        for amb_id, amb in simulator.ambulances.items():
            if amb.status.value == 'available':
                available_ambulances.append({
                    'id': amb_id,
                    'current_h3': amb.current_h3_index,
                    'station_h3': amb.station_h3_index,
                    'status': 'available'
                })
        
        if not available_ambulances:
            print(f"\n事案#{idx}: 利用可能な救急車なし（スキップ）")
            continue
        
        # 事案情報
        incident = {
            'id': str(call.id),
            'h3_index': call.h3_index,
            'severity': call.severity
        }
        
        # トレース実行
        result = tracer.trace_single_dispatch(
            incident, available_ambulances, simulator, idx
        )
        
        if result.get('ppo_used'):
            traced_count += 1
    
    # サマリー表示
    tracer.print_summary()
    
    # ログ保存
    tracer.save_trace_logs('debug_output/ppo_io_trace.json')
    
    print("\n" + "=" * 80)
    print("診断完了")
    print("=" * 80)
    print("\n次のステップ:")
    print("  1. トレースログを確認: debug_output/ppo_io_trace.json")
    print("  2. 最適解一致率が低い場合、ID対応表またはモデルに問題")
    print("  3. マッピング失敗が多い場合、ID対応表の修正が必要")
    print("=" * 80)

if __name__ == "__main__":
    main()

