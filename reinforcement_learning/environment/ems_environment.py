"""
ems_environment.py
ValidationSimulatorと同等の詳細度を持つ、統一EMS強化学習環境。

この環境は、高速な学習と現実的なシミュレーションの両立を目指し、
以下の高精度なコンポーネントを統合しています。
- フェーズ別に最適化された移動時間行列
- 実績データに基づく階層的なサービス時間生成器
- 傷病度に応じた確率的・決定論的病院選択モデル
- 現実的な初期稼働状態を再現するリセット機能
- 学習目標を明確にする詳細な報酬関数
"""

import numpy as np
import pandas as pd
import json
import pickle
import h3
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# --- 外部コンポーネントのインポート ---
# プロジェクトルートからの相対パスでコンポーネントをインポート
# このファイルの場所に応じてパス調整が必要な場合があります
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# 1. 高機能サービス時間生成器
try:
    # service_time_generator_enhanced.py が data/tokyo/service_time_analysis/ にあることを想定
    from data.tokyo.service_time_analysis.service_time_generator_enhanced import ServiceTimeGeneratorEnhanced
    USE_ENHANCED_GENERATOR = True
except ImportError as e:
    print(f"警告: ServiceTimeGeneratorEnhancedのインポートに失敗しました: {e}")
    # フォールバックとしてvalidation_simulation内の旧版を試みる
    try:
        from validation_simulation import ServiceTimeGenerator
    except ImportError as e_vs:
        print(f"致命的エラー: ServiceTimeGeneratorも見つかりません: {e_vs}")
        sys.exit(1)
    USE_ENHANCED_GENERATOR = False

# 2. プロジェクト内モジュール
from data_cache import get_emergency_data_cache
from reinforcement_learning.config_utils import load_config_with_inheritance
from reinforcement_learning.environment.reward_designer import RewardDesigner
from reinforcement_learning.environment.dispatch_logger import DispatchLogger
from constants import is_severe_condition


@dataclass
class StepResult:
    """ステップ実行結果を格納するデータクラス"""
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]

class EMSEnvironment:
    """ValidationSimulatorと同等の詳細度を持つ統一EMS環境"""
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "train"):
        """
        環境の初期化を行います。

        Args:
            config_path (str): 設定ファイルのパス。
            mode (str): "train" または "eval"。
        """
        self.config = load_config_with_inheritance(config_path)
        self.mode = mode
        self.base_dir = Path("data/tokyo")
        
        self._initialize_core_components()
        self._load_base_data()
        
        # 状態・行動空間の次元を定義
        self.action_dim = len(self.ambulance_data)
        from .state_encoder import StateEncoder
        self.state_encoder = StateEncoder(
            config=self.config,
            max_ambulances=self.action_dim,
            travel_time_matrix=self.travel_time_matrices.get('response'),
            grid_mapping=self.grid_mapping
        )
        self.state_dim = self.state_encoder.state_dim
        print(f"✓ 状態空間: {self.state_dim}次元, 行動空間: {self.action_dim}次元")

        # 報酬設計とログ機能を初期化
        self.reward_designer = RewardDesigner(self.config)
        self.dispatch_logger = DispatchLogger(enabled=True)

        # ハイブリッドモードの設定
        self.hybrid_mode = self.config.get('hybrid_mode', {}).get('enabled', False)
        if self.hybrid_mode:
            self.severe_conditions = self.config.get('hybrid_mode', {}).get('severity_classification', {}).get('severe_conditions', ['重症', '重篤', '死亡'])
            print("✓ ハイブリッドモード有効")

    def _initialize_core_components(self):
        """シミュレーションに必要なコアコンポーネントを読み込みます。"""
        print("=" * 60)
        print("統合EMS環境の初期化を開始します")
        print("=" * 60)
        
        # データキャッシュの準備
        self.data_cache = get_emergency_data_cache()
        if not self.data_cache.get_cache_info()["cached"]:
            self.data_cache.load_data()
        
        # H3グリッドマッピングの読み込み
        with open(self.base_dir / "processed/grid_mapping_res9.json", 'r', encoding='utf-8') as f:
            self.grid_mapping = json.load(f)
        print(f"✓ H3グリッド読み込み完了: {len(self.grid_mapping)}グリッド")

        # 各種モデルの読み込み
        self._load_travel_time_matrices()
        self._load_hospital_selection_model()
        self._load_service_time_generator()
    
    def _load_travel_time_matrices(self):
        """ValidationSimulatorと互換性のある、最適な移動時間行列を読み込みます。"""
        print("\n[1/4] 移動時間行列の読み込み中...")
        self.travel_time_matrices = {}
        calibration_dir = self.base_dir / "calibration2"
        stats_path = calibration_dir / 'travel_time_statistics_all_phases.json'
        if not stats_path.exists():
            raise FileNotFoundError(f"移動時間統計ファイルが見つかりません: {stats_path}")

        with open(stats_path, 'r', encoding='utf-8') as f:
            phase_stats_data = json.load(f)
        
        for phase in ['response', 'transport', 'return']:
            matrix_filename, model_type = self._get_best_travel_time_model(phase_stats_data, phase)
            if matrix_filename:
                matrix_path = calibration_dir / matrix_filename
                if matrix_path.exists():
                    self.travel_time_matrices[phase] = np.load(matrix_path)
                    print(f"  ✓ {phase.capitalize()} Matrix: '{model_type}'モデルをロード")
        
        if len(self.travel_time_matrices) < 3:
             raise RuntimeError("必須の移動時間行列が全て読み込めませんでした。")
        print("✓ 移動時間行列の読み込み完了")

    def _get_best_travel_time_model(self, stats_data, phase):
        """JSON統計ファイルから最も精度の高いモデルのファイル名を取得します。"""
        if phase in stats_data and 'calibrated' in stats_data[phase]:
            model_type = stats_data[phase]['calibrated'].get('model_type')
            if model_type in ['linear', 'log', 'uncalibrated']:
                filename = f"{model_type}_calibrated_{phase}.npy" if model_type != 'uncalibrated' else f"uncalibrated_travel_time_{phase}.npy"
                return filename, model_type
        return None, None

    def _load_hospital_selection_model(self):
        """実績データに基づく確率的病院選択モデルを読み込みます。"""
        print("\n[2/4] 病院選択モデルの読み込み中...")
        self.hospital_data = pd.read_csv(self.base_dir / "import/hospital_master.csv", encoding='utf-8')
        # ValidationSimulatorと同じ列名変更処理
        self.hospital_data = self.hospital_data.rename(columns={'hospital_latitude': 'latitude', 'hospital_longitude': 'longitude'})
        model_path = self.base_dir / "processed/hospital_selection_model_revised.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f: model_data = pickle.load(f)
            self.hospital_selection_model = model_data.get('selection_probabilities')
            self.static_fallback_model = model_data.get('static_fallback_model')
            print("✓ 確率的病院選択モデル読み込み完了")
        else:
            self.hospital_selection_model, self.static_fallback_model = None, None
            print("⚠ 確率モデルなし。決定論的選択にフォールバックします。")
    
    def _load_service_time_generator(self):
        """階層的パラメータを持つ高精度なサービス時間生成器を初期化します。"""
        print("\n[3/4] サービス時間生成器の初期化中...")
        hierarchical_path = self.base_dir / "service_time_analysis/lognormal_parameters_hierarchical.json"
        if USE_ENHANCED_GENERATOR and hierarchical_path.exists():
            self.service_time_generator = ServiceTimeGeneratorEnhanced(str(hierarchical_path))
            print("✓ 階層的サービス時間生成器を初期化")
        else:
            self.service_time_generator = None
            print("⚠ サービス時間パラメータなし。簡易計算を使用します。")
    
    def _load_base_data(self):
        """ValidationSimulatorと同一のフィルタリングを適用し、救急車データを準備します。"""
        print("\n[4/4] 救急車データの読み込みとフィルタリング中...")
        amb_data = pd.read_csv(self.base_dir / "import/amb_place_master.csv", encoding='utf-8')
        
        # フィルタリングチェーン
        amb_data = amb_data[amb_data['special_flag'] == 1].copy()
        if 'team_name' in amb_data.columns: amb_data = amb_data[amb_data['team_name'] != '救急隊なし']
        if self.config.get('data', {}).get('exclude_daytime_ambulances', True):
            amb_data = amb_data[~amb_data['team_name'].str.contains('デイタイム', na=False)]
        
        self.ambulance_data = amb_data.reset_index(drop=True)
        print(f"✓ 救急車データ準備完了: {len(self.ambulance_data)}台")

    def step(self, action: int) -> StepResult:
        """
        環境を1ステップ進めます。エージェントの行動に基づき、状態遷移と報酬計算を行います。

        Args:
            action (int): エージェントが選択した行動（救急車ID）。

        Returns:
            StepResult: (次の観測, 報酬, 終了フラグ, 追加情報) のタプル。
        """
        reward, info, dispatch_result = 0.0, {}, None
        current_incident = self.pending_call

        if current_incident:
            # ハイブリッドモードの場合、重症案件は最適行動（最近接）を強制
            if self.hybrid_mode and is_severe_condition(current_incident['severity']):
                action_to_take = self.get_optimal_action()
                info['dispatch_type'] = 'direct_closest'
            else:
                action_to_take = action
                info['dispatch_type'] = 'ppo_learning'
            
            # 選択された救急車が利用可能か確認し、配車
            ambulance_id = self._mask_and_select_ambulance(action_to_take)
            
            if ambulance_id is not None and self.ambulance_states[ambulance_id]['status'] == 'available':
                # 詳細な活動完了時間と各フェーズの所要時間を計算
                completion_time_sec, details = self._calculate_ambulance_completion_time(ambulance_id, current_incident)
                
                # 報酬計算（学習対象のアクションのみ）
                if info['dispatch_type'] == 'ppo_learning':
                    reward = self._calculate_reward_detailed(details['response_time'], current_incident['severity'])
                
                # 救急車を「出動中」状態に更新
                self.ambulance_states[ambulance_id]['status'] = 'dispatched'
                self.ambulance_states[ambulance_id]['completion_time'] = self.episode_step_seconds + completion_time_sec
                self.ambulance_states[ambulance_id]['calls_today'] += 1
                
                # 統計とログを更新（傷病度情報を追加）
                details['severity'] = current_incident['severity']
                self._update_statistics(details)
                info.update(details)
            else:
                # 配車失敗（利用可能な救急車がいない、または無効なアクション）
                reward = -100.0 # 大きなペナルティ
                self.unhandled_calls.append(current_incident)
                self._handle_unresponsive_call()
        
        # 次の事案に進む
        self._advance_to_next_call()
        
        observation = self._get_observation()
        done = self._is_episode_done()
        info['episode_stats'] = self.get_episode_statistics()
        
        return StepResult(observation, reward, done, info)

    def reset(self) -> np.ndarray:
        """
        エピソードをリセットし、初期状態に戻します。

        Returns:
            np.ndarray: 初期観測ベクトル。
        """
        # 学習/評価期間からランダムに一つの期間を選択
        periods = self.config['data']['train_periods'] if self.mode == "train" else self.config['data']['eval_periods']
        period = periods[np.random.randint(len(periods))]
        calls_df = self.data_cache.get_period_data(period['start_date'], period['end_date'])
        
        # 選択した期間から、エピソード長に合わせた事案データを準備
        self.current_episode_calls = self._prepare_episode_calls(calls_df, self.config['data']['episode_duration_hours'])
        
        # 時間、統計、事案キューをリセット
        self.episode_step_seconds = 0
        self.step_in_episode = 0
        self.pending_call, self.unhandled_calls = None, []
        self._reset_statistics()
        
        # 全車出動中の状況記録用
        self.all_busy_events = []  # (time_seconds, duration_seconds, hour_of_day)
        
        # 全車出動中のトラッキング変数をクリア
        if hasattr(self, '_all_busy_start_time'):
            delattr(self, '_all_busy_start_time')
        if hasattr(self, '_mask_debug_count'):
            delattr(self, '_mask_debug_count')
        
        # 救急車を現実的な初期稼働状態で配置
        self._initialize_ambulances_realistic()
        
        # 最初の事案を設定
        if self.current_episode_calls:
            self.pending_call = self.current_episode_calls[0]
            self.episode_step_seconds = (self.pending_call['datetime'] - self.episode_start_time).total_seconds()
        
        return self._get_observation()
    
    def set_mode(self, mode: str):
        """
        環境のモードを切り替える（トレーニング/評価）
        
        Args:
            mode: "train" または "eval"
            
        Raises:
            ValueError: 無効なモードが指定された場合
        """
        if mode not in ["train", "eval"]:
            raise ValueError(f"無効なモード: {mode}. 'train' または 'eval' を指定してください。")
        
        old_mode = self.mode
        self.mode = mode
        
        # モード切り替え時の通知
        if old_mode != mode:
            print(f"環境モード切り替え: {old_mode} → {mode}")
        
    def _initialize_ambulances_realistic(self):
        """ValidationSimulator互換の、現実的な救急車初期化処理。"""
        self.ambulance_states = {}
        # DataFrameのindexではなく、0から始まる連続した番号を使用（元のファイルと同じ）
        for amb_id, (_, row) in enumerate(self.ambulance_data.iterrows()):
            if amb_id >= self.action_dim:
                break
                
            station_h3 = h3.latlng_to_cell(row['latitude'], row['longitude'], 9)
            
            # 50-70%の確率で初期活動中とする
            is_busy = np.random.uniform() < np.random.uniform(0.5, 0.7)
            
            self.ambulance_states[amb_id] = {
                'status': 'dispatched' if is_busy else 'available',
                'completion_time': np.random.uniform(0, 1800) if is_busy else 0, # 0-30分後に復帰
                'current_h3': station_h3, 'station_h3': station_h3,
                'calls_today': 1 if is_busy else 0
            }
        
        # 初期化の確認
        available_count = sum(1 for st in self.ambulance_states.values() if st['status'] == 'available')
        print(f"  救急車初期化完了: {len(self.ambulance_states)}台 (available: {available_count}台, dispatched: {len(self.ambulance_states) - available_count}台)")

    def _calculate_ambulance_completion_time(self, ambulance_id, call):
        """ValidationSimulator互換の詳細な活動完了時間と各フェーズの所要時間を計算します。"""
        amb_state = self.ambulance_states[ambulance_id]
        
        # 各フェーズの時間（分単位）を計算
        response_t = self._get_travel_time(amb_state['current_h3'], call['h3_index'], 'response')
        on_scene_t = self.service_time_generator.generate_time(call['severity'], 'on_scene_time', call['datetime']) if self.service_time_generator else 15.0
        hospital_h3 = self._select_hospital_probabilistic(call['h3_index'], call['severity'])
        transport_t = self._get_travel_time(call['h3_index'], hospital_h3, 'transport')
        hospital_t = self.service_time_generator.generate_time(call['severity'], 'hospital_time', call['datetime']) if self.service_time_generator else 20.0
        return_t = self._get_travel_time(hospital_h3, amb_state['station_h3'], 'return')
        
        total_seconds = (response_t + on_scene_t + transport_t + hospital_t + return_t) * 60.0
        details = {'response_time': response_t, 'on_scene_time': on_scene_t, 'transport_time': transport_t, 'hospital_time': hospital_t, 'return_time': return_t}
        return total_seconds, details

    def _calculate_reward_detailed(self, response_time_minutes, severity):
        """目標時間やカバレッジを考慮した、より詳細な報酬関数。"""
        reward = 10.0 # 基本報酬
        if is_severe_condition(severity): # 重症系
            if response_time_minutes <= 6.0: reward += 20.0 # 6分以内ボーナス
            else: reward -= min((response_time_minutes - 6.0) * 2.0, 30.0) # 超過ペナルティ
        else: # 軽症系
            if response_time_minutes <= 13.0: reward += 5.0 # 13分以内ボーナス
            else: reward -= min((response_time_minutes - 13.0) * 0.5, 10.0) # 超過ペナルティ
        
        # カバレッジ（利用可能率）維持ボーナス
        available_ratio = sum(1 for s in self.ambulance_states.values() if s['status'] == 'available') / self.action_dim
        if available_ratio > 0.3: reward += 5.0 * available_ratio
        return reward

    def _get_travel_time(self, from_h3, to_h3, phase):
        """H3インデックス間の移動時間を分単位で返します。"""
        from_idx, to_idx = self.grid_mapping.get(from_h3), self.grid_mapping.get(to_h3)
        if from_idx is None or to_idx is None or phase not in self.travel_time_matrices: return 10.0 # 10分のデフォルト値
        time_sec = self.travel_time_matrices[phase][from_idx, to_idx]
        return time_sec / 60.0 if 0 < time_sec <= 7200 else 10.0

    def _select_hospital_probabilistic(self, scene_h3, severity):
        """確率モデルに基づき搬送先病院を選択します。モデルがない場合は最近接を選択します。"""
        if self.hospital_selection_model:
            # 簡易的な確率モデルの参照ロジック
            key = (scene_h3, severity) # 仮のキー
            if key in self.hospital_selection_model:
                probs = self.hospital_selection_model[key]
                return np.random.choice(list(probs.keys()), p=list(probs.values()))
        return self._select_nearest_hospital(scene_h3, severity)
        
    def _select_nearest_hospital(self, scene_h3, severity):
        """傷病度に応じてフィルタリングし、最近接の病院を選択します。"""
        target_hospitals = self.hospital_data
        if is_severe_condition(severity):
            severe_hospitals = self.hospital_data[self.hospital_data['genre_code'] == 1]
            if not severe_hospitals.empty: target_hospitals = severe_hospitals
        
        min_time, selected_h3 = float('inf'), scene_h3
        for _, hospital in target_hospitals.iterrows():
            if pd.notna(hospital['latitude']) and pd.notna(hospital['longitude']):
                hosp_h3 = h3.latlng_to_cell(hospital['latitude'], hospital['longitude'], 9)
                if hosp_h3 in self.grid_mapping:
                    time = self._get_travel_time(scene_h3, hosp_h3, 'transport')
                    if time < min_time: min_time, selected_h3 = time, hosp_h3
        return selected_h3

    def get_optimal_action(self) -> Optional[int]:
        """(再統合) 現在の事案に対し、最も早く到着できる救急車（最適行動）を返します。"""
        if not self.pending_call: return None
        best_action, min_time = None, float('inf')
        for amb_id, state in self.ambulance_states.items():
            if state['status'] == 'available':
                time = self._get_travel_time(state['current_h3'], self.pending_call['h3_index'], 'response')
                if time < min_time: min_time, best_action = time, amb_id
        return best_action

    def render(self, mode='human'):
        """(再統合) 環境の現在の状態をコンソールに簡易表示します。"""
        if mode == 'human':
            print(f"\n--- Step {self.step_in_episode} (Time: {self.episode_step_seconds/60.0:.1f} min) ---")
            if self.pending_call: print(f"  Incident: {self.pending_call['severity']} at {self.pending_call['h3_index']}")
            available = sum(1 for s in self.ambulance_states.values() if s['status'] == 'available')
            print(f"  Available Ambulances: {available}/{self.action_dim}")
            if self.episode_stats['response_times']:
                print(f"  Avg Response Time: {np.mean(self.episode_stats['response_times']):.2f} min")

    def get_episode_statistics(self) -> Dict:
        """(再統合) 現在のエピソードの集計済み統計情報を返します。"""
        stats = self.episode_stats.copy()
        if stats['response_times']:
            stats['mean_response_time'] = np.mean(stats['response_times'])
            total = len(stats['response_times'])
            stats['achieved_6min_rate'] = stats['achieved_6min'] / total if total > 0 else 0
        
        # エピソード終了時に継続中の全車出動中期間を記録
        if hasattr(self, '_all_busy_start_time'):
            duration = self.episode_step_seconds - self._all_busy_start_time
            if duration > 0:  # 正の値のみ記録
                hour_of_day = int(self._all_busy_start_time / 3600) % 24
                if not hasattr(self, 'all_busy_events'):
                    self.all_busy_events = []
                self.all_busy_events.append({
                    'start_time': self._all_busy_start_time,
                    'duration': duration,
                    'hour': hour_of_day
                })
        
        # 全車出動中の統計を追加
        if hasattr(self, 'all_busy_events') and self.all_busy_events:
            stats['all_busy_count'] = len(self.all_busy_events)
            stats['all_busy_total_duration'] = sum(e['duration'] for e in self.all_busy_events)
            stats['all_busy_events'] = self.all_busy_events.copy()
        else:
            stats['all_busy_count'] = 0
            stats['all_busy_total_duration'] = 0
            stats['all_busy_events'] = []
        
        return stats

    def _mask_and_select_ambulance(self, action):
        """アクションが有効か確認し、無効なら有効なアクションからランダムに選択します。"""
        mask = self.get_action_mask()
        if action is not None and mask[action]: return action
        valid_actions = np.where(mask)[0]
        return np.random.choice(valid_actions) if len(valid_actions) > 0 else None
        
    def get_action_mask(self) -> np.ndarray:
        """現在利用可能な救急車のマスクを返します。"""
        mask = np.zeros(self.action_dim, dtype=bool)
        
        # 元のファイルと同じロジック：action_dim未満のIDのみ処理
        for amb_id, state in self.ambulance_states.items():
            if amb_id < self.action_dim and state['status'] == 'available':
                mask[amb_id] = True
        
        # 全車出動中の状況を記録
        if not np.any(mask):
            current_hour = int(self.episode_step_seconds / 3600) % 24
            
            # 開始時刻を記録（連続した全車出動中期間の開始）
            if not hasattr(self, '_all_busy_start_time'):
                self._all_busy_start_time = self.episode_step_seconds
                
            # デバッグ：最初の3回のみ詳細情報を表示
            if not hasattr(self, '_mask_debug_count'):
                self._mask_debug_count = 0
            
            if self._mask_debug_count < 3:
                available_count = sum(1 for aid, st in self.ambulance_states.items() if st['status'] == 'available')
                total_count = len(self.ambulance_states)
                dispatched_count = sum(1 for aid, st in self.ambulance_states.items() if st['status'] == 'dispatched')
                
                print(f"\n[全車出動中 {self._mask_debug_count+1}/3] 時刻: {self.episode_step_seconds/3600:.2f}時間 ({current_hour}時台)")
                print(f"  救急車: available=0, dispatched={dispatched_count}")
                
                # 最も早く戻ってくる救急車を表示
                next_return_times = sorted([(aid, st['completion_time']) for aid, st in self.ambulance_states.items()], 
                                          key=lambda x: x[1])[:3]
                print(f"  次の復帰予定:")
                for aid, comp_time in next_return_times:
                    wait_time = comp_time - self.episode_step_seconds
                    print(f"    救急車{aid}: {wait_time:.0f}秒後 ({wait_time/60:.1f}分後)")
                
                self._mask_debug_count += 1
        else:
            # 全車出動中から回復した場合
            if hasattr(self, '_all_busy_start_time'):
                duration = self.episode_step_seconds - self._all_busy_start_time
                hour_of_day = int(self._all_busy_start_time / 3600) % 24
                self.all_busy_events.append({
                    'start_time': self._all_busy_start_time,
                    'duration': duration,
                    'hour': hour_of_day
                })
                delattr(self, '_all_busy_start_time')
        
        return mask
        
    def _advance_to_next_call(self):
        """時間を進め、次の事案をセットします。"""
        self.step_in_episode += 1
        # 次の事案の時刻まで時間を進める
        if self.step_in_episode < len(self.current_episode_calls):
            next_call = self.current_episode_calls[self.step_in_episode]
            self.episode_step_seconds = (next_call['datetime'] - self.episode_start_time).total_seconds()
            self.pending_call = next_call
        else:
            self.pending_call = None
        
        # 救急車の状態を更新
        self._update_ambulance_availability()

    def _update_ambulance_availability(self):
        """現在の時刻に基づき、出動中の救急車が利用可能になるか更新します。"""
        for state in self.ambulance_states.values():
            if state['status'] == 'dispatched' and self.episode_step_seconds >= state['completion_time']:
                state['status'] = 'available'
                state['current_h3'] = state['station_h3']

    def _prepare_episode_calls(self, df, duration_hours):
        """データフレームからエピソード一つ分の事案リストを作成します。"""
        df = df.sort_values('出場年月日時分')
        if len(df) == 0: return []
        start_time, end_time = df['出場年月日時分'].iloc[0], df['出場年月日時分'].iloc[-1]
        max_start = end_time - pd.Timedelta(hours=duration_hours)
        if start_time >= max_start: self.episode_start_time = start_time
        else: self.episode_start_time = start_time + pd.Timedelta(seconds=np.random.uniform(0, (max_start - start_time).total_seconds()))
        episode_end = self.episode_start_time + pd.Timedelta(hours=duration_hours)
        episode_df = df[(df['出場年月日時分'] >= self.episode_start_time) & (df['出場年月日時分'] <= episode_end)]
        calls = []
        for _, row in episode_df.iterrows():
            try: calls.append({'id': str(row['救急事案番号キー']), 'h3_index': h3.latlng_to_cell(row['Y_CODE'], row['X_CODE'], 9), 'severity': row.get('収容所見程度', 'その他'), 'datetime': row['出場年月日時分']})
            except: continue
        return calls

    def _is_episode_done(self): 
        """エピソードが終了したか判定します。"""
        return self.pending_call is None
        
    def _get_observation(self):
        """現在の環境状態から観測ベクトルを生成します。"""
        state_dict = {
            'ambulances': self.ambulance_states,
            'pending_call': self.pending_call,
            'episode_step': self.step_in_episode,
            'time_of_day': self.pending_call['datetime'].hour if self.pending_call else 12
        }
        return self.state_encoder.encode_state(state_dict)

    def _reset_statistics(self): 
        """エピソードの統計情報をリセットします。"""
        self.episode_stats = {
            'response_times': [],
            'achieved_6min': 0,
            'achieved_13min': 0,
            'critical_6min': 0,
            'critical_total': 0,
            'unhandled_calls': 0
        }
        
    def _handle_unresponsive_call(self): 
        """対応不能事案の統計を更新します。"""
        self.episode_stats['unhandled_calls'] += 1
        
    def _update_statistics(self, details): 
        """ステップごとの統計を更新します。"""
        rt = details['response_time']
        severity = details.get('severity', '軽症')  # デフォルトは軽症
        
        self.episode_stats['response_times'].append(rt)
        
        # 6分達成率
        if rt <= 6.0:
            self.episode_stats['achieved_6min'] += 1
        
        # 13分達成率
        if rt <= 13.0:
            self.episode_stats['achieved_13min'] += 1
        
        # 重症系の統計（重篤、重症、死亡）
        if is_severe_condition(severity):
            self.episode_stats['critical_total'] += 1
            if rt <= 6.0:
                self.episode_stats['critical_6min'] += 1