"""
ems_environment.py
救急隊ディスパッチのための強化学習環境
"""

import numpy as np
import torch
import yaml
import json
import h3
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from data_cache import get_emergency_data_cache
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from validation_simulation import (
    ValidationSimulator,
    EventType,
    AmbulanceStatus,
    EmergencyCall,
    Event
)

# 設定ユーティリティのインポート
try:
    from ..config_utils import load_config_with_inheritance
except ImportError:
    # スタンドアロン実行時のフォールバック
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_utils import load_config_with_inheritance

@dataclass
class StepResult:
    """ステップ実行結果"""
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]

class EMSEnvironment:
    """
    PPO学習用のEMS環境
    OpenAI Gym形式のインターフェースを提供
    """
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "train"):
        """
        Args:
            config_path: 設定ファイルのパス
            mode: "train" or "eval"
        """
        # 設定読み込み（継承機能付き）
        self.config = load_config_with_inheritance(config_path)
        
        self.mode = mode
        self.current_period_idx = 0
        
        # ログ制御フラグ
        self._first_period_logged = False
        self._episode_count = 0
        
        print("=" * 60)
        print(f"EMS環境初期化 (モード: {mode})")
        print(f"設定ファイル: {config_path}")
        print("=" * 60)
        
        # データキャッシュの初期化
        print("データキャッシュを初期化中...")
        self.data_cache = get_emergency_data_cache()
        
        # 初回データ読み込み（起動時に一度だけ）
        print("初期データ読み込み中...")
        self.data_cache.load_data()
        print("データキャッシュ準備完了")
        
        # 傷病度設定の初期化
        self._setup_severity_mapping()
        
        # データパスの設定
        self.base_dir = Path("data/tokyo")
        self._load_base_data()
        
        # 移動時間行列の読み込み（ValidationSimulatorと同じ方法）
        self.travel_time_matrices = {}
        self.travel_distance_matrices = {}
        
        calibration_dir = self.base_dir / "calibration2"
        travel_time_stats_path = calibration_dir / 'travel_time_statistics_all_phases.json'
        
        if travel_time_stats_path.exists():
            with open(travel_time_stats_path, 'r', encoding='utf-8') as f:
                phase_stats_data = json.load(f)
            
            # ValidationSimulatorと同じロジックで行列を読み込み
            for phase in ['response', 'transport', 'return']:
                matrix_filename = None
                
                if phase in phase_stats_data and 'calibrated' in phase_stats_data[phase]:
                    model_type = phase_stats_data[phase]['calibrated'].get('model_type')
                    
                    if model_type == "uncalibrated":
                        matrix_filename = f"uncalibrated_travel_time_{phase}.npy"
                    elif model_type in ['linear', 'log']:
                        matrix_filename = f"{model_type}_calibrated_{phase}.npy"
                    
                    if matrix_filename:
                        matrix_path = calibration_dir / matrix_filename
                        if matrix_path.exists():
                            self.travel_time_matrices[phase] = np.load(matrix_path)
                            print(f"  移動時間行列読み込み: {phase} ({model_type})")
        
        # 距離行列も同様に読み込み
        distance_matrix_path = self.base_dir / "processed/travel_distance_matrix_res9.npy"
        if distance_matrix_path.exists():
            travel_distance_matrix = np.load(distance_matrix_path)
            # ValidationSimulatorと同じ形式に変換
            self.travel_distance_matrices = {
                'dispatch_to_scene': travel_distance_matrix,
                'scene_to_hospital': travel_distance_matrix,
                'hospital_to_station': travel_distance_matrix
            }
        
        # シミュレータの初期化は reset() で行う
        self.simulator = None
        self.current_episode_calls = []
        self.pending_call = None
        self.episode_step = 0
        self.max_steps_per_episode = None
        
        # デバッグ用のverbose_logging属性を初期化
        self.verbose_logging = False
        
        # 状態・行動空間の次元
        self.state_dim = self._calculate_state_dim()
        self.action_dim = 192  # 最大救急車数*(デイタイム救急抜き)
        
        print(f"状態空間次元: {self.state_dim}")
        print(f"行動空間次元: {self.action_dim}")
        
        # 統計情報の初期化
        self.episode_stats = self._init_episode_stats()
        
        # RewardDesignerを一度だけ初期化
        from .reward_designer import RewardDesigner
        self.reward_designer = RewardDesigner(self.config)        
        
    def _setup_severity_mapping(self):
        """傷病度マッピングの設定"""
        self.severity_to_category = {}
        self.severity_weights = {}
        
        for category, info in self.config['severity']['categories'].items():
            for condition in info['conditions']:
                self.severity_to_category[condition] = category
                self.severity_weights[condition] = info['reward_weight']
        
        print("傷病度設定:")
        for category, info in self.config['severity']['categories'].items():
            conditions = ', '.join(info['conditions'])
            weight = info['reward_weight']
            print(f"  {category}: {conditions} (重み: {weight})")
    
    def _load_base_data(self):
        """基本データの読み込み"""
        print("\n基本データ読み込み中...")
        
        # 救急署データ
        firestation_path = self.base_dir / "import/amb_place_master.csv"
        self.ambulance_data = pd.read_csv(firestation_path, encoding='utf-8')
        self.ambulance_data = self.ambulance_data[self.ambulance_data['special_flag'] == 1]
        print(f"  救急署数: {len(self.ambulance_data)}")
        
        # 病院データ
        hospital_path = self.base_dir / "import/hospital_master.csv"
        self.hospital_data = pd.read_csv(hospital_path, encoding='utf-8')
        print(f"  病院数: {len(self.hospital_data)}")
        
        # グリッドマッピング
        grid_mapping_path = self.base_dir / "processed/grid_mapping_res9.json"
        with open(grid_mapping_path, 'r', encoding='utf-8') as f:
            self.grid_mapping = json.load(f)
        print(f"  H3グリッド数: {len(self.grid_mapping)}")
        
        # 移動時間行列（軽量版 - 学習用）
        # 実際のファイルパスに合わせて修正してください
        self.travel_time_matrices = {}
        calibration_dir = self.base_dir / "calibration2"
        for phase in ['response', 'transport', 'return']:
            matrix_path = calibration_dir / f"linear_calibrated_{phase}.npy"
            if matrix_path.exists():
                self.travel_time_matrices[phase] = np.load(matrix_path)
        
        # 距離行列
        distance_matrix_path = self.base_dir / "processed/travel_distance_matrix_res9.npy"
        self.travel_distance_matrix = np.load(distance_matrix_path)
        
    def _calculate_state_dim(self):
        """状態空間の次元を計算"""
        # 192台の救急車 × 4特徴
        ambulance_features = 192 * 4  # 768
        
        # 事案情報
        incident_features = 10
        
        # 時間情報
        temporal_features = 8
        
        # 空間情報
        spatial_features = 20
        
        total = ambulance_features + incident_features + temporal_features + spatial_features
        return total  # 806
    
    def reset(self, period_index: Optional[int] = None) -> np.ndarray:
        """
        環境のリセット
        
        Returns:
            初期観測
        """
        # 期間の選択
        if self.mode == "train":
            periods = self.config['data']['train_periods']
        else:
            periods = self.config['data']['eval_periods']
        
        if period_index is None:
            period_index = np.random.randint(len(periods))
        
        self.current_period_idx = period_index
        period = periods[period_index]
        
        # エピソード開始情報は最初の期間のみ表示
        if not self._first_period_logged:
            print(f"\nエピソード開始: {period['start_date']} - {period['end_date']}")
            self._first_period_logged = True
        
        # エピソードカウンタをインクリメント
        self._episode_count += 1
        
        # シミュレータの初期化
        self._init_simulator_for_period(period)
        
        # エピソード統計のリセット
        self.episode_stats = self._init_episode_stats()
        
        # 最初の事案を設定（重要！）
        if len(self.current_episode_calls) > 0:
            self.episode_step = 0
            self.pending_call = self.current_episode_calls[0]
        else:
            print("警告: エピソードに事案がありません")
            self.pending_call = None
        
        # 初期観測を返す
        return self._get_observation()
    
    def _init_simulator_for_period(self, period: Dict):
        """指定期間用のシミュレータを初期化"""
        # 救急事案データの読み込み
        calls_df = self._load_calls_for_period(period)
        
        # エピソード用の事案を準備
        self.current_episode_calls = self._prepare_episode_calls(calls_df)
        self.max_steps_per_episode = len(self.current_episode_calls)
        
        print(f"読み込まれた事案数: {len(self.current_episode_calls)}")
        
        # 救急車状態の初期化
        self._init_ambulance_states()
        
        # エピソードカウンタ初期化（重要！）
        self.episode_step = 0
        self.pending_call = None
        
    def _load_calls_for_period(self, period: Dict) -> pd.DataFrame:
        """
        指定期間の救急事案データを読み込み（最適化版）
        キャッシュからデータを取得するため高速
        """
        start_date = str(period['start_date'])
        end_date = str(period['end_date'])
        
        # 最初の期間のみ詳細情報を表示
        if not self._first_period_logged:
            print(f"期間データ取得中: {start_date} - {end_date}")
        
        # キャッシュから高速取得
        filtered_df = self.data_cache.get_period_data(start_date, end_date)
        
        if not self._first_period_logged:
            print(f"期間内の事案数: {len(filtered_df)}件")
        
        # 必要なカラムの存在確認
        required_columns = ['救急事案番号キー', 'Y_CODE', 'X_CODE', '収容所見程度', '出場年月日時分']
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        if missing_columns:
            print(f"警告: 必要なカラムが不足: {missing_columns}")
            return pd.DataFrame()
        
        if not self._first_period_logged:
            print(f"最終的な事案数: {len(filtered_df)}件")
            
            if len(filtered_df) > 0:
                # 傷病度の分布を表示
                severity_counts = filtered_df['収容所見程度'].value_counts()
                print("傷病度分布:")
                for severity, count in severity_counts.head().items():
                    print(f"  {severity}: {count}件")
            print(f"エピソード長: {self.config['data']['episode_duration_hours']}時間")
        
        return filtered_df
    
    def _prepare_episode_calls(self, calls_df: pd.DataFrame) -> List[Dict]:
        """エピソード用の事案リストを準備"""
        import h3
        import numpy as np
        import pandas as pd
        
        if len(calls_df) == 0:
            print("警告: 事案データが空です")
            return []
        
        episode_calls = []
        
        # エピソード長の設定（時間）
        episode_hours = self.config['data']['episode_duration_hours']
        print(f"エピソード長: {episode_hours}時間")
        
        # 時刻でソート
        calls_df = calls_df.sort_values('出場年月日時分')
        
        # エピソードの開始時刻をランダムに選択
        start_time = calls_df['出場年月日時分'].iloc[0]
        end_time = calls_df['出場年月日時分'].iloc[-1]
        
        # エピソード期間内のデータを選択できる開始時刻の範囲
        max_start_time = end_time - pd.Timedelta(hours=episode_hours)
        
        if start_time >= max_start_time:
            # データが短すぎる場合は全体を使用
            episode_start = start_time
            episode_end = end_time
            print(f"警告: データ期間が短いため、全期間を使用")
        else:
            # ランダムな開始時刻を選択
            time_range = (max_start_time - start_time).total_seconds()
            random_offset = np.random.uniform(0, time_range)
            episode_start = start_time + pd.Timedelta(seconds=random_offset)
            episode_end = episode_start + pd.Timedelta(hours=episode_hours)
        
        # エピソード期間内の事案を抽出
        mask = (calls_df['出場年月日時分'] >= episode_start) & (calls_df['出場年月日時分'] <= episode_end)
        episode_df = calls_df[mask].copy()
        
        # 毎回表示する情報（簡潔版）
        print(f"エピソード期間: {episode_start.strftime('%Y-%m-%d %H:%M')} ～ {episode_end.strftime('%Y-%m-%d %H:%M')}")
        print(f"エピソード内事案数: {len(episode_df)}件")
        
        for _, row in episode_df.iterrows():
            # H3インデックスの計算
            try:
                # 座標の有効性チェック
                lat = float(row['Y_CODE'])
                lng = float(row['X_CODE'])
                
                if -90 <= lat <= 90 and -180 <= lng <= 180:
                    h3_index = h3.latlng_to_cell(lat, lng, 9)
                else:
                    continue  # 無効な座標はスキップ
            except Exception as e:
                continue  # 変換エラーはスキップ
            
            call_info = {
                'id': str(row['救急事案番号キー']),
                'h3_index': h3_index,
                'severity': row.get('収容所見程度', 'その他'),
                'datetime': row['出場年月日時分'],
                'location': (lat, lng)
            }
            episode_calls.append(call_info)
        
        # 時間順にソート
        episode_calls.sort(key=lambda x: x['datetime'])
        
        print(f"有効な事案数: {len(episode_calls)}件")
        
        return episode_calls
    
    def _init_ambulance_states(self):
        """救急車の状態を初期化"""
        self.ambulance_states = {}
        
        for idx, row in self.ambulance_data.iterrows():
            if idx >= self.action_dim:
                break
            
            h3_index = h3.latlng_to_cell(row['latitude'], row['longitude'], 9)
            
            self.ambulance_states[idx] = {
                'id': f"amb_{idx}",
                'station_h3': h3_index,
                'current_h3': h3_index,
                'status': 'available',
                'calls_today': 0,
                'last_dispatch_time': None
            }
    
    def step(self, action: int) -> StepResult:
        """
        環境のステップ実行
        
        Args:
            action: 選択された救急車のインデックス
            
        Returns:
            StepResult: 観測、報酬、終了フラグ、追加情報
        """
        try:
            # デバッグ用: 最適行動との比較を出力
            if hasattr(self, 'verbose_logging') and self.verbose_logging:
                optimal_action = self.get_optimal_action()
                if optimal_action is not None and action != optimal_action:
                    optimal_time = self._calculate_travel_time(
                        self.ambulance_states[optimal_action]['current_h3'],
                        self.pending_call['h3_index']
                    )
                    actual_time = self._calculate_travel_time(
                        self.ambulance_states[action]['current_h3'],
                        self.pending_call['h3_index']
                    )
                    print(f"[選択比較] PPO選択: 救急車{action}({actual_time/60:.1f}分) "
                        f"vs 最適: 救急車{optimal_action}({optimal_time/60:.1f}分)")
            
            # 行動の実行（救急車の配車）
            dispatch_result = self._dispatch_ambulance(action)
            
            # 報酬の計算
            reward = self._calculate_reward(dispatch_result)
            
            # 統計情報の更新
            self._update_statistics(dispatch_result)
            
            # 次の事案へ進む
            self._advance_to_next_call()
            
            # エピソード終了判定
            done = self._is_episode_done()
            
            # 次の観測を取得
            observation = self._get_observation()
            
            # 追加情報
            info = {
                'dispatch_result': dispatch_result,
                'episode_stats': self.episode_stats.copy(),
                'step': self.episode_step
            }
            
            # StepResultオブジェクトを返す
            return StepResult(
                observation=observation,
                reward=reward,
                done=done,
                info=info
            )
        except Exception as e:
            print(f"❌ step()メソッドでエラー発生: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_optimal_action(self) -> Optional[int]:
        """
        現在の事案に対して最適な救急車を選択（最近接）
        ValidationSimulatorのfind_closest_available_ambulanceと同じロジック
        
        Returns:
            最適な救急車のID、または None
        """
        if self.pending_call is None:
            return None
        
        best_action = None
        min_travel_time = float('inf')
        
        # 全ての救急車をチェック
        for amb_id, amb_state in self.ambulance_states.items():
            # 利用可能な救急車のみ対象
            if amb_state['status'] != 'available':
                continue
            
            try:
                # 現在位置から事案発生地点への移動時間を計算
                travel_time = self._calculate_travel_time(
                    amb_state['current_h3'],
                    self.pending_call['h3_index']
                )
                
                # より近い救急車を発見
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    best_action = amb_id
                    
            except Exception as e:
                # エラーが発生した場合はスキップ
                continue
        
        # デバッグ情報の出力（verboseモード時）
        if best_action is not None and hasattr(self, 'verbose_logging') and self.verbose_logging:
            print(f"[最適選択] 救急車{best_action}を選択 (移動時間: {min_travel_time/60:.1f}分)")
        
        return best_action


    
    def _dispatch_ambulance(self, action: int) -> Dict:
        """救急車を配車"""
        if self.pending_call is None:
            return {'success': False, 'reason': 'no_pending_call'}
        
        # 行動の妥当性チェック
        if action >= len(self.ambulance_states):
            return {'success': False, 'reason': 'invalid_action'}
        
        amb_state = self.ambulance_states[action]
        
        # 利用可能性チェック
        if amb_state['status'] != 'available':
            return {'success': False, 'reason': 'ambulance_busy'}
        
        # 移動時間の計算（修正版）
        travel_time = self._calculate_travel_time(
            amb_state['current_h3'],
            self.pending_call['h3_index']
        )
        
        # 配車実行
        amb_state['status'] = 'dispatched'
        amb_state['calls_today'] += 1
        amb_state['last_dispatch_time'] = self.episode_step
        
        result = {
            'success': True,
            'ambulance_id': action,
            'call_id': self.pending_call['id'],
            'severity': self.pending_call['severity'],
            'response_time': travel_time,
            'response_time_minutes': travel_time / 60.0
        }
        
        # 簡易的に一定時間後に利用可能に戻す
        # 実際の実装では、現場活動時間、搬送時間等を考慮
        return_time = self.episode_step + np.random.randint(30, 90)  # 30-90分後
        self._schedule_ambulance_return(action, return_time)
        
        return result
    
    # _calculate_travel_timeメソッドの修正
    def _calculate_travel_time(self, from_h3: str, to_h3: str) -> float:
        """
        移動時間を計算（秒単位）
        ValidationSimulatorのget_travel_timeと同じロジックを使用
        """
        # phaseは'response'をデフォルトとする（救急車選択時）
        phase = 'response'
        
        from_idx = self.grid_mapping.get(from_h3)
        to_idx = self.grid_mapping.get(to_h3)
        
        if from_idx is None or to_idx is None:
            # グリッドマッピングにない場合のフォールバック
            return 600.0  # デフォルト10分
        
        # 移動時間行列から取得
        current_travel_time_matrix = self.travel_time_matrices.get(phase)
        
        if current_travel_time_matrix is None:
            # responseフェーズの行列がない場合
            return 600.0  # デフォルト10分
        
        try:
            travel_time = current_travel_time_matrix[from_idx, to_idx]
            
            # 異常値チェック（ValidationSimulatorにはないが、安全のため）
            if travel_time <= 0 or travel_time > 3600:  # 1時間以上は異常
                return 600.0  # デフォルト10分
            
            return travel_time
        except:
            return 600.0  # エラー時のデフォルト
    
    def _schedule_ambulance_return(self, amb_id: int, return_time: int):
        """救急車の帰還をスケジュール"""
        # 簡易実装: return_timeステップ後に利用可能に戻す
        # 実際の実装では、イベントキューを使用
        pass
    
    def _calculate_reward(self, dispatch_result: Dict) -> float:
        """報酬を計算"""
        if not dispatch_result['success']:
            return -10.0  # 配車失敗ペナルティ
        
        severity = dispatch_result['severity']
        response_time = dispatch_result['response_time']
        
        # 既に初期化済みのreward_designerを使用
        reward = self.reward_designer.calculate_reward(
            severity=severity,
            response_time=response_time,
            coverage_impact=0.0
        )
        
        return reward
    
    def _update_statistics(self, dispatch_result: Dict):
        """統計情報を更新"""
        if not dispatch_result['success']:
            self.episode_stats['failed_dispatches'] += 1
            return
        
        self.episode_stats['total_dispatches'] += 1
        
        # 応答時間統計
        rt_minutes = dispatch_result['response_time_minutes']
        self.episode_stats['response_times'].append(rt_minutes)
        
        # 傷病度別統計
        severity = dispatch_result['severity']
        if severity not in self.episode_stats['response_times_by_severity']:
            self.episode_stats['response_times_by_severity'][severity] = []
        self.episode_stats['response_times_by_severity'][severity].append(rt_minutes)
        
        # 閾値達成率
        if rt_minutes <= 6.0:
            self.episode_stats['achieved_6min'] += 1
        if rt_minutes <= 13.0:
            self.episode_stats['achieved_13min'] += 1
        
        # 重症系の6分達成率
        if severity in self.config['severity']['categories']['critical']['conditions']:
            self.episode_stats['critical_total'] += 1
            if rt_minutes <= 6.0:
                self.episode_stats['critical_6min'] += 1
    
    def _advance_to_next_call(self):
        """次の事案へ進む"""
        self.episode_step += 1
        
        if self.episode_step < len(self.current_episode_calls):
            self.pending_call = self.current_episode_calls[self.episode_step]
            
            # 時間経過に伴う救急車状態の更新
            self._update_ambulance_availability()
        else:
            self.pending_call = None
    
    def _update_ambulance_availability(self):
        """救急車の利用可能性を更新"""
        # 簡易実装: 一定時間経過後に自動的に利用可能に
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_state['status'] == 'dispatched':
                if amb_state['last_dispatch_time'] is not None:
                    elapsed = self.episode_step - amb_state['last_dispatch_time']
                    if elapsed > 60:  # 60分経過で自動復帰
                        amb_state['status'] = 'available'
                        amb_state['current_h3'] = amb_state['station_h3']
    
    def _is_episode_done(self) -> bool:
        """エピソード終了判定"""
        # 全事案を処理したら終了
        if self.pending_call is None:
            return True
        
        # ステップ数が最大値を超えたら終了
        if self.episode_step >= len(self.current_episode_calls):
            return True
        
        # 設定された最大ステップ数を超えたら終了（オプション）
        max_steps = self.config.get('max_steps_per_episode', 1000)
        if self.episode_step >= max_steps:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """現在の観測を取得"""
        # 状態エンコーダを使用
        from .state_encoder import StateEncoder
        
        # configにnetworkセクションがない場合のデフォルト値
        if 'network' not in self.config:
            self.config['network'] = {
                'state_encoder': {
                    'ambulance_features': 4,
                    'incident_features': 10,
                    'spatial_features': 20,
                    'temporal_features': 8
                }
            }
        
        encoder = StateEncoder(self.config)
        
        state_dict = {
            'ambulances': self.ambulance_states,
            'pending_call': self.pending_call,
            'episode_step': self.episode_step,
            'time_of_day': self._get_time_of_day()
        }
        
        observation = encoder.encode_state(state_dict, self.grid_mapping)
        
        return observation
    
    def _get_time_of_day(self) -> int:
        """現在の時刻を取得（0-23）"""
        if self.pending_call and 'datetime' in self.pending_call:
            return self.pending_call['datetime'].hour
        return 12  # デフォルト
    
    def _init_episode_stats(self) -> Dict:
        """エピソード統計の初期化"""
        return {
            'total_dispatches': 0,
            'failed_dispatches': 0,
            'response_times': [],
            'response_times_by_severity': {},
            'achieved_6min': 0,
            'achieved_13min': 0,
            'critical_total': 0,
            'critical_6min': 0
        }
    
    def get_action_mask(self) -> np.ndarray:
        """利用可能な行動のマスクを取得"""
        mask = np.zeros(self.action_dim, dtype=bool)
        
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_id < self.action_dim and amb_state['status'] == 'available':
                mask[amb_id] = True
        
        return mask
 
    def get_best_action_for_call(self) -> Optional[int]:
        """
        現在の事案に対して最適な救急車（行動）を選択
        学習初期はこれを教師として使用できる
        """
        if self.pending_call is None:
            return None
        
        best_action = None
        min_travel_time = float('inf')
        
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_state['status'] != 'available':
                continue
            
            # 移動時間を計算
            travel_time = self._calculate_travel_time(
                amb_state['current_h3'],
                self.pending_call['h3_index']
            )
            
            if travel_time < min_travel_time:
                min_travel_time = travel_time
                best_action = amb_id
        
        return best_action
    
    def render(self, mode: str = 'human'):
        """環境の可視化（オプション）"""
        if mode == 'human':
            print(f"\nStep {self.episode_step}")
            if self.pending_call:
                print(f"  事案: {self.pending_call['severity']} at {self.pending_call['h3_index']}")
            
            available_count = sum(1 for a in self.ambulance_states.values() if a['status'] == 'available')
            print(f"  利用可能救急車: {available_count}/{len(self.ambulance_states)}")
            
            if self.episode_stats['total_dispatches'] > 0:
                avg_rt = np.mean(self.episode_stats['response_times'])
                rate_6min = self.episode_stats['achieved_6min'] / self.episode_stats['total_dispatches'] * 100
                print(f"  平均応答時間: {avg_rt:.1f}分")
                print(f"  6分達成率: {rate_6min:.1f}%")