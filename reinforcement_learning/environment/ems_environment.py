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
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.mode = mode
        self.current_period_idx = 0
        
        print("=" * 60)
        print(f"EMS環境初期化 (モード: {mode})")
        print(f"設定ファイル: {config_path}")
        print("=" * 60)
        
        # 傷病度設定の初期化
        self._setup_severity_mapping()
        
        # データパスの設定
        self.base_dir = Path("data/tokyo")
        self._load_base_data()
        
        # シミュレータの初期化は reset() で行う
        self.simulator = None
        self.current_episode_calls = []
        self.pending_call = None
        self.episode_step = 0
        self.max_steps_per_episode = None
        
        # 状態・行動空間の次元
        self.state_dim = self._calculate_state_dim()
        self.action_dim = 192  # 最大救急車数*(デイタイム救急抜き)
        
        print(f"状態空間次元: {self.state_dim}")
        print(f"行動空間次元: {self.action_dim}")
        
        # 統計情報の初期化
        self.episode_stats = self._init_episode_stats()
        
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
        # 状態の構成要素
        # - 救急車情報: 最大300台 × 4特徴 = 1200
        # - 事案情報: 10特徴
        # - 時間情報: 8特徴
        # - 地域統計: 20特徴
        # 合計: 約1238次元
        
        ambulance_features = 192 * 4  # 位置、状態、出動回数、経過時間
        incident_features = 10  # 位置、傷病度、etc
        temporal_features = 8  # 時刻、曜日、etc
        spatial_features = 20  # 地域の混雑度、etc
        
        return ambulance_features + incident_features + temporal_features + spatial_features
    
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
        
        print(f"\nエピソード開始: {period['start_date']} - {period['end_date']}")
        
        # シミュレータの初期化
        self._init_simulator_for_period(period)
        
        # エピソード統計のリセット
        self.episode_stats = self._init_episode_stats()
        self.episode_step = 0
        
        # 最初の事案を取得
        self._advance_to_next_call()
        
        # 初期観測を返す
        return self._get_observation()
    
    def _init_simulator_for_period(self, period: Dict):
        """指定期間用のシミュレータを初期化"""
        # 簡易版: ValidationSimulatorの軽量ラッパーを作成
        # 実際の実装では、ValidationSimulatorを適切に初期化
        
        # 救急事案データの読み込み
        calls_df = self._load_calls_for_period(period)
        
        # エピソード用の事案を準備
        self.current_episode_calls = self._prepare_episode_calls(calls_df)
        self.max_steps_per_episode = len(self.current_episode_calls)
        
        # 救急車状態の初期化
        self._init_ambulance_states()
        
    def _load_calls_for_period(self, period: Dict) -> pd.DataFrame:
        """指定期間の救急事案データを読み込み"""
        # 実際のデータ読み込み処理
        # ここでは簡易版
        calls_path = "C:/Users/tetsu/OneDrive - Yokohama City University/30_データカタログ/tfd_data/hanso_special_wards.csv"
        calls_df = pd.read_csv(calls_path, encoding='utf-8')
        
        # 期間でフィルタリング
        calls_df['出場年月日時分'] = pd.to_datetime(calls_df['出場年月日時分'])
        start_date = pd.to_datetime(period['start_date'])
        end_date = pd.to_datetime(period['end_date'])
        
        mask = (calls_df['出場年月日時分'] >= start_date) & (calls_df['出場年月日時分'] <= end_date)
        calls_df = calls_df[mask].copy()
        
        # 「その他」を除外
        calls_df = calls_df[calls_df['収容所見程度'] != 'その他']
        
        return calls_df
    
    def _prepare_episode_calls(self, calls_df: pd.DataFrame) -> List[Dict]:
        """エピソード用の事案リストを準備"""
        episode_calls = []
        
        for _, row in calls_df.iterrows():
            # H3インデックスの計算
            h3_index = h3.latlng_to_cell(row['Y_CODE'], row['X_CODE'], 9)
            
            call_info = {
                'id': str(row['救急事案番号キー']),
                'h3_index': h3_index,
                'severity': row.get('収容所見程度', 'その他'),
                'datetime': row['出場年月日時分'],
                'location': (row['Y_CODE'], row['X_CODE'])
            }
            episode_calls.append(call_info)
        
        # 時間順にソート
        episode_calls.sort(key=lambda x: x['datetime'])
        
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
        
        return StepResult(observation, reward, done, info)
    
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
        
        # 移動時間の計算
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
    
    def _calculate_travel_time(self, from_h3: str, to_h3: str) -> float:
        """移動時間を計算（秒）"""
        # グリッドマッピングからインデックスを取得
        from_idx = self.grid_mapping.get(from_h3)
        to_idx = self.grid_mapping.get(to_h3)
        
        if from_idx is None or to_idx is None:
            return 600.0  # デフォルト10分
        
        # 移動時間行列から取得
        if 'response' in self.travel_time_matrices:
            travel_time = self.travel_time_matrices['response'][from_idx, to_idx]
        else:
            travel_time = 600.0  # デフォルト
        
        return travel_time
    
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
        
        # 報酬設計クラスを使用
        from .reward_designer import RewardDesigner
        reward_designer = RewardDesigner(self.config)
        
        reward = reward_designer.calculate_reward(
            severity=severity,
            response_time=response_time,
            coverage_impact=0.0  # 簡易版では0
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
        return self.pending_call is None or self.episode_step >= self.max_steps_per_episode
    
    def _get_observation(self) -> np.ndarray:
        """現在の観測を取得"""
        # 状態エンコーダを使用
        from .state_encoder import StateEncoder
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