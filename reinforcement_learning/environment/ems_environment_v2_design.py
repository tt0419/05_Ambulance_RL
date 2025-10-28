"""
ems_environment_v2_design.py
ValidationSimulator完全統合型PPO学習環境の設計ドキュメント

このファイルは実装の青写真として、修正の方向性を明確にします。
"""

import heapq
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

# ============================================================================
# イベント定義（ValidationSimulatorと同一）
# ============================================================================

class EventType(Enum):
    """イベントタイプの定義"""
    NEW_CALL = "new_call"              # 新規事案の発生
    AMBULANCE_RETURN = "ambulance_return"  # 救急車の復帰
    EPISODE_END = "episode_end"        # エピソード終了
    CHECKPOINT = "checkpoint"          # 学習用チェックポイント

@dataclass
class Event:
    """イベントデータクラス"""
    time: float                        # イベント発生時刻（秒）
    event_type: EventType              # イベントタイプ
    data: Dict[str, Any]              # イベント固有データ
    
    def __lt__(self, other):
        """優先度付きキュー用の比較演算子"""
        return self.time < other.time

# ============================================================================
# 統合環境のコアクラス
# ============================================================================

class ValidationIntegratedEMSEnvironment:
    """
    ValidationSimulatorの精度とPPO学習の両立を実現する統合環境
    
    設計原則:
    1. 内部時間管理は連続時間（秒単位float）
    2. イベント駆動による正確なシミュレーション
    3. 外部インターフェースはGym互換のstep/reset
    4. フェーズ別移動時間行列の完全サポート
    """
    
    def __init__(self, config_path: str, mode: str = "train"):
        """
        環境の初期化
        
        Args:
            config_path: 設定ファイルのパス
            mode: "train" または "eval"
        """
        self.config = self._load_config(config_path)
        self.mode = mode
        
        # ============================================================
        # Phase 1: イベント駆動システムの初期化
        # ============================================================
        self.current_time = 0.0           # 現在時刻（秒）
        self.event_queue = []             # イベントキュー（heapq）
        self.episode_start_time = None    # エピソード開始の実時間
        
        # ============================================================
        # Phase 2: ValidationSimulator互換コンポーネント
        # ============================================================
        self._load_travel_time_matrices()    # フェーズ別行列
        self._load_service_time_generator()  # サービス時間生成器
        self._load_hospital_selection_model() # 病院選択モデル
        
        # ============================================================
        # Phase 3: PPO学習用インターフェース
        # ============================================================
        self.pending_call = None          # 現在処理中の事案
        self.ambulance_states = {}        # 救急車状態辞書
        self.action_dim = 192             # 行動空間次元
        
        # 状態エンコーダの初期化
        from .state_encoder import StateEncoder
        self.state_encoder = StateEncoder(
            config=self.config,
            max_ambulances=self.action_dim,
            travel_time_matrix=self.travel_time_matrices.get('response'),
            grid_mapping=self.grid_mapping
        )
        self.state_dim = self.state_encoder.state_dim
        
        print(f"✓ ValidationSimulator統合環境を初期化")
        print(f"  - 状態空間: {self.state_dim}次元")
        print(f"  - 行動空間: {self.action_dim}次元")
        print(f"  - モード: {self.mode}")
    
    # ========================================================================
    # Phase 1: イベント駆動システムの実装
    # ========================================================================
    
    def _schedule_event(self, event: Event):
        """
        イベントをキューに追加
        
        ValidationSimulatorと同じ優先度付きキュー管理
        """
        heapq.heappush(self.event_queue, event)
    
    def _process_next_event(self) -> Optional[Event]:
        """
        次のイベントを処理
        
        Returns:
            処理したEvent、またはキューが空の場合None
        """
        if not self.event_queue:
            return None
        
        event = heapq.heappop(self.event_queue)
        old_time = self.current_time
        self.current_time = event.time
        
        # イベントタイプに応じた処理
        if event.event_type == EventType.NEW_CALL:
            self._handle_new_call_event(event)
        elif event.event_type == EventType.AMBULANCE_RETURN:
            self._handle_ambulance_return_event(event)
        elif event.event_type == EventType.CHECKPOINT:
            self._handle_checkpoint_event(event)
        
        return event
    
    def _handle_new_call_event(self, event: Event):
        """
        新規事案イベントの処理
        
        ValidationSimulatorのロジックを踏襲しつつ、
        PPO学習用にpending_callとして保持
        """
        call_data = event.data
        self.pending_call = {
            'id': call_data['id'],
            'h3_index': call_data['h3_index'],
            'severity': call_data['severity'],
            'datetime': call_data['datetime'],
            'arrival_time': event.time  # 事案到着時刻
        }
        
        # PPO学習のためにここで一時停止
        # step()が呼ばれるまで待機
    
    def _handle_ambulance_return_event(self, event: Event):
        """
        救急車復帰イベントの処理
        
        ValidationSimulatorと同様に、救急車を利用可能状態に戻す
        """
        ambulance_id = event.data['ambulance_id']
        
        if ambulance_id in self.ambulance_states:
            self.ambulance_states[ambulance_id]['status'] = 'available'
            self.ambulance_states[ambulance_id]['current_h3'] = \
                self.ambulance_states[ambulance_id]['station_h3']
            
            # デバッグログ
            if self.config.get('verbose_logging', False):
                print(f"[{self.current_time/3600:.2f}h] 救急車{ambulance_id}が復帰")
    
    def _handle_checkpoint_event(self, event: Event):
        """
        学習用チェックポイントイベント
        
        定期的に状態を保存・統計を更新
        """
        pass
    
    # ========================================================================
    # Phase 2: ValidationSimulator互換の移動時間・サービス時間計算
    # ========================================================================
    
    def _calculate_ambulance_activity_time(
        self, 
        ambulance_id: int, 
        call: Dict
    ) -> Tuple[float, Dict]:
        """
        救急車の活動時間を詳細に計算（ValidationSimulator互換）
        
        重要な変更点:
        1. フェーズ別移動時間行列を使用
        2. ServiceTimeGeneratorで確率的に時間を生成
        3. 全フェーズの詳細を返す
        
        Args:
            ambulance_id: 救急車ID
            call: 事案情報
        
        Returns:
            (合計時間(秒), 詳細辞書)
        """
        amb_state = self.ambulance_states[ambulance_id]
        current_time_obj = call['datetime']
        
        # Phase 1: Response (救急車 → 現場)
        response_time_sec = self._get_travel_time_by_phase(
            amb_state['current_h3'], 
            call['h3_index'], 
            'response'  # ★フェーズ指定
        )
        
        # Phase 2: On-Scene (現場での活動)
        on_scene_time_min = self.service_time_generator.generate_time(
            call['severity'], 
            'on_scene_time',
            current_time_obj  # 時刻情報を渡す
        )
        on_scene_time_sec = on_scene_time_min * 60.0
        
        # Phase 3: 病院選択（確率的モデル使用）
        hospital_h3 = self._select_hospital_probabilistic(
            call['h3_index'], 
            call['severity']
        )
        
        # Phase 4: Transport (現場 → 病院)
        transport_time_sec = self._get_travel_time_by_phase(
            call['h3_index'], 
            hospital_h3, 
            'transport'  # ★フェーズ指定
        )
        
        # Phase 5: Hospital (病院での活動)
        hospital_time_min = self.service_time_generator.generate_time(
            call['severity'], 
            'hospital_time',
            current_time_obj
        )
        hospital_time_sec = hospital_time_min * 60.0
        
        # Phase 6: Return (病院 → 基地)
        return_time_sec = self._get_travel_time_by_phase(
            hospital_h3, 
            amb_state['station_h3'], 
            'return'  # ★フェーズ指定
        )
        
        # 合計時間
        total_time_sec = (
            response_time_sec + 
            on_scene_time_sec + 
            transport_time_sec + 
            hospital_time_sec + 
            return_time_sec
        )
        
        # 詳細情報
        details = {
            'response_time': response_time_sec / 60.0,  # 分単位
            'on_scene_time': on_scene_time_min,
            'transport_time': transport_time_sec / 60.0,
            'hospital_time': hospital_time_min,
            'return_time': return_time_sec / 60.0,
            'total_time': total_time_sec / 60.0,
            'hospital_h3': hospital_h3
        }
        
        return total_time_sec, details
    
    def _get_travel_time_by_phase(
        self, 
        from_h3: str, 
        to_h3: str, 
        phase: str
    ) -> float:
        """
        フェーズ別移動時間行列から移動時間を取得
        
        ValidationSimulatorと同じロジック:
        - response: 出場時の移動時間
        - transport: 搬送時の移動時間
        - return: 帰署時の移動時間
        
        Args:
            from_h3: 出発地H3インデックス
            to_h3: 目的地H3インデックス
            phase: 'response', 'transport', 'return'
        
        Returns:
            移動時間（秒）
        """
        # H3インデックスをマトリックスインデックスに変換
        from_idx = self.grid_mapping.get(from_h3)
        to_idx = self.grid_mapping.get(to_h3)
        
        if from_idx is None or to_idx is None:
            # フォールバック: 10分
            return 600.0
        
        # フェーズに応じた行列を選択
        if phase not in self.travel_time_matrices:
            print(f"警告: フェーズ'{phase}'の移動時間行列が見つかりません")
            phase = 'response'  # デフォルトにフォールバック
        
        matrix = self.travel_time_matrices[phase]
        time_sec = matrix[from_idx, to_idx]
        
        # 異常値チェック
        if time_sec <= 0 or time_sec > 7200:  # 2時間以上は異常
            return 600.0
        
        return time_sec
    
    # ========================================================================
    # Phase 3: PPO学習インターフェース
    # ========================================================================
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        PPO学習用のステップ実行
        
        設計:
        1. actionで指定された救急車を配車
        2. 復帰イベントをスケジュール
        3. 次の事案イベントまでシミュレーションを進める
        4. 観測・報酬・終了フラグを返す
        
        Args:
            action: 選択された救急車のID
        
        Returns:
            (observation, reward, done, info)
        """
        if self.pending_call is None:
            # 次のイベントまで進める
            self._advance_to_next_call()
        
        # 事案が存在しない場合（エピソード終了）
        if self.pending_call is None:
            observation = self._get_observation()
            return observation, 0.0, True, {}
        
        # 配車の実行
        reward, info = self._execute_dispatch(action, self.pending_call)
        
        # 次の事案まで進める
        self._advance_to_next_call()
        
        # 観測の取得
        observation = self._get_observation()
        done = self._is_episode_done()
        
        return observation, reward, done, info
    
    def _execute_dispatch(self, action: int, call: Dict) -> Tuple[float, Dict]:
        """
        救急車の配車を実行し、復帰イベントをスケジュール
        
        Args:
            action: 救急車ID
            call: 事案情報
        
        Returns:
            (報酬, 情報辞書)
        """
        # アクションマスクのチェック
        mask = self.get_action_mask()
        if not mask[action]:
            # 無効なアクション: 大きなペナルティ
            return -100.0, {'success': False, 'reason': 'invalid_action'}
        
        # 活動時間の詳細計算
        total_time_sec, details = self._calculate_ambulance_activity_time(
            action, call
        )
        
        # 救急車を出動中状態に更新
        self.ambulance_states[action]['status'] = 'dispatched'
        self.ambulance_states[action]['calls_today'] += 1
        
        # 復帰イベントをスケジュール
        return_time = self.current_time + total_time_sec
        return_event = Event(
            time=return_time,
            event_type=EventType.AMBULANCE_RETURN,
            data={'ambulance_id': action}
        )
        self._schedule_event(return_event)
        
        # 報酬の計算
        reward = self._calculate_reward(details['response_time'], call['severity'])
        
        # 統計の更新
        self._update_statistics(details)
        
        info = {
            'success': True,
            'ambulance_id': action,
            'response_time': details['response_time'],
            'total_time': details['total_time'],
            'severity': call['severity']
        }
        
        return reward, info
    
    def _advance_to_next_call(self):
        """
        次の事案イベントまでシミュレーションを進める
        
        重要な処理:
        1. 事案間で発生する救急車復帰イベントを全て処理
        2. 次のNEW_CALLイベントでpending_callを更新
        """
        self.pending_call = None
        
        # 次のNEW_CALLイベントまでイベントを処理
        while self.event_queue:
            next_event = self.event_queue[0]  # peekのみ
            
            if next_event.event_type == EventType.NEW_CALL:
                # 次の事案に到達
                self._process_next_event()
                break
            elif next_event.event_type == EventType.EPISODE_END:
                # エピソード終了
                self._process_next_event()
                break
            else:
                # 救急車復帰などの中間イベントを処理
                self._process_next_event()
    
    def reset(self) -> np.ndarray:
        """
        エピソードのリセット
        
        処理:
        1. イベントキューのクリア
        2. 救急車状態の初期化
        3. 事案データの読み込み
        4. 全事案をイベントとしてスケジュール
        
        Returns:
            初期観測ベクトル
        """
        # イベントキューとタイマーのリセット
        self.event_queue = []
        self.current_time = 0.0
        
        # 期間の選択
        periods = (self.config['data']['train_periods'] if self.mode == "train" 
                  else self.config['data']['eval_periods'])
        period = periods[np.random.randint(len(periods))]
        
        # 事案データの読み込み
        calls_df = self._load_calls_for_period(period)
        episode_calls = self._prepare_episode_calls(calls_df)
        
        if not episode_calls:
            print("警告: 事案データが空です")
            return np.zeros(self.state_dim)
        
        self.episode_start_time = episode_calls[0]['datetime']
        
        # 全事案をイベントとしてスケジュール
        for call in episode_calls:
            event_time = (call['datetime'] - self.episode_start_time).total_seconds()
            event = Event(
                time=event_time,
                event_type=EventType.NEW_CALL,
                data=call
            )
            self._schedule_event(event)
        
        # エピソード終了イベント
        episode_duration_sec = self.config['data']['episode_duration_hours'] * 3600
        end_event = Event(
            time=episode_duration_sec,
            event_type=EventType.EPISODE_END,
            data={}
        )
        self._schedule_event(end_event)
        
        # 救急車の初期化
        self._initialize_ambulances_realistic()
        
        # 統計のリセット
        self._reset_statistics()
        
        # 最初の事案まで進める
        self._advance_to_next_call()
        
        return self._get_observation()
    
    def get_action_mask(self) -> np.ndarray:
        """
        利用可能な救急車のマスクを返す
        
        Returns:
            boolean配列（True=利用可能）
        """
        mask = np.zeros(self.action_dim, dtype=bool)
        
        for amb_id, state in self.ambulance_states.items():
            if amb_id < self.action_dim and state['status'] == 'available':
                mask[amb_id] = True
        
        return mask
    
    # ========================================================================
    # ヘルパーメソッド
    # ========================================================================
    
    def _load_config(self, config_path: str) -> Dict:
        """設定ファイルの読み込み"""
        from ..config_utils import load_config_with_inheritance
        return load_config_with_inheritance(config_path)
    
    def _load_travel_time_matrices(self):
        """フェーズ別移動時間行列の読み込み"""
        # 実装は既存のコードを使用
        pass
    
    def _load_service_time_generator(self):
        """サービス時間生成器の初期化"""
        # 実装は既存のコードを使用
        pass
    
    def _load_hospital_selection_model(self):
        """病院選択モデルの読み込み"""
        # 実装は既存のコードを使用
        pass
    
    def _initialize_ambulances_realistic(self):
        """救急車の現実的な初期配置"""
        # 実装は既存のコードを使用
        pass
    
    def _get_observation(self) -> np.ndarray:
        """現在の状態を観測ベクトルに変換"""
        # StateEncoderを使用
        state_dict = {
            'ambulances': self.ambulance_states,
            'pending_call': self.pending_call,
            'current_time': self.current_time,  # ★連続時間を渡す
            'time_of_day': int(self.current_time / 3600) % 24
        }
        return self.state_encoder.encode_state(state_dict)
    
    def _calculate_reward(self, response_time_min: float, severity: str) -> float:
        """報酬の計算"""
        # RewardDesignerを使用
        return self.reward_designer.calculate_reward({
            'response_time': response_time_min,
            'severity': severity
        })
    
    def _update_statistics(self, details: Dict):
        """統計情報の更新"""
        # 既存のコードを使用
        pass
    
    def _reset_statistics(self):
        """統計情報のリセット"""
        # 既存のコードを使用
        pass
    
    def _is_episode_done(self) -> bool:
        """エピソード終了判定"""
        # 事案がなく、かつエピソード終了イベントを処理済み
        return self.pending_call is None and not any(
            e.event_type == EventType.NEW_CALL for e in self.event_queue
        )


# ============================================================================
# 使用例
# ============================================================================

if __name__ == "__main__":
    # 環境の作成
    env = ValidationIntegratedEMSEnvironment(
        config_path="reinforcement_learning/experiments/config_continuous.yaml",
        mode="train"
    )
    
    # エピソードの実行
    obs = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    
    while not done:
        # アクションマスクの取得
        action_mask = env.get_action_mask()
        
        # ランダムアクション（マスク考慮）
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            break
        action = np.random.choice(valid_actions)
        
        # ステップ実行
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
    
    print(f"エピソード完了: {step_count}ステップ, 合計報酬: {total_reward:.2f}")

