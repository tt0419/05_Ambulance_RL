"""
dispatch_strategies.py
救急隊ディスパッチ戦略の実装

このモジュールは様々なディスパッチ戦略を実装し、
validation_simulation.pyと統合して使用される。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import numpy as np
import h3
from collections import defaultdict
import json
import yaml
import torch
import sys
import os

# 統一された傷病度定数をインポート
from constants import (
    SEVERITY_GROUPS, SEVERITY_PRIORITY, SEVERITY_TIME_LIMITS,
    is_severe_condition, is_mild_condition, get_severity_time_limit
)

# PPOエージェントと関連モジュールをインポート
# プロジェクトのルートパスを一時的に追加して、RLモジュールをインポート
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    from reinforcement_learning.agents.ppo_agent import PPOAgent
    from reinforcement_learning.environment.state_encoder import StateEncoder
    PPO_AVAILABLE = True
except ImportError as e:
    print(f"警告: PPOモジュールのインポートに失敗しました: {e}")
    PPO_AVAILABLE = False


class DispatchPriority(Enum):
    """緊急度優先度（数値が小さいほど緊急度が高い）"""
    CRITICAL = 2  # 重篤
    HIGH = 3      # 重症
    MEDIUM = 4    # 中等症
    FATAL = 1     # 死亡（最優先）
    LOW = 5       # 軽症

@dataclass
class EmergencyRequest:
    """救急要請データクラス"""
    id: str
    h3_index: str
    severity: str
    time: float
    priority: DispatchPriority
    call_datetime: Optional[Any] = None
    
    def get_urgency_score(self) -> float:
        """緊急度スコアを計算（低い値ほど緊急）"""
        return self.priority.value

@dataclass
class AmbulanceInfo:
    """救急車情報データクラス"""
    id: str
    current_h3: str
    station_h3: str
    status: str
    last_call_time: Optional[float] = None
    total_calls_today: int = 0
    current_workload: float = 0.0

class DispatchContext:
    """ディスパッチ決定時のコンテキスト情報"""
    def __init__(self):
        self.current_time: float = 0.0
        self.hour_of_day: int = 0
        self.total_ambulances: int = 0
        self.available_ambulances: int = 0
        self.recent_call_density: Dict[str, float] = {}
        self.grid_mapping: Dict[str, int] = {}
        self.all_h3_indices: Set[str] = set()
        # ★★★ PPO戦略用の属性を追加 ★★★
        self.all_ambulances: Dict[str, Any] = {}  # 全救急車の状態情報

class DispatchStrategy(ABC):
    """ディスパッチ戦略の抽象基底クラス"""
    
    def __init__(self, name: str, strategy_type: str):
        self.name = name
        self.strategy_type = strategy_type
        self.metrics = {}
        self.config = {}
        
    @abstractmethod
    def select_ambulance(self, 
                        request: EmergencyRequest,
                        available_ambulances: List[AmbulanceInfo],
                        travel_time_func: callable,
                        context: DispatchContext) -> Optional[AmbulanceInfo]:
        """救急車を選択する"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict):
        """戦略固有の初期化"""
        pass
    
    def requires_training(self) -> bool:
        """学習が必要かどうか"""
        return self.strategy_type in ['reinforcement_learning', 'optimization']
    
    def get_severity_priority(self, severity: str) -> DispatchPriority:
        """傷病度から優先度を取得（統一された定数を使用）"""
        severity_map = {
            '重篤': DispatchPriority.CRITICAL,
            '重症': DispatchPriority.HIGH,
            '中等症': DispatchPriority.MEDIUM,
            '死亡': DispatchPriority.FATAL,  # 死亡は最優先
            '軽症': DispatchPriority.LOW
        }
        return severity_map.get(severity, DispatchPriority.LOW)

class ClosestAmbulanceStrategy(DispatchStrategy):
    """最寄り救急車戦略（現行）"""
    
    def __init__(self):
        super().__init__("closest", "rule_based")
        
    def initialize(self, config: Dict):
        """初期化（特に設定なし）"""
        self.config = config
        
    def select_ambulance(self,
                        request: EmergencyRequest,
                        available_ambulances: List[AmbulanceInfo],
                        travel_time_func: callable,
                        context: DispatchContext) -> Optional[AmbulanceInfo]:
        """最も近い救急車を選択"""
        if not available_ambulances:
            return None
            
        min_time = float('inf')
        closest_ambulance = None
        
        for ambulance in available_ambulances:
            travel_time = travel_time_func(ambulance.current_h3, request.h3_index, 'response')
            if travel_time < min_time:
                min_time = travel_time
                closest_ambulance = ambulance
                
        return closest_ambulance

class SeverityBasedStrategy(DispatchStrategy):
    """傷病度考慮型戦略"""
    
    def __init__(self):
        super().__init__("severity_based", "rule_based")
        # 統一された定数を使用
        self.severe_conditions = SEVERITY_GROUPS['severe_conditions']
        self.mild_conditions = SEVERITY_GROUPS['mild_conditions']
        self.coverage_radius_km = 5.0
        self.time_threshold_6min = 360
        self.time_threshold_13min = 780
        
        # デフォルトのパラメータをここで定義
        self.time_score_weight = 0.6  # デフォルトは60%
        self.coverage_loss_weight = 0.4 # デフォルトは40%
        self.mild_time_limit_sec = SEVERITY_TIME_LIMITS['軽症']  # 統一された定数を使用
        self.moderate_time_limit_sec = SEVERITY_TIME_LIMITS['中等症']  # 統一された定数を使用
        
    def initialize(self, config: Dict):
        """戦略の初期化"""
        self.config = config
        # カスタム設定があれば上書き
        if 'coverage_radius_km' in config:
            self.coverage_radius_km = config['coverage_radius_km']
        if 'severe_conditions' in config:
            self.severe_conditions = config['severe_conditions']
        if 'mild_conditions' in config:
            self.mild_conditions = config['mild_conditions']
        
        #重みパラメータと時間制限の設定
        self.time_score_weight = config.get('time_score_weight', self.time_score_weight)
        self.coverage_loss_weight = config.get('coverage_loss_weight', self.coverage_loss_weight)
        self.mild_time_limit_sec = config.get('mild_time_limit_sec', self.mild_time_limit_sec)
        self.moderate_time_limit_sec = config.get('moderate_time_limit_sec', self.moderate_time_limit_sec)
    
    def select_ambulance(self,
                        request: EmergencyRequest,
                        available_ambulances: List[AmbulanceInfo],
                        travel_time_func: callable,
                        context: DispatchContext) -> Optional[AmbulanceInfo]:
        """傷病度に応じた救急車選択"""
        if not available_ambulances:
            return None
        
        # 重症系の場合は最寄りを選択
        if is_severe_condition(request.severity):
            return self._select_closest(request, available_ambulances, travel_time_func)
        
        # 軽症系の場合はカバレッジを考慮
        return self._select_with_coverage(request, available_ambulances, travel_time_func, context)
    
    def _select_closest(self,
                       request: EmergencyRequest,
                       available_ambulances: List[AmbulanceInfo],
                       travel_time_func: callable) -> Optional[AmbulanceInfo]:
        """最寄りの救急車を選択"""
        min_time = float('inf')
        closest_ambulance = None
        
        for ambulance in available_ambulances:
            travel_time = travel_time_func(ambulance.current_h3, request.h3_index, 'response')
            if travel_time < min_time:
                min_time = travel_time
                closest_ambulance = ambulance
        
        return closest_ambulance
    
    def _select_with_coverage(self,
                            request: EmergencyRequest,
                            available_ambulances: List[AmbulanceInfo],
                            travel_time_func: callable,
                            context: DispatchContext) -> Optional[AmbulanceInfo]:
        """カバレッジを考慮した救急車選択（修正版）"""
        
        # ===== 修正箇所1: 傷病度別の時間制限 =====
        # 元のコード:
        # candidates = []
        # for amb in available_ambulances:
        #     travel_time = travel_time_func(amb.current_h3, request.h3_index, 'response')
        #     if travel_time <= self.time_threshold_13min:
        #         candidates.append((amb, travel_time))
        
        # 傷病度に応じて制限時間を設定（統一された定数を使用）
        time_limit = get_severity_time_limit(request.severity)
        
        candidates = []
        for amb in available_ambulances:
            travel_time = travel_time_func(amb.current_h3, request.h3_index, 'response')
            if travel_time <= time_limit:
                candidates.append((amb, travel_time))
        

        
        # 13分以内の候補がない場合は最寄りを選択
        if not candidates:
            return self._select_closest(request, available_ambulances, travel_time_func)
        
        # ===== 修正箇所3: スコア計算の重み調整 =====
        best_ambulance = None
        best_score = float('inf')
        
        for amb, travel_time in candidates:
            # カバレッジ損失を計算
            coverage_loss = self._calculate_coverage_loss(
                amb, available_ambulances, travel_time_func, context
            )
            
            # 元のコード:
            # time_score = travel_time / self.time_threshold_13min
            # combined_score = time_score * 0.4 + coverage_loss * 0.6
            
            # 複合スコア（重みは外部設定可能）
            # 応答時間は13分で正規化
            time_score = travel_time / self.time_threshold_13min
            combined_score = (time_score * self.time_score_weight + 
                              coverage_loss * self.coverage_loss_weight)  # ★修正: 外部設定可能に
            
            if combined_score < best_score:
                best_score = combined_score
                best_ambulance = amb
        
        return best_ambulance
    
    def _calculate_coverage_loss(self,
                                ambulance: AmbulanceInfo,
                                all_available: List[AmbulanceInfo],
                                travel_time_func: callable,
                                context: DispatchContext) -> float:
        """
        救急車が出動した場合のカバレッジ損失を計算
        
        Returns:
            float: 0-1の範囲の損失スコア（高いほど損失大）
        """
        # その救急車を除いた利用可能な救急車リスト
        remaining_ambulances = [amb for amb in all_available if amb.id != ambulance.id]
        
        if not remaining_ambulances:
            return 1.0  # 他に救急車がない場合は最大損失
        
        # 簡易的なカバレッジ計算
        # 救急車の担当エリア周辺のグリッドをサンプリング
        coverage_points = self._get_coverage_sample_points(ambulance.station_h3, context)
        
        if not coverage_points:
            # サンプルポイントが取得できない場合は、近隣救急車数ベースの簡易計算
            nearby_count = self._count_nearby_ambulances(
                ambulance.station_h3, remaining_ambulances, travel_time_func
            )
            # 近隣救急車が多いほど損失は小さい
            return 1.0 / (nearby_count + 1)
        
        # 6分・13分カバレッジへの影響を計算
        coverage_6min_before = 0
        coverage_13min_before = 0
        coverage_6min_after = 0
        coverage_13min_after = 0
        
        for point_h3 in coverage_points:
            # 現在の状態でのカバレッジ
            min_time_before = self._get_min_response_time(
                point_h3, all_available, travel_time_func
            )
            if min_time_before <= self.time_threshold_6min:
                coverage_6min_before += 1
            if min_time_before <= self.time_threshold_13min:
                coverage_13min_before += 1
            
            # 救急車が出動した後のカバレッジ
            min_time_after = self._get_min_response_time(
                point_h3, remaining_ambulances, travel_time_func
            )
            if min_time_after <= self.time_threshold_6min:
                coverage_6min_after += 1
            if min_time_after <= self.time_threshold_13min:
                coverage_13min_after += 1
        
        # カバレッジ率の変化を計算
        total_points = len(coverage_points)
        if total_points == 0:
            return 0.5  # デフォルト値
        
        # 6分カバレッジと13分カバレッジの損失を重み付け合成
        loss_6min = (coverage_6min_before - coverage_6min_after) / total_points
        loss_13min = (coverage_13min_before - coverage_13min_after) / total_points
        
        # 6分カバレッジの損失により重みを置く
        combined_loss = loss_6min * 0.5 + loss_13min * 0.5 #v2 loss_6min0.7, loss_13min0.3から変更
        
        # 0-1の範囲にクリップ
        return max(0.0, min(1.0, combined_loss))
    
    def _get_coverage_sample_points(self,
                                   center_h3: str,
                                   context: DispatchContext,
                                   sample_size: int = 20) -> List[str]:
        """カバレッジ計算用のサンプルポイントを取得"""
        try:
            # 中心から2リング以内のグリッドを取得
            nearby_grids = h3.grid_disk(center_h3, 2)
            
            # context.grid_mappingに存在するグリッドのみを使用
            valid_grids = [g for g in nearby_grids if g in context.grid_mapping]
            
            # サンプルサイズを調整
            if len(valid_grids) <= sample_size:
                return valid_grids
            
            # ランダムサンプリング
            import random
            return random.sample(valid_grids, sample_size)
            
        except Exception:
            # エラーの場合は空リストを返す
            return []
    
    def _count_nearby_ambulances(self,
                                station_h3: str,
                                ambulances: List[AmbulanceInfo],
                                travel_time_func: callable,
                                threshold_time: float = 600) -> int:
        """近隣の救急車数をカウント（10分以内）"""
        count = 0
        for amb in ambulances:
            travel_time = travel_time_func(amb.current_h3, station_h3, 'response')
            if travel_time <= threshold_time:
                count += 1
        return count
    
    def _get_min_response_time(self,
                              target_h3: str,
                              ambulances: List[AmbulanceInfo],
                              travel_time_func: callable) -> float:
        """指定地点への最小応答時間を取得"""
        if not ambulances:
            return float('inf')
        
        min_time = float('inf')
        for amb in ambulances:
            travel_time = travel_time_func(amb.current_h3, target_h3, 'response')
            if travel_time < min_time:
                min_time = travel_time
        
        return min_time

"""
advanced_severity_strategy.py
重症系を強力に優先する高度な傷病度考慮戦略
"""

class AdvancedSeverityStrategy(SeverityBasedStrategy):
    """高度な傷病度優先戦略：重症系への強い優先度付け"""
    
    def __init__(self):
        super().__init__()
        # 親クラスの設定を上書き
        self.name = "advanced_severity"
        self.strategy_type = "rule_based"
        
        # 傷病度カテゴリ（統一された定数を使用）
        self.critical_conditions = SEVERITY_GROUPS['critical_conditions']  # 最優先
        self.severe_conditions = SEVERITY_GROUPS['severe_conditions']  # 高優先
        self.moderate_conditions = SEVERITY_GROUPS['moderate_conditions']  # 中優先
        self.mild_conditions = SEVERITY_GROUPS['mild_conditions']  # 低優先
        
        # 戦略パラメータ
        self.params = {
            # 重篤・重症用
            'critical_search_radius': 480,  # 8分以内の救急車を全て考慮
            'severe_search_radius': 540,    # 9分以内の救急車を考慮
            
            # 中等症用
            'moderate_time_limit': 900,     # 15分制限
            'moderate_coverage_weight': 0.3, # カバレッジ重視度を下げる
            
            # 軽症用
            'mild_time_limit': 900,        # 15分制限（大幅緩和）
            'mild_coverage_weight': 0.2,    # カバレッジ最小限
            'mild_delay_threshold': 480,    # 8分以上かかる救急車を積極利用
            
            # 繁忙期判定
            'high_utilization': 0.75,       # 65%で繁忙期判定（早めに切り替え）
            'critical_utilization': 0.85,   # 80%で緊急モード
        }
        
    def select_ambulance(self,
                        request: EmergencyRequest,
                        available_ambulances: List[AmbulanceInfo],
                        travel_time_func: callable,
                        context: DispatchContext) -> Optional[AmbulanceInfo]:
        """傷病度に応じた差別化された救急車選択"""
        
        if not available_ambulances:
            return None
        
        # 稼働率を計算
        utilization = self._calculate_utilization_rate(context)
        
        # 傷病度別の処理
        if request.severity in self.critical_conditions:
            # 重篤：直近隊
            return self._get_closest(request, available_ambulances, travel_time_func)
        elif request.severity in self.severe_conditions:
            # 重症・死亡：直近隊
            return self._get_closest(request, available_ambulances, travel_time_func)
        elif request.severity in self.moderate_conditions:
            # 中等症：稼働率による分岐
            if utilization > 0.75:
                return self._get_closest(request, available_ambulances, travel_time_func)
            else:
                return self._select_with_coverage(request, available_ambulances, travel_time_func, context)
        else:  # 軽症
            # 軽症：稼働率による分岐
            if utilization > 0.75:
                return self._get_closest(request, available_ambulances, travel_time_func)
            else:
                return self._select_with_coverage(request, available_ambulances, travel_time_func, context)
        
    def _select_for_critical(self, request, ambulances, travel_time_func, utilization):
        """
        重篤用：最速到着を絶対優先
        複数の近い救急車から最適を選択
        """
        candidates = []
        for amb in ambulances:
            travel_time = travel_time_func(amb.current_h3, request.h3_index, 'response')
            candidates.append((amb, travel_time))
        
        candidates.sort(key=lambda x: x[1])
        
        # 重篤は常に最速
        return candidates[0][0] if candidates else None
    
    def _select_for_severe(self, request, ambulances, travel_time_func, utilization):
        """
        重症・死亡用：準最適解を許容
        7分以内の救急車から、次の影響が最小のものを選択
        """
        candidates = []
        for amb in ambulances:
            travel_time = travel_time_func(amb.current_h3, request.h3_index, 'response')
            if travel_time <= self.params['severe_search_radius']:
                candidates.append((amb, travel_time))
        
        if not candidates:
            # 7分以内がなければ最寄り
            return self._get_closest(request, ambulances, travel_time_func)
        
        # 繁忙期は最速、平常期は2番目まで考慮
        if utilization > self.params['critical_utilization']:
            return candidates[0][0]
        
        # 最速から15%以内の範囲で、出動回数が少ない救急車を優先
        fastest_time = candidates[0][1]
        threshold = fastest_time * 1.15
        
        best_amb = candidates[0][0]
        best_score = candidates[0][0].total_calls_today + candidates[0][1] / 60
        
        for amb, travel_time in candidates[:3]:  # 上位3台のみ
            if travel_time <= threshold:
                score = amb.total_calls_today + travel_time / 60
                if score < best_score:
                    best_score = score
                    best_amb = amb
        
        return best_amb
    
    def _select_for_moderate(self, request, ambulances, travel_time_func, utilization, context):
        """中等症: SeverityBasedと同じカバレッジ考慮ロジック"""
        return self._select_with_coverage(request, ambulances, travel_time_func, context)
    
    def _select_for_mild(self, request, ambulances, travel_time_func, utilization, context):
        """軽症: SeverityBasedと同じカバレッジ考慮ロジック"""
        return self._select_with_coverage(request, ambulances, travel_time_func, context)
    
    def _select_with_coverage(self, request, available_ambulances, travel_time_func, context):
        """SeverityBasedStrategyと同じカバレッジ考慮ロジック"""
        # SeverityBasedStrategyの_select_with_coverageメソッドの内容をそのままコピー
        time_limit = get_severity_time_limit(request.severity)
        
        candidates = []
        for amb in available_ambulances:
            travel_time = travel_time_func(amb.current_h3, request.h3_index, 'response')
            if travel_time <= time_limit:
                candidates.append((amb, travel_time))
        
        if not candidates:
            return self._get_closest(request, available_ambulances, travel_time_func)
        
        best_ambulance = None
        best_score = float('inf')
        
        for amb, travel_time in candidates:
            coverage_loss = self._calculate_coverage_loss(
                amb, available_ambulances, travel_time_func, context
            )
            
            time_score = travel_time / 780  # 13分で正規化
            combined_score = time_score * 0.6 + coverage_loss * 0.4
            
            if combined_score < best_score:
                best_score = combined_score
                best_ambulance = amb
        
        return best_ambulance
    
    def _calculate_utilization_rate(self, context: DispatchContext) -> float:
        """稼働率計算"""
        if context.total_ambulances == 0:
            return 1.0
        return 1.0 - (context.available_ambulances / context.total_ambulances)
    
    def _get_closest(self, request, ambulances, travel_time_func):
        """最寄りの救急車を取得"""
        min_time = float('inf')
        closest = None
        for amb in ambulances:
            travel_time = travel_time_func(amb.current_h3, request.h3_index, 'response')
            if travel_time < min_time:
                min_time = travel_time
                closest = amb
        return closest
    
    def _count_nearby_available(self, ambulance, all_ambulances, travel_time_func):
        """近隣の利用可能救急車数をカウント"""
        count = 0
        for amb in all_ambulances:
            if amb.id != ambulance.id:
                travel_time = travel_time_func(
                    amb.current_h3, ambulance.station_h3, 'response'
                )
                if travel_time <= 600:  # 10分以内
                    count += 1
        return count
    
    def initialize(self, config: Dict):
        """戦略固有の初期化"""
        # デフォルト設定を更新
        if config:
            for key, value in config.items():
                if key in self.params:
                    self.params[key] = value
                else:
                    # 新しいパラメータを追加
                    self.params[key] = value


# パラメータ調整用の設定辞書
STRATEGY_CONFIGS = {
    "conservative": {
        # 保守的設定（v2相当）
        'mild_time_limit': 900,  # 15分
        'mild_delay_threshold': 480,  # 8分
        'high_utilization': 0.7,
    },
    "aggressive": {
        # 積極的設定（推奨）
        'mild_time_limit': 1080,  # 18分
        'mild_delay_threshold': 600,  # 10分
        'high_utilization': 0.65,
        'moderate_time_limit': 900,  # 15分
    },
    "extreme": {
        # 極端設定（実験用）
        'mild_time_limit': 1200,  # 20分
        'mild_delay_threshold': 720,  # 12分
        'high_utilization': 0.6,
        'moderate_time_limit': 1080,  # 18分
    },
    # ← 以下を追加
    "second_ride_default": {
        # デフォルト設定（2番目選択、時間制限なし）
        'alternative_rank': 2,
        'enable_time_limit': False,
        'time_limit_seconds': 780
    },
    "second_ride_conservative": {
        # 保守的設定（2番目選択、13分制限あり）
        'alternative_rank': 2,
        'enable_time_limit': True,
        'time_limit_seconds': 780
    },
    "second_ride_aggressive": {
        # 積極的設定（3番目選択、時間制限なし）
        'alternative_rank': 3,
        'enable_time_limit': False,
        'time_limit_seconds': 780
    },
    "second_ride_time_limited": {
        # 時間制限設定（2番目選択、10分制限）
        'alternative_rank': 2,
        'enable_time_limit': True,
        'time_limit_seconds': 600
    }
}

# ★★★【追加箇所②】★★★
# SecondRideStrategy クラスを追加
class SecondRideStrategy(DispatchStrategy):
    """
    2番目優先配車戦略
    - 軽症系（軽症・中等症）: 2番目に近い救急車を配車
    - 重症系（重症・重篤・死亡）: 最寄りの救急車を配車（従来通り）
    """
    
    def __init__(self):
        super().__init__("second_ride", "rule_based")
        
        # 傷病度分類（統一された定数を使用）
        self.severe_conditions = SEVERITY_GROUPS['severe_conditions']
        self.mild_conditions = SEVERITY_GROUPS['mild_conditions']
        
        # デフォルトパラメータ
        self.alternative_rank = 2  # 軽症系で選択する順位（2番目）
        self.enable_time_limit = True  # 13分制限機能（デフォルトオフ）
        self.time_limit_seconds = 780  # 13分制限の閾値（秒）
        
    def initialize(self, config: Dict):
        """戦略の初期化"""
        self.config = config
        
        # 設定可能パラメータの読み込み
        self.alternative_rank = config.get('alternative_rank', self.alternative_rank)
        self.enable_time_limit = config.get('enable_time_limit', self.enable_time_limit)
        self.time_limit_seconds = config.get('time_limit_seconds', self.time_limit_seconds)
        
        # パラメータの妥当性チェック
        if self.alternative_rank < 1:
            print(f"警告: alternative_rank ({self.alternative_rank}) は1以上である必要があります。デフォルト値2を使用します。")
            self.alternative_rank = 2
            
        if self.time_limit_seconds <= 0:
            print(f"警告: time_limit_seconds ({self.time_limit_seconds}) は正の値である必要があります。デフォルト値780秒を使用します。")
            self.time_limit_seconds = 780
    
    def select_ambulance(self,
                        request: EmergencyRequest,
                        available_ambulances: List[AmbulanceInfo],
                        travel_time_func: callable,
                        context: DispatchContext) -> Optional[AmbulanceInfo]:
        """救急車を選択する"""
        
        if not available_ambulances:
            return None
        
        # 傷病度による戦略分岐
        if request.severity in self.severe_conditions:
            # 重症系: 最寄りの救急車を選択（従来通り）
            return self._select_closest(request, available_ambulances, travel_time_func)
        elif request.severity in self.mild_conditions:
            # 軽症系: 2番目に近い救急車を選択
            return self._select_alternative_rank(request, available_ambulances, travel_time_func)
        else:
            # その他の傷病度: 最寄りを選択（フォールバック）
            return self._select_closest(request, available_ambulances, travel_time_func)
    
    def _select_closest(self,
                       request: EmergencyRequest,
                       available_ambulances: List[AmbulanceInfo],
                       travel_time_func: callable) -> Optional[AmbulanceInfo]:
        """最寄りの救急車を選択"""
        
        min_time = float('inf')
        closest_ambulance = None
        
        for ambulance in available_ambulances:
            travel_time = travel_time_func(ambulance.current_h3, request.h3_index, 'response')
            if travel_time < min_time:
                min_time = travel_time
                closest_ambulance = ambulance
        
        return closest_ambulance
    
    def _select_alternative_rank(self,
                               request: EmergencyRequest,
                               available_ambulances: List[AmbulanceInfo],
                               travel_time_func: callable) -> Optional[AmbulanceInfo]:
        """指定順位の救急車を選択（軽症系用）"""
        
        # 利用可能な救急車が指定順位未満の場合は最寄りを選択
        if len(available_ambulances) < self.alternative_rank:
            return self._select_closest(request, available_ambulances, travel_time_func)
        
        # 各救急車の移動時間を計算してソート
        ambulance_times = []
        for ambulance in available_ambulances:
            travel_time = travel_time_func(ambulance.current_h3, request.h3_index, 'response')
            ambulance_times.append((ambulance, travel_time))
        
        # 移動時間の昇順でソート
        ambulance_times.sort(key=lambda x: x[1])
        
        # 指定順位の救急車を取得（インデックスは0ベースなので-1）
        target_ambulance, target_time = ambulance_times[self.alternative_rank - 1]
        
        # 13分制限機能がオンの場合の処理
        if self.enable_time_limit:
            if target_time > self.time_limit_seconds:
                # 13分を超える場合は最寄りを選択
                return ambulance_times[0][0]  # 最寄り（1番目）を選択
        
        return target_ambulance
    
    def get_strategy_info(self) -> Dict:
        """戦略の情報を取得"""
        return {
            'name': self.name,
            'strategy_type': self.strategy_type,
            'alternative_rank': self.alternative_rank,
            'enable_time_limit': self.enable_time_limit,
            'time_limit_seconds': self.time_limit_seconds,
            'time_limit_minutes': self.time_limit_seconds / 60.0,
            'severe_conditions': self.severe_conditions,
            'mild_conditions': self.mild_conditions
        }

# PPOエージェントを戦略として使用する新しいクラス
class PPOStrategy(DispatchStrategy):
    """学習済みPPOエージェントを使用する戦略"""
    
    def __init__(self):
        super().__init__("ppo_agent", "reinforcement_learning")
        self.agent = None
        self.state_encoder = None
        self.config = None
        
    def initialize(self, config: Dict):
        """学習済みPPOエージェントをロードして初期化する"""
        if not PPO_AVAILABLE:
            raise ImportError("PPOモジュールが利用できません。reinforcement_learningパッケージを確認してください。")
        
        print("PPO戦略を初期化中...")
        
        model_path = config.get('model_path')
        config_path = config.get('config_path')
        if not model_path or not config_path:
            raise ValueError("PPOStrategyには 'model_path' と 'config_path' が必要です。")
        
        # 設定ファイルの読み込み
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        except Exception as e:
            raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}")
        
        # StateEncoderの初期化に必要なデータを準備
        data_paths_config = self.config.get('data_paths', {})
        if not data_paths_config:
            raise ValueError(f"config.yamlに'data_paths'セクションが見つかりません。ファイル: {config_path}")
        
        # グリッドマッピングの読み込み
        grid_mapping_path = data_paths_config.get('grid_mapping')
        if not grid_mapping_path:
            raise ValueError("config.yamlに'grid_mapping'パスが見つかりません。")
        
        with open(grid_mapping_path, 'r', encoding='utf-8') as f:
            grid_mapping = json.load(f)
        
        # 移動時間行列の読み込み
        matrix_path = data_paths_config.get('travel_time_matrix')
        if not matrix_path:
            raise ValueError("config.yamlに'travel_time_matrix'パスが見つかりません。")
        
        travel_time_matrix = np.load(matrix_path)
        
        # ★★★ エリア制限の確認と次元数の設定（正しいパスから取得）★★★
        data_config = self.config.get('data', {})
        area_restriction = data_config.get('area_restriction', {})
        if area_restriction.get('enabled', False):
            # 設定ファイルで明示的に定義された次元数を使用
            action_dim = area_restriction.get('action_dim', area_restriction.get('num_ambulances_in_area', 25))
            state_dim = area_restriction.get('state_dim', None)
            print(f"エリア制限有効: 行動次元={action_dim}, 状態次元={state_dim}")
        else:
            action_dim = 192  # 全体
            state_dim = None
        
        # StateEncoderを初期化
        self.state_encoder = StateEncoder(
            config=self.config,
            max_ambulances=action_dim,
            travel_time_matrix=travel_time_matrix,
            grid_mapping=grid_mapping
        )
        
        # ★★★ 状態次元数の決定（設定ファイル優先）★★★
        if state_dim is None:
            state_dim = self.state_encoder.state_dim
        else:
            print(f"設定ファイルから状態次元数を取得: {state_dim}")
        
        # PPOエージェントを初期化
        ppo_config = self.config.get('ppo', {})
        self.agent = PPOAgent(state_dim, action_dim, ppo_config)
        self.agent.load(model_path)
        self.agent.actor.eval()  # 評価モードに設定
        
        print(f"PPOモデル '{model_path}' の読み込み完了。")
    
    def select_ambulance(self,
                        request: EmergencyRequest,
                        available_ambulances: List[AmbulanceInfo],
                        travel_time_func: callable,
                        context: DispatchContext) -> Optional[AmbulanceInfo]:
        """シミュレータの状態からPPOエージェントが最適な救急車を選択する"""
        if not available_ambulances:
            return None
        
        if self.agent is None or self.state_encoder is None:
            print("警告: PPOエージェントが初期化されていません。フォールバックします。")
            return self._fallback_selection(request, available_ambulances, travel_time_func)
        
        try:
            # 1. シミュレータの状態を、RL環境が理解できる形式 (state_dict) に変換
            state_dict = {
                'ambulances': self._get_ambulance_states(context),
                'pending_call': {
                    'h3_index': request.h3_index,
                    'severity': request.severity
                },
                'episode_step': context.current_time,
                'time_of_day': context.hour_of_day
            }
            
            # 2. state_dictを状態ベクトルにエンコード
            state_vector = self.state_encoder.encode_state(state_dict)
            
            # 3. 利用可能な救急車のマスクを作成
            action_dim = self.state_encoder.max_ambulances
            action_mask = np.zeros(action_dim, dtype=bool)
            
            # available_ambulancesのIDを整数インデックスにマッピング
            available_indices = []
            for amb_info in available_ambulances:
                try:
                    # 'AMB_1' -> 1
                    amb_idx = int(amb_info.id.split('_')[-1])
                    if amb_idx < action_dim:
                        action_mask[amb_idx] = True
                        available_indices.append(amb_idx)
                except (ValueError, IndexError):
                    continue
            
            # 利用可能な隊がいない場合は最近接を返す（フォールバック）
            if not available_indices:
                return self._fallback_selection(request, available_ambulances, travel_time_func)
            
            # 4. PPOエージェントに行動を選択させる (決定的モード)
            with torch.no_grad():
                selected_action_idx, _, _ = self.agent.select_action(
                    state_vector,
                    action_mask,
                    deterministic=True
                )
            
            # 5. 選択された行動インデックスをAmbulanceInfoオブジェクトに変換
            for amb_info in available_ambulances:
                try:
                    amb_idx = int(amb_info.id.split('_')[-1])
                    if amb_idx == selected_action_idx:
                        return amb_info
                except (ValueError, IndexError):
                    continue
            
            # もし選択された救急車が利用不可能なリストにいた場合（稀なケース）、
            # 利用可能な中から最近接を返す（安全のためのフォールバック）
            print(f"警告: PPOが選択した救急車 {selected_action_idx} は利用不可能でした。フォールバックします。")
            return self._fallback_selection(request, available_ambulances, travel_time_func)
            
        except Exception as e:
            print(f"PPO戦略でエラーが発生しました: {e}")
            return self._fallback_selection(request, available_ambulances, travel_time_func)
    
    def _get_ambulance_states(self, context: DispatchContext) -> Dict:
        """コンテキストから救急車の状態を取得"""
        states = {}
        for amb_id_str, amb_obj in context.all_ambulances.items():
            try:
                # 救急車IDを整数インデックスに変換
                if isinstance(amb_id_str, str) and '_' in amb_id_str:
                    # 'AMB_1' -> 1
                    amb_idx = int(amb_id_str.split('_')[-1])
                else:
                    # 数値の場合はそのまま使用
                    amb_idx = int(amb_id_str)
                
                # 救急車の状態情報を取得
                states[amb_idx] = {
                    'id': getattr(amb_obj, 'id', amb_id_str),
                    'station_h3': getattr(amb_obj, 'station_h3_index', getattr(amb_obj, 'station_h3', '')),
                    'current_h3': getattr(amb_obj, 'current_h3_index', getattr(amb_obj, 'current_h3', '')),
                    'status': self._get_status_string(amb_obj),
                    'calls_today': getattr(amb_obj, 'num_calls_handled', getattr(amb_obj, 'calls_today', 0)),
                }
            except (ValueError, IndexError, AttributeError) as e:
                print(f"警告: 救急車 {amb_id_str} の状態取得に失敗: {e}")
                continue
        return states
    
    def _get_status_string(self, amb_obj) -> str:
        """救急車の状態を文字列で取得"""
        status = getattr(amb_obj, 'status', None)
        if status is None:
            return 'unknown'
        
        if hasattr(status, 'name'):
            return status.name.lower()
        elif hasattr(status, 'value'):
            return status.value.lower()
        else:
            return str(status).lower()
    
    def _fallback_selection(self, request: EmergencyRequest, available_ambulances: List[AmbulanceInfo], 
                           travel_time_func: callable) -> Optional[AmbulanceInfo]:
        """フォールバック用の最近接選択"""
        if not available_ambulances:
            return None
        return min(available_ambulances, 
                   key=lambda amb: travel_time_func(amb.current_h3, request.h3_index, 'response'))

class StrategyFactory:
    """戦略の動的生成を行うファクトリークラス"""
    
    _strategies = {
        'closest': ClosestAmbulanceStrategy,
        'severity_based': SeverityBasedStrategy,
        'advanced_severity': AdvancedSeverityStrategy,
        'ppo_agent': PPOStrategy,
        'second_ride': SecondRideStrategy,  # ← この行を追加
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict = None) -> DispatchStrategy:
        """戦略を生成"""
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(cls._strategies.keys())}")
        
        strategy = cls._strategies[strategy_name]()
        
        # PPO戦略の場合は特別な初期化処理
        if strategy_name == 'ppo_agent':
            if not config:
                raise ValueError("PPO戦略には 'model_path' と 'config_path' を含む設定が必要です。")
            strategy.initialize(config)
        else:
            # その他の戦略は通常の初期化
            if config:
                strategy.initialize(config)
            else:
                strategy.initialize({})
        
        return strategy
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """新しい戦略を登録"""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def list_available_strategies(cls) -> List[str]:
        """利用可能な戦略のリストを返す"""
        return list(cls._strategies.keys())