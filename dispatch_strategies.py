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

# データキャッシュのインポート
from data_cache import get_emergency_data_cache

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

# PPOエージェントを戦略として使用する新しいクラス（改良版）
class PPOStrategy(DispatchStrategy):
    """学習済みPPOエージェントを使用する戦略（状態エンコーディング改良版）"""
    
    def __init__(self):
        super().__init__("ppo_agent", "reinforcement_learning")
        self.agent = None
        self.state_encoder = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ハイブリッドモード関連
        self.hybrid_mode = False
        self.severe_conditions = []
        self.mild_conditions = []
        
        # 次元数とデータ
        self.action_dim = None
        self.state_dim = None
        self.travel_time_matrix = None
        self.grid_mapping = None
        
    def initialize(self, config: Dict):
        """PPOモデルと関連コンポーネントの初期化"""
        if not PPO_AVAILABLE:
            raise ImportError("PPOモジュールが利用できません。reinforcement_learningパッケージを確認してください。")
        
        print("PPO戦略を初期化中（改良版）...")
        
        # ハイブリッドモード設定
        self.hybrid_mode = config.get('hybrid_mode', False)
        if self.hybrid_mode:
            self.severe_conditions = config.get('severe_conditions', ['重症', '重篤', '死亡'])
            self.mild_conditions = config.get('mild_conditions', ['軽症', '中等症'])
            print(f"  ハイブリッドモード有効: 重症系={self.severe_conditions}は直近隊")
        
        # モデルパスの取得と検証
        model_path = config.get('model_path')
        if not model_path:
            raise ValueError("PPOStrategy requires 'model_path' in config")
        
        from pathlib import Path
        model_file = Path(model_path)
        if not model_file.exists():
            # 相対パスの可能性を考慮
            alternative_paths = [
                Path(f"reinforcement_learning/{model_path}"),
                Path(f"../{model_path}")
            ]
            for alt_path in alternative_paths:
                if alt_path.exists():
                    model_file = alt_path
                    break
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"  モデルファイル: {model_file}")
        
        # 設定の読み込み（チェックポイントまたはYAMLファイル）
        checkpoint = torch.load(model_file, map_location=self.device)
        saved_config = checkpoint.get('config', {})
        
        # config.yamlが指定されている場合はそちらを優先
        config_path = config.get('config_path')
        if config_path:
            print(f"  設定ファイルから読み込み: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                saved_config = yaml.safe_load(f)
        elif not saved_config:
            # チェックポイントにもconfigがない場合、デフォルト設定を使用
            print("  警告: 設定情報が見つかりません。デフォルト設定を使用します")
            saved_config = self._create_default_config()
        
        self.config = saved_config
        
        # 次元数の決定（学習時の設定を優先）
        if 'data' in saved_config:
            area_config = saved_config['data'].get('area_restriction', {})
            if area_config.get('enabled'):
                self.action_dim = area_config.get('num_ambulances_in_area', 192)
                self.state_dim = area_config.get('state_dim', None)
            else:
                self.action_dim = saved_config.get('state_prediction', {}).get('action_dim', 192)
                self.state_dim = saved_config.get('state_prediction', {}).get('state_dim', None)
        else:
            self.action_dim = 192
            self.state_dim = None
        
        print(f"  行動次元: {self.action_dim}")
        
        # データパスの取得
        data_paths = saved_config.get('data_paths', {})
        travel_time_path = Path(data_paths.get('travel_time_matrix', 'data/tokyo/calibration2/linear_calibrated_response.npy'))
        grid_mapping_path = Path(data_paths.get('grid_mapping', 'data/tokyo/processed/grid_mapping_res9.json'))
        
        # 移動時間行列の読み込み
        if travel_time_path.exists():
            self.travel_time_matrix = np.load(travel_time_path)
            print(f"  移動時間行列読み込み完了: {self.travel_time_matrix.shape}")
        else:
            print(f"  警告: 移動時間行列が見つかりません: {travel_time_path}")
        
        # グリッドマッピングの読み込み
        if grid_mapping_path.exists():
            with open(grid_mapping_path, 'r') as f:
                self.grid_mapping = json.load(f)
            print(f"  グリッドマッピング読み込み完了: {len(self.grid_mapping)}グリッド")
        else:
            print(f"  警告: グリッドマッピングが見つかりません: {grid_mapping_path}")
        
        # StateEncoderの初期化
        self.state_encoder = StateEncoder(
            config=saved_config,
            max_ambulances=self.action_dim,
            travel_time_matrix=self.travel_time_matrix,
            grid_mapping=self.grid_mapping
        )
        
        # 状態次元数の決定
        if self.state_dim is None:
            self.state_dim = self.state_encoder.state_dim
        print(f"  状態次元: {self.state_dim}")
        
        # PPOエージェントの作成と読み込み
        ppo_config = saved_config.get('ppo', {})
        if not ppo_config:
            print("  警告: PPO設定が見つかりません。デフォルト設定を使用します")
            ppo_config = self._create_default_ppo_config()
        
        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=ppo_config,
            device=str(self.device)
        )
        self.agent.load(model_file)
        self.agent.actor.eval()
        self.agent.critic.eval()
        
        print(f"PPOモデルの読み込み完了 (hybrid={self.hybrid_mode})")
    
    def _create_default_config(self) -> Dict:
        """デフォルト設定を作成"""
        return {
            'data_paths': {
                'travel_time_matrix': 'data/tokyo/calibration2/linear_calibrated_response.npy',
                'grid_mapping': 'data/tokyo/processed/grid_mapping_res9.json'
            },
            'state_prediction': {
                'action_dim': 192,
                'state_dim': 999
            }
        }
    
    def _create_default_ppo_config(self) -> Dict:
        """デフォルトPPO設定を作成"""
        return {
            'learning_rate': {
                'actor': 3e-4,
                'critic': 1e-3
            },
            'clip_epsilon': 0.2,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'n_epochs': 10,
            'batch_size': 64,
            'entropy_coef': 0.01,
            'network': {
                'hidden_dims': [256, 256]
            }
        }
    
    def select_ambulance(self,
                        request: EmergencyRequest,
                        available_ambulances: List[AmbulanceInfo],
                        travel_time_func: callable,
                        context: DispatchContext) -> Optional[AmbulanceInfo]:
        """救急車選択ロジック（ハイブリッドモード対応）"""
        if not available_ambulances:
            return None
        
        # ハイブリッドモード：重症度で分岐
        if self.hybrid_mode and request.severity in self.severe_conditions:
            return self._select_closest(request, available_ambulances, travel_time_func)
        
        # 軽症系またはハイブリッドでない場合 → PPO選択
        return self._select_with_ppo(request, available_ambulances, travel_time_func, context)
    
    def _select_closest(self, request, available_ambulances, travel_time_func):
        """直近隊選択（重症系用）"""
        best_ambulance = None
        min_time = float('inf')
        
        for amb_info in available_ambulances:
            travel_time = travel_time_func(amb_info.current_h3, request.h3_index, 'response')
            if travel_time < min_time:
                min_time = travel_time
                best_ambulance = amb_info
        
        return best_ambulance
    
    def _select_with_ppo(self, request, available_ambulances, travel_time_func, context):
        """PPOモデルによる選択（学習時と同じ状態エンコーディング）"""
        if self.agent is None or self.state_encoder is None:
            print("警告: PPOエージェント未初期化、フォールバック")
            return self._select_closest(request, available_ambulances, travel_time_func)
        
        try:
            # 1. ValidationSimulatorの状態を学習環境形式に変換
            state_dict = self._build_state_dict(request, available_ambulances, context)
            
            # 2. StateEncoderで状態ベクトルに変換
            state_vector = self.state_encoder.encode_state(state_dict)
            
            # 3. 行動マスクの作成
            action_mask = self._create_action_mask(available_ambulances)
            
            # 4. PPOエージェントで行動選択
            with torch.no_grad():
                action, _, _ = self.agent.select_action(
                    state_vector,
                    action_mask,
                    deterministic=True
                )
            
            # 5. 選択された行動を救急車にマッピング
            selected_amb = self._map_action_to_ambulance(action, available_ambulances)
            
            if selected_amb:
                return selected_amb
            else:
                print(f"警告: Action {action} マッピング失敗、フォールバック")
                return self._select_closest(request, available_ambulances, travel_time_func)
                
        except Exception as e:
            print(f"PPO選択エラー: {e}")
            import traceback
            traceback.print_exc()
            return self._select_closest(request, available_ambulances, travel_time_func)
    
    def _build_state_dict(self, request, available_ambulances, context):
        """ValidationSimulatorの状態を学習環境形式に変換"""
        ambulances = {}
        
        # 全救急車情報を取得
        for amb_id, amb_obj in context.all_ambulances.items():
            try:
                # IDを整数に変換
                if isinstance(amb_id, str) and '_' in amb_id:
                    idx = int(amb_id.split('_')[-1])
                else:
                    idx = int(amb_id)
                
                # 状態の取得
                ambulances[idx] = {
                    'current_h3': amb_obj.current_h3_index,
                    'status': 'available' if amb_obj.status.value == 'available' and not amb_obj.is_on_break else 'busy',
                    'calls_today': amb_obj.num_calls_handled,
                    'station_h3': amb_obj.station_h3_index
                }
            except (ValueError, AttributeError):
                continue
        
        # 事案情報
        pending_call = {
            'h3_index': request.h3_index,
            'severity': request.severity,
            'wait_time': 0,
            'priority': request.priority.value if hasattr(request.priority, 'value') else 0.5
        }
        
        # 時間情報
        episode_step = int(context.current_time / 60) if context.current_time else 0
        time_of_day = context.hour_of_day if context.hour_of_day is not None else 12
        
        return {
            'ambulances': ambulances,
            'pending_call': pending_call,
            'episode_step': episode_step,
            'time_of_day': time_of_day
        }
    
    def _create_action_mask(self, available_ambulances):
        """利用可能な救急車のマスクを作成"""
        mask = np.zeros(self.action_dim, dtype=bool)
        
        for amb_info in available_ambulances:
            try:
                if '_' in amb_info.id:
                    idx = int(amb_info.id.split('_')[-1])
                else:
                    idx = int(amb_info.id)
                
                if 0 <= idx < self.action_dim:
                    mask[idx] = True
            except (ValueError, AttributeError):
                continue
        
        if not mask.any():
            print("警告: マスク内に利用可能な救急車がありません")
        
        return mask
    
    def _map_action_to_ambulance(self, action: int, available_ambulances):
        """選択された行動インデックスを救急車オブジェクトにマッピング"""
        for amb_info in available_ambulances:
            try:
                if '_' in amb_info.id:
                    idx = int(amb_info.id.split('_')[-1])
                else:
                    idx = int(amb_info.id)
                
                if idx == action:
                    return amb_info
            except (ValueError, AttributeError):
                continue
        
        return None


# ============== MEXCLP戦略の実装 ==============

class DemandDistributionCalculator:
    """過去の救急事案データから需要分布を計算するヘルパークラス"""
    
    def __init__(self, grid_mapping: Dict):
        self.grid_mapping = grid_mapping
        self.demand_distribution = None  # キャッシュ用
        
    def get_demand_distribution(self) -> Dict[str, float]:
        """需要分布を取得（キャッシュ済みならそれを返す）"""
        if self.demand_distribution is None:
            self.demand_distribution = self._calculate_demand_distribution()
        return self.demand_distribution
    
    def _calculate_demand_distribution(self) -> Dict[str, float]:
        """実際の需要分布計算
        
        validation_simulation.pyと同じ方法でh3インデックスを取得
        """
        import time
        import pandas as pd
        start_time = time.time()
        
        # データキャッシュから全データを取得
        data_cache = get_emergency_data_cache()
        calls_df = data_cache.load_data()
        
        print(f"需要分布計算開始: {len(calls_df)}件の事案データから計算")
        
        # validation_simulation.pyと同じ方法でh3インデックスを計算
        # Y_CODE（緯度）、X_CODE（経度）からh3インデックスを生成
        h3_counts = {}
        total_calls = 0
        invalid_coords = 0
        out_of_grid = 0
        
        for _, call in calls_df.iterrows():
            try:
                # 座標データの確認
                if pd.notna(call.get('Y_CODE')) and pd.notna(call.get('X_CODE')):
                    lat = float(call['Y_CODE'])
                    lng = float(call['X_CODE'])
                    
                    # h3インデックスを生成（validation_simulation.pyと同じ解像度9）
                    h3_idx = h3.latlng_to_cell(lat, lng, 9)
                    
                    # グリッドマッピングに含まれているか確認
                    if h3_idx in self.grid_mapping:
                        h3_counts[h3_idx] = h3_counts.get(h3_idx, 0) + 1
                        total_calls += 1
                    else:
                        out_of_grid += 1
                else:
                    invalid_coords += 1
                    
            except (ValueError, TypeError, KeyError) as e:
                invalid_coords += 1
                continue
        
        if total_calls == 0:
            raise ValueError("需要分布計算エラー: 有効な事案データが見つかりません")
        
        # 需要割合に変換（全グリッドに対して計算）
        demand_distribution = {}
        for h3_idx in self.grid_mapping.keys():
            count = h3_counts.get(h3_idx, 0)
            demand_distribution[h3_idx] = count / total_calls
        
        elapsed_time = time.time() - start_time
        
        print(f"需要分布計算完了:")
        print(f"  処理時間: {elapsed_time:.2f}秒")
        print(f"  有効事案数: {total_calls}件")
        print(f"  無効座標: {invalid_coords}件")
        print(f"  グリッド外: {out_of_grid}件")
        print(f"  需要があるグリッド数: {len(h3_counts)}/{len(self.grid_mapping)}")
        print(f"  最大需要グリッド: {max(demand_distribution.values()):.4f}")
        
        return demand_distribution


class MEXCLPStrategy(DispatchStrategy):
    """Dynamic MEXCLP戦略の実装（最適化版）
    
    Jagtenberg et al. (2017) に基づく実装
    パフォーマンス最適化：需要の累積90%をカバーする主要グリッドのみで計算
    """
    
    def __init__(self):
        super().__init__("mexclp", "optimization_based")
        # デフォルトパラメータ
        self.busy_fraction = 0.3
        self.time_threshold_seconds = 780
        self.demand_calculator = None
        self.travel_time_matrix = None
        self.grid_mapping = None
        
        # パフォーマンス最適化用
        self.significant_demand_grids = None  # 需要の高い主要グリッド
        self.coverage_cache = {}  # カバレッジ計算のキャッシュ
        self.cumulative_threshold = 0.9  # 累積需要の90%をカバー
        
    def initialize(self, config: Dict):
        """戦略の初期化"""
        self.config = config
        self.busy_fraction = config.get('busy_fraction', 0.3)
        self.time_threshold_seconds = config.get('time_threshold', 780) # 13分
        self.cumulative_threshold = config.get('cumulative_threshold', 0.9)
        
        print(f"MEXCLPStrategy初期化開始...")
        
        # グリッドマッピングの読み込み
        try:
            with open('data/tokyo/processed/grid_mapping_res9.json', 'r', encoding='utf-8') as f:
                self.grid_mapping = json.load(f)
            print(f"  グリッドマッピング読み込み成功: {len(self.grid_mapping)}グリッド")
        except Exception as e:
            raise RuntimeError(f"グリッドマッピングの読み込みエラー: {e}")
        
        # 移動時間行列の読み込み
        try:
            import numpy as np
            matrix_path = 'data/tokyo/calibration2/log_calibrated_response.npy'
            self.travel_time_matrix = np.load(matrix_path)
            print(f"  移動時間行列読み込み成功: shape={self.travel_time_matrix.shape}")
        except Exception as e:
            raise RuntimeError(f"移動時間行列の読み込みエラー: {e}")
        
        # 需要分布計算器の初期化
        try:
            self.demand_calculator = DemandDistributionCalculator(self.grid_mapping)
            print(f"  需要分布計算器初期化成功")
            
            # 主要需要グリッドの事前計算
            self._precompute_significant_grids()
            
        except Exception as e:
            raise RuntimeError(f"需要分布計算器の初期化エラー: {e}")
        
        print(f"MEXCLPStrategy初期化完了:")
        print(f"  busy_fraction: {self.busy_fraction}")
        print(f"  time_threshold: {self.time_threshold_seconds}秒")
        print(f"  計算対象グリッド数: {len(self.significant_demand_grids)}/{len(self.grid_mapping)}")
    
    def _precompute_significant_grids(self):
        """需要の累積X%をカバーする主要グリッドを事前計算"""
        demand_distribution = self.demand_calculator.get_demand_distribution()
        
        # 需要の高い順にソート
        sorted_demands = sorted(demand_distribution.items(), 
                              key=lambda x: x[1], reverse=True)
        
        # 累積需要がthresholdを超えるまでのグリッドを選択
        cumulative_demand = 0.0
        self.significant_demand_grids = []
        
        for h3_idx, demand in sorted_demands:
            if demand == 0:
                break
            self.significant_demand_grids.append((h3_idx, demand))
            cumulative_demand += demand
            if cumulative_demand >= self.cumulative_threshold:
                break
        
        print(f"  主要需要グリッド: {len(self.significant_demand_grids)}個")
        print(f"  カバー需要割合: {cumulative_demand:.2%}")
    
    def select_ambulance(self, request: EmergencyRequest,
                        available_ambulances: List[AmbulanceInfo],
                        travel_time_func: callable,
                        context: DispatchContext) -> Optional[AmbulanceInfo]:
        """MEXCLPアルゴリズムによる救急車選択（最適化版）"""
        
        if not available_ambulances:
            return None
        
        # 初期化チェック
        if not self.grid_mapping or self.travel_time_matrix is None:
            raise RuntimeError("MEXCLPStrategy: 初期化が完了していません")
        
        # キャッシュのクリア（毎回の配車で状況が変わるため）
        self.coverage_cache.clear()
        
        # Step 1: 閾値時間内到着可能な救急車を分類
        within_threshold = []
        beyond_threshold = []
        
        for ambulance in available_ambulances:
            travel_time = travel_time_func(ambulance.current_h3, request.h3_index, 'response')
            if travel_time <= self.time_threshold_seconds:
                within_threshold.append(ambulance)
            else:
                beyond_threshold.append(ambulance)
        
        candidates = within_threshold if within_threshold else beyond_threshold
        
        # Step 2: 高速化されたカバレッジ評価
        import time
        calc_start = time.time()
        
        best_ambulance = None
        max_remaining_coverage = -float('inf')
        
        # バッチ処理用のデータ準備
        remaining_lists = []
        for candidate in candidates:
            remaining = [a for a in available_ambulances if a.id != candidate.id]
            remaining_lists.append((candidate, remaining))
        
        # 各候補の残存カバレッジを計算
        for candidate, remaining in remaining_lists:
            coverage = self._calculate_remaining_coverage_optimized(remaining)
            
            if coverage > max_remaining_coverage:
                max_remaining_coverage = coverage
                best_ambulance = candidate
        
        calc_time = time.time() - calc_start
        
        if calc_time > 0.5:  # 0.5秒以上かかった場合のみログ出力
            print(f"MEXCLP: {len(candidates)}台の評価に{calc_time:.3f}秒")
        
        return best_ambulance
    
    def _calculate_remaining_coverage_optimized(self, 
                                               remaining: List[AmbulanceInfo]) -> float:
        """最適化された残存カバレッジ計算
        
        最適化手法：
        1. 主要需要グリッドのみで計算
        2. numpy配列演算の活用
        3. キャッシングの活用
        """
        
        # 残存車両の位置をキーとしたキャッシュ
        cache_key = tuple(sorted(a.id for a in remaining))
        if cache_key in self.coverage_cache:
            return self.coverage_cache[cache_key]
        
        import numpy as np
        
        # 残存車両のグリッドインデックスを事前計算
        remaining_indices = []
        for amb in remaining:
            if amb.current_h3 in self.grid_mapping:
                remaining_indices.append(self.grid_mapping[amb.current_h3])
        
        if not remaining_indices:
            return 0.0
        
        remaining_indices = np.array(remaining_indices)
        
        total_coverage = 0.0
        
        # 主要需要グリッドのみで計算
        for demand_h3, demand_fraction in self.significant_demand_grids:
            if demand_h3 not in self.grid_mapping:
                continue
                
            demand_idx = self.grid_mapping[demand_h3]
            
            # numpy配列演算で一括計算
            # 各救急車から需要点への移動時間を取得
            travel_times = self.travel_time_matrix[remaining_indices, demand_idx]
            
            # 閾値時間内にカバーできる救急車数
            k = np.sum(travel_times <= self.time_threshold_seconds)
            
            if k > 0:
                # MEXCLPカバレッジ公式
                coverage = demand_fraction * (1 - self.busy_fraction ** k)
                total_coverage += coverage
        
        # キャッシュに保存
        self.coverage_cache[cache_key] = total_coverage
        
        return total_coverage

# ============== MEXCLP戦略の実装終了 ==============

class StrategyFactory:
    """戦略の動的生成を行うファクトリークラス"""
    
    _strategies = {
        'closest': ClosestAmbulanceStrategy,
        'severity_based': SeverityBasedStrategy,
        'advanced_severity': AdvancedSeverityStrategy,
        'ppo_agent': PPOStrategy,
        'second_ride': SecondRideStrategy,
        'mexclp': MEXCLPStrategy,  # ← MEXCLP戦略を追加
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