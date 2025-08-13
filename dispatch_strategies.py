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

class DispatchPriority(Enum):
    """緊急度優先度"""
    CRITICAL = 1  # 重篤
    HIGH = 2      # 重症
    MEDIUM = 3    # 中等症・死亡
    LOW = 4       # 軽症

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
        """傷病度から優先度を取得"""
        severity_map = {
            '重篤': DispatchPriority.CRITICAL,
            '重症': DispatchPriority.HIGH,
            '中等症': DispatchPriority.MEDIUM,
            '死亡': DispatchPriority.MEDIUM,
            '軽症': DispatchPriority.LOW,
            'その他': DispatchPriority.LOW
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
        self.severe_conditions = ['重症', '重篤', '死亡']
        self.mild_conditions = ['軽症', '中等症']
        self.coverage_radius_km = 5.0
        self.time_threshold_6min = 360
        self.time_threshold_13min = 780
        
        # 新規追加: 傷病度別の時間制限（設定可能に）
        self.time_limits = {
            '軽症': 1080,    # 18分
            '中等症': 900,   # 15分
            'その他': 780    # 13分（デフォルト）
        }
        
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
    
    def select_ambulance(self,
                        request: EmergencyRequest,
                        available_ambulances: List[AmbulanceInfo],
                        travel_time_func: callable,
                        context: DispatchContext) -> Optional[AmbulanceInfo]:
        """傷病度に応じた救急車選択"""
        if not available_ambulances:
            return None
        
        # 重症・重篤・死亡の場合は最寄りを選択
        if request.severity in self.severe_conditions:
            return self._select_closest(request, available_ambulances, travel_time_func)
        
        # 軽症・中等症の場合はカバレッジを考慮
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
        
        # 修正後:
        # 傷病度に応じて制限時間を設定
        if request.severity == '軽症':
            time_limit = 1080  # 18分（780秒から大幅緩和）
        elif request.severity == '中等症':
            time_limit = 900   # 15分（少し緩和）
        else:
            time_limit = self.time_threshold_13min  # 13分（デフォルト）
        
        candidates = []
        for amb in available_ambulances:
            travel_time = travel_time_func(amb.current_h3, request.h3_index, 'response')
            if travel_time <= time_limit:
                candidates.append((amb, travel_time))
        
        # ===== 修正箇所2: 軽症の場合、近い救急車を避ける =====
        # 新規追加:
        if request.severity == '軽症' and len(candidates) > 3:
            # 6分以内の救急車を除外（重症用に温存）
            filtered_candidates = [(amb, tt) for amb, tt in candidates if tt > 360]
            if filtered_candidates:
                candidates = filtered_candidates
        
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
            
            # 修正後: 傷病度別の重み付け
            time_score = travel_time / time_limit  # 制限時間で正規化
            
            if request.severity == '軽症':
                # 軽症：時間をあまり重視しない（遠くてもOK）
                combined_score = time_score * 0.2 + coverage_loss * 0.8
            elif request.severity == '中等症':
                # 中等症：バランス型
                combined_score = time_score * 0.5 + coverage_loss * 0.5
            else:
                # その他：時間重視（元の設定）
                combined_score = time_score * 0.6 + coverage_loss * 0.4
            
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

class StrategyFactory:
    """戦略の動的生成を行うファクトリークラス"""
    
    _strategies = {
        'closest': ClosestAmbulanceStrategy,
        'severity_based': SeverityBasedStrategy,
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict = None) -> DispatchStrategy:
        """戦略を生成"""
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(cls._strategies.keys())}")
        
        strategy = cls._strategies[strategy_name]()
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