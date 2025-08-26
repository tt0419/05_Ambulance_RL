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
        
        # デフォルトのパラメータをここで定義
        self.time_score_weight = 0.6  # デフォルトは60%
        self.coverage_loss_weight = 0.4 # デフォルトは40%
        self.mild_time_limit_sec = 780  # 軽症のデフォルトは13分
        self.moderate_time_limit_sec = 780 # 中等症のデフォルトは13分
        
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
        
        # ★★★ 新規追加: 重みパラメータと時間制限の設定 ★★★
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
        
        # 傷病度に応じて制限時間を設定
        if request.severity == '軽症':
            time_limit = self.mild_time_limit_sec  # ★修正: 外部設定可能に
        elif request.severity == '中等症':
            time_limit = self.moderate_time_limit_sec  # ★修正: 外部設定可能に
        else:
            time_limit = self.time_threshold_13min  # 13分（デフォルト）
        
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

class AdvancedSeverityStrategy(DispatchStrategy):
    """高度な傷病度優先戦略：重症系への強い優先度付け"""
    
    def __init__(self):
        super().__init__("advanced_severity", "rule_based")
        
        # 傷病度カテゴリ
        self.critical_conditions = ['重篤']  # 最優先
        self.severe_conditions = ['重症', '死亡']  # 高優先
        self.moderate_conditions = ['中等症']  # 中優先
        self.mild_conditions = ['軽症']  # 低優先
        
        # 戦略パラメータ
        self.params = {
            # 重篤・重症用
            'critical_search_radius': 480,  # 8分以内の救急車を全て考慮
            'severe_search_radius': 420,    # 7分以内の救急車を考慮
            
            # 中等症用
            'moderate_time_limit': 900,     # 15分制限
            'moderate_coverage_weight': 0.3, # カバレッジ重視度を下げる
            
            # 軽症用
            'mild_time_limit': 1080,        # 18分制限（大幅緩和）
            'mild_coverage_weight': 0.2,    # カバレッジ最小限
            'mild_delay_threshold': 600,    # 10分以上かかる救急車を積極利用
            
            # 繁忙期判定
            'high_utilization': 0.65,       # 65%で繁忙期判定（早めに切り替え）
            'critical_utilization': 0.80,   # 80%で緊急モード
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
            return self._select_for_critical(
                request, available_ambulances, travel_time_func, utilization
            )
        elif request.severity in self.severe_conditions:
            return self._select_for_severe(
                request, available_ambulances, travel_time_func, utilization
            )
        elif request.severity in self.moderate_conditions:
            return self._select_for_moderate(
                request, available_ambulances, travel_time_func, utilization
            )
        else:  # 軽症
            return self._select_for_mild(
                request, available_ambulances, travel_time_func, utilization, context
            )
    
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
    
    def _select_for_moderate(self, request, ambulances, travel_time_func, utilization):
        """
        中等症用：バランス型だが軽症より優先
        15分以内で、重症系の邪魔をしない救急車を選択
        """
        candidates = []
        for amb in ambulances:
            travel_time = travel_time_func(amb.current_h3, request.h3_index, 'response')
            if travel_time <= self.params['moderate_time_limit']:
                candidates.append((amb, travel_time))
        
        if not candidates:
            return self._get_closest(request, ambulances, travel_time_func)
        
        # 繁忙期：8分以上かかる救急車を優先
        if utilization > self.params['high_utilization']:
            # 遠い救急車から選ぶ（重症用を温存）
            candidates.sort(key=lambda x: x[1], reverse=True)
            for amb, travel_time in candidates:
                if travel_time >= 480:  # 8分以上
                    return amb
        
        # 平常期：バランス考慮
        best_amb = None
        best_score = float('inf')
        
        for amb, travel_time in candidates:
            # 6分以内の救急車にはペナルティ
            time_penalty = 0 if travel_time > 360 else (360 - travel_time) / 60
            score = travel_time / 900 + time_penalty * 2
            
            if score < best_score:
                best_score = score
                best_amb = amb
        
        return best_amb or candidates[0][0]
    
    def _select_for_mild(self, request, ambulances, travel_time_func, utilization, context):
        """
        軽症用：重症系リソースを避ける
        遠い救急車を積極的に活用
        """
        # 全救急車を距離でソート
        all_candidates = []
        for amb in ambulances:
            travel_time = travel_time_func(amb.current_h3, request.h3_index, 'response')
            all_candidates.append((amb, travel_time))
        
        all_candidates.sort(key=lambda x: x[1])
        
        # 繁忙期：10分以上かかる救急車を最優先
        if utilization > self.params['high_utilization']:
            for amb, travel_time in all_candidates:
                if travel_time >= self.params['mild_delay_threshold']:
                    if travel_time <= self.params['mild_time_limit']:
                        return amb
            
            # 10分以上がなければ、6分以上から選択
            for amb, travel_time in all_candidates:
                if travel_time >= 360 and travel_time <= self.params['mild_time_limit']:
                    return amb
        
        # 平常期：18分以内で最も遠い利用可能な救急車
        valid_candidates = [
            (amb, tt) for amb, tt in all_candidates 
            if tt <= self.params['mild_time_limit']
        ]
        
        if not valid_candidates:
            # 18分以内がない場合のみ最寄り
            return all_candidates[0][0] if all_candidates else None
        
        # カバレッジを少し考慮しつつ、遠めを選択
        best_amb = None
        best_score = float('-inf')  # 高いスコアを選ぶ
        
        for amb, travel_time in valid_candidates:
            # 遠いほど高スコア
            distance_score = travel_time / self.params['mild_time_limit']
            
            # 6分以内の救急車は強くペナルティ
            if travel_time < 360:
                distance_score *= 0.3
            elif travel_time < 480:
                distance_score *= 0.6
            
            # 簡易カバレッジチェック（近隣に救急車が多いか）
            coverage_bonus = self._count_nearby_available(
                amb, ambulances, travel_time_func
            ) * 0.1
            
            total_score = distance_score + coverage_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_amb = amb
        
        return best_amb
    
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
    }
}

class StrategyFactory:
    """戦略の動的生成を行うファクトリークラス"""
    
    _strategies = {
        'closest': ClosestAmbulanceStrategy,
        'severity_based': SeverityBasedStrategy,
        'advanced_severity': AdvancedSeverityStrategy,
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