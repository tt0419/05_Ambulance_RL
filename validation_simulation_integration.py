"""
validation_simulation.pyへの統合修正内容

このファイルは、validation_simulation.pyに追加・修正が必要な部分を示します。
"""

# ============================================================
# 1. インポート部分に追加
# ============================================================
"""
# 既存のインポートの後に追加
from dispatch_strategies import (
    DispatchStrategy, 
    StrategyFactory,
    EmergencyRequest,
    AmbulanceInfo,
    DispatchContext,
    DispatchPriority
)
"""

# ============================================================
# 2. ValidationSimulatorクラスの__init__メソッドを修正
# ============================================================
def __init__(self, 
             travel_time_matrices: Dict[str, np.ndarray],
             travel_distance_matrices: Dict[str, np.ndarray],
             grid_mapping: Dict,
             service_time_generator: ServiceTimeGenerator,
             hospital_h3_indices: List[str],
             hospital_data: Optional[pd.DataFrame] = None,
             use_probabilistic_selection: bool = True,
             enable_breaks: bool = True,
             dispatch_strategy: str = 'closest',  # 追加
             strategy_config: Dict = None):  # 追加
    
    # 既存の初期化コード...
    
    # ディスパッチ戦略の初期化を追加
    self.dispatch_strategy = StrategyFactory.create_strategy(
        dispatch_strategy, 
        strategy_config or {}
    )
    
    # コンテキスト情報の初期化を追加
    self.dispatch_context = DispatchContext()
    self.dispatch_context.grid_mapping = self.grid_mapping
    self.dispatch_context.all_h3_indices = set(grid_mapping.keys())
    
    # 既存のコードの続き...

# ============================================================
# 3. find_closest_available_ambulanceメソッドを置き換え
# ============================================================
def find_closest_available_ambulance(self, call_h3: str, severity: str = None) -> Optional[Ambulance]:
    """
    ディスパッチ戦略を使用して最適な救急車を選択
    
    Args:
        call_h3: 事案発生地点のH3インデックス
        severity: 傷病度（オプション）
    
    Returns:
        選択された救急車オブジェクト、またはNone
    """
    # 利用可能な救急車を取得（休憩中を除外）
    available_ambulances = [
        amb for amb in self.ambulances.values() 
        if amb.status == AmbulanceStatus.AVAILABLE and not amb.is_on_break
    ]
    
    if not available_ambulances:
        if self.verbose_logging:
            print(f"[INFO] No available ambulances for call at {call_h3} at time {self.current_time:.2f}")
        return None
    
    # AmbulanceInfoオブジェクトのリストを作成
    ambulance_infos = []
    for amb in available_ambulances:
        amb_info = AmbulanceInfo(
            id=amb.id,
            current_h3=amb.current_h3_index,
            station_h3=amb.station_h3_index,
            status=amb.status.value,
            total_calls_today=amb.num_calls_handled,
            current_workload=0.0  # 必要に応じて計算
        )
        ambulance_infos.append(amb_info)
    
    # EmergencyRequestオブジェクトを作成
    priority = self.dispatch_strategy.get_severity_priority(severity) if severity else DispatchPriority.LOW
    request = EmergencyRequest(
        id=f"temp_{self.current_time}",  # 仮ID
        h3_index=call_h3,
        severity=severity or "その他",
        time=self.current_time,
        priority=priority
    )
    
    # DispatchContextを更新
    self.dispatch_context.current_time = self.current_time
    self.dispatch_context.hour_of_day = int((self.current_time / 3600) % 24)
    self.dispatch_context.total_ambulances = len(self.ambulances)
    self.dispatch_context.available_ambulances = len(available_ambulances)
    
    # 戦略を使用して救急車を選択
    selected_info = self.dispatch_strategy.select_ambulance(
        request=request,
        available_ambulances=ambulance_infos,
        travel_time_func=self.get_travel_time,
        context=self.dispatch_context
    )
    
    if selected_info:
        # AmbulanceInfoからAmbulanceオブジェクトを取得
        selected_ambulance = next(
            (amb for amb in available_ambulances if amb.id == selected_info.id),
            None
        )
        
        if selected_ambulance and self.verbose_logging:
            travel_time = self.get_travel_time(
                selected_ambulance.current_h3_index, 
                call_h3, 
                phase='response'
            )
            print(f"[INFO] Call at {call_h3} (severity: {severity}): "
                  f"Selected ambulance {selected_ambulance.id} at {selected_ambulance.current_h3_index}. "
                  f"Est. travel time: {travel_time:.2f}s. "
                  f"Strategy: {self.dispatch_strategy.name}")
        
        return selected_ambulance
    
    return None

# ============================================================
# 4. _handle_new_callメソッドの修正
# ============================================================
def _handle_new_call(self, event: Event):
    """新規事案の処理（傷病度情報を追加）"""
    call = EmergencyCall(
        id=event.data['call_id'],
        time=event.time,
        h3_index=event.data['h3_index'],
        severity=event.data['severity'],
        call_datetime=event.data.get('call_datetime')
    )
    self.calls[call.id] = call
    
    self.statistics['total_calls'] += 1
    hour = int(event.time // 3600) % 24
    self.statistics['calls_by_hour'][hour] += 1
    
    # 傷病度を渡すように修正
    ambulance = self.find_closest_available_ambulance(call.h3_index, call.severity)
    
    # 以下既存のコード...

# ============================================================
# 5. run_validation_simulation関数の修正
# ============================================================
def run_validation_simulation(
    target_date_str: str,
    output_dir: str,
    initial_active_rate_min: float = 0.5,
    initial_active_rate_max: float = 0.7,
    initial_availability_time_min_minutes: int = 0,
    initial_availability_time_max_minutes: int = 30,
    simulation_duration_hours: int = 24,
    random_seed: int = 42,
    verbose_logging: bool = False,
    enable_detailed_travel_time_analysis: bool = False,
    use_probabilistic_selection: bool = True,
    enable_breaks: bool = True,
    dispatch_strategy: str = 'closest',  # 追加
    strategy_config: Dict = None  # 追加
) -> None:
    """
    検証用シミュレーションを実行
    
    Args:
        ... (既存の引数)
        dispatch_strategy: 使用するディスパッチ戦略 ('closest' or 'severity_based')
        strategy_config: 戦略固有の設定
    """
    
    # ... 既存のコード ...
    
    # シミュレータの初期化部分を修正
    simulator = ValidationSimulator(
        travel_time_matrices=travel_time_matrices,
        travel_distance_matrices={'default': travel_distance_matrix},
        grid_mapping=grid_mapping,
        service_time_generator=service_time_generator,
        hospital_h3_indices=hospital_h3_indices,
        hospital_data=hospital_data,
        use_probabilistic_selection=use_probabilistic_selection,
        enable_breaks=enable_breaks,
        dispatch_strategy=dispatch_strategy,  # 追加
        strategy_config=strategy_config  # 追加
    )
    
    # ... 既存のコード ...

# ============================================================
# 6. メイン実行部分の修正例
# ============================================================
"""
if __name__ == "__main__":
    # ... 既存の設定 ...
    
    # ディスパッチ戦略の選択
    # dispatch_strategy = 'closest'  # 従来の最寄り戦略
    dispatch_strategy = 'severity_based'  # 新しい傷病度考慮戦略
    
    # 戦略固有の設定（オプション）
    strategy_config = {
        'coverage_radius_km': 5.0,
        'severe_conditions': ['重症', '重篤', '死亡'],
        'mild_conditions': ['軽症', '中等症']
    }
    
    run_validation_simulation(
        target_date_str=target_day_formatted,
        output_dir=output_dir,
        # ... 既存のパラメータ ...
        dispatch_strategy=dispatch_strategy,
        strategy_config=strategy_config
    )
"""