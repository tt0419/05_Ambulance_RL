# 訓練環境とテスト環境の融合方針

## 1. 時間管理の融合方針

### 推奨方針：固定時間ステップ制の導入

```python
class UnifiedTimeManagement:
    """
    1ステップ = 1分の固定時間制
    訓練とテストで同一の時間管理を実現
    """
    
    def __init__(self):
        self.time_per_step = 60  # 秒
        self.current_step = 0
        self.current_time_seconds = 0.0
        
        # イベントキューは保持（ValidationSim互換）
        self.event_queue = []
    
    def step(self, action):
        """1分間のシミュレーションを実行"""
        start_time = self.current_time_seconds
        end_time = start_time + self.time_per_step
        
        # この1分間に発生するイベントを処理
        events_in_minute = []
        while self.event_queue:
            if self.event_queue[0].time <= end_time:
                event = heapq.heappop(self.event_queue)
                events_in_minute.append(event)
                self._process_event(event)
            else:
                break
        
        # 時間を進める
        self.current_step += 1
        self.current_time_seconds = end_time
        
        return self._get_observation()
```

**メリット**：
- 訓練とテストで同一の時間粒度
- 事案間の時間も適切に処理
- ValidationSimのイベント処理と互換

## 2. 学習を阻害する重要な相違点と対処法

### 2.1 サービス時間の相違【影響度：高】

**問題**：
- 訓練環境：固定値（現場15分、病院20分）
- テスト環境：傷病度・時間帯別の確率分布

**対処法**：
```python
# 訓練環境に同一のサービス時間生成器を統合
class TrainingEnvironment:
    def __init__(self):
        # ValidationSimと同じ生成器を使用
        self.service_time_generator = ServiceTimeGeneratorEnhanced(
            "lognormal_parameters_hierarchical.json"
        )
    
    def calculate_activity_time(self, severity, call_datetime):
        # 各フェーズで同じ分布から生成
        on_scene = self.service_time_generator.generate_time(
            severity, 'on_scene_time', call_datetime
        )
        hospital = self.service_time_generator.generate_time(
            severity, 'hospital_time', call_datetime
        )
        return on_scene, hospital
```

### 2.2 病院選択ロジックの相違【影響度：高】

**問題**：
- 訓練環境：最近接またはランダム
- テスト環境：確率的選択モデル

**対処法**：
```python
# 同一の病院選択モデルを使用
class UnifiedHospitalSelection:
    def __init__(self):
        with open("hospital_selection_model.json") as f:
            self.model = json.load(f)
    
    def select_hospital(self, scene_h3, severity):
        # 訓練・テスト共通のロジック
        key = f"{scene_h3}_{severity}"
        if key in self.model['selection_probabilities']:
            hospitals = self.model['selection_probabilities'][key]
            return np.random.choice(
                list(hospitals.keys()),
                p=list(hospitals.values())
            )
        return self._select_nearest_fallback(scene_h3, severity)
```

### 2.3 行動マスキングの実装差異【影響度：中】

**問題**：
- 訓練環境：全救急車のマスクを生成
- テスト環境：利用可能な救急車のみ考慮

**対処法**：
```python
# 統一されたマスキング実装
def get_action_mask_unified(self):
    mask = np.zeros(self.num_ambulances, dtype=bool)
    
    for amb_id, state in self.ambulance_states.items():
        # 両環境で同じ条件
        if state['status'] == 'available' and not state.get('on_break', False):
            mask[amb_id] = True
    
    return mask
```

### 2.4 報酬関数の不一致【影響度：中】

**問題**：
- 訓練時の報酬と実環境での評価指標が異なる

**対処法**：
```python
class UnifiedReward:
    """訓練とテストで同一の報酬計算"""
    
    def calculate_reward(self, response_time, severity):
        reward = 0.0
        
        # 6分・13分達成の統一基準
        if is_severe_condition(severity):
            target = 6.0
            penalty_rate = 2.0
        else:
            target = 13.0
            penalty_rate = 0.5
        
        if response_time <= target:
            reward += 10.0
        else:
            reward -= (response_time - target) * penalty_rate
        
        return reward
```

## 3. 段階的な融合実装計画

### Phase 1: 最小限の統一（1週間）
```python
class MinimalUnifiedEnvironment(ems_environment1027):
    """既存環境に最小限の修正"""
    
    def __init__(self):
        super().__init__()
        # 同じサービス時間生成器を追加
        self.service_time_generator = load_from_validation()
        # 同じ病院選択モデルを追加
        self.hospital_model = load_hospital_model()
```

### Phase 2: 時間管理の統一（2週間）
```python
class TimeUnifiedEnvironment:
    """1ステップ = 1分の固定時間制"""
    
    def step(self, action):
        # 1分間の処理
        self._process_minute()
        
        # 救急車状態の更新
        self._update_ambulances_by_time()
        
        # 事案処理
        if self._has_pending_call():
            self._dispatch_ambulance(action)
```

### Phase 3: 完全統合環境（1ヶ月）
```python
class FullyIntegratedEnvironment:
    """ValidationSimをコアに持つRL環境"""
    
    def __init__(self):
        # ValidationSimulatorをコアエンジンとして使用
        self.core_sim = ValidationSimulator(...)
        # RL用のラッパー
        self.rl_interface = RLInterface()
    
    def step(self, action):
        # アクションをValidationSim形式に変換
        sim_action = self.rl_interface.convert_action(action)
        
        # 1分間シミュレーション実行
        events = self.core_sim.run_for_duration(60)
        
        # RL形式の観測・報酬を生成
        obs = self.rl_interface.create_observation(self.core_sim.state)
        reward = self.rl_interface.calculate_reward(events)
        
        return obs, reward
```

## 4. 実装優先順位

### 即座に実装すべき（影響大）
1. **サービス時間生成の統一**
   - 最も影響が大きく、実装も容易
   
2. **病院選択ロジックの統一**
   - 搬送時間に直接影響

### 次に実装すべき（影響中）
3. **固定時間ステップ制**
   - 時間管理の根本的解決
   
4. **報酬関数の統一**
   - 学習目標の一致

### 将来的に実装（影響小）
5. **休憩モデル**
   - 精度向上には寄与するが必須ではない

## 5. 検証方法

### 統合環境の検証手順
```python
def validate_integration():
    """統合環境の妥当性検証"""
    
    # 1. 同一シナリオでの比較
    scenario = load_test_scenario()
    
    # 訓練環境での実行
    train_env = UnifiedTrainingEnv()
    train_results = run_scenario(train_env, scenario)
    
    # ValidationSimでの実行
    val_sim = ValidationSimulator()
    val_results = run_scenario(val_sim, scenario)
    
    # 2. 主要指標の比較
    metrics = {
        'avg_response_time': compare_response_times(train_results, val_results),
        'ambulance_utilization': compare_utilization(train_results, val_results),
        'service_times': compare_service_times(train_results, val_results)
    }
    
    # 3. 許容誤差内かチェック
    assert metrics['avg_response_time']['diff'] < 0.1  # 0.1分以内
    assert metrics['ambulance_utilization']['diff'] < 0.05  # 5%以内
    
    return metrics
```

## 6. 推奨実装順序

### 短期（1-2週間）
1. ServiceTimeGeneratorEnhancedの統合
2. 病院選択モデルの統合
3. 統一報酬関数の実装

### 中期（3-4週間）
4. 固定時間ステップ制の導入
5. 統一マスキング実装
6. 検証テストの実施

### 長期（1-2ヶ月）
7. ValidationSimコアの統合環境
8. 完全な状態表現の統一
9. 性能最適化

## 7. リスクと対策

| リスク | 影響 | 対策 |
|-------|------|------|
| 学習の不安定化 | 高 | 段階的実装で各ステップを検証 |
| 計算コストの増大 | 中 | 必要な精度レベルを見極めて最適化 |
| 既存モデルとの非互換 | 高 | 移行期間は両環境を並行運用 |
| デバッグの困難化 | 中 | 詳細なログ機能を実装 |

## まとめ

**最優先で実装すべき統合**：
1. サービス時間生成器の統一
2. 病院選択モデルの統一
3. 1ステップ = 1分の固定時間制

これら3つの実装により、訓練環境とテスト環境の主要な相違が解消され、学習した方策が実環境で正しく動作するようになります。
