# ValidationSimulator完全統合計画書（修正版）

**⚠️ 重要な更新**: 他のLLMからの評価を受けて、実用機能の統合を最優先とする方針に変更しました。

---

## 📋 目的

現在のPPO学習環境(`ems_environment.py`)とValidationSimulatorの環境差を解消し、学習精度と検証精度を一致させる。

## 🔄 計画の修正点（2024年10月28日更新）

### 当初の計画の問題点
- ✅ イベント駆動アーキテクチャの設計は理論的に正しい
- ❌ **PPO学習に必須の実用機能が大幅に欠落**
  - RewardDesigner（報酬計算）
  - ハイブリッドモード（重症/軽症の振り分け）
  - get_episode_statistics（統計収集）
  - get_optimal_action（最適行動の取得）
  - render（可視化）

### 修正後の方針
```
┌─────────────────────────────────────────┐
│ 優先順位1: 既存の実用機能を100%維持     │
│ ├─ RewardDesigner                       │
│ ├─ DispatchLogger                       │
│ ├─ ハイブリッドモード                   │
│ ├─ 統計収集機能                         │
│ └─ 学習補助メソッド                     │
└─────────────────────────────────────────┘
            ↓ 統合
┌─────────────────────────────────────────┐
│ 優先順位2: イベント駆動コアの追加       │
│ ├─ heapqによる優先度付きキュー          │
│ ├─ 連続時間管理                         │
│ └─ イベント処理ループ                   │
└─────────────────────────────────────────┘
```

**実装アプローチ**:
- ❌ ゼロから新規実装
- ✅ **既存のems_environment.pyをベースに、イベント駆動機能を追加**

## 🎯 達成目標

### 定量的目標
| 指標 | 現状 | 目標 |
|------|------|------|
| 平均応答時間 | 21.67分 | 8.0分以下 |
| 6分達成率 | 2.8% | 40%以上 |
| 全車出動中頻度 | 50-72回/エピソード | 10回以下/エピソード |
| ValidationSimulatorとの性能差 | PPO -48%悪化 | ±5%以内 |

### 定性的目標
- イベント駆動型シミュレーションによる正確な時間管理
- フェーズ別移動時間行列の完全サポート
- ServiceTimeGeneratorの統合による確率的活動時間生成
- ValidationSimulatorとのアーキテクチャ統一

---

## 🏗️ アーキテクチャ概要

### 現状の問題（3層構造）

```
┌─────────────────────────────────────────────────────┐
│ レイヤー1: 時間管理の不整合                           │
│ ────────────────────────────────────────────────── │
│ 問題: 事案間の時間ジャンプにより救急車復帰が遅延      │
│ 影響: 全車出動中が頻発（50-72回/エピソード）         │
└─────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────┐
│ レイヤー2: 移動時間計算の簡略化                       │
│ ────────────────────────────────────────────────── │
│ 問題: 単一の'response'行列のみ使用                   │
│ 正解: response/transport/returnで異なる行列          │
│ 影響: 移動時間の推定誤差が累積                       │
└─────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────┐
│ レイヤー3: サービス時間の固定値使用                   │
│ ────────────────────────────────────────────────── │
│ 問題: 簡易的な固定値（15分、20分など）               │
│ 正解: 傷病度・時間帯・地域を考慮した確率的生成       │
│ 影響: 活動時間の不正確さ                            │
└─────────────────────────────────────────────────────┘
```

### 新しいアーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│           PPO学習インターフェース層                   │
│  ─────────────────────────────────────────────────  │
│  - step(action) → (obs, reward, done, info)        │
│  - reset() → obs                                    │
│  - get_action_mask() → bool[192]                    │
│  - Gym互換API                                       │
└─────────────────────────────────────────────────────┘
                      ↓ 時間の抽象化
┌─────────────────────────────────────────────────────┐
│       イベント駆動シミュレーションコア                │
│  ─────────────────────────────────────────────────  │
│  - 優先度付きキュー（heapq）                         │
│  - 連続時間管理（float秒）                           │
│  - イベント処理ループ                               │
│    • NEW_CALL: 事案発生                             │
│    • AMBULANCE_RETURN: 救急車復帰                   │
│    • CHECKPOINT: 学習用チェックポイント              │
└─────────────────────────────────────────────────────┘
                      ↓ ValidationSimulator互換
┌─────────────────────────────────────────────────────┐
│        ValidationSimulator互換コンポーネント         │
│  ─────────────────────────────────────────────────  │
│  - フェーズ別移動時間行列                            │
│    • response: 出場時                               │
│    • transport: 搬送時                              │
│    • return: 帰署時                                 │
│  - ServiceTimeGeneratorEnhanced                     │
│    • 傷病度別パラメータ                             │
│    • 時間帯・地域考慮                               │
│  - 確率的病院選択モデル                             │
└─────────────────────────────────────────────────────┘
```

---

## 📝 実装計画（修正版：既存機能統合型）

### ⚠️ 重要な認識修正

**当初の設計案の問題点**:
- イベント駆動アーキテクチャは理論的に正しいが、**実用機能が大幅に欠落**
- RewardDesigner、ハイブリッドモード、統計機能などPPO学習に必須の機能が未実装
- 「設計の青写真」であり、そのままでは学習を実行できない

**修正方針**:
```
✅ 正しいアプローチ: イベント駆動 + 既存実用機能の統合
❌ 誤ったアプローチ: 理想的な設計だけで実用機能を無視
```

---

### Phase 1: イベント駆動コア + 既存機能の統合 ⭐ 最優先

**目的**: ValidationSimulatorと同じ時間管理 + PPO学習に必要な全機能

**実装内容**:

#### 1-A. イベント駆動システムの基礎
```python
# イベントクラスの定義
@dataclass
class Event:
    time: float              # イベント発生時刻（秒）
    event_type: EventType    # NEW_CALL, AMBULANCE_RETURN, etc.
    data: Dict[str, Any]    # イベント固有データ

# イベントキューの管理
self.event_queue = []  # heapq
self.current_time = 0.0  # 連続時間
```

#### 1-B. 既存の実用機能を統合（★重要★）

**移植元**: `reinforcement_learning/environment/ems_environment.py`（現行版）

**必須で移植すべきメソッド**:

1. **初期化関連**（`__init__`に追加）:
```python
# RewardDesignerの初期化（既存コードそのまま使用）
from .reward_designer import RewardDesigner
self.reward_designer = RewardDesigner(self.config)

# DispatchLoggerの初期化
from .dispatch_logger import DispatchLogger  
self.dispatch_logger = DispatchLogger(enabled=True)

# ハイブリッドモードの設定
self.hybrid_mode = self.config.get('hybrid_mode', {}).get('enabled', False)
if self.hybrid_mode:
    self.severe_conditions = self.config.get('hybrid_mode', {}).get(
        'severity_classification', {}
    ).get('severe_conditions', ['重症', '重篤', '死亡'])
    self.direct_dispatch_count = 0
    self.ppo_dispatch_count = 0
    print("✓ ハイブリッドモード有効")
```

2. **報酬計算**（現行版の`_calculate_reward_detailed`をそのまま移植）:
```python
def _calculate_reward_detailed(self, response_time_minutes, severity):
    """
    目標時間やカバレッジを考慮した、より詳細な報酬関数。
    
    ★既存のems_environment.pyから完全移植★
    """
    reward = 10.0  # 基本報酬
    if is_severe_condition(severity):  # 重症系
        if response_time_minutes <= 6.0: 
            reward += 20.0  # 6分以内ボーナス
        else: 
            reward -= min((response_time_minutes - 6.0) * 2.0, 30.0)
    else:  # 軽症系
        if response_time_minutes <= 13.0: 
            reward += 5.0
        else: 
            reward -= min((response_time_minutes - 13.0) * 0.5, 10.0)
    
    # カバレッジボーナス
    available_ratio = sum(
        1 for s in self.ambulance_states.values() if s['status'] == 'available'
    ) / self.action_dim
    if available_ratio > 0.3: 
        reward += 5.0 * available_ratio
    
    return reward
```

3. **ハイブリッドモードのロジック**（`step`メソッドに組み込む）:
```python
def step(self, action: int):
    """
    PPO学習用のステップ実行（ハイブリッドモード対応）
    """
    # 事案が存在しない場合の処理
    if self.pending_call is None:
        self._advance_to_next_call()
    
    if self.pending_call is None:
        return self._get_observation(), 0.0, True, {}
    
    current_incident = self.pending_call
    
    # ★ハイブリッドモード: 重症系は直近隊を強制★
    if self.hybrid_mode and is_severe_condition(current_incident['severity']):
        # 直近隊を選択
        action_to_take = self.get_optimal_action()
        self.direct_dispatch_count += 1
        
        # 配車実行
        reward, info = self._execute_dispatch(action_to_take, current_incident)
        reward = 0.0  # 学習対象外なので報酬は0
        info['dispatch_type'] = 'direct_closest'
        info['skipped_learning'] = True
    else:
        # PPOで学習
        action_to_take = action
        self.ppo_dispatch_count += 1
        
        # 配車実行
        reward, info = self._execute_dispatch(action_to_take, current_incident)
        info['dispatch_type'] = 'ppo_learning'
    
    # 次の事案へ進む
    self._advance_to_next_call()
    
    observation = self._get_observation()
    done = self._is_episode_done()
    info['episode_stats'] = self.get_episode_statistics()
    
    return observation, reward, done, info
```

4. **学習補助メソッド**（現行版からそのまま移植）:
```python
def get_optimal_action(self) -> Optional[int]:
    """
    現在の事案に対し、最も早く到着できる救急車を返す
    
    ★既存のems_environment.pyから完全移植★
    """
    if not self.pending_call: 
        return None
    
    best_action, min_time = None, float('inf')
    for amb_id, state in self.ambulance_states.items():
        if state['status'] == 'available':
            time = self._get_travel_time(
                state['current_h3'], 
                self.pending_call['h3_index'], 
                'response'
            )
            if time < min_time: 
                min_time, best_action = time, amb_id
    
    return best_action

def render(self, mode='human'):
    """
    環境の現在の状態を表示
    
    ★既存のems_environment.pyから完全移植★
    """
    if mode == 'human':
        print(f"\n--- Time: {self.current_time/3600.0:.2f}h ---")
        if self.pending_call: 
            print(f"  Incident: {self.pending_call['severity']} at {self.pending_call['h3_index']}")
        available = sum(1 for s in self.ambulance_states.values() if s['status'] == 'available')
        print(f"  Available Ambulances: {available}/{self.action_dim}")

def get_episode_statistics(self) -> Dict:
    """
    エピソード統計を取得
    
    ★元のems_environment1027.pyから完全移植★
    詳細な統計情報を返す（約50行のメソッド）
    """
    # ... 既存の実装をそのまま使用 ...
    pass
```

5. **現実的な初期化**（`reset`メソッドで使用）:
```python
def _initialize_ambulances_realistic(self):
    """
    現実的な救急車初期化処理
    
    ★既存のems_environment.pyから完全移植★
    一部の救急車が活動中の状態を再現
    """
    self.ambulance_states = {}
    
    for amb_id, (_, row) in enumerate(self.ambulance_data.iterrows()):
        if amb_id >= self.action_dim:
            break
        
        station_h3 = h3.latlng_to_cell(row['latitude'], row['longitude'], 9)
        
        # 50-70%の確率で初期活動中とする
        is_busy = np.random.uniform() < np.random.uniform(0.5, 0.7)
        
        self.ambulance_states[amb_id] = {
            'status': 'dispatched' if is_busy else 'available',
            'completion_time': self.current_time + np.random.uniform(0, 1800) if is_busy else 0,
            'current_h3': station_h3,
            'station_h3': station_h3,
            'calls_today': 1 if is_busy else 0
        }
        
        # ★イベント駆動: 初期活動中の救急車の復帰イベントをスケジュール★
        if is_busy:
            return_event = Event(
                time=self.ambulance_states[amb_id]['completion_time'],
                event_type=EventType.AMBULANCE_RETURN,
                data={'ambulance_id': amb_id}
            )
            self._schedule_event(return_event)
    
    available_count = sum(1 for st in self.ambulance_states.values() if st['status'] == 'available')
    print(f"  救急車初期化完了: {len(self.ambulance_states)}台 (available: {available_count}台)")
```

**期待効果**:
- ✅ 全車出動中問題の解消（50-72回 → 10回以下）
- ✅ PPO学習に必要な全機能が動作
- ✅ ハイブリッドモードの継続動作
- ✅ 既存のトレーニングスクリプトとの互換性維持

**修正ファイル**:
- `reinforcement_learning/environment/ems_environment_v2.py`（新規作成）
  - 既存の`ems_environment.py`をベースに、イベント駆動コアを統合

**検証方法**:
```python
# test_integrated_environment.py
def test_all_features():
    """統合環境の全機能テスト"""
    env = ValidationIntegratedEMSEnvironment(...)
    
    # 1. イベント駆動の動作確認
    env.reset()
    assert len(env.event_queue) > 0, "イベントキューが空"
    
    # 2. ハイブリッドモードの動作確認
    if env.hybrid_mode:
        # 重症事案で直近隊が選択されるか
        pass
    
    # 3. 報酬計算の動作確認
    obs, reward, done, info = env.step(0)
    assert 'dispatch_type' in info, "dispatch_typeが欠落"
    
    # 4. 統計情報の取得確認
    stats = env.get_episode_statistics()
    assert 'response_times' in stats, "統計情報が不完全"
    
    # 5. 全車出動中の頻度確認
    all_busy_count = 0
    for _ in range(1000):
        if not env.get_action_mask().any():
            all_busy_count += 1
        env.step(np.random.choice(np.where(env.get_action_mask())[0]))
    
    assert all_busy_count < 10, f"全車出動中が多すぎる: {all_busy_count}回"
    print("✅ 全機能テスト合格")
```

---

### Phase 2: フェーズ別移動時間行列の統合

**目的**: response/transport/returnで異なる移動時間を使用

**実装内容**:
```python
def _get_travel_time_by_phase(self, from_h3, to_h3, phase):
    """
    フェーズに応じた移動時間行列を使用
    
    phase: 'response', 'transport', 'return'
    """
    matrix = self.travel_time_matrices[phase]  # ★フェーズ別
    from_idx = self.grid_mapping[from_h3]
    to_idx = self.grid_mapping[to_h3]
    return matrix[from_idx, to_idx]

def _calculate_ambulance_activity_time(self, ambulance_id, call):
    # Response phase
    response_time = self._get_travel_time_by_phase(
        amb_h3, call_h3, 'response'  # ★
    )
    
    # Transport phase
    transport_time = self._get_travel_time_by_phase(
        call_h3, hospital_h3, 'transport'  # ★
    )
    
    # Return phase
    return_time = self._get_travel_time_by_phase(
        hospital_h3, station_h3, 'return'  # ★
    )
```

**期待効果**:
- ✅ 移動時間の精度向上
- ✅ ValidationSimulatorとの一致率向上
- ✅ 平均応答時間の改善（21.67分 → 8分以下）

**修正ファイル**:
- `reinforcement_learning/environment/ems_environment.py`
  - `_calculate_ambulance_activity_time()` メソッド
  - `_get_travel_time_by_phase()` メソッド（新規追加）

---

### Phase 3: ServiceTimeGeneratorの完全統合

**目的**: 確率的なサービス時間生成

**実装内容**:
```python
# 既存のServiceTimeGeneratorEnhancedを使用
from data.tokyo.service_time_analysis.service_time_generator_enhanced import ServiceTimeGeneratorEnhanced

def __init__(self, ...):
    # 階層的パラメータファイルを読み込む
    params_path = "data/tokyo/service_time_analysis/lognormal_parameters_hierarchical.json"
    self.service_time_generator = ServiceTimeGeneratorEnhanced(params_path)

def _calculate_ambulance_activity_time(self, ambulance_id, call):
    # 現場活動時間（確率的）
    on_scene_time_min = self.service_time_generator.generate_time(
        severity=call['severity'],
        phase='on_scene_time',
        current_time=call['datetime']  # 時刻情報を渡す
    )
    
    # 病院活動時間（確率的）
    hospital_time_min = self.service_time_generator.generate_time(
        severity=call['severity'],
        phase='hospital_time',
        current_time=call['datetime']
    )
```

**期待効果**:
- ✅ 活動時間の変動性を再現
- ✅ 傷病度別の活動時間特性を反映
- ✅ 時間帯・地域の影響を考慮

**修正ファイル**:
- `reinforcement_learning/environment/ems_environment.py`
  - `_calculate_ambulance_activity_time()` メソッド

---

### Phase 4: 状態エンコーダの適応

**目的**: 連続時間対応の状態エンコーディング

**実装内容**:
```python
# state_encoder.pyの修正
def _encode_temporal(self, state_dict):
    """
    時間特徴量のエンコーディング
    
    変更点: episode_stepの代わりにcurrent_timeを使用
    """
    current_time = state_dict.get('current_time', 0.0)  # 秒単位
    
    # エピソード進行度（0→1）
    episode_duration = self.config['data']['episode_duration_hours'] * 3600
    progress = min(current_time / episode_duration, 1.0)
    
    # 時刻（0→1）
    hour_of_day = (current_time / 3600) % 24
    time_of_day = hour_of_day / 24.0
    
    return np.array([progress, time_of_day, ...])
```

**期待効果**:
- ✅ 時間情報の正確な表現
- ✅ エピソード進行度の連続的な変化
- ✅ 学習の安定性向上

**修正ファイル**:
- `reinforcement_learning/environment/state_encoder.py`
  - `_encode_temporal()` メソッド
  - `encode_state()` メソッド

---

### Phase 5: 報酬関数の調整

**目的**: 新しい環境に適した報酬設計

**実装内容**:
```python
def _calculate_reward(self, response_time_min, severity):
    """
    連続報酬関数（既存のRewardDesignerを使用）
    
    調整点:
    - カバレッジボーナスの重み調整
    - 時間ペナルティの勾配調整
    """
    reward = self.reward_designer.calculate_reward({
        'response_time': response_time_min,
        'severity': severity,
        'coverage_ratio': self._calculate_coverage_ratio()
    })
    
    return reward
```

**期待効果**:
- ✅ 環境の精度向上に伴う報酬の再調整
- ✅ 学習の収束性向上

**修正ファイル**:
- `reinforcement_learning/environment/reward_designer.py`（微調整のみ）

---

## 🔧 実装手順（修正版：既存機能優先統合）

### Step 1: 既存ファイルのベースコピーと基本構造の追加

```bash
# 既存の実装をベースに新しいファイルを作成
cp reinforcement_learning/environment/ems_environment.py \
   reinforcement_learning/environment/ems_environment_v2.py
```

**実装内容**:

1. **イベントクラスの追加**（ファイル冒頭に追加）:
```python
import heapq
from enum import Enum
from dataclasses import dataclass

class EventType(Enum):
    NEW_CALL = "new_call"
    AMBULANCE_RETURN = "ambulance_return"
    EPISODE_END = "episode_end"

@dataclass
class Event:
    time: float
    event_type: EventType
    data: Dict[str, Any]
    
    def __lt__(self, other):
        return self.time < other.time
```

2. **`__init__`メソッドに追加**:
```python
# イベントキューの初期化
self.event_queue = []
self.current_time = 0.0

# ★既存の初期化コードはそのまま維持★
# RewardDesigner、DispatchLogger、hybrid_modeなど
```

3. **イベント管理メソッドの追加**（クラスの末尾に追加）:
```python
def _schedule_event(self, event: Event):
    """イベントをキューに追加"""
    heapq.heappush(self.event_queue, event)

def _process_next_event(self) -> Optional[Event]:
    """次のイベントを処理"""
    if not self.event_queue:
        return None
    
    event = heapq.heappop(self.event_queue)
    old_time = self.current_time
    self.current_time = event.time
    
    if event.event_type == EventType.AMBULANCE_RETURN:
        self._handle_ambulance_return_event(event)
    elif event.event_type == EventType.NEW_CALL:
        self._handle_new_call_event(event)
    
    return event

def _handle_ambulance_return_event(self, event: Event):
    """救急車復帰イベントの処理"""
    ambulance_id = event.data['ambulance_id']
    if ambulance_id in self.ambulance_states:
        self.ambulance_states[ambulance_id]['status'] = 'available'
        self.ambulance_states[ambulance_id]['current_h3'] = \
            self.ambulance_states[ambulance_id]['station_h3']

def _handle_new_call_event(self, event: Event):
    """新規事案イベントの処理"""
    self.pending_call = event.data
```

**チェックリスト**:
- ☑ イベントクラスの定義完了
- ☑ イベントキューの初期化完了
- ☑ 基本的なイベント処理メソッドの追加完了
- ☑ 既存の初期化コード（RewardDesigner、DispatchLogger等）が維持されている

**期間**: 半日

---

### Step 2: `reset()`メソッドの改修（イベント駆動対応）

**実装内容**:

1. **イベントキューのリセット処理を追加**:
```python
def reset(self) -> np.ndarray:
    """エピソードのリセット（イベント駆動版）"""
    # イベントキューとタイマーのクリア
    self.event_queue = []
    self.current_time = 0.0
    
    # ★既存のリセット処理はそのまま維持★
    # 期間選択、データ読み込みなど
    periods = (self.config['data']['train_periods'] if self.mode == "train" 
              else self.config['data']['eval_periods'])
    period = periods[np.random.randint(len(periods))]
    calls_df = self.data_cache.get_period_data(period['start_date'], period['end_date'])
    self.current_episode_calls = self._prepare_episode_calls(calls_df, ...)
    
    if not self.current_episode_calls:
        return np.zeros(self.state_dim)
    
    self.episode_start_time = self.current_episode_calls[0]['datetime']
    
    # ★新規追加: 全事案をイベントとしてスケジュール★
    for call in self.current_episode_calls:
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
    
    # ★既存の救急車初期化を維持（イベントスケジューリングを追加）★
    self._initialize_ambulances_realistic()
    
    # 統計のリセット（既存）
    self._reset_statistics()
    
    # 最初の事案まで進める
    self._advance_to_next_call()
    
    return self._get_observation()
```

2. **`_initialize_ambulances_realistic()`の改修**:
```python
def _initialize_ambulances_realistic(self):
    """現実的な救急車初期化（イベント駆動対応版）"""
    self.ambulance_states = {}
    
    for amb_id, (_, row) in enumerate(self.ambulance_data.iterrows()):
        if amb_id >= self.action_dim:
            break
        
        station_h3 = h3.latlng_to_cell(row['latitude'], row['longitude'], 9)
        is_busy = np.random.uniform() < np.random.uniform(0.5, 0.7)
        
        self.ambulance_states[amb_id] = {
            'status': 'dispatched' if is_busy else 'available',
            'completion_time': self.current_time + np.random.uniform(0, 1800) if is_busy else 0,
            'current_h3': station_h3,
            'station_h3': station_h3,
            'calls_today': 1 if is_busy else 0
        }
        
        # ★新規追加: 初期活動中の救急車の復帰イベントをスケジュール★
        if is_busy:
            return_event = Event(
                time=self.ambulance_states[amb_id]['completion_time'],
                event_type=EventType.AMBULANCE_RETURN,
                data={'ambulance_id': amb_id}
            )
            self._schedule_event(return_event)
    
    available_count = sum(1 for st in self.ambulance_states.values() if st['status'] == 'available')
    print(f"  救急車初期化完了: {len(self.ambulance_states)}台 (available: {available_count}台)")
```

**チェックリスト**:
- ☑ `reset()`でイベントキューをクリア
- ☑ 全事案をイベントとしてスケジュール
- ☑ 初期活動中の救急車の復帰イベントをスケジュール
- ☑ 既存のデータ読み込みロジックが維持されている

**期間**: 半日

---

### Step 3: `step()`メソッドの改修（イベント駆動 + ハイブリッドモード）

**実装内容**:

```python
def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
    """
    PPO学習用のステップ実行（イベント駆動 + ハイブリッドモード対応）
    """
    # 事案が存在しない場合、次の事案まで進める
    if self.pending_call is None:
        self._advance_to_next_call()
    
    if self.pending_call is None:
        return self._get_observation(), 0.0, True, {}
    
    current_incident = self.pending_call
    
    # ★ハイブリッドモード（既存ロジックを維持）★
    if self.hybrid_mode and is_severe_condition(current_incident['severity']):
        # 重症系: 直近隊を強制
        action_to_take = self.get_optimal_action()
        self.direct_dispatch_count += 1
        dispatch_type = 'direct_closest'
        skipped_learning = True
    else:
        # 軽症系: PPOで学習
        action_to_take = action
        self.ppo_dispatch_count += 1
        dispatch_type = 'ppo_learning'
        skipped_learning = False
    
    # マスクチェック
    mask = self.get_action_mask()
    if not mask[action_to_take]:
        # 無効なアクションの場合、有効なアクションからランダム選択
        valid_actions = np.where(mask)[0]
        if len(valid_actions) > 0:
            action_to_take = np.random.choice(valid_actions)
        else:
            # 全車出動中
            return self._get_observation(), -100.0, False, {
                'success': False,
                'reason': 'all_busy'
            }
    
    # ★配車実行（既存の詳細計算を使用）★
    total_time_sec, details = self._calculate_ambulance_completion_time(
        action_to_take, current_incident
    )
    
    # 救急車を出動中状態に更新
    self.ambulance_states[action_to_take]['status'] = 'dispatched'
    self.ambulance_states[action_to_take]['calls_today'] += 1
    
    # ★新規追加: 復帰イベントをスケジュール★
    return_time = self.current_time + total_time_sec
    return_event = Event(
        time=return_time,
        event_type=EventType.AMBULANCE_RETURN,
        data={'ambulance_id': action_to_take}
    )
    self._schedule_event(return_event)
    
    # ★報酬計算（既存のRewardDesignerを使用）★
    if skipped_learning:
        reward = 0.0  # 学習対象外
    else:
        reward = self._calculate_reward_detailed(
            details['response_time'], 
            current_incident['severity']
        )
    
    # 統計の更新（既存）
    details['severity'] = current_incident['severity']
    self._update_statistics(details)
    
    # 次の事案へ進む
    self._advance_to_next_call()
    
    observation = self._get_observation()
    done = self._is_episode_done()
    
    info = {
        'success': True,
        'ambulance_id': action_to_take,
        'response_time': details['response_time'],
        'dispatch_type': dispatch_type,
        'skipped_learning': skipped_learning,
        'episode_stats': self.get_episode_statistics()
    }
    
    return observation, reward, done, info
```

**チェックリスト**:
- ☑ ハイブリッドモードのロジックが維持されている
- ☑ 復帰イベントが正しくスケジュールされる
- ☑ 既存の報酬計算（RewardDesigner）が使用されている
- ☑ 既存の統計更新が動作している

**期間**: 1日

---

### Step 4: `_advance_to_next_call()`メソッドの改修

**実装内容**:

```python
def _advance_to_next_call(self):
    """
    次の事案イベントまでシミュレーションを進める（イベント駆動版）
    
    重要: 事案間で発生する救急車復帰イベントを全て処理
    """
    self.pending_call = None
    
    # 次のNEW_CALLイベントまでイベントを処理
    while self.event_queue:
        next_event = self.event_queue[0]  # peek
        
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
```

**チェックリスト**:
- ☑ 事案間で救急車復帰イベントが処理される
- ☑ エピソード終了イベントが正しく処理される

**期間**: 半日

---

### Step 5: 統合テストと検証

```python
# test_ems_environment_v2.py
def test_event_driven_time_management():
    """イベント駆動時間管理のテスト"""
    env = ValidationIntegratedEMSEnvironment(...)
    env.reset()
    
    # 全車出動中の頻度をテスト
    all_busy_count = 0
    for _ in range(1000):
        mask = env.get_action_mask()
        if not mask.any():
            all_busy_count += 1
        # ...
    
    assert all_busy_count < 10, f"全車出動中が多すぎる: {all_busy_count}回"

def test_phase_specific_travel_times():
    """フェーズ別移動時間のテスト"""
    env = ValidationIntegratedEMSEnvironment(...)
    
    # response, transport, returnで異なる時間が返されることを確認
    response_time = env._get_travel_time_by_phase(h3_a, h3_b, 'response')
    transport_time = env._get_travel_time_by_phase(h3_a, h3_b, 'transport')
    return_time = env._get_travel_time_by_phase(h3_a, h3_b, 'return')
    
    # 通常、response < transport（緊急走行 vs 通常走行）
    assert response_time <= transport_time

def test_service_time_generation():
    """サービス時間生成のテスト"""
    env = ValidationIntegratedEMSEnvironment(...)
    
    # 重症と軽症で異なる時間が生成されることを確認
    critical_times = [
        env.service_time_generator.generate_time('重症', 'on_scene_time', datetime.now())
        for _ in range(100)
    ]
    mild_times = [
        env.service_time_generator.generate_time('軽症', 'on_scene_time', datetime.now())
        for _ in range(100)
    ]
    
    # 重症の方が活動時間が長い傾向
    assert np.mean(critical_times) > np.mean(mild_times)
```

**期間**: 2日

---

### Step 5: トレーニングスクリプトの更新

```python
# train_ppo_v2.py
from reinforcement_learning.environment.ems_environment_v2 import ValidationIntegratedEMSEnvironment

def main():
    # 新しい環境を使用
    env = ValidationIntegratedEMSEnvironment(
        config_path="reinforcement_learning/experiments/config_continuous.yaml",
        mode="train"
    )
    
    # トレーナーの初期化（既存のTrainerクラスを使用可能）
    trainer = PPOTrainer(
        env=env,
        config=config
    )
    
    # 学習実行
    trainer.train()
```

**期間**: 1日

---

### Step 6: 比較実験と検証

**実験内容**:
1. **旧環境 vs 新環境の比較**
   - 全車出動中頻度
   - 平均応答時間
   - 6分達成率

2. **新環境 vs ValidationSimulatorの比較**
   - 同じ事案データで実行
   - 統計指標の一致度確認

**検証スクリプト**:
```python
# compare_environments.py
def compare_old_vs_new():
    """旧環境と新環境の比較"""
    # 旧環境
    env_old = EMSEnvironment(config_path, mode="eval")
    stats_old = run_episode(env_old)
    
    # 新環境
    env_new = ValidationIntegratedEMSEnvironment(config_path, mode="eval")
    stats_new = run_episode(env_new)
    
    print("比較結果:")
    print(f"  全車出動中: {stats_old['all_busy_count']}回 → {stats_new['all_busy_count']}回")
    print(f"  平均応答時間: {stats_old['mean_response']:.2f}分 → {stats_new['mean_response']:.2f}分")
    print(f"  6分達成率: {stats_old['6min_rate']:.1f}% → {stats_new['6min_rate']:.1f}%")
```

**期間**: 2日

---

## 📊 マイグレーション戦略

### オプション1: 段階的移行（推奨）

```
Week 1: Phase 1実装 → テスト
Week 2: Phase 2-3実装 → テスト
Week 3: Phase 4-5実装 → 統合テスト
Week 4: 比較実験 → 本番移行
```

**メリット**:
- ✅ リスク分散
- ✅ 各段階で検証可能
- ✅ 問題の早期発見

### オプション2: 並行開発

```
旧環境: ems_environment.py（維持）
新環境: ems_environment_v2.py（開発）
→ 検証完了後に置き換え
```

**メリット**:
- ✅ 既存の学習を継続可能
- ✅ 比較実験が容易

---

## ⚠️ リスクと対策

### リスク1: 学習の不安定化

**原因**: 環境の精度向上により、難易度が上昇

**対策**:
- ハイパーパラメータの再調整（学習率、エントロピー係数）
- カリキュラム学習の導入（簡単な事案から開始）

### リスク2: 計算コストの増加

**原因**: イベント駆動処理のオーバーヘッド

**対策**:
- イベント処理の最適化
- 並列化の検討（複数エピソードの並行実行）

### リスク3: 既存モデルの互換性喪失

**原因**: 状態表現の変更

**対策**:
- 新環境で再学習（避けられない）
- Transfer Learningの検討

---

## 📈 成功指標

### 短期（Phase 1-3完了時）

| 指標 | 目標値 |
|------|--------|
| 全車出動中頻度 | 10回以下/エピソード |
| 平均応答時間 | 10分以下 |
| 単体テスト通過率 | 100% |

### 中期（Phase 4-5完了時）

| 指標 | 目標値 |
|------|--------|
| 6分達成率 | 30%以上 |
| ValidationSimulatorとの応答時間差 | ±10%以内 |
| 学習の収束 | 100エピソード以内 |

### 長期（本番運用時）

| 指標 | 目標値 |
|------|--------|
| ValidationSimulatorとの性能差 | ±5%以内 |
| 6分達成率 | 40%以上 |
| 学習の安定性 | 1000エピソード連続収束 |

---

## 🎯 次のアクション

1. **即座に実施**:
   - ✅ 設計ドキュメントのレビュー（完了）
   - ⬜ Phase 1の実装開始（イベント駆動コア）

2. **今週中**:
   - ⬜ Phase 1の単体テスト
   - ⬜ Phase 2の実装開始

3. **来週**:
   - ⬜ Phase 3-4の実装
   - ⬜ 統合テスト

4. **2週間後**:
   - ⬜ Phase 5の実装
   - ⬜ 比較実験の実施
   - ⬜ 本番環境への移行判断

---

## 📚 参考資料

1. **reportファイル**:
   - `old/bu/detailed_analysis_report.md`: 時間管理問題の詳細分析
   - `old/bu/ems_environment_investigation_report.md`: 学習収束問題の調査

2. **既存実装**:
   - `validation_simulation.py`: イベント駆動アーキテクチャの参考実装
   - `old/bu/ems_environment1027.py`: 元のステップベース実装

3. **設計ドキュメント**:
   - `reinforcement_learning/environment/ems_environment_v2_design.py`: 新しいアーキテクチャの詳細

---

## 💬 質問と回答

**Q1: なぜ既存の環境を修正するのではなく、新しく作り直すのか？**

A: 時間管理の根本的なアーキテクチャが異なるため、部分的な修正では対応できません。イベント駆動型への変更は全体の再設計が必要です。

**Q2: 既存のモデルは使えなくなるのか？**

A: 状態表現が変わるため、既存モデルは使用できません。ただし、新環境の方が精度が高いため、再学習により性能向上が期待できます。

**Q3: ValidationSimulatorとの完全な一致は可能か？**

A: アーキテクチャレベルでは一致しますが、乱数シードの違いなどにより、個別の事案レベルでは差異が生じます。統計的な一致（±5%）を目標とします。

**Q4: 実装にどのくらいの期間がかかるか？**

A: 段階的実装で3-4週間を想定しています。Phase 1（イベント駆動コア）が最も重要で、1週間程度で完了予定です。

---

---

## 📌 実装の優先順位マトリックス

### 必須コンポーネント（Phase 1で完全実装）

| コンポーネント | 移植元 | 役割 | 実装難易度 |
|--------------|--------|------|-----------|
| **RewardDesigner** | `ems_environment.py` L93 | 報酬計算 | ★☆☆ (そのまま使用) |
| **DispatchLogger** | `ems_environment.py` L94 | ログ記録 | ★☆☆ (そのまま使用) |
| **ハイブリッドモード** | `ems_environment.py` L96-100 | 重症/軽症振り分け | ★★☆ (step()に統合) |
| **get_optimal_action** | `ems_environment.py` L402-410 | 最適行動取得 | ★☆☆ (そのまま使用) |
| **get_episode_statistics** | `ems_environment1027.py` L1428-1478 | 統計収集 | ★☆☆ (そのまま使用) |
| **_initialize_ambulances_realistic** | `ems_environment.py` L314-336 | 現実的初期化 | ★★☆ (イベント追加) |
| **render** | `ems_environment.py` L412-420 | 可視化 | ★☆☆ (そのまま使用) |

### 新規追加コンポーネント（イベント駆動）

| コンポーネント | 役割 | 実装難易度 |
|--------------|------|-----------|
| **Event, EventType** | イベント定義 | ★☆☆ (データクラス) |
| **event_queue** | イベントキュー | ★☆☆ (heapq使用) |
| **_schedule_event** | イベント追加 | ★☆☆ (heappush) |
| **_process_next_event** | イベント処理 | ★★☆ (ループ制御) |
| **_handle_ambulance_return_event** | 救急車復帰処理 | ★☆☆ (状態更新) |
| **_advance_to_next_call** (改修) | 事案間イベント処理 | ★★★ (ロジック統合) |

### 実装の3段階アプローチ

```
┌─────────────────────────────────────────────────┐
│ Stage 1: 基礎（半日）                            │
│ ├─ イベントクラス定義                           │
│ ├─ イベントキュー初期化                         │
│ └─ 基本イベント処理メソッド                     │
│    → 既存機能は100%維持                         │
└─────────────────────────────────────────────────┘
            ↓ テスト・検証
┌─────────────────────────────────────────────────┐
│ Stage 2: 統合（1.5日）                          │
│ ├─ reset()メソッドの改修                        │
│ ├─ step()メソッドの改修                         │
│ └─ _advance_to_next_call()の改修                │
│    → イベント駆動 + 既存機能の融合              │
└─────────────────────────────────────────────────┘
            ↓ テスト・検証
┌─────────────────────────────────────────────────┐
│ Stage 3: 最適化（1日）                          │
│ ├─ フェーズ別移動時間行列の活用                 │
│ ├─ ServiceTimeGeneratorの統合                   │
│ └─ 性能検証と微調整                             │
│    → ValidationSimulatorとの一致度検証          │
└─────────────────────────────────────────────────┘
```

---

## 🚨 実装時の重要な注意事項

### ❌ やってはいけないこと

1. **既存メソッドの削除や大幅変更**
   - `_calculate_reward_detailed` は絶対に変更しない
   - `get_episode_statistics` のロジックを保持
   - `hybrid_mode` の振り分けロジックを維持

2. **新規設計による実装**
   - "きれいな設計"を優先して既存機能を無視しない
   - 理想的なアーキテクチャより、動作する統合を優先

3. **テストなしの大規模変更**
   - 各Stageで必ず動作確認
   - 既存機能が壊れていないか検証

### ✅ やるべきこと

1. **既存コードの最大限の再利用**
   - コピー&ペーストを積極的に活用
   - 動いているコードは変更しない

2. **段階的な統合**
   - イベントクラス追加 → テスト
   - reset()改修 → テスト
   - step()改修 → テスト

3. **後方互換性の維持**
   - 既存のトレーニングスクリプトが動作すること
   - 既存の設定ファイル（config.yaml）が使えること

---

## 📝 実装チェックリスト

### Phase 1完了時の確認項目

- [ ] イベントクラス（Event, EventType）が定義されている
- [ ] `__init__`でイベントキューが初期化されている
- [ ] `_schedule_event`, `_process_next_event`が実装されている
- [ ] RewardDesignerが初期化され、動作している
- [ ] DispatchLoggerが初期化され、動作している
- [ ] hybrid_modeの設定が読み込まれている
- [ ] get_optimal_actionがそのまま使用可能
- [ ] get_episode_statisticsがそのまま使用可能
- [ ] renderがそのまま使用可能

### Phase 2完了時の確認項目

- [ ] reset()で全事案がイベントとしてスケジュールされる
- [ ] reset()で初期活動中の救急車の復帰イベントがスケジュールされる
- [ ] step()でハイブリッドモードが正しく動作する
- [ ] step()で復帰イベントが正しくスケジュールされる
- [ ] _advance_to_next_call()で事案間のイベントが処理される
- [ ] 全車出動中の頻度が10回以下/エピソードになっている

### Phase 3完了時の確認項目

- [ ] フェーズ別移動時間行列が使用されている
- [ ] ServiceTimeGeneratorが統合されている
- [ ] ValidationSimulatorとの応答時間差が±10%以内
- [ ] 既存のトレーニングスクリプトが動作する
- [ ] 学習が正常に収束する

---

**承認者**: _______________  **日付**: _______________

