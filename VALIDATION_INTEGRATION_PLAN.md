# ValidationSimulator完全統合計画書

## 📋 目的

現在のPPO学習環境(`ems_environment.py`)とValidationSimulatorの環境差を解消し、学習精度と検証精度を一致させる。

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

## 📝 実装計画（5段階）

### Phase 1: イベント駆動コアの実装 ⭐ 最優先

**目的**: ValidationSimulatorと同じ時間管理システムの導入

**実装内容**:
```python
# 1. イベントクラスの定義
@dataclass
class Event:
    time: float              # イベント発生時刻（秒）
    event_type: EventType    # NEW_CALL, AMBULANCE_RETURN, etc.
    data: Dict[str, Any]    # イベント固有データ

# 2. イベントキューの管理
self.event_queue = []  # heapq
self.current_time = 0.0  # 連続時間

# 3. イベント処理ループ
def _process_next_event(self):
    event = heapq.heappop(self.event_queue)
    self.current_time = event.time  # 時間を進める
    
    if event.event_type == EventType.AMBULANCE_RETURN:
        # 救急車を復帰させる
        self.ambulance_states[event.data['ambulance_id']]['status'] = 'available'
```

**期待効果**:
- ✅ 全車出動中問題の解消（50-72回 → 10回以下）
- ✅ 救急車の正確な復帰タイミング
- ✅ 事案間での状態更新の保証

**修正ファイル**:
- `reinforcement_learning/environment/ems_environment.py` (全面改修)

**検証方法**:
```python
# テストスクリプト
env = EMSEnvironment(...)
obs = env.reset()

# 全車出動中の発生回数をカウント
all_busy_count = 0
for step in range(1000):
    mask = env.get_action_mask()
    if not mask.any():
        all_busy_count += 1
    # ...

print(f"全車出動中: {all_busy_count}回")  # 目標: 10回以下
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

## 🔧 実装手順（ステップバイステップ）

### Step 1: 新ファイルの作成と基本構造の実装

```bash
# 設計ドキュメントから実装ファイルを作成
cp reinforcement_learning/environment/ems_environment_v2_design.py \
   reinforcement_learning/environment/ems_environment_v2.py
```

**実装内容**:
1. イベントクラスの定義
2. イベントキューの管理機能
3. 基本的なイベント処理ループ

**期間**: 1日

---

### Step 2: ValidationSimulator互換コンポーネントの移植

**実装内容**:
1. フェーズ別移動時間行列の読み込み
2. ServiceTimeGeneratorの統合
3. 病院選択モデルの統合

**修正箇所**:
- `_load_travel_time_matrices()`: 3つの行列を読み込む
- `_get_travel_time_by_phase()`: フェーズ指定対応
- `_calculate_ambulance_activity_time()`: 詳細計算

**期間**: 2日

---

### Step 3: PPO学習インターフェースの実装

**実装内容**:
1. `step()` メソッドの実装
2. `reset()` メソッドの実装
3. `get_action_mask()` の実装
4. 事案イベントのスケジューリング

**期間**: 2日

---

### Step 4: 単体テストと統合テスト

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

**承認者**: _______________  **日付**: _______________

