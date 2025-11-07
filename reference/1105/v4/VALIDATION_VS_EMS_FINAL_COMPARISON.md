# ValidationSimulator vs EMSEnvironment 最終比較

**作成日**: 2025年11月7日  
**目的**: 両システムの動作と性能の完全な比較

---

## 📊 ValidationSimulator の実際の動作

### debug_output/baseline/simulation_report.json（直近隊運用）

```json
{
  "total_calls": 1526,
  "completed_calls": 1442,  ← 94.5%
  "mean_response_time": 7.47分,
  "6min_achievement_rate": 35.08%
}
```

**重要な発見:**
1. ✅ **completed_calls: 1442件（94.5%）**
2. ❌ **84件（5.5%）は完了していない**
3. ✅ 平均応答時間: 7.47分
4. ✅ 6分達成率: 35.08%

---

### debug_output/simulation_report.json（古い実装）

```json
{
  "total_calls": 1526,
  "completed_calls": 1421,  ← 93.1%
  "mean_response_time": 14.22分,
  "6min_achievement_rate": 4.57%
}
```

**重要な発見:**
1. ✅ completed_calls: 1421件（93.1%）
2. ❌ **105件（6.9%）は完了していない**
3. ❌ 平均応答時間: 14.22分（遅い）
4. ❌ 6分達成率: 4.57%（低い）

---

## 📊 EMSEnvironment の実際の動作

### 5000エピソード学習結果（最終）

```
total_calls: 1168件（76.6%）
タイムアウト: 358件（23.4%）
mean_response_time: 19.70分
6min_achievement_rate: 3.60%
```

### 初期エピソード（教師あり学習中）

```
total_calls: 約1200件（78.6%）
mean_response_time: 12.62分
6min_achievement_rate: 27.5%
```

---

## 🔍 完了率の比較

| システム | 完了件数 | 完了率 | 未完了 |
|----------|----------|--------|--------|
| **ValidationSimulator (baseline)** | 1442件 | **94.5%** | 84件 |
| **ValidationSimulator (古い)** | 1421件 | 93.1% | 105件 |
| **EMSEnvironment (初期)** | 1200件 | 78.6% | 326件 |
| **EMSEnvironment (最終)** | 1168件 | **76.6%** | 358件 |

**結論:**
- ✅ ValidationSimulatorでも5-7%は未完了
- ❌ **EMSEnvironmentは23.4%が未完了（3-4倍多い）**

---

## 🔍 応答時間の比較

| システム | 平均応答時間 | 6分達成率 | 備考 |
|----------|--------------|-----------|------|
| **ValidationSimulator (baseline)** | **7.47分** | **35.08%** | 直近隊運用 |
| **ValidationSimulator (古い)** | 14.22分 | 4.57% | 古い戦略 |
| **EMSEnvironment (初期)** | 12.62分 | 27.5% | 教師あり学習中 |
| **EMSEnvironment (最終)** | 19.70分 | 3.60% | PPO単独 |

**重要な発見:**
1. ✅ **Validation baseline: 7.47分（最高性能）**
2. ❌ **EMS初期（教師あり）: 12.62分（1.7倍遅い）**
3. ❌ **EMS最終（PPO単独）: 19.70分（2.6倍遅い）**

---

## 🚨 重大な問題の特定

### 問題1: EMSEnvironmentの完了率が低い（23.4%未完了）

**原因候補:**

**A. タイムアウトが早すぎる**
```python
max_wait_time:
  重症: 10分
  中等症: 20分
  軽症: 45分
```

**検証:**
- ログでは132-289分待機でタイムアウト
- つまり、タイムアウト時間を大幅に超えている
- **タイムアウトチェックが遅れている**

**B. 復帰が遅い**
- 利用可能台数が0-5台まで低下
- 全隊出場中の状態が長く続く
- 復帰イベントが処理されていない可能性

---

### 問題2: EMSEnvironmentの応答時間が遅い

**比較:**
- Validation baseline: 7.47分（直近隊運用）
- EMS初期（教師あり）: 12.62分（**1.7倍遅い**）

**これは異常です。**

**原因候補:**

**A. `advance_time()`で時間が進んでいる影響**

```python
# 事案発生時刻: 0.0分
advance_time()
  # イベント処理
  # current_time: 1.0分に進む

step(action)
  # 配車時刻: 1.0分
  # 本来は0.0分で配車すべき
```

**影響:**
- 配車が1分遅れる
- 応答時間が1分余分にかかる
- 平均応答時間が悪化

**B. 復帰時刻の計算ミス**

```python
call_start_time = 0.0分
current_time = 1.0分

completion_time = call_start_time + 総活動時間
return_event.time = completion_time

# しかし、復帰イベント処理時に
# current_timeがさらに進んでいる可能性
```

---

## 🎯 問題の優先順位

### 優先度1: 応答時間が1.7倍遅い（最重要）

**目標:**
- Validation baseline: 7.47分
- EMS現在（教師あり）: 12.62分
- **差: 5.15分（69%遅い）**

**これは受け入れられません。**

**修正が必要:**
- `advance_time()`と配車のタイミング
- 復帰時刻の計算
- 時刻管理の見直し

---

### 優先度2: 完了率が低い（76.6%）

**目標:**
- Validation baseline: 94.5%
- EMS現在: 76.6%
- **差: 17.9ポイント**

**修正が必要:**
- タイムアウトチェックのタイミング
- 復帰イベントの処理
- 全隊出場中の対応

---

### 優先度3: PPO学習が収束しない

**現象:**
- 初期（教師あり）: 12.62分, 27.5%
- 最終（PPO単独）: 19.70分, 3.60%
- **性能が悪化している**

**原因:**
- PPOエージェントが学習できていない
- 報酬設計の問題
- 状態表現の問題

---

## 🔧 具体的な修正案

### 修正1: 配車タイミングの修正

**問題:**
```python
advance_time()  # 時間が1分進む
step(action)  # 配車が1分遅れる
```

**解決策A: advance_time()内で時間を戻す**
```python
def advance_time(self):
    # イベント処理
    # pending_call設定時にcall_start_timeを記録
    
    # ★時間を戻す★
    if self.pending_call:
        self.current_time_seconds = self.call_start_times[self.pending_call['id']]
```

**解決策B: step()で配車時刻を補正**
```python
def step(self, action):
    if self.pending_call:
        # 配車時刻を事案発生時刻に設定
        self.current_time_seconds = self.call_start_times[self.pending_call['id']]
        
        # 配車処理
        dispatch_result = self._dispatch_ambulance(action)
        
        # 時間を元に戻す
        self.current_time_seconds = original_time
```

**解決策C: _calculate_ambulance_completion_time()を修正（既に実施済み）**
```python
# 既に実施済み
call_start_time = self.call_start_times.get(call['id'], ...)
arrive_scene_time = call_start_time + response_time
```

**評価:**
- 解決策Cは既に実施済みだが、応答時間が改善していない
- 解決策A or Bの追加実装が必要

---

### 修正2: 復帰イベント処理の確認

**デバッグログで確認すべきこと:**
1. 復帰イベントが実際に処理されているか
2. 復帰時刻が正しいか
3. 利用可能台数が増加しているか

**次回実行で確認:**
```
[復帰時刻] 救急車137:
  総活動時間: XX.X分
  復帰予定: XX.X分

[復帰スケジュール] 救急車137:
  活動時間: XX.X分  ← マイナスでないか確認

[復帰処理] 救急車137が復帰:
  利用可能: XX台  ← 増加しているか確認
```

---

## 📋 検証チェックリスト

### ValidationSimulator

- ✅ total_calls: 1526件
- ✅ completed_calls: 1442件（94.5%）
- ✅ 未完了: 84件（5.5%）
- ✅ タイムアウト処理: **存在しない**
- ✅ 全隊出場中: **callをスキップ**

### EMSEnvironment

- ❌ total_calls: 1168件（76.6%）
- ❌ タイムアウト: 358件（23.4%）
- ❌ 応答時間: 12.62分（ValidationBaselineの1.7倍）
- ✅ タイムアウト処理: 存在する
- ✅ 全隊出場中: 再試行→タイムアウト

---

## 🎯 次のアクション

### 1. デバッグログの確認（最優先）

次回の学習実行で、以下を確認:
- `[復帰時刻]`, `[復帰スケジュール]`, `[復帰処理]`ログ
- 活動時間が正常範囲か
- 復帰イベントが処理されているか

### 2. 応答時間の差（5.15分）の原因特定

**仮説:**
- 配車が1分遅れている
- 復帰時刻の計算がずれている
- ServiceTimeGeneratorが長い時間を生成している

**検証方法:**
- デバッグログで各フェーズの時間を確認
- ValidationSimulatorと比較

### 3. 設計方針の決定

**選択肢A: ValidationSimulatorに完全一致させる**
- タイムアウト処理を削除
- pending_callをスキップ
- 完了率を94%に改善

**選択肢B: 現実的な挙動を優先**
- タイムアウト処理を保持
- 完了率を改善（タイミングの修正）
- 応答時間の差を解消

---

**結論**: 
ValidationSimulatorでも5.5%は未完了ですが、EMSEnvironmentは23.4%が未完了です。また、応答時間が1.7倍遅いという重大な問題があります。次回の実行で、デバッグログを確認し、原因を特定する必要があります。

