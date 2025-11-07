# 救急隊復帰問題の調査

**作成日**: 2025年11月6日  
**問題**: 復帰が異常に遅く、利用可能台数が0台になる

---

## 🚨 観測された問題

### 異常な動作

```
ステップ750-850: 利用可能台数が0-5台（異常に少ない）
ステップ1300-1440: 大量のタイムアウト（200件以上）
total_calls: 894件（58%） ← 1526件中632件が未処理

[TIMEOUT] ステップ879: 事案(軽症)を132.0分待機
[TIMEOUT] ステップ1321: 事案(重症)を251.0分待機
```

**異常な点:**
- 6月15日の夜間で全隊出場中は通常起こらない
- ValidationSimulatorでは発生しなかった
- 旧ems_environmentでも発生しなかった

---

## 🔍 調査のための仮説

### 仮説1: 復帰時刻の計算がずれている

**メカニズム:**

```python
# advance_time()
current_time: 0秒
NEW_CALLイベント処理（time=0.0）
  → _process_next_event()
  → current_time = 0秒（イベントの時刻）
  → call_start_times[id] = 0秒

AMBULANCE_AVAILABLEイベント処理（time=15.5秒）
  → _process_next_event()
  → current_time = 15.5秒（イベントの時刻）

AMBULANCE_AVAILABLEイベント処理（time=32.3秒）
  → _process_next_event()
  → current_time = 32.3秒

イベント処理終了
current_time = 60秒（end_time）

# step(action)
_calculate_ambulance_completion_time()
  call_start_time = 0秒（正しい）
  arrive_scene_time = 0秒 + response_time（正しい）
  completion_time = arrive_scene_time + ... （正しいはず）

復帰イベントのスケジュール:
  return_time = completion_time
  
デバッグログ:
  活動時間 = return_time - current_time_seconds
           = completion_time - 60秒
           = (0秒 + 総活動時間) - 60秒
           = 総活動時間 - 60秒（ずれている？）
```

**確認ポイント:**
- `completion_time`の値が正しいか
- デバッグログで「活動時間」がマイナスになっていないか
- `call_start_time`が正しく使われているか

---

### 仮説2: ServiceTimeGeneratorが異常に長い時間を生成

**メカニズム:**

```python
on_scene_time_minutes = service_time_generator.generate_time(severity, 'on_scene_time')
hospital_time_minutes = service_time_generator.generate_time(severity, 'hospital_time')
```

**確率的生成:**
- 対数正規分布（μ, σ）
- 極端に長い値が生成される可能性

**確認ポイント:**
- デバッグログで各フェーズの時間を確認
- 異常に長い時間（100分以上）が生成されていないか

---

### 仮説3: 復帰イベントが処理されていない

**メカニズム:**

```python
# advance_time()
while event_queue and event_queue[0].time <= end_time:
    if event.event_type == EventType.AMBULANCE_AVAILABLE:
        self._process_next_event()  # 復帰処理
```

**確認ポイント:**
- 復帰イベントが実際に処理されているか
- デバッグログで「[復帰処理]」が表示されるか
- イベントキューから正しくpopされているか

---

### 仮説4: イベントキューの順序が壊れている

**メカニズム:**

```python
heapq.heappush(self.event_queue, event)
```

**確認ポイント:**
- イベントキューがheapqで正しく管理されているか
- 復帰イベントの時刻が正しいか

---

## 🔧 追加したデバッグログ

### 1. 復帰時刻計算の詳細（`_calculate_ambulance_completion_time`）

```python
print(f"\n[復帰時刻] 救急車{ambulance_id}:")
print(f"  事案発生: {call_start_time/60:.1f}分, 配車時刻: {self.current_time_seconds/60:.1f}分")
print(f"  応答: {response_time/60:.1f}分, 現場: {on_scene_time/60:.1f}分")
print(f"  搬送: {transport_time/60:.1f}分, 病院: {hospital_time/60:.1f}分, 帰署: {return_time/60:.1f}分")
print(f"  総活動時間: {total_activity_time/60:.1f}分")
print(f"  復帰予定: {completion_time/60:.1f}分")
```

**確認ポイント:**
- 各フェーズの時間が正常範囲か（応答:3-30分、現場:10-40分、病院:15-60分）
- 総活動時間が正常範囲か（40-150分）
- 復帰予定時刻が正しいか

---

### 2. 復帰イベントのスケジュール（`step()`）

```python
print(f"[復帰スケジュール] 救急車{amb_id}:")
print(f"  現在時刻: {self.current_time_seconds/60:.1f}分")
print(f"  復帰予定: {return_time/60:.1f}分")
print(f"  活動時間: {(return_time - self.current_time_seconds)/60:.1f}分")
```

**確認ポイント:**
- 復帰予定時刻が正しいか
- **活動時間がマイナスになっていないか**
- 活動時間が正常範囲か（40-150分）

---

### 3. 復帰イベントの処理（`_handle_ambulance_return_event`）

```python
print(f"[復帰処理] 救急車{amb_id}が復帰:")
print(f"  イベント時刻: {event.time/60:.1f}分")
print(f"  現在時刻: {self.current_time_seconds/60:.1f}分")
print(f"  利用可能: {available_count}台")
```

**確認ポイント:**
- 復帰イベントが実際に処理されているか
- イベント時刻と現在時刻が一致するか（`_process_next_event`で同期）
- 利用可能台数が増えているか

---

## 📋 次回実行時の確認手順

### 1. 復帰時刻計算の確認

**期待されるログ:**
```
[復帰時刻] 救急車137:
  事案発生: 0.0分, 配車時刻: 1.0分
  応答: 3.5分, 現場: 15.2分
  搬送: 8.3分, 病院: 25.1分, 帰署: 6.2分
  総活動時間: 58.3分
  復帰予定: 58.3分
```

**異常なパターン:**
```
[復帰時刻] 救急車137:
  総活動時間: 150.0分  ← 異常に長い
  復帰予定: 200.0分  ← 異常に遠い
```

---

### 2. 復帰スケジュールの確認

**期待されるログ:**
```
[復帰スケジュール] 救急車137:
  現在時刻: 1.0分
  復帰予定: 59.3分
  活動時間: 58.3分  ← 正常範囲
```

**異常なパターン:**
```
[復帰スケジュール] 救急車137:
  現在時刻: 1.0分
  復帰予定: 59.3分
  活動時間: -2.0分  ← ★マイナス（問題）★
```

または：

```
[復帰スケジュール] 救急車137:
  現在時刻: 1.0分
  復帰予定: 200.0分
  活動時間: 199.0分  ← 異常に長い
```

---

### 3. 復帰処理の確認

**期待されるログ:**
```
[復帰処理] 救急車137が復帰:
  イベント時刻: 59.3分
  現在時刻: 59.3分  ← 一致
  利用可能: 82台  ← 増加
```

**異常なパターン:**
```
[復帰処理]ログが出ない  ← 復帰イベントが処理されていない
```

---

## 🎯 修正した内容

### 1. `call_start_time`を基準にする

**ファイル**: `ems_environment.py`  
**場所**: `_calculate_ambulance_completion_time()`（1444-1449行目）

**修正:**
```python
# 修正前
current_time = self.current_time_seconds
arrive_scene_time = current_time + response_time

# 修正後
call_start_time = self.call_start_times.get(call['id'], self.current_time_seconds)
arrive_scene_time = call_start_time + response_time
```

**効果:**
- ✅ 事案の実際の発生時刻を基準にする
- ✅ `advance_time()`で時間が進んでいても正しく計算

---

### 2. デバッグログの追加

- 復帰時刻計算の詳細
- 復帰イベントのスケジュール情報
- 復帰イベントの処理情報

---

## 📊 期待される改善

### 修正前（現在）

```
利用可能台数: 0-5台（異常に少ない）
復帰時刻: 不明（デバッグなし）
total_calls: 894件（58%）
```

### 修正後（期待）

```
利用可能台数: 30-80台（正常範囲）
復帰時刻: 40-100分後（正常範囲）
total_calls: 1500件以上（98%以上）
```

---

## 🚀 次のアクション

### 1. 学習を実行

```bash
python train_ppo.py --config config_hybrid_continuous.yaml
```

### 2. ログを確認

**重要な確認ポイント:**

1. **[復帰時刻]ログ:**
   - 各フェーズの時間が正常範囲か
   - 総活動時間が40-150分の範囲か
   - 復帰予定時刻が正しいか

2. **[復帰スケジュール]ログ:**
   - **活動時間がマイナスでないか**
   - 活動時間が正常範囲か

3. **[復帰処理]ログ:**
   - 復帰イベントが処理されているか
   - 利用可能台数が増加しているか

### 3. 問題の特定

**パターンA: 活動時間がマイナス**
→ 時刻計算のロジックエラー
→ さらなる修正が必要

**パターンB: 活動時間が異常に長い（150分超）**
→ ServiceTimeGeneratorの問題
→ パラメータの確認が必要

**パターンC: 復帰イベントが処理されない**
→ イベント処理ループの問題
→ advance_time()の実装を再検討

---

**結論**: 
デバッグログを追加しました。次回の実行で、復帰が遅い原因を特定できます。

