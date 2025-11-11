# 時間管理システム修正：修正前後の動作比較

## 修正前（旧システム）：「1ステップ = 1事案」

```
時刻の進行:
ステップ0   ステップ1   ステップ2   ステップ3
   |           |           |           |
事案A発生   事案B発生   事案C発生   事案D発生
   |           |           |           |
配車(救急車1) 配車(救急車2) 配車(救急車3) 配車(救急車4)
   |           |           |           |
   +-----67ステップ後に復帰------+
                          (事案67の時点で初めて復帰)

問題点:
✗ 事案間の時間が無視される（時間がジャンプ）
✗ 救急車の復帰が非現実的に遅延
✗ 実時間の概念がない（ステップ数=事案数）
```

## 修正後（新システム）：「1ステップ = 1分（60秒）」

```
時刻の進行:
0分   1分   2分   3分   ...   67分   68分
 |     |     |     |           |      |
 |  事案A  事案B        事案C        救急車1
 |     |     |           |           復帰
 |  配車(救急車1)    配車(救急車2)     ↓
 |     |     |           |        利用可能
 |     |     |           |
 +-----活動時間67分-----+
       (実時間で計算)

イベントキュー:
t=0:    NEW_CALL(事案A)
t=2:    NEW_CALL(事案B)
t=15:   NEW_CALL(事案C)
t=67:   AMBULANCE_AVAILABLE(救急車1)
t=72:   AMBULANCE_AVAILABLE(救急車2)

利点:
✓ 事案間の時間を適切に処理
✓ 救急車の復帰が現実的なタイミング
✓ ValidationSimulatorと同じ時間粒度
```

## ステップごとの詳細動作比較

### 旧システム（修正前）
```python
step(action):
    1. current_incidentを取得
    2. actionで配車
    3. 報酬を計算
    4. 次の事案へジャンプ（_advance_to_next_call）
    5. 救急車の復帰チェック（_update_ambulance_availability）
       → episode_step >= call_completion_time で判定
    6. 観測を返す
```

### 新システム（修正後）
```python
step(action):
    1. 60秒間のイベントを処理
       while event_queue and event.time <= end_time:
           - NEW_CALL → pending_callにセット
           - AMBULANCE_AVAILABLE → 救急車復帰
    
    2. pending_callがあれば:
       - actionで配車
       - 報酬を計算
       - 復帰イベントをスケジュール
    
    3. 時間を60秒進める（current_time_seconds += 60）
    4. ステップをインクリメント（episode_step += 1）
    5. 観測を返す
```

## 事案間の時間処理の違い

### 例：事案Aが0分、事案Bが15分に発生する場合

**旧システム**:
```
ステップ0: 事案A処理 → 即座にステップ1へ
ステップ1: 事案B処理（0分と15分の間の時間は無視）
```

**新システム**:
```
ステップ0（0分）:   事案A処理、配車
ステップ1（1分）:   事案なし、時間だけ進む
ステップ2（2分）:   事案なし、時間だけ進む
...
ステップ15（15分）: 事案B処理、配車
```

## 救急車の復帰処理の違い

### 例：活動時間67分の救急車

**旧システム**:
```python
# 配車時
amb_state['call_completion_time'] = episode_step + 67

# 復帰判定（_update_ambulance_availability内）
if episode_step >= amb_state['call_completion_time']:
    amb_state['status'] = 'available'
    
問題: episode_stepは事案数なので、
      67件目の事案が来るまで復帰しない
```

**新システム**:
```python
# 配車時
return_time = current_time_seconds + (67 * 60)  # 4020秒後
return_event = Event(
    time=return_time,
    event_type=AMBULANCE_AVAILABLE,
    data={'ambulance_id': amb_id}
)
event_queue.append(return_event)

# 復帰判定（step内のイベント処理）
while event_queue and event.time <= end_time:
    if event.type == AMBULANCE_AVAILABLE:
        amb_state['status'] = 'available'

利点: 実時間で67分後に確実に復帰
```

## 統合のメリット

| 観点 | 旧システム | 新システム |
|------|-----------|-----------|
| 時間粒度 | 事案数（不規則） | 1分（規則的） |
| 事案間の時間 | 無視される | 適切に処理 |
| 救急車復帰 | 非現実的に遅延 | 現実的なタイミング |
| テスト環境との整合性 | 不整合 | 整合 |
| 学習の質 | 低い | 高い |
| ValidationSimとの互換性 | 低い | 高い |

## コード例：イベント処理の実装

```python
# イベントのスケジューリング
def _schedule_event(self, event: Event):
    """イベントをキューに追加"""
    heapq.heappush(self.event_queue, event)

# イベントの処理
def _process_next_event(self) -> Optional[Event]:
    """次のイベントを処理"""
    if not self.event_queue:
        return None
    
    event = heapq.heappop(self.event_queue)
    
    if event.event_type == EventType.NEW_CALL:
        self._handle_new_call_event(event)
    elif event.event_type == EventType.AMBULANCE_AVAILABLE:
        self._handle_ambulance_return_event(event)
    
    return event
```

## まとめ

修正後のシステムは:
1. **ValidationSimulatorと同じ時間管理** → テストでの性能低下を防ぐ
2. **現実的なシミュレーション** → 学習の質が向上
3. **イベント駆動型の処理** → 複雑な時間管理に対応可能

これにより、訓練環境で学習したモデルがテスト環境で正しく動作するようになります。
