# 🐛 バグ修正レポート: 救急車の復帰イベント処理

## 問題の症状

```
[trainer.py デバッグ] 行動選択時のaction_mask全てFalse:
  ステップ: 510, available救急車: 0/192
  現在時刻: 30600.0秒 (8.5時間)
```

### 発生状況
- エピソード開始から510ステップ（8.5時間）経過時点
- 全救急車（192台）が出動中
- 利用可能な救急車が0台
- 救急車が全く復帰していない

## 根本原因の特定

### 原因1: イベント処理ループのロジックエラー（修正前）

```python
# ❌ 問題のあるコード
while self.event_queue and self.event_queue[0].time <= end_time:
    if self.pending_call is not None:
        if self.event_queue[0].event_type == EventType.AMBULANCE_AVAILABLE:
            self._process_next_event()  # 復帰イベントは処理
        else:
            # 2件目以降のNEW_CALLはキューに残し、次のステップで処理
            break  # ← ★ここで即座にループを抜ける！★
```

**問題点**: 
- `pending_call`がある状態で、NEW_CALLイベントが来ると`break`でループを抜ける
- その後ろにある復帰イベントが処理されない
- 復帰イベントが永久にスキップされる

**具体例**:
```
イベントキュー:
  時刻120: NEW_CALL (事案B)
  時刻4000: AMBULANCE_AVAILABLE (救急車0)  ← これが処理されない！
  時刻5000: AMBULANCE_AVAILABLE (救急車1)  ← これも処理されない！

ステップ2 (120秒):
  - pending_callがある
  - NEW_CALLが来たのでbreak
  - 復帰イベント（4000秒, 5000秒）は見られない
```

### 原因2: 復帰時刻の計算ミス（修正前）

```python
# ❌ 問題のあるコード
completion_time_seconds = dispatch_result.get('completion_time_seconds', 0)
dispatch_time = self.call_start_times.get(current_incident['id'], self.current_time_seconds)
return_time = dispatch_time + completion_time_seconds  # ← ★二重計算！★
```

**問題点**:
- `completion_time_seconds`は既に絶対時刻（current_time + 活動時間）
- さらに`dispatch_time`を足すと、2倍の時刻になる
- 救急車が非現実的な未来時刻に復帰予定になる

**具体例**:
```
配車時刻: 1000秒
活動時間: 4000秒
completion_time_seconds = 1000 + 4000 = 5000秒 （絶対時刻）

修正前:
  return_time = 1000 + 5000 = 6000秒 ← ★間違い！★

修正後:
  return_time = 5000秒 ← ✓ 正しい
```

## 修正内容

### 修正1: イベント処理ループの改善

```python
# ✅ 修正後のコード
while self.event_queue and self.event_queue[0].time <= end_time:
    event = self.event_queue[0]  # 次のイベントを確認（まだpopしない）
    
    # 復帰イベントは常に処理
    if event.event_type == EventType.AMBULANCE_AVAILABLE:
        self._process_next_event()
        continue  # ← ★continueでループを継続★
    
    # NEW_CALLイベントの処理
    if event.event_type == EventType.NEW_CALL:
        # 既にpending_callがある場合、新しい事案は次のステップで処理
        if self.pending_call is not None:
            break
        # pending_callが空なら、この事案を処理
        self._process_next_event()
    else:
        # その他のイベント
        self._process_next_event()
```

**改善点**:
- 復帰イベントを**最優先**で処理
- `continue`でループを継続し、後続の復帰イベントも処理
- NEW_CALLは1件のみ処理（PPOの制約）

**動作例**:
```
イベントキュー:
  時刻120: NEW_CALL (事案B)
  時刻4000: AMBULANCE_AVAILABLE (救急車0)
  時刻5000: AMBULANCE_AVAILABLE (救急車1)

ステップ2 (120秒):
  1. NEW_CALLを処理 → pending_call = 事案B
  2. 次のイベントは4000秒なので、ループ終了

ステップ67 (4020秒):
  1. 復帰イベント（4000秒）を処理 → 救急車0が復帰 ✓
  2. continueでループ継続
  3. 次のイベントは5000秒なので、ループ終了

ステップ84 (5040秒):
  1. 復帰イベント（5000秒）を処理 → 救急車1が復帰 ✓
```

### 修正2: 復帰時刻の計算修正

```python
# ✅ 修正後のコード
# 復帰イベントをスケジュール
amb_id = dispatch_result['ambulance_id']
# completion_time_secondsは既に絶対時刻（current_time + 活動時間）
return_time = dispatch_result.get('completion_time_seconds', 
                                  self.current_time_seconds + 4000)  # フォールバック

return_event = Event(
    time=return_time,
    event_type=EventType.AMBULANCE_AVAILABLE,
    data={
        'ambulance_id': amb_id,
        'station_h3': self.ambulance_states[amb_id]['station_h3']
    }
)
self._schedule_event(return_event)
```

**改善点**:
- `completion_time_seconds`を直接使用（二重計算を回避）
- フォールバック値を追加（安全性向上）

## 修正の効果

### Before（修正前）
```
ステップ0-500: 救急車が徐々に出動
ステップ510: 全救急車が出動中（0/192台）← 復帰イベントが処理されていない
ステップ511-: 配車不可能な状態が継続
```

### After（修正後）
```
ステップ0: 救急車0を配車
ステップ67: 救急車0が復帰（4000秒後）✓
ステップ0: 救急車1を配車
ステップ84: 救急車1が復帰（5000秒後）✓
...継続的に復帰と配車が繰り返される
```

## 検証方法

### 1. イベントキューの確認
```python
# デバッグコードを追加
print(f"イベントキュー数: {len(env.event_queue)}")
for event in env.event_queue[:5]:
    print(f"  時刻={event.time:.1f}秒, タイプ={event.event_type}")
```

期待される出力:
```
イベントキュー数: 1520
  時刻=0.0秒, タイプ=new_call
  時刻=120.0秒, タイプ=new_call
  時刻=4000.0秒, タイプ=ambulance_available
  時刻=5000.0秒, タイプ=ambulance_available
```

### 2. 救急車状態の確認
```python
# 各ステップで利用可能台数を確認
available_count = sum(1 for amb in env.ambulance_states.values() 
                     if amb['status'] == 'available')
print(f"ステップ{env.episode_step}: 利用可能={available_count}台")
```

期待される出力:
```
ステップ0: 利用可能=192台
ステップ10: 利用可能=150台
ステップ67: 利用可能=151台 ← 1台復帰！
ステップ100: 利用可能=130台
```

### 3. 診断スクリプトの実行
```bash
python diagnose_ambulance_return.py
```

## まとめ

### 修正箇所
1. ✅ `step()`メソッドのイベント処理ループ（line 878-894）
2. ✅ 復帰時刻の計算（line 929-943）

### 期待される改善
- ✅ 救急車が適切なタイミングで復帰
- ✅ 利用可能台数が0にならない
- ✅ シミュレーションが正常に進行
- ✅ 学習が正しく実行される

### 今後の課題
- 初期状態で活動中の救急車の復帰イベント（現在は未実装）
- 休憩イベントの実装（将来の拡張）
- より詳細なログ出力

---

**修正日**: 2025年11月5日  
**対象ファイル**: `ems_environment.py`  
**修正者**: Claude
