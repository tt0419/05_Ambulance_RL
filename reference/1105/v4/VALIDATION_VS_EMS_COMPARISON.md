# ValidationSimulator vs EMSEnvironment 設計比較

**作成日**: 2025年11月6日  
**目的**: 全隊出場中の処理の違いを明確化

---

## 🔍 全隊出場中の処理の違い

### ValidationSimulator の処理

**コード確認:**
```python
# validation_simulation.py: _handle_new_call()
ambulance = self.find_closest_available_ambulance(call.h3_index, call.severity)

if ambulance:
    # 配車処理
    ambulance.status = AmbulanceStatus.DISPATCHED
    # 次のイベント（ARRIVE_SCENE）をスケジュール
else:
    if self.verbose_logging:
        print(f"[WARN] Call {call.id}: No ambulance available for dispatch at {event.time:.2f}. Call queued implicitly.")
    # ★何もしない★
```

**動作:**
1. ✅ NEW_CALLイベントを処理
2. ✅ 利用可能な救急車がない場合、**ログのみ出力**
3. ✅ そのまま次のイベントに進む
4. ✅ **callは処理されないままイベントキューから消える**

**重要な点:**
- ❌ **タイムアウト処理は存在しない**
- ❌ **callは再試行されない**
- ❌ **単にスキップされる**

**結果:**
- 全隊出場中のcallは統計に含まれない
- total_callsが実際のcall数より少なくなる
- しかし、シミュレーションは続行する

---

### EMSEnvironment の処理（現在）

**コード確認:**
```python
# ems_environment.py: step()
# advance_time()でイベント処理
if event.event_type == EventType.NEW_CALL:
    if self.pending_call is None:
        self._process_next_event()  # pending_callに設定
    else:
        break  # 次のステップで処理

# step(action)で配車処理
if self.pending_call is not None:
    dispatch_result = self._dispatch_ambulance(action)
    
    if dispatch_result['success']:
        self.pending_call = None  # クリア
    else:
        # pending_callを保持（再試行）
        
# 次のstep()でタイムアウトチェック
if wait_time > max_wait:
    self._handle_unresponsive_call(...)
    self.pending_call = None
```

**動作:**
1. ✅ NEW_CALLイベントを処理してpending_callに設定
2. ✅ 配車失敗してもpending_callを保持
3. ✅ 次のステップで再試行
4. ✅ タイムアウト（10-45分）で放棄

**重要な点:**
- ✅ **再試行メカニズムが存在する**
- ✅ **タイムアウト処理が存在する**
- ✅ **pending_callが長時間保持される**

**結果:**
- 全隊出場中のcallも再試行される
- しかし、タイムアウトが多発する
- total_callsが減少する

---

## 📊 数値比較

### ValidationSimulator（推定）

```
total_calls: 1526件（すべてのNEW_CALLイベントが処理される）
未配車: 不明（統計に含まれない）
タイムアウト: 0件（機能が存在しない）
全隊出場中の処理: スキップ
```

### EMSEnvironment（観測済み）

```
total_calls: 1168件（処理された事案のみ）
タイムアウト: 358件（1526 - 1168）
タイムアウト率: 23.5%
全隊出場中の処理: 再試行→タイムアウト
```

---

## 🚨 重大な設計上の違い

### ValidationSimulator: イベント駆動（完全）

**特徴:**
- すべてのイベントを順番に処理
- 利用可能な救急車がない場合、callはスキップ
- タイムアウトという概念がない
- シミュレーションは必ず完了する

**利点:**
- シンプル
- 確実に全イベントを処理
- デッドロックしない

**欠点:**
- 全隊出場中のcallが統計に含まれない可能性
- 現実の救急システムと異なる（再試行がない）

---

### EMSEnvironment: イベント駆動 + 強化学習

**特徴:**
- イベント処理とPPOエージェントのステップを統合
- pending_callメカニズムで再試行
- タイムアウト処理で現実的な挙動を再現
- 固定時間ステップ（1ステップ=1分）

**利点:**
- 現実的（再試行メカニズム）
- PPOに適した設計（1ステップ=1行動）

**欠点:**
- タイムアウトが多発
- 複雑
- ValidationSimulatorと動作が異なる

---

## 🔍 ValidationSimulatorでのタイムアウト確認

### 実際のログを確認

validation_simulationの実行ログを確認する必要があります。

**確認ポイント:**
1. `[WARN] Call: No ambulance available`が出ているか
2. その後、そのcallは処理されているか
3. total_callsが1526件になっているか
4. 未処理のcallがあるか

---

## 💡 根本的な問題の特定

### 問題: 設計の矛盾

**ValidationSimulator:**
- すべてのイベントを処理
- callがスキップされる場合、統計に含まれない
- total_calls ≠ 1526件の可能性

**EMSEnvironment:**
- pending_callを保持して再試行
- タイムアウトで強制的に放棄
- total_calls = 処理された事案のみ

**矛盾:**
- ValidationSimulatorとの一致を目指しているが、設計が異なる
- どちらが正しいのか不明確

---

## 🎯 検証すべきこと

### 1. ValidationSimulatorの実際の動作

**確認方法:**
```bash
python validation_simulation.py
```

**確認ポイント:**
- `[WARN] No ambulance available`が何回出るか
- total_callsが何件か
- 未配車のcallが何件あるか

---

### 2. EMSEnvironmentの設計方針の確認

**選択肢A: ValidationSimulatorに合わせる**
- タイムアウト処理を削除
- pending_callをスキップ
- 利用可能な救急車がない場合は何もしない

**選択肢B: 現実的な挙動を優先**
- タイムアウト処理を保持
- 再試行メカニズムを保持
- ValidationSimulatorとは異なる設計を認める

**選択肢C: ValidationSimulatorを修正**
- タイムアウト処理を追加
- 再試行メカニズムを追加
- EMSEnvironmentと同じ挙動にする

---

## 📝 次のアクション

### 1. ValidationSimulatorのログ確認

既存のvalidation_simulationの実行結果を確認します。

**確認する実行結果:**
```bash
# 既存の実行結果を確認
ls logs/validation/
cat logs/validation/validation_*.log | grep "No ambulance available"
```

---

### 2. ValidationSimulatorを実行

```bash
python validation_simulation.py
```

**確認ログ:**
```
[WARN] Call: No ambulance available  ← 何回出るか
total_calls: XXXX件  ← 1526件か？
未配車: YYY件  ← 何件あるか
```

---

### 3. 設計方針の決定

ValidationSimulatorの実際の動作を確認した上で、設計方針を決定します。

---

**結論**: 
ValidationSimulatorとEMSEnvironmentで全隊出場中の処理が根本的に異なります。ValidationSimulatorの実際の動作を確認し、どちらの設計が正しいかを判断する必要があります。

