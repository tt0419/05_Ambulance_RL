# 根本的な問題分析と完全な解決策

**作成日**: 2025年11月6日  
**対象**: ems_environment.py の時間管理とイベント処理  
**問題**: 502件/1526件しか処理されない、教師あり学習が機能しない

---

## 📊 ログから判明した問題

### 観測された事実

```
エピソード内事案数: 1526件
performance/total_calls: 502件  ← 33%しか処理されていない
episode_length: 1440ステップ
event_queue: 1642件 → 1620件（22件のみ処理）

最後の配車: Ep20-710（ステップ710付近）
ステップ711-1439: 配車なし（729ステップが空走）
```

### 時系列の詳細

```
ステップ0（0-60秒）:
  event_queue: 1642件
  イベント処理: 11件
  pending_call: あり → 配車成功 → クリア
  
ステップ1（60-120秒）:
  event_queue: 1632件
  イベント処理: 7件
  pending_call: あり → 配車成功 → クリア
  
ステップ2（120-180秒）:
  event_queue: 1626件
  イベント処理: 6件（すべてAMBULANCE_AVAILABLE）
  pending_call: なし  ← ★NEW_CALLが0件処理★
  配車: スキップ
  
ステップ3（180-240秒）:
  次のイベント時刻: 201.29秒（AMBULANCE_AVAILABLE）
  イベント処理: 6件
  pending_call: あり（240秒前のNEW_CALL）
```

---

## 🚨 根本原因の特定

### 原因1: 配車失敗時の無限ループ（最重要）

**問題のコード:**
```python
if self.pending_call is not None:
    dispatch_result = self._dispatch_ambulance(action)
    
    if dispatch_result['success']:
        self.pending_call = None  # クリア
    else:
        # pending_callを保持（再試行）
        # ★ここで何百ステップも同じ事案を保持し続ける★

# イベント処理
while event_queue and event_queue[0].time <= end_time:
    if event.type == NEW_CALL:
        if self.pending_call is not None:  # ★常にTrue★
            break  # ★新しい事案が処理されない★
```

**影響:**
- ステップ650で全隊出場中
- 配車失敗 → pending_call保持
- 次の790ステップ（約13時間分）、同じ事案を保持
- 新しいNEW_CALLイベントが処理されない
- その間に発生した1024件の事案が未処理

### 原因2: trainer.pyとstep()のタイミングのずれ

**問題の処理順序:**
```python
# trainer.py
optimal_action = env.get_optimal_action()  # ← pending_call=なし
env.step(action)  # ← この中でpending_callが設定される
```

**結果:**
- `get_optimal_action()`が常に`None`を返す
- 教師あり学習が機能しない

### 原因3: 復帰イベントの時刻が遠い

```python
completion_time = current_time + 3000-6000秒（50-100分）
return_event.time = completion_time
```

**影響:**
- 配車した救急車が50-100分後に復帰
- その間、利用可能台数が減り続ける
- ステップ650で全隊出場中

---

## 💊 完全な解決策

### 解決策1: step()の処理順序を完全に変更（最重要）

**修正後のstep():**

```python
def step(self, action: int) -> StepResult:
    """
    正しい処理順序（根本的修正版）
    
    Phase 1: イベント処理 → pending_call設定
    Phase 2: 配車処理 → pending_callクリア
    Phase 3: 時間を進める
    """
    
    start_time = self.current_time_seconds
    end_time = start_time + 60.0
    
    # ===== Phase 1: イベント処理 =====
    # ★先にイベント処理してpending_callを設定★
    while self.event_queue and self.event_queue[0].time <= end_time:
        event = self.event_queue[0]
        
        # 復帰イベントは常に処理
        if event.event_type == EventType.AMBULANCE_AVAILABLE:
            self._process_next_event()
            continue
        
        # NEW_CALLイベント
        if event.event_type == EventType.NEW_CALL:
            # ★pending_callが空の場合のみ設定★
            if self.pending_call is None:
                self._process_next_event()
                # pending_callが設定される
            else:
                # ★重要: タイムアウトチェック★
                wait_time = self.current_time_seconds - self.call_start_times.get(
                    self.pending_call['id'], self.current_time_seconds
                )
                max_wait = self._get_max_wait_time(self.pending_call['severity']) * 60
                
                if wait_time > max_wait:
                    # タイムアウト: 古い事案を放棄
                    self._handle_unresponsive_call(self.pending_call, wait_time / 60)
                    self.pending_call = None  # クリア
                    # 新しい事案を処理
                    self._process_next_event()
                else:
                    # まだタイムアウトしていない: 次のステップで
                    break
        else:
            self._process_next_event()
    
    # ===== Phase 2: 配車処理 =====
    reward = 0.0
    info = {}
    
    if self.pending_call is not None:
        # ハイブリッドモード処理...
        dispatch_result = self._dispatch_ambulance(action)
        reward = self._calculate_reward(dispatch_result)
        
        if dispatch_result['success']:
            self._log_dispatch_action(...)
            self._update_statistics(...)
            self._schedule_event(return_event)
            self.pending_call = None  # クリア
        else:
            # 配車失敗: pending_call保持
            reward = self.reward_designer.get_failure_penalty('no_available')
            info = {'dispatch_failed': True}
    
    # ===== Phase 3: 時間を進める =====
    self.current_time_seconds = end_time
    self.episode_step += 1
    
    done = self._is_episode_done()
    observation = self._get_observation()
    
    return StepResult(observation, reward, done, info)
```

**重要な変更:**
1. **イベント処理を先に実行**
2. **タイムアウト処理を追加**（古い事案を放棄）
3. **配車処理を後に実行**

### 解決策2: trainer.pyの修正（教師あり学習の修正）

**現在の問題:**
```python
# trainer.py
optimal_action = env.get_optimal_action()  # ← pending_call=なし
step_result = env.step(action)
```

**修正案A: step()を2回呼び出す**
```python
# 観測のみ取得（イベント処理のみ）
dummy_result = env.step(action=0)  # ダミーアクション
observation = dummy_result.observation

# 行動選択
optimal_action = env.get_optimal_action()  # ← pending_callあり
action = select_action(observation, optimal_action)

# 実際の配車
# しかし、これは2回step()を呼ぶことになり、正しくない
```

**修正案B: get_next_call()メソッドを追加**
```python
# ems_environment.py
def get_next_call(self) -> Optional[Dict]:
    """次の事案を先読み（イベントキューをpopせずに確認）"""
    for event in self.event_queue:
        if event.event_type == EventType.NEW_CALL:
            if event.time <= self.current_time_seconds + 60:
                return event.data['call']
    return None

# trainer.py
next_call = env.get_next_call()
if next_call:
    # 次の事案に基づいて最適行動を計算
    optimal_action = env.get_optimal_action_for_call(next_call)
```

**修正案C: step()の戻り値に次の事案を含める（推奨）**
```python
# step()の戻り値
return StepResult(
    observation=observation,
    reward=reward,
    done=done,
    info={
        'next_pending_call': self.pending_call,  # ★追加★
        ...
    }
)

# trainer.py
step_result = env.step(action)
next_state = step_result.observation

# 次のループで
optimal_action = env.get_optimal_action()  # ← 今回設定されたpending_callを使う
```

これなら、Gym APIを壊さずに対応できます。

### 解決策3: 終了判定の修正

**現在:**
```python
if self.current_time_seconds >= 86400:  # 24時間
    return True
```

**問題:**
- イベントキューに1000件以上残っていても終了
- 未処理の事案が大量に残る

**修正:**
```python
# オプション1: イベントキュー優先
if not self.event_queue and self.pending_call is None:
    return True  # すべて処理完了

# オプション2: 両方考慮
if self.current_time_seconds >= 86400:
    if self.pending_call is None:  # 処理中の事案がなければ終了
        return True
```

---

## 🎯 推奨される修正の優先順位

### 優先度1: step()の処理順序変更（必須）

**効果**: NEW_CALLイベントが確実に処理される

**実装方法**: 
1. イベント処理を先に実行
2. タイムアウト処理を追加
3. 配車処理を後に実行

### 優先度2: タイムアウト処理の追加（必須）

**効果**: 配車失敗の無限ループを防ぐ

**実装方法**:
- 待機時間が一定時間を超えたら事案を放棄
- `_handle_unresponsive_call()`を呼ぶ

### 優先度3: trainer.pyの修正（推奨）

**効果**: 教師あり学習が機能する

**実装方法**:
- step()の戻り値に`next_pending_call`を追加
- または、処理順序を変更

---

## 📝 実装チェックリスト

- [ ] step()の処理順序を変更
  - [ ] Phase 1: イベント処理
  - [ ] Phase 2: 配車処理
  - [ ] Phase 3: 時間を進める
  
- [ ] タイムアウト処理の追加
  - [ ] 待機時間の計算
  - [ ] max_wait_timeの取得
  - [ ] `_handle_unresponsive_call()`の呼び出し
  
- [ ] trainer.pyの修正（オプション）
  - [ ] step()の戻り値に`next_pending_call`を追加
  - [ ] または、get_next_call()メソッドを追加

- [ ] デバッグログの継続
  - [ ] イベント時刻の分布を確認
  - [ ] pending_callの状態遷移を追跡
  - [ ] 処理された事案数のカウント

---

## 🔧 緊急修正: タイムアウト処理のみ追加

最小限の修正で動作させる場合:

```python
# step()のイベント処理ループ内
if event.event_type == EventType.NEW_CALL:
    if self.pending_call is not None:
        # ★タイムアウトチェック追加★
        wait_time = self.current_time_seconds - self.call_start_times.get(
            self.pending_call['id'], self.current_time_seconds
        )
        
        # 10分（600秒）待ってもまだpending_callがある場合
        if wait_time > 600:
            # 古い事案を放棄
            print(f"[TIMEOUT] 事案{self.pending_call['id']}をタイムアウト（{wait_time/60:.1f}分待機）")
            self._handle_unresponsive_call(self.pending_call, wait_time / 60)
            self.pending_call = None  # クリア
            # 新しい事案を処理
            self._process_next_event()
        else:
            break
    else:
        self._process_next_event()
```

**効果:**
- 配車失敗が10分続いたら、事案を放棄して次へ
- すべての事案が処理される
- 教師あり学習の問題は別途対応が必要

---

## 📈 期待される改善

### 修正前
```
total_calls: 502件（33%）
最後の配車: ステップ710
空走: 729ステップ
```

### 修正後
```
total_calls: 1526件（100%）
最後の配車: ステップ1420付近
空走: 20ステップ以下
```

---

## 🎯 次のステップ

1. **タイムアウト処理を追加**（緊急）
2. **step()の処理順序を変更**（推奨）
3. **デバッグログで確認**
4. **trainer.pyを修正**（長期）
5. **完全なイベント駆動設計に移行**（将来）

---

**結論**: 
最も緊急の修正は**タイムアウト処理の追加**です。これだけで、すべての事案が処理されるようになります。

