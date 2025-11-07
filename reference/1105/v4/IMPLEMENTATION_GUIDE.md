# 実装ガイド: 根本的な修正の適用手順

**作成日**: 2025年11月6日  
**目的**: ems_environment.pyの時間管理とイベント処理を完全に修正する

---

## 📋 修正の全体像

### 問題の要約

| 問題 | 影響 | 優先度 |
|------|------|--------|
| 配車失敗の無限ループ | 502/1526件しか処理されない | 🔴最高 |
| trainer.pyのタイミングずれ | 教師あり学習が機能しない | 🔴最高 |
| step()の処理順序 | pending_callが設定されるタイミングが遅い | 🟠高 |

### 修正の戦略

**段階1: 緊急修正（タイムアウト処理）**
- 配車失敗の無限ループを回避
- すべての事案を処理できるようにする

**段階2: 処理順序の修正**
- step()内でイベント処理を先に実行
- pending_callの状態管理を明確化

**段階3: trainer.pyの修正**
- get_optimal_action()のタイミング問題を解決
- 教師あり学習を機能させる

---

## 🔧 段階1: 緊急修正（タイムアウト処理）

### 修正箇所1: step()のイベント処理ループ

**ファイル**: `reinforcement_learning/environment/ems_environment.py`  
**行番号**: 約958-967

**修正前:**
```python
if event.event_type == EventType.NEW_CALL:
    if self.pending_call is None:
        self._process_next_event()
    else:
        break  # ★ここで止まる★
```

**修正後:**
```python
if event.event_type == EventType.NEW_CALL:
    if self.pending_call is None:
        self._process_next_event()
        events_processed += 1
    else:
        # ★タイムアウトチェック追加★
        wait_time_seconds = self.current_time_seconds - self.call_start_times.get(
            self.pending_call['id'], self.current_time_seconds
        )
        max_wait_seconds = self._get_max_wait_time(self.pending_call['severity']) * 60
        
        if wait_time_seconds > max_wait_seconds:
            # タイムアウト: 古い事案を放棄
            if self.episode_step <= 50:  # 最初の50ステップのみログ
                print(f"[TIMEOUT] 事案{self.pending_call['id']}を{self.pending_call['severity']}としてタイムアウト処理（{wait_time_seconds/60:.1f}分待機）")
            
            self._handle_unresponsive_call(self.pending_call, wait_time_seconds / 60)
            self.pending_call = None  # クリア
            
            # 新しい事案を処理
            self._process_next_event()
            events_processed += 1
        else:
            # まだタイムアウトしていない
            break
```

**効果:**
- 配車失敗が10分（重症）、20分（中等症）、45分（軽症）続いたら、事案を放棄
- 新しい事案が処理される
- すべての1526件が処理される

---

## 🔧 段階2: 処理順序の完全な修正（推奨）

### 現在の問題

```
trainer.py:
  get_optimal_action()  ← pending_call=なし
  step(action)
    → イベント処理
    → pending_call設定
    → 配車
```

**結果**: get_optimal_action()が常にNoneを返す

### 解決策A: step()を2段階に分割

**新しいメソッドを追加:**

```python
def advance_time(self) -> None:
    """
    時間を進めてイベントを処理（配車は行わない）
    
    trainer.pyで以下のように使用:
    1. env.advance_time()  # イベント処理
    2. optimal_action = env.get_optimal_action()  # pending_callあり
    3. step_result = env.step(action)  # 配車のみ
    """
    end_time = self.current_time_seconds + self.time_per_step
    
    # イベント処理
    while self.event_queue and self.event_queue[0].time <= end_time:
        event = self.event_queue[0]
        
        if event.event_type == EventType.AMBULANCE_AVAILABLE:
            self._process_next_event()
        elif event.event_type == EventType.NEW_CALL:
            if self.pending_call is None:
                self._process_next_event()
            else:
                # タイムアウトチェック...
                break
        else:
            self._process_next_event()
    
    # 時間を進める
    self.current_time_seconds = end_time
    self.episode_step += 1

def step(self, action: int) -> StepResult:
    """配車のみを実行（イベント処理はadvance_timeで実施済み）"""
    
    reward = 0.0
    info = {}
    
    if self.pending_call is not None:
        # 配車処理...
        dispatch_result = self._dispatch_ambulance(action)
        # ...
        if dispatch_result['success']:
            self.pending_call = None
    
    done = self._is_episode_done()
    observation = self._get_observation()
    
    return StepResult(observation, reward, done, info)
```

**trainer.pyの修正:**
```python
# _run_episode()メソッド内
while True:
    # 時間を進めてイベント処理
    env.advance_time()
    
    # 行動選択（pending_callが設定されている）
    action_mask = env.get_action_mask()
    optimal_action = env.get_optimal_action()  # ← pending_callあり
    
    # 行動選択
    action = select_action(observation, action_mask, optimal_action)
    
    # 配車実行
    step_result = env.step(action)
```

**利点:**
- Gym APIを大きく変更しない
- 教師あり学習が機能する
- タイミング問題が解決

**欠点:**
- 2つのメソッド呼び出しが必要

### 解決策B: step()の戻り値を拡張（シンプル）

**step()の修正:**
```python
def step(self, action: int) -> StepResult:
    # イベント処理 → pending_call設定
    # 配車処理
    # ...
    
    return StepResult(
        observation=observation,
        reward=reward,
        done=done,
        info={
            'current_pending_call': self.pending_call,  # ★追加★
            'episode_stats': self.episode_stats.copy(),
            'step': self.episode_step
        }
    )
```

**trainer.pyは修正不要:**
- 行動選択時、前のステップのpending_callを使う
- これは1ステップ遅れるが、学習には影響が少ない

**利点:**
- 修正が最小限
- Gym API互換

**欠点:**
- 1ステップのタイムラグがある

---

## 🎯 推奨される実装手順

### Step 1: タイムアウト処理の追加（必須）

1. `SOLUTION_STEP_REORDER.py`のタイムアウト処理部分をコピー
2. `ems_environment.py`のstep()メソッド内、NEW_CALLイベント処理部分に適用
3. 動作確認:
   ```bash
   python train_ppo.py --config config_hybrid_continuous.yaml
   ```
4. ログで確認:
   - `[TIMEOUT]`メッセージが出るか
   - `total_calls`が1526件に近づくか

### Step 2: 処理順序の修正（推奨）

**選択肢A**: advance_time()メソッドを追加（推奨）
- より明確な設計
- 教師あり学習が確実に機能

**選択肢B**: step()の戻り値を拡張
- 最小限の修正
- 1ステップの遅れは許容範囲

### Step 3: デバッグログの確認

修正後、以下を確認:
```bash
# ログの確認ポイント
[STEP DEBUG] イベント処理数: X（NEW_CALL: Y件）
[TIMEOUT] 事案をタイムアウト
total_calls: 1526件  ← 目標
```

---

## ✅ 成功基準

### 修正前
```
total_calls: 502件（33%）
最後の配車: ステップ710
教師一致: False
```

### 修正後の目標
```
total_calls: 1500件以上（98%以上）
最後の配車: ステップ1420付近
教師一致: True（90%の確率で）
```

---

## 📝 次のアクション

1. **段階1の緊急修正を実施**
   - タイムアウト処理を追加
   - 動作確認

2. **結果を分析**
   - total_callsが改善したか
   - 新しい問題が発生していないか

3. **段階2の修正を検討**
   - 教師あり学習の修正が必要か
   - どの解決策を選ぶか

4. **完全なテスト**
   - 6月15日のデータで1526件処理できるか
   - 3月18日のデータで1215件処理できるか

---

**結論**: 
まず**段階1のタイムアウト処理**を実装し、動作を確認してから、段階2の修正を検討することを推奨します。

