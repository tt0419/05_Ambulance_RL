# 教師あり学習の修正 - 適用完了

**実施日**: 2025年11月6日  
**修正内容**: `advance_time()`メソッドを追加し、教師あり学習を機能させる

---

## 📊 問題の確認

### 修正前の状況

```
[ACTION DEBUG] Step 0
  teacher_prob: 0.8992  ← 教師あり確率90%
  get_optimal_action呼び出し: True
  optimal_action: None  ← pending_callがないため取得できない
  [ERROR] pending_call is None or doesn't exist!
  use_teacher: False  ← 教師あり学習が機能しない
```

**影響:**
- 教師あり学習が全く機能していない
- すべての行動がPPOエージェントによる選択
- 直近隊運用（最適行動）が学習されない
- 応答時間が改善しない（20分前後で停滞）

---

## ✅ 実施した修正

### 修正1: `advance_time()`メソッドの追加

**ファイル**: `reinforcement_learning/environment/ems_environment.py`  
**場所**: 926-998行目（`step()`の前）

**実装内容:**

```python
def advance_time(self) -> None:
    """
    時間を進めてイベントを処理（配車は行わない）
    
    このメソッドは教師あり学習のために追加されました。
    trainer.pyで以下の順序で呼び出されます：
    1. advance_time()  # イベント処理、pending_call設定
    2. get_optimal_action()  # pending_callが存在する
    3. step(action)  # 配車処理のみ
    """
    end_time = self.current_time_seconds + self.time_per_step
    
    # Phase 0: タイムアウトチェック
    if self.pending_call is not None:
        wait_time_seconds = current_time - call_start_time
        if wait_time_seconds > max_wait_seconds:
            self._handle_unresponsive_call(...)
            self.pending_call = None
    
    # Phase 1: イベント処理
    while event_queue and event_queue[0].time <= end_time:
        # AMBULANCE_AVAILABLEイベント処理
        # NEW_CALLイベント処理（pending_callが空の場合のみ）
    
    # Phase 2: 時間を進める
    self.current_time_seconds = end_time
    self.episode_step += 1
    self._time_advanced_externally = True  # フラグを設定
```

**効果:**
- ✅ イベント処理と配車処理を分離
- ✅ `pending_call`が設定された後に`get_optimal_action()`を呼べる
- ✅ Gym API互換性を維持

---

### 修正2: `step()`メソッドの簡略化

**ファイル**: `reinforcement_learning/environment/ems_environment.py`  
**場所**: 999-1130行目

**実装内容:**

```python
def step(self, action: int) -> StepResult:
    """
    環境のステップ実行
    
    動作モード:
    1. advance_time()が事前に呼ばれている場合: 配車処理のみ
    2. そうでない場合: 時間を進めてから配車処理（後方互換性）
    """
    try:
        # フラグをチェック: advance_time()が呼ばれていなければ内部で呼ぶ
        if not getattr(self, '_time_advanced_externally', False):
            self.advance_time()
        
        # フラグをリセット
        self._time_advanced_externally = False
        
        # 配車処理のみ実行
        if self.pending_call is not None:
            # ハイブリッドモード処理
            # 配車実行
            # 復帰イベントのスケジュール
        
        # 終了判定と観測の取得
        done = self._is_episode_done()
        observation = self._get_observation()
        
        return StepResult(observation, reward, done, info)
```

**重要な変更:**
- ✅ イベント処理部分を削除（`advance_time()`で実施）
- ✅ 時間の進行を削除（`advance_time()`で実施済み）
- ✅ 配車処理のみを実行
- ✅ 後方互換性を維持（`advance_time()`が呼ばれていない場合は内部で呼ぶ）

---

### 修正3: `trainer.py`の修正

**ファイル**: `reinforcement_learning/training/trainer.py`  
**場所**: 213-214行目

**修正前:**
```python
# 行動選択
action_mask = self.env.get_action_mask()
optimal_action = self.env.get_optimal_action()  # ← pending_call=なし
step_result = self.env.step(action)  # ← この中でpending_call設定
```

**修正後:**
```python
# ★★★ 時間を進めてイベント処理（教師あり学習のため） ★★★
self.env.advance_time()  # ← pending_call設定

# 行動選択
action_mask = self.env.get_action_mask()
optimal_action = self.env.get_optimal_action()  # ← pending_callが存在する
step_result = self.env.step(action)  # ← 配車処理のみ
```

**効果:**
- ✅ `get_optimal_action()`が呼ばれる時点で`pending_call`が存在
- ✅ 教師あり学習が機能する
- ✅ 直近隊運用が学習される

---

## 📈 期待される改善

### 教師あり学習

| 指標 | 修正前 | 修正後（予測） |
|------|--------|----------------|
| `optimal_action` | 常に`None` | 正常に取得 |
| `use_teacher` | 常に`False` | `True`（90%の確率） |
| 教師一致率 | 0% | 85-90% |

### 応答時間

| 指標 | 修正前 | 修正後（予測） |
|------|--------|----------------|
| 平均応答時間 | 20.21分 | **8-10分** ✨ |
| 6分達成率 | 1.8% | **30-40%** ✨ |
| 13分達成率 | 15.6% | **70-80%** ✨ |

### 傷病度別（重症系）

| 指標 | 修正前 | 修正後（予測） |
|------|--------|----------------|
| 重症6分達成率 | 19.7% | **80-90%** ✨ |
| 重症平均応答時間 | 11.0分 | **4-6分** ✨ |

---

## 🔍 動作確認ポイント

### 1. 教師あり学習の動作確認

**確認ログ（期待）:**
```
[ADVANCE_TIME] ステップ0開始
  current_time: 0.0秒
  イベント処理: 7件（NEW_CALL: 1件）
  pending_call: あり

[ACTION DEBUG] Step 0
  teacher_prob: 0.8992
  optimal_action: 159  ← ★取得できる★
  use_teacher: True  ← ★機能する★
  
[STEP] ステップ1: 配車処理
  pending_call: あり
  配車成功: pending_callをクリア
```

### 2. 応答時間の改善確認

**確認メトリクス（wandb）:**
```
performance/mean_response_time: 8-10分  ← 目標
performance/6min_achievement_rate: 0.30-0.40  ← 目標
severity/severe_6min_rate: 0.80-0.90  ← 重症系
```

### 3. ログの確認

**配車ログ:**
```
[配車] Ep1-0: 重症 → 三田救急(実車) 3.5分 (利用可能:75台) ← ★直近隊★
[配車] Ep1-10: 軽症 → 根津救急(実車) 5.2分 (利用可能:110台) ← ★直近隊★
```

**期待:**
- 応答時間が大幅に短縮（3-6分）
- 直近隊が選ばれている

---

## 🎯 次のステップ

### 即座に実施

1. **学習の実行**
   ```bash
   python train_ppo.py --config config_hybrid_continuous.yaml
   ```

2. **ログの確認**
   - `[ADVANCE_TIME]`ログが表示される
   - `optimal_action`が取得できる（`None`でない）
   - `use_teacher: True`が表示される

3. **wandbメトリクスの確認**
   - `mean_response_time`が8-10分に改善
   - `6min_achievement_rate`が30-40%に改善

### 改善が確認できたら

4. **タイムアウト問題の再検討**
   - 教師あり学習が機能すると、全隊出場中の状況が減る
   - タイムアウト発生頻度が減る
   - 処理件数が改善する可能性

5. **長期学習の実施**
   - 100エピソード以上で学習
   - 収束を確認
   - テスト環境で評価

---

## 📝 修正ファイルのサマリー

### 変更したファイル

1. **`reinforcement_learning/environment/ems_environment.py`**
   - `advance_time()`メソッドを追加（926-998行目）
   - `step()`メソッドを簡略化（999-1130行目）
   - タイムアウト統計の記録を追加

2. **`reinforcement_learning/training/trainer.py`**
   - `_run_episode()`メソッドを修正（213-214行目）
   - `env.advance_time()`呼び出しを追加

---

## 🚀 期待される結果

### 学習の収束

**修正前:**
- 教師あり学習が機能しない
- ランダムな行動のみ
- 学習が収束しない

**修正後:**
- 教師あり学習が機能する
- 直近隊運用を学習
- 学習が収束する（100エピソード程度で）

### テスト環境での性能

**修正前:**
- 平均応答時間: 20分
- 6分達成率: 2%
- 実用に耐えない

**修正後:**
- 平均応答時間: 8-10分
- 6分達成率: 30-40%
- 実用レベル

---

**結論**: 
教師あり学習の修正は完了しました。次回の学習実行で、`optimal_action`が取得でき、応答時間が大幅に改善することを期待します。

