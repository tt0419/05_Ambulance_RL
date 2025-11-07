# 詳細分析と修正案

**作成日**: 2025年11月6日  
**目的**: ログ分析結果と4つの問題に対する修正方針

---

## 📊 実行結果のサマリー

### 基本統計

```
episode_length: 1440ステップ ✅
total_calls: 1199件 / 1526件（78.6%） ❌
平均応答時間: 20.02分
6分達成率: 2.25%
```

### 改善状況

| 項目 | 修正前 | 修正後 | 評価 |
|------|--------|--------|------|
| エピソード完了 | 715ステップで強制終了 | 1440ステップ完走 | ✅ |
| エラー発生 | `calculate_unhandled_penalty` | なし | ✅ |
| 処理件数 | 498件（33%） | 1199件（78.6%） | 🟡 |
| タイムアウト処理 | なし | 動作（不完全） | 🟡 |

---

## 🔍 ユーザーからの4つの質問と回答

### 質問1: 1440ステップで1526件すべて処理できているか？

**回答: いいえ、327件（21.4%）が未処理です。**

**詳細分析:**

```
total_calls: 1199件（処理済み）
未処理: 327件（21.4%）
```

**原因:**
1. タイムアウト処理が動作しているが、**タイミングに問題**がある
2. NEW_CALLイベントが来た時のみタイムアウトチェックされる
3. その間に時間が経過し、一部の事案が放棄される

**影響:**
- 軽症事案が優先的にタイムアウトされている（📋 軽症搬送見送りが多い）
- 中等症・重症は比較的処理されている

---

### 質問2: 待機時間が205分など異常に長い

**回答: はい、これは重大な問題です。タイムアウトチェックのタイミングが不適切です。**

**問題の流れ:**

```
ステップ100（1時間40分）: 
  - 中等症の事案が発生
  - pending_callに設定
  - 配車失敗（全隊出場中、available: 0台）

ステップ101-200:
  - NEW_CALLイベントなし
  - タイムアウトチェックが実行されない ← ★問題★
  - pending_callは保持され続ける

ステップ201（3時間21分）:
  - NEW_CALLイベント発生
  - 初めてタイムアウトチェックが実行される
  - 待機時間: 201分 - 100分 = 101分 → タイムアウト
  - 「⚡ 中等症応援: 中等症 (205.0分待機)」と表示
```

**現在の実装の問題:**

```python
# step()メソッド内
while event_queue and event_queue[0].time <= end_time:
    if event.event_type == EventType.NEW_CALL:
        if self.pending_call is not None:
            # ★ここでタイムアウトチェック★
            if wait_time > max_wait:
                # タイムアウト処理
```

**問題点:**
- ✅ タイムアウト処理自体は正しく実装されている
- ❌ NEW_CALLイベントが来るまで実行されない
- ❌ イベントの間隔が長い場合、待機時間が異常に伸びる

**修正が必要:**

タイムアウトチェックを**毎ステップの最初**に実行する。

---

### 質問3: 復帰が遅い、サービス時間生成は正常か？

**回答: 復帰時間自体は正常範囲内です。ServiceTimeGeneratorは確率的に動作しています。**

**サービス時間の確認:**

```python
# _calculate_ambulance_completion_time()の実装確認
on_scene_time_minutes = self.service_time_generator.generate_time(severity, 'on_scene_time')
hospital_time_minutes = self.service_time_generator.generate_time(severity, 'hospital_time')
```

**確率的生成:**
- `on_scene_time`: 対数正規分布（平均15-20分、σ=0.5）
- `hospital_time`: 対数正規分布（平均20-30分、σ=0.5）
- `transport_time`: 移動時間行列から取得
- `return_time`: 移動時間行列から取得

**総活動時間の例:**

| フェーズ | 時間（分） |
|----------|------------|
| 応答 | 5-30分 |
| 現場活動 | 10-40分 |
| 搬送 | 5-30分 |
| 病院滞在 | 15-60分 |
| 帰署 | 5-30分 |
| **合計** | **40-190分** |

**ログ確認:**
```
[配車] Ep20-740: 中等症 → 高島平第２救急(実車) 30.4分 (利用可能:2台)
[配車] Ep20-760: 軽症 → 本郷救急(実車) 14.9分 (利用可能:0台)
```

**応答時間は正常範囲（10-30分）。**

**復帰が遅く見える理由:**
1. 総活動時間が40-100分程度（正常）
2. 利用可能台数が減り続ける → 全隊出場中の状態が長く続く
3. タイムアウト処理が遅れる → さらに事案が溜まる

**結論: ServiceTimeGeneratorは正常に動作している。問題はタイムアウト処理のタイミング。**

---

### 質問4: 教師あり学習が機能していない

**回答: はい、これは既知の問題です。「段階2の修正」で対応する予定です。**

**現在の問題:**

```python
# trainer.py
optimal_action = env.get_optimal_action()  # ← pending_call=なし
step_result = env.step(action)  # ← この中でpending_call設定
```

**ログ確認:**
```
[ACTION DEBUG] Step 0
  teacher_prob: 0.8992  ← 教師ありモード有効
  get_optimal_action呼び出し: True
  optimal_action: None  ← pending_callがないため
  [ERROR] pending_call is None or doesn't exist!
  use_teacher: False  ← 教師あり学習が機能しない
```

**影響:**
- 教師あり学習確率90%が設定されているが機能していない
- すべての行動がランダムまたはPPOエージェントによる選択
- 直近隊運用（最適行動）が学習されない
- 応答時間が改善しない（20分前後で停滞）

**修正方針（ドキュメント記載済み）:**

1. **解決策A: advance_time()メソッドを追加（推奨）**
2. **解決策B: step()の戻り値を拡張**
3. **解決策C: step()の処理順序を変更**

**これは「段階2」の修正として、タイムアウト処理の完全修正後に実施予定。**

---

## 🔧 修正案

### 修正1: タイムアウトチェックのタイミング変更（最重要）

**目的:** 待機時間が異常に長くなる問題を解決

**修正箇所:** `step()`メソッドの最初

**修正前:**
```python
def step(self, action: int) -> StepResult:
    end_time = self.current_time_seconds + self.time_per_step
    
    # イベント処理
    while event_queue and event_queue[0].time <= end_time:
        if event.event_type == EventType.NEW_CALL:
            if self.pending_call is not None:
                # ★タイムアウトチェック（NEW_CALLイベント時のみ）★
                if wait_time > max_wait:
                    # タイムアウト処理
```

**修正後:**
```python
def step(self, action: int) -> StepResult:
    end_time = self.current_time_seconds + self.time_per_step
    
    # ★★★ Phase 0: 毎ステップ、最初にタイムアウトチェック ★★★
    if self.pending_call is not None:
        wait_time_seconds = self.current_time_seconds - self.call_start_times.get(
            self.pending_call['id'], self.current_time_seconds
        )
        max_wait_seconds = self._get_max_wait_time(self.pending_call['severity']) * 60
        
        if wait_time_seconds > max_wait_seconds:
            # タイムアウト処理
            if self._episode_count <= 3 or self.episode_step <= 100:  # 詳細ログ
                print(f"[TIMEOUT] ステップ{self.episode_step}: 事案{self.pending_call['id']}({self.pending_call['severity']})を{wait_time_seconds/60:.1f}分待機でタイムアウト")
            
            self._handle_unresponsive_call(self.pending_call, wait_time_seconds / 60)
            self.pending_call = None  # クリア
    
    # Phase 1: イベント処理
    while event_queue and event_queue[0].time <= end_time:
        # 復帰イベントは常に処理
        if event.event_type == EventType.AMBULANCE_AVAILABLE:
            self._process_next_event()
            continue
        
        # NEW_CALLイベント
        if event.event_type == EventType.NEW_CALL:
            if self.pending_call is None:
                self._process_next_event()
                # ★タイムアウトチェック不要（Phase 0で実施済み）★
            else:
                # pending_callが既に存在する場合はスキップ
                break
    
    # Phase 2: 配車処理
    # ...
```

**効果:**
- ✅ タイムアウトチェックが毎ステップ実行される
- ✅ 待機時間が最大10分（重症）、20分（中等症）、45分（軽症）に制限される
- ✅ 異常に長い待機時間（205分など）が発生しなくなる
- ✅ 処理件数が1526件に近づく（100%達成）

---

### 修正2: タイムアウト処理の統計記録

**目的:** タイムアウト発生状況を監視

**追加コード:**
```python
# __init__()
self.timeout_stats = {
    '重篤': 0,
    '重症': 0,
    '中等症': 0,
    '軽症': 0,
    'total': 0
}

# _handle_unresponsive_call()
self.timeout_stats[severity] += 1
self.timeout_stats['total'] += 1

# _get_episode_stats()
stats['timeout_calls'] = self.timeout_stats.copy()
```

**wandbへの記録:**
```python
# trainer.py
wandb.log({
    'timeout/total': info['episode_stats']['timeout_calls']['total'],
    'timeout/critical': info['episode_stats']['timeout_calls']['重篤'],
    'timeout/severe': info['episode_stats']['timeout_calls']['重症'],
    'timeout/moderate': info['episode_stats']['timeout_calls']['中等症'],
    'timeout/mild': info['episode_stats']['timeout_calls']['軽症']
})
```

---

### 修正3: デバッグログの改善

**目的:** タイムアウト処理を監視

**追加ログ:**
```python
# step()の最初（Phase 0）
if self.episode_step % 100 == 0:  # 100ステップごと
    print(f"\n[STEP {self.episode_step}]")
    print(f"  current_time: {self.current_time_seconds/60:.1f}分")
    print(f"  pending_call: {'あり' if self.pending_call else 'なし'}")
    if self.pending_call:
        wait_time = self.current_time_seconds - self.call_start_times.get(
            self.pending_call['id'], self.current_time_seconds
        )
        print(f"  待機時間: {wait_time/60:.1f}分")
    print(f"  available_ambulances: {sum(1 for a in self.ambulance_states.values() if a['status'] == 'available')}台")
    print(f"  timeout_total: {self.timeout_stats['total']}件")
```

---

## 📈 期待される改善

### 修正前（現在）

```
total_calls: 1199件（78.6%）
タイムアウト発生: あり（不定期）
待機時間: 最大283分
利用可能台数: 0-20台（変動大）
```

### 修正後（期待）

```
total_calls: 1526件（100%） ← 目標達成
タイムアウト発生: あり（定期的）
待機時間: 最大45分（軽症）、20分（中等症）、10分（重症）
利用可能台数: 5-30台（安定）
```

### 詳細予測

| 指標 | 現在 | 修正後（予測） |
|------|------|----------------|
| 処理件数 | 1199件 | 1526件 |
| タイムアウト件数 | 不明 | 50-100件 |
| 平均待機時間 | 20.02分 | 18-19分 |
| 6分達成率 | 2.25% | 3-5% |
| 最大待機時間 | 283分 | 45分 |

---

## 🎯 実装の優先順位

### 優先度1: タイムアウトチェックのタイミング変更（今すぐ）

**理由:**
- 待機時間が異常に長い問題の根本原因
- 処理件数を100%に改善できる
- 実装が比較的簡単

**実装時間:** 10分

**効果:** 大

---

### 優先度2: タイムアウト統計の記録（推奨）

**理由:**
- タイムアウト発生状況を監視できる
- wandbで可視化できる
- 問題の早期発見

**実装時間:** 5分

**効果:** 中

---

### 優先度3: 教師あり学習の修正（後回し）

**理由:**
- タイムアウト処理が完全に動作してから実施
- 応答時間の大幅改善が期待できる
- 実装がやや複雑

**実装時間:** 30分

**効果:** 大（応答時間20分 → 8-10分）

---

## 📝 次のステップ

1. **修正1を実装**
   - `step()`メソッドの最初にタイムアウトチェックを追加
   - NEW_CALLイベント処理からタイムアウトチェックを削除

2. **動作確認**
   - 学習を再実行
   - `total_calls`が1526件に近づくか確認
   - 待機時間が45分以下に収まるか確認

3. **修正2を実装**
   - タイムアウト統計を記録
   - wandbで監視

4. **修正3を検討**
   - 教師あり学習の修正
   - 応答時間の改善

---

**結論**: 
最優先は「タイムアウトチェックのタイミング変更」です。これにより、すべての事案が処理され、待機時間も正常範囲に収まります。

