# 今後の実施計画

**作成日**: 2025年11月7日  
**目的**: EMSEnvironmentの検証と改善の優先順位付け

---

## 📋 優先順位付けされた作業リスト

### 【最優先】Phase 1: デバッグログで動作検証

**目的**: 現在の実装が正しく動作しているか確認

**作業内容:**
1. 学習を実行してデバッグログを取得
2. 復帰時刻計算を確認
3. 復帰イベント処理を確認
4. 応答時間の差の原因を特定

**実施方法:**
```bash
python train_ppo.py --config config_hybrid_continuous.yaml
```

**確認ポイント:**
```
[復帰時刻] 救急車137:
  事案発生: 0.0分, 配車時刻: 1.0分  ← 差を確認
  総活動時間: XX.X分  ← 正常範囲か（40-100分）

[復帰スケジュール] 救急車137:
  活動時間: XX.X分  ← マイナスでないか

[復帰処理] 救急車137が復帰:
  利用可能: XX台  ← 増加しているか
```

**期待される結果:**
- 各フェーズの時間が正常範囲
- 復帰イベントが正常に処理される
- 利用可能台数が適切に推移

**判断基準:**
- ✅ 正常 → Phase 2へ
- ❌ 異常 → 修正してPhase 1をやり直し

---

### 【高優先】Phase 2: ValidationSimulatorに配車失敗率の記録を追加

**目的**: ValidationSimulatorで何%がスルーされているか明確化

**作業内容:**
1. ValidationSimulatorに統計追加
   ```python
   self.statistics['dispatch_failures'] = 0
   
   if ambulance:
       # 配車処理
   else:
       self.statistics['dispatch_failures'] += 1
       print(f"[WARN] Call {call.id}: No ambulance available")
   ```

2. レポートに追加
   ```json
   {
     "dispatch_failures": {
       "total": XX,
       "rate": X.X%
     }
   }
   ```

3. ValidationSimulatorを再実行

**期待される結果:**
- 配車失敗率が明確になる
- EMSEnvironmentのタイムアウト率（23.4%）と比較できる

**判断基準:**
- ValidationSimulatorの配車失敗率が5-10% → EMSの23.4%は異常
- ValidationSimulatorの配車失敗率が20%以上 → EMSは正常

---

### 【高優先】Phase 3: EMSEnvironmentと ValidationSimulatorの動作比較

**目的**: 両システムが同じ条件で同じ結果を出すか検証

**作業内容:**
1. 同じデータ（6月15日）で実行
2. 同じ戦略（直近隊運用）で実行
3. 結果を比較

**比較項目:**
| 項目 | ValidationSimulator | EMSEnvironment | 判定 |
|------|---------------------|----------------|------|
| 完了件数 | 1442件 | ??? | |
| 平均応答時間 | 7.47分 | ??? | |
| 6分達成率 | 35.08% | ??? | |
| 配車失敗率 | ???% | 23.4% | |

**判断基準:**
- 応答時間の差が1分以内 → 許容範囲
- 応答時間の差が5分以上 → 設計の問題

---

### 【中優先】Phase 4: PPO学習戦略の見直し

**目的**: 教師あり学習の問題を解決し、PPOが収束するようにする

**ユーザーの洞察を踏まえた設計:**

#### A. 教師あり学習の廃止または変更

**現在の問題:**
```
教師あり学習（直近隊運用）
  ↓ 90% → 5%に減衰
PPO学習
  ↓
直近隊運用をコピーするだけで、それ以上の最適化ができない
```

**解決策1: 教師あり学習を廃止**
```yaml
teacher:
  enabled: false  # 完全にゼロから学習
```

**利点:**
- PPOが自由に探索できる
- 最適解（重症優先+軽症カバレッジ）を発見できる可能性

**欠点:**
- 学習に時間がかかる
- 収束しない可能性

---

**解決策2: 報酬設計を変更（推奨）**

**現在の報酬:**
```python
reward = -response_time * weight
```

**問題:**
- 応答時間のみを最小化
- 直近隊運用が最適解になる
- カバレッジが考慮されない

**改善案:**
```python
# 重症系: 応答時間重視（100%）
if severity in ['重篤', '重症']:
    reward = -response_time * 1.0

# 軽症系: 応答時間とカバレッジのバランス（50% + 50%）
else:
    time_reward = -response_time * 0.5
    coverage_reward = calculate_coverage_impact() * 0.5
    reward = time_reward + coverage_reward
```

**効果:**
- 重症系は直近隊運用を学習（教師ありと同じ）
- 軽症系はカバレッジを考慮した配車を学習
- バランスの取れた最適解に到達

---

**解決策3: カリキュラム学習**

**段階的な学習:**
```yaml
stage1:
  episodes: 0-1000
  teacher_prob: 0% # ゼロから学習
  reward: simple  # 応答時間のみ
  
stage2:
  episodes: 1000-3000
  teacher_prob: 0%
  reward: hybrid  # 応答時間 + カバレッジ
  
stage3:
  episodes: 3000-5000
  teacher_prob: 0%
  reward: advanced  # 複雑な報酬
```

---

#### B. ハイブリッドモードの見直し

**現在の設計:**
```python
if severity in ['重篤', '重症']:
    # 直近隊運用（学習しない）
    action = get_optimal_action()
    reward = 0.0
else:
    # PPOで学習
    action = ppo_action
    reward = calculate_reward()
```

**問題:**
- 重症系は学習しない
- 軽症系のみ学習するが、カバレッジの概念がない
- バランスが取れない

**改善案1: 重症系も学習**
```python
if severity in ['重篤', '重症']:
    # 教師あり学習（高確率で直近隊を選ぶ）
    optimal_action = get_optimal_action()
    if random() < 0.9:
        action = optimal_action
    else:
        action = ppo_action
    reward = calculate_reward()  # 学習する
```

**改善案2: カバレッジ報酬の追加**
```python
# すべての傷病度でカバレッジを考慮
if severity in ['重篤', '重症']:
    weight_rt = 0.9
    weight_coverage = 0.1
else:
    weight_rt = 0.5
    weight_coverage = 0.5

reward = -response_time * weight_rt + coverage_score * weight_coverage
```

---

### 【低優先】Phase 5: タイムアウト処理の完全修正

**現状の問題:**
- タイムアウトが遅れる（132-289分待機）
- 完了率が76.6%（ValidationSimulatorの94.5%より低い）

**修正方針:**
1. Phase 1のデバッグログで原因特定
2. 必要に応じて設計を見直す

**ただし:**
- これは最優先ではない
- PPO学習の収束が先

---

## 🎯 推奨される実施順序

### Week 1: 動作検証と基盤の修正

**Day 1-2: Phase 1実施**
- デバッグログで動作検証
- 復帰処理が正常か確認
- 応答時間の差の原因特定

**Day 3: Phase 2実施**
- ValidationSimulatorに統計追加
- 配車失敗率を明確化
- 比較分析

**Day 4-5: Phase 3実施**
- 両システムで同じ条件で実行
- 結果を比較
- 差異を分析

---

### Week 2: PPO学習の改善

**Day 6-7: Phase 4準備**
- 報酬設計の見直し
- カバレッジ計算の実装
- ハイブリッドモードの改善

**Day 8-10: Phase 4実施**
- 新しい報酬設計で学習
- 100-500エピソード実行
- 収束を確認

**Day 11-12: 評価**
- テスト環境で評価
- baselineと比較
- 最終調整

---

### Week 3以降: タイムアウト処理の改善（必要に応じて）

**Phase 5実施**
- タイムアウトチェックのタイミング修正
- 完了率を94%以上に改善

---

## 📝 具体的な次のアクション

### 今すぐ実施

**1. デバッグログの確認（Phase 1）**
```bash
python train_ppo.py --config config_hybrid_continuous.yaml
```

**確認内容:**
- 復帰時刻計算が正しいか
- 復帰イベントが処理されているか
- 活動時間が正常範囲か

**所要時間:** 30分（学習） + 30分（分析） = 1時間

---

### 次に実施（Phase 1の結果に応じて）

**2-A. Phase 1で異常が見つかった場合**
- 修正を実施
- Phase 1を再実行

**2-B. Phase 1で正常だった場合**
- Phase 2へ進む（ValidationSimulatorに統計追加）
- Phase 3へ進む（両システムの比較）

---

## 🔍 Phase 4の詳細設計

### 報酬設計の改善（推奨）

**目標:**
- 重症系: 応答時間を最小化（直近隊運用）
- 軽症系: 応答時間とカバレッジのバランス

**実装:**
```python
def calculate_reward(self, dispatch_result):
    severity = dispatch_result['severity']
    response_time = dispatch_result['response_time_minutes']
    
    # 重症系: 応答時間のみ（100%）
    if severity in ['重篤', '重症']:
        reward = -response_time * 1.0
        
        # 6分達成ボーナス
        if response_time <= 6:
            reward += 10.0
    
    # 軽症・中等症: バランス重視
    else:
        # A. 応答時間報酬（50%）
        time_reward = -response_time * 0.5
        
        # B. カバレッジ報酬（50%）
        coverage_before = self.calculate_coverage()
        # 配車後の予測カバレッジ
        coverage_after = self.predict_coverage_after_dispatch(
            dispatch_result['ambulance_id']
        )
        coverage_impact = coverage_before - coverage_after
        
        # カバレッジが維持されていればボーナス
        if coverage_impact < 0.05:  # 5%以内の低下
            coverage_reward = 5.0
        elif coverage_impact < 0.10:  # 10%以内
            coverage_reward = 0.0
        else:  # 10%以上低下
            coverage_reward = -coverage_impact * 50.0
        
        reward = time_reward + coverage_reward
    
    return reward
```

---

### 教師あり学習の廃止（推奨）

**現在の問題:**
- 教師あり学習で直近隊運用を覚える
- そこから先に進めない
- 収束しない

**解決策:**
```yaml
# config_hybrid_continuous.yaml
teacher:
  enabled: false  # 教師あり学習を廃止
  
# ゼロから学習
# 報酬設計で最適解を誘導
```

**効果:**
- PPOが自由に探索
- 最適解を発見できる可能性
- ただし、学習に時間がかかる（1000エピソード以上）

---

### ハイブリッドモードの改善（オプション）

**現在の設計:**
```python
if severity in ['重症', '重篤']:
    action = get_optimal_action()  # 学習しない
    reward = 0.0
else:
    action = ppo_action  # 学習する
    reward = calculate_reward()
```

**改善案:**
```python
# 重症系も学習するが、報酬設計で誘導
if severity in ['重症', '重篤']:
    action = ppo_action
    reward = -response_time * 1.0  # 応答時間のみ重視
    # → PPOが直近隊運用を学習
else:
    action = ppo_action
    reward = calculate_reward_with_coverage()  # バランス重視
    # → PPOがカバレッジも考慮
```

**効果:**
- 重症系も学習される
- 報酬設計で最適解を誘導
- よりフレキシブル

---

## 📊 期待される改善

### Phase 1-3完了後

| 指標 | 現在 | 目標 |
|------|------|------|
| 完了率 | 76.6% | 94%以上 |
| 平均応答時間 | 12.62分（教師あり）| 7-8分 |
| 6分達成率 | 27.5%（教師あり）| 35%以上 |

---

### Phase 4完了後（PPO改善）

| 指標 | 目標（重症系） | 目標（軽症系） |
|------|----------------|----------------|
| 平均応答時間 | 7-8分 | 10-12分 |
| 6分達成率 | 35-40% | 20-25% |
| カバレッジ維持率 | - | 95%以上 |

---

## 🚀 実装の優先度

### 今すぐ実施（必須）

1. **Phase 1: デバッグログの確認**
   - 所要時間: 1時間
   - リスク: 低
   - 効果: 高（問題の特定）

---

### 今週中に実施（推奨）

2. **Phase 2: ValidationSimulatorに統計追加**
   - 所要時間: 30分
   - リスク: 低
   - 効果: 中（配車失敗率の明確化）

3. **Phase 3: 両システムの比較**
   - 所要時間: 1時間
   - リスク: 低
   - 効果: 高（設計の検証）

---

### 来週実施（重要）

4. **Phase 4-A: 報酬設計の改善**
   - 所要時間: 2-3時間
   - リスク: 中
   - 効果: 高（PPO学習の改善）

5. **Phase 4-B: 教師あり学習の廃止**
   - 所要時間: 10分（設定変更のみ）
   - リスク: 低
   - 効果: 中（探索の自由度向上）

6. **Phase 4-C: 学習の実行と評価**
   - 所要時間: 5-10時間（学習）
   - リスク: 低
   - 効果: 高（収束の確認）

---

### 必要に応じて実施

7. **Phase 5: タイムアウト処理の完全修正**
   - 所要時間: 2-4時間
   - リスク: 中
   - 効果: 中（完了率の改善）

---

## 📝 各Phaseの詳細

### Phase 1: デバッグログの確認

**実施内容:**
```bash
# 1. 学習を実行（最初の1エピソードのみでも可）
python train_ppo.py --config config_hybrid_continuous.yaml

# 2. ログを確認
# - [復帰時刻]ログ: 各フェーズの時間
# - [復帰スケジュール]ログ: 活動時間
# - [復帰処理]ログ: 復帰イベントの処理

# 3. 問題を特定
# - 活動時間がマイナス → 時刻計算の問題
# - 活動時間が異常に長い → ServiceTimeGeneratorの問題
# - 復帰イベントが処理されない → イベント処理の問題
```

---

### Phase 2: ValidationSimulatorに統計追加

**実施内容:**
```python
# validation_simulation.py

# __init__()
self.statistics['dispatch_failures'] = 0
self.statistics['dispatch_success'] = 0

# _handle_new_call()
if ambulance:
    self.statistics['dispatch_success'] += 1
    # 配車処理
else:
    self.statistics['dispatch_failures'] += 1
    if self.verbose_logging:
        print(f"[WARN] Call {call.id}: No ambulance available")

# generate_report()
report['dispatch_stats'] = {
    'success': self.statistics['dispatch_success'],
    'failures': self.statistics['dispatch_failures'],
    'failure_rate': self.statistics['dispatch_failures'] / 
                   (self.statistics['dispatch_success'] + self.statistics['dispatch_failures']) * 100
}
```

**実行:**
```bash
python validation_simulation.py
```

---

### Phase 3: 両システムの比較

**実施内容:**
```bash
# 1. ValidationSimulator（baseline）を実行
python validation_simulation.py

# 2. EMSEnvironmentを直近隊運用で実行
# config_hybrid_continuous.yamlを変更:
teacher:
  enabled: true
  initial_prob: 1.0  # 100%教師あり
  final_prob: 1.0
  
# 実行
python train_ppo.py --config config_hybrid_continuous.yaml

# 3. 結果を比較
```

---

### Phase 4: PPO学習の改善

**実施内容:**

**4-A: 報酬設計の実装**
```python
# reinforcement_learning/environment/reward_designer.py
def calculate_step_reward_with_coverage(self, ...):
    # 重症系: 応答時間のみ
    # 軽症系: 応答時間 + カバレッジ
```

**4-B: 設定の変更**
```yaml
# config_hybrid_continuous.yaml
teacher:
  enabled: false  # 廃止

reward:
  mode: hybrid_coverage  # 新しいモード
  coverage_enabled: true
  weights:
    critical_rt: 1.0
    mild_rt: 0.5
    mild_coverage: 0.5
```

**4-C: 学習の実行**
```bash
python train_ppo.py --config config_hybrid_continuous.yaml --episodes 1000
```

---

## 🎯 成功基準

### Phase 1-3完了時

- ✅ 復帰処理が正常に動作
- ✅ 応答時間の差が説明できる
- ✅ 両システムの動作の違いが明確

### Phase 4完了時

- ✅ PPO学習が収束する
- ✅ 応答時間が改善する（教師なしでも）
- ✅ カバレッジが維持される

### 最終目標

- ✅ 平均応答時間: 8-10分
- ✅ 6分達成率: 30%以上
- ✅ 完了率: 90%以上
- ✅ テスト環境で使用可能

---

**結論**: 
まずPhase 1でデバッグログを確認し、復帰処理が正常か検証します。その後、Phase 2-3でValidationSimulatorとの比較を行い、Phase 4でPPO学習を改善します。

