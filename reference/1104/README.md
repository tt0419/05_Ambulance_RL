# ems_environment.py 時間管理システム修正 - 成果物

## 📦 納品物一覧

| ファイル名 | 説明 | 重要度 |
|-----------|------|--------|
| **ems_environment.py** | 修正済みの環境ファイル（本体） | ⭐⭐⭐ |
| **IMPLEMENTATION_GUIDE.md** | 実装手順と動作確認方法 | ⭐⭐⭐ |
| **MODIFICATION_SUMMARY.md** | 修正内容の詳細サマリー | ⭐⭐ |
| **BEFORE_AFTER_COMPARISON.md** | 修正前後の動作比較 | ⭐⭐ |
| **test_time_management.py** | 動作確認用テストスクリプト | ⭐ |

---

## 🎯 修正の目的

訓練環境（`ems_environment.py`）とテスト環境（`validation_simulation.py`）の時間管理システムを統一し、学習したモデルがテスト環境で正しく動作するようにする。

### 主な問題（修正前）
- ❌ 訓練環境: 「1ステップ=1事案」で事案間の時間が無視される
- ❌ 救急車の復帰が非現実的に遅延（事案67件目まで復帰しない）
- ❌ テスト環境との時間管理が不整合
- ❌ 学習したモデルがテストで性能を発揮できない

### 解決策（修正後）
- ✅ 「1ステップ=1分（60秒）」の固定時間ステップ制
- ✅ イベント駆動型の処理（heapqによる優先度付きキュー）
- ✅ 救急車が実時間で復帰（例: 67分後）
- ✅ ValidationSimulatorと同じ時間管理
- ✅ 訓練とテストで性能が一致

---

## 🚀 クイックスタート

### 1. ファイルの配置

```bash
# バックアップを作成
cp reinforcement_learning/environment/ems_environment.py \
   reinforcement_learning/environment/ems_environment_backup.py

# 修正済みファイルをコピー
cp ems_environment.py \
   reinforcement_learning/environment/ems_environment.py
```

### 2. 動作確認

```bash
# テストスクリプトを実行
python test_time_management.py
```

期待される出力:
```
✓ テスト1成功: 時間が正しく進行
✓ テスト2成功: イベントが時刻順に処理される
✓ テスト3成功: 救急車が適切なタイミングで復帰
✓ テスト4成功: 事案間の時間が適切に処理される
✓ テスト5成功: エピソードが適切に終了
全テスト完了！
```

### 3. 学習の実行

```bash
# デバッグモードで短時間実行（動作確認）
python train_ppo.py --config config.yaml --debug

# 本番の学習実行
python train_ppo.py --config config.yaml
```

### 4. 検証

```bash
# ValidationSimulatorでテスト
python validation_simulation.py \
    --model models/ppo_latest.pth \
    --strategy ppo \
    --output results/validation_test/
```

---

## 📊 主要な変更内容

### 1. 時間管理変数の追加
```python
self.time_per_step = 60.0  # 1ステップ = 60秒
self.current_time_seconds = 0.0  # 経過秒数
self.event_queue = []  # イベント優先度付きキュー
```

### 2. `step`メソッドの書き換え
**旧システム**:
```python
# 1事案を処理して次へジャンプ
dispatch_ambulance(action)
advance_to_next_call()
```

**新システム**:
```python
# 60秒間のイベントを処理
while event_queue and event.time <= end_time:
    process_event()
if pending_call:
    dispatch_ambulance(action)
    schedule_return_event()
current_time_seconds += 60
```

### 3. イベント処理メソッドの追加
- `_schedule_event()`: イベントをキューに追加
- `_process_next_event()`: 次のイベントを処理
- `_handle_new_call_event()`: 事案発生を処理
- `_handle_ambulance_return_event()`: 救急車復帰を処理

### 4. 削除したメソッド
- `_advance_to_next_call()`: 不要（イベント処理に統合）
- `_update_ambulance_availability()`: 不要（イベントで自動処理）

---

## 📈 期待される効果

### 訓練環境での改善
- ✅ 事案間の時間が適切に処理される
- ✅ 救急車が現実的なタイミングで復帰
- ✅ より現実的なシミュレーション
- ✅ 学習の質が向上

### テスト環境での改善
- ✅ 訓練時とテスト時の性能が一致
- ✅ 応答時間が現実的な値
- ✅ ValidationSimulatorとの整合性

### 具体例: 救急車の復帰タイミング

**旧システム**:
```
配車（ステップ0） → 復帰（ステップ67 = 事案67件目）
問題: 事案が少ない場合、長時間復帰しない
```

**新システム**:
```
配車（0分） → 復帰（67分後）
利点: 実時間で確実に復帰
```

---

## 🔍 動作比較

### 事案間の時間処理

**旧システム**: 事案Aから事案Bへ瞬時にジャンプ
```
ステップ0: 事案A（0分）
ステップ1: 事案B（15分）← 時間が飛ぶ
```

**新システム**: 1分ずつ時間が進む
```
ステップ0: 事案A（0分）
ステップ1～14: 事案なし、時間だけ進む
ステップ15: 事案B（15分）
```

---

## ⚠️ 重要な注意事項

### 1. 既存モデルの互換性
- **旧バージョンで学習したモデルは使用不可**
- 時間管理が根本的に変更されているため、学習を最初からやり直す必要がある

### 2. 変更していない機能
以下の機能は**変更なし**で動作します:
- ✅ サービス時間生成（ServiceTimeGenerator）
- ✅ 病院選択モデル（HospitalSelection）
- ✅ 報酬計算（RewardDesigner）
- ✅ ハイブリッドモード
- ✅ 状態表現（StateEncoder）

### 3. 設定ファイル（config.yaml）
- **既存の設定ファイルをそのまま使用可能**
- APIは変更されていない

---

## 📚 ドキュメント詳細

### IMPLEMENTATION_GUIDE.md
実装手順、動作確認、トラブルシューティングを詳しく解説。
- ステップバイステップの実装手順
- 動作確認チェックリスト
- よくある問題と解決策
- FAQ

### MODIFICATION_SUMMARY.md
修正内容の技術的な詳細を解説。
- 各メソッドの変更内容
- コード例
- 設計思想
- 互換性情報

### BEFORE_AFTER_COMPARISON.md
修正前後の動作を図解で比較。
- 時間進行の違い
- イベント処理の違い
- 救急車復帰タイミングの違い
- 具体例とコード

### test_time_management.py
修正内容を検証するテストスクリプト。
- 時間ステップの進行テスト
- イベントキューの動作テスト
- 救急車復帰タイミングのテスト
- 事案間の時間処理テスト

---

## 🎓 使用方法

### 基本的な使い方

```python
from environment.ems_environment import EMSEnvironment

# 環境の初期化
env = EMSEnvironment(config_path="config.yaml", mode="train")

# リセット
observation = env.reset()

# ステップ実行
action = 0  # 救急車0を選択
result = env.step(action)

print(f"報酬: {result.reward}")
print(f"終了: {result.done}")
print(f"現在時刻: {env.current_time_seconds}秒 ({env.current_time_seconds/60:.1f}分)")
```

### デバッグモード

```python
env = EMSEnvironment(config_path="config.yaml", mode="train")
env.verbose_logging = True  # 詳細ログを有効化

observation = env.reset()
result = env.step(0)

# 詳細な活動時間が出力される:
# 救急車0活動時間計算（秒単位）:
#   応答: 480.0秒, 現場: 900.0秒
#   搬送: 720.0秒, 病院: 1200.0秒, 帰署: 720.0秒
#   総活動時間: 67.0分
```

---

## 🔧 トラブルシューティング

### 問題: 時間が進まない
```python
# step()メソッドで時間を進めているか確認
print(f"ステップ前: {env.current_time_seconds}秒")
result = env.step(action)
print(f"ステップ後: {env.current_time_seconds}秒")
# 期待: 60秒増加
```

### 問題: 救急車が復帰しない
```python
# イベントキューを確認
print(f"イベントキュー数: {len(env.event_queue)}")
for event in env.event_queue[:5]:
    print(f"  時刻={event.time:.1f}秒, タイプ={event.event_type}")
```

### 問題: エピソードがすぐ終了
```python
# 終了条件を確認
print(f"イベントキュー: {len(env.event_queue)}件")
print(f"pending_call: {env.pending_call is not None}")
print(f"現在時刻: {env.current_time_seconds}秒")
print(f"最大時刻: {24*3600}秒")
```

---

## 📞 サポート

問題が解決しない場合:
1. **test_time_management.py**を実行して基本動作を確認
2. **IMPLEMENTATION_GUIDE.md**のトラブルシューティングセクションを参照
3. バックアップファイルから元に戻して再試行

---

## ✅ チェックリスト

実装前:
- [ ] 元のファイルをバックアップ
- [ ] ドキュメントを一読

実装後:
- [ ] test_time_management.pyが成功
- [ ] 学習が正常に実行される
- [ ] 時間が1分ずつ進行
- [ ] 救急車が適切なタイミングで復帰
- [ ] ValidationSimulatorでテスト成功

---

## 📝 まとめ

この修正により、訓練環境とテスト環境の時間管理が統一され、学習したモデルがテスト環境で正しく動作するようになります。

### ビフォー・アフター
```
【修正前】
訓練: 1ステップ=1事案（時間が飛ぶ）
  ↓ 学習
  ↓
テスト: 1ステップ=1分（実時間）
  ↓
結果: 性能が大幅に低下 ❌

【修正後】
訓練: 1ステップ=1分（実時間）
  ↓ 学習
  ↓
テスト: 1ステップ=1分（実時間）
  ↓
結果: 性能が一致 ✅
```

---

**作成日**: 2025年11月4日  
**バージョン**: 1.0  
**対応環境**: ems_environment.py, validation_simulation.py
