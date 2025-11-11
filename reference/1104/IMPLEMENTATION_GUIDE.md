# ems_environment.py 時間管理システム修正 実装ガイド

## 📋 目次
1. [修正の概要](#修正の概要)
2. [ファイル構成](#ファイル構成)
3. [実装手順](#実装手順)
4. [動作確認](#動作確認)
5. [トラブルシューティング](#トラブルシューティング)

---

## 修正の概要

### 目的
訓練環境（`ems_environment.py`）の時間管理を、テスト環境（`validation_simulation.py`）と整合性のある**「1ステップ=1分（60秒）の固定時間ステップ制」**に変更する。

### 背景
- **旧システム**: 「1ステップ=1事案」で、事案間の時間が無視される
- **問題**: 救急車の復帰が非現実的に遅延し、学習したモデルがテスト環境で性能を発揮できない
- **解決策**: ValidationSimulatorと同じ時間管理システムを採用

### 主要な変更
1. ✅ 固定時間ステップ制（1ステップ=60秒）の導入
2. ✅ イベント駆動型の処理（heapqによる優先度付きキュー）
3. ✅ 救急車の復帰イベントのスケジューリング
4. ✅ 事案間の時間を適切に処理

---

## ファイル構成

```
/mnt/user-data/outputs/
├── ems_environment.py              # 【主要】修正済みの環境ファイル
├── MODIFICATION_SUMMARY.md         # 修正内容のサマリー
├── BEFORE_AFTER_COMPARISON.md      # 修正前後の動作比較
├── test_time_management.py         # 動作確認テスト
└── IMPLEMENTATION_GUIDE.md         # このファイル
```

---

## 実装手順

### ステップ1: バックアップの作成

元のファイルをバックアップします。

```bash
# 元のファイルをバックアップ
cp reinforcement_learning/environment/ems_environment.py \
   reinforcement_learning/environment/ems_environment_backup.py

echo "バックアップ完成: ems_environment_backup.py"
```

### ステップ2: 修正済みファイルの配置

修正済みファイルを適切な場所に配置します。

```bash
# 修正済みファイルをコピー
cp /mnt/user-data/outputs/ems_environment.py \
   reinforcement_learning/environment/ems_environment.py

echo "修正済みファイルを配置完了"
```

### ステップ3: インポートの確認

修正済みファイルが正しくインポートできるか確認します。

```python
# Python環境で実行
import sys
sys.path.append('reinforcement_learning')

try:
    from environment.ems_environment import EMSEnvironment
    print("✓ インポート成功")
except Exception as e:
    print(f"❌ インポートエラー: {e}")
```

### ステップ4: 基本動作の確認

簡単なテストで基本動作を確認します。

```python
# test_basic.py
from environment.ems_environment import EMSEnvironment
import numpy as np

# 環境の初期化
env = EMSEnvironment(config_path="config.yaml", mode="train")

# リセット
obs = env.reset()
print(f"初期観測の形状: {obs.shape}")

# 1ステップ実行
action = 0  # 救急車0を選択
result = env.step(action)

print(f"報酬: {result.reward}")
print(f"終了フラグ: {result.done}")
print(f"✓ 基本動作確認完了")
```

### ステップ5: 時間管理の確認

新しい時間管理システムが正しく動作しているか確認します。

```python
# test_time_management.py（提供済み）
python /mnt/user-data/outputs/test_time_management.py
```

期待される出力:
```
テスト1: 時間ステップの進行確認
✓ テスト1成功: 時間が正しく進行

テスト2: イベントキューの順序確認
✓ テスト2成功: イベントが時刻順に処理される

...

全テスト完了！
```

### ステップ6: 学習の実行

修正された環境で学習を実行します。

```bash
# デバッグモードで短時間実行
python train_ppo.py --config config.yaml --debug

# 期待される動作:
# - 時間が1分ずつ進行
# - 事案間の時間が適切に処理される
# - 救急車が現実的なタイミングで復帰
```

### ステップ7: ValidationSimulatorとの比較

学習したモデルをValidationSimulatorでテストします。

```bash
# 学習済みモデルでテスト
python validation_simulation.py \
    --model models/ppo_latest.pth \
    --strategy ppo \
    --output results/validation_test/

# 期待される結果:
# - 訓練時とテスト時の性能が一致
# - 応答時間が現実的な値
```

---

## 動作確認

### チェックリスト

#### ✅ 基本機能
- [ ] 環境が正常に初期化される
- [ ] `reset()`が正常に動作する
- [ ] `step()`が正常に動作する
- [ ] 観測が正しい形状で返される
- [ ] 報酬が計算される

#### ✅ 時間管理
- [ ] `current_time_seconds`が60秒ずつ増加
- [ ] `episode_step`が1ずつ増加
- [ ] イベントキューが正しく動作
- [ ] 事案が時刻順に処理される
- [ ] 事案間の時間が適切に処理される

#### ✅ 救急車管理
- [ ] 配車時に救急車が`dispatched`状態になる
- [ ] 復帰イベントがスケジュールされる
- [ ] 救急車が適切なタイミングで復帰（`available`に変更）
- [ ] 活動時間が現実的な値（30～90分程度）

#### ✅ エピソード管理
- [ ] エピソードが設定時間で終了
- [ ] 全事案を処理した場合に終了
- [ ] 最大ステップ数で終了

#### ✅ 既存機能の維持
- [ ] サービス時間生成が正常に動作
- [ ] 病院選択が正常に動作
- [ ] 報酬計算が正常に動作
- [ ] ハイブリッドモードが正常に動作（有効な場合）

### デバッグ用ログの有効化

詳細なログを確認したい場合:

```python
env = EMSEnvironment(config_path="config.yaml", mode="train")
env.verbose_logging = True  # 詳細ログを有効化

obs = env.reset()
result = env.step(0)
```

期待される出力:
```
救急車0活動時間計算（秒単位）:
  応答: 480.0秒, 現場: 900.0秒
  搬送: 720.0秒, 病院: 1200.0秒, 帰署: 720.0秒
  総活動時間: 67.0分
```

---

## トラブルシューティング

### 問題1: インポートエラー

**症状**:
```
ImportError: cannot import name 'Event' from 'validation_simulation'
```

**解決策**:
```python
# validation_simulation.pyが正しいパスにあるか確認
import sys
sys.path.append('path/to/validation_simulation')
```

### 問題2: 時間が進まない

**症状**: `current_time_seconds`が0のまま

**原因**: `step()`内で時間を進めるコードが実行されていない

**解決策**:
```python
# step()メソッドの最後で時間を進めているか確認
self.current_time_seconds = end_time
self.episode_step += 1
```

### 問題3: 救急車が復帰しない

**症状**: 救急車がずっと`dispatched`状態のまま

**原因**: 復帰イベントがスケジュールされていない、またはイベント処理が実行されていない

**解決策**:
```python
# 配車時に復帰イベントをスケジュールしているか確認
return_event = Event(
    time=return_time,
    event_type=EventType.AMBULANCE_AVAILABLE,
    data={'ambulance_id': amb_id, 'station_h3': station_h3}
)
self._schedule_event(return_event)

# step()内でイベント処理ループが実行されているか確認
while self.event_queue and self.event_queue[0].time <= end_time:
    event = self._process_next_event()
```

### 問題4: エピソードがすぐに終了する

**症状**: 1～2ステップで`done=True`になる

**原因**: `_is_episode_done()`の判定条件が厳しすぎる

**解決策**:
```python
# _is_episode_done()の条件を確認
def _is_episode_done(self) -> bool:
    # イベントキューが空 AND pending_callがない場合のみ終了
    if not self.event_queue and self.pending_call is None:
        return True
    
    # 時間制限チェック
    episode_hours = self.config['data'].get('episode_duration_hours', 24)
    max_time_seconds = episode_hours * 3600.0
    if self.current_time_seconds >= max_time_seconds:
        return True
    
    return False
```

### 問題5: 性能が低下した

**症状**: 修正後、応答時間が悪化した

**原因**: 時間管理の変更により、救急車の稼働状況が変化

**解決策**:
- 学習を最初からやり直す（古い方策を使わない）
- ハイパーパラメータを調整する
- エピソード長を調整する

### 問題6: メモリ使用量が増加

**症状**: メモリ使用量が大幅に増加

**原因**: イベントキューに大量のイベントが蓄積

**解決策**:
```python
# イベントキューのサイズを定期的にチェック
if len(self.event_queue) > 10000:
    print(f"警告: イベントキューが大きい: {len(self.event_queue)}件")

# 古いイベントを定期的にクリーンアップ（必要な場合）
```

---

## よくある質問

### Q1: 旧バージョンのモデルは使えますか？

**A**: いいえ、使えません。時間管理が根本的に変更されているため、旧バージョンで学習したモデルは新システムでは正しく動作しません。学習を最初からやり直してください。

### Q2: ステップあたりの時間を変更できますか？

**A**: 可能ですが、推奨しません。`self.time_per_step`を変更すれば可能ですが、ValidationSimulatorとの整合性が失われます。

### Q3: 既存の設定ファイル（config.yaml）は使えますか？

**A**: はい、そのまま使えます。APIは変更されていないため、既存の設定ファイルが動作します。

### Q4: どのくらいの性能向上が期待できますか？

**A**: テスト環境での性能が向上します（訓練時とテスト時の性能が一致）。ただし、訓練環境での絶対的な報酬値は変化する可能性があります。

### Q5: ハイブリッドモードは影響を受けますか？

**A**: いいえ、ハイブリッドモードのロジックは変更されていません。そのまま使用できます。

---

## 参考資料

1. **MODIFICATION_SUMMARY.md**: 修正内容の詳細
2. **BEFORE_AFTER_COMPARISON.md**: 修正前後の動作比較
3. **integration_strategy.md**: 統合方針（元の設計書）
4. **time_management_detailed_comparison.md**: 時間管理の詳細比較

---

## サポート

問題が解決しない場合:
1. `test_time_management.py`を実行して、基本的な動作を確認
2. `verbose_logging = True`でデバッグログを確認
3. バックアップファイルから元に戻して再度試行

---

## まとめ

この修正により:
- ✅ 訓練環境とテスト環境の時間管理が統一
- ✅ 事案間の時間を適切に処理
- ✅ 救急車の復帰が現実的なタイミングに
- ✅ ValidationSimulatorとの整合性が向上
- ✅ 学習の質が向上

修正後は、学習したモデルがテスト環境で正しく動作するようになります。
