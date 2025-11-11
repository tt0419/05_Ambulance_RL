# ems_environment.py 時間管理システム修正サマリー

## 修正日時
2025年11月4日

## 修正の目的
訓練環境（`ems_environment.py`）の時間管理を、テスト環境（`validation_simulation.py`）と整合性のある**「1ステップ=1分（60秒）の固定時間ステップ制」**に変更し、学習したモデルがテスト環境で正しく動作するようにする。

## 主要な変更点

### 1. 時間管理変数の追加（`__init__`メソッド）
```python
# 固定時間ステップ制: 1ステップ = 1分 = 60秒
self.time_per_step = 60.0  # 秒
self.current_time_seconds = 0.0  # エピソード開始からの経過秒数
self.event_queue = []  # イベント優先度付きキュー（heapq使用）
```

**理由**: ValidationSimulatorと同じ時間粒度で動作させるため。

### 2. `reset`メソッドの修正
**変更内容**:
- `self.current_time_seconds = 0.0` と `self.event_queue = []` を初期化
- 全ての事案を`NEW_CALL`イベントとしてキューに追加
- `pending_call`は`step()`内で設定するように変更

**理由**: イベント駆動型のシミュレーションに対応するため。

### 3. `step`メソッドの完全書き換え
**旧ロジック**: 「1ステップ=1事案」で、事案ごとに時間がジャンプ

**新ロジック**: 「1ステップ=1分（60秒）」で、固定時間ずつ進行

**処理フロー**:
```python
1. start_time と end_time を定義（60秒間）
2. この1分間に発生するイベントを処理（while ループ）
   - NEW_CALLイベント → pending_callにセット
   - AMBULANCE_AVAILABLEイベント → 救急車を復帰
3. pending_callがあれば、actionで配車
4. 配車成功なら、復帰イベントをスケジュール
5. current_time_secondsを60秒進める
6. episode_stepをインクリメント
7. 観測を返す
```

**理由**: 事案間の時間も適切に処理し、救急車の復帰タイミングを現実的にするため。

### 4. イベント処理メソッドの追加
新規追加したメソッド:
- `_schedule_event(event)`: イベントをキューに追加
- `_process_next_event()`: 次のイベントを処理
- `_handle_new_call_event(event)`: NEW_CALLイベントの処理
- `_handle_ambulance_return_event(event)`: 救急車復帰イベントの処理

**理由**: ValidationSimulatorと同じイベント駆動型の処理を実現するため。

### 5. 削除したメソッド
- `_advance_to_next_call()`: 不要（step内のイベント処理に統合）
- `_update_ambulance_availability()`: 不要（イベントキューの復帰イベントで処理）

**理由**: 古い時間管理システムの一部であり、新システムでは不要。

### 6. `_dispatch_ambulance`メソッドの修正
**変更内容**:
- `last_dispatch_time`を`self.current_time_seconds`（秒単位）に変更
- `completion_time_seconds`を返すように修正

**理由**: 秒単位の時間管理に対応するため。

### 7. `_calculate_ambulance_completion_time`メソッドの修正
**変更内容**:
- 全ての時間計算を秒単位に統一
- `ServiceTimeGenerator`が返す分単位の値を秒に変換（×60）

**理由**: 秒単位の時間管理に対応し、ValidationSimulatorと整合性を保つため。

### 8. `_is_episode_done`メソッドの修正
**旧ロジック**: 事案数とステップ数で判定

**新ロジック**: 
- イベントキューが空で pending_call もない場合
- エピソード時間制限（秒単位）を超えた場合
- 最大ステップ数を超えた場合

**理由**: 時間ベースとステップ数ベースの両方で終了判定を行うため。

## 重要な制約（変更していない箇所）

以下の機能は、すでにValidationSimulatorと統一されており、正常に動作しているため、**一切変更していません**:

1. `_init_service_time_generator` - サービス時間生成
2. `_init_hospital_selection` - 病院選択モデル
3. `_calculate_reward` - 報酬計算
4. `hybrid_mode` - ハイブリッドモードのロジック
5. `get_optimal_action` - 最適行動の取得
6. `StateEncoder` - 状態表現

## テスト推奨事項

1. **単一事案テスト**: 1つの事案で救急車が正しく復帰するか確認
2. **複数事案テスト**: 事案間の時間が適切に処理されるか確認
3. **救急車復帰テスト**: 活動時間が現実的な値になっているか確認
4. **エピソード終了テスト**: 時間制限で正しく終了するか確認

## 期待される効果

1. **訓練環境とテスト環境の時間管理が統一** → 学習したモデルがテストで正しく動作
2. **事案間の時間を適切に処理** → 救急車の復帰タイミングが現実的に
3. **より現実的なシミュレーション** → 学習の質が向上

## 互換性

- **PPOTrainerとの互換性**: `step()`と`reset()`のAPIは変更なし
- **ConfigファイルとStep互換性**: 既存のconfig.yamlをそのまま使用可能
- **既存機能との互換性**: サービス時間生成、病院選択など、既存機能はすべて維持

## 注意事項

1. **ステップあたりの時間**: 1ステップ = 1分（60秒）固定
2. **エピソード長**: configの`episode_duration_hours`で制御（デフォルト24時間）
3. **イベント処理**: `heapq`を使用した優先度付きキュー
4. **時間単位**: 内部は秒単位、表示は分単位を併用
