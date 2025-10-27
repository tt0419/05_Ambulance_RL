# PPOStrategy Phase 3 修正内容

## 概要
救急隊のIDが正しく認識されるように`PPOStrategy`クラスを修正しました。
常に192台分の完全な状態辞書を構築することで、ValidationSimulatorから不完全な情報が渡されても対応できるようになりました。

## 主な変更点

### 1. 全救急車の静的情報を保持
```python
# __init__メソッドに追加
self.ambulance_static_info = {}  # {action_idx: {'station_h3': str, 'team_name': str, 'validation_id': str}}
```

**目的**: 
- 各救急車の所属署、H3インデックス、チーム名などの静的情報を保持
- ValidationSimulatorから情報が渡されない救急車のデフォルト状態を提供

### 2. 静的情報読み込みメソッドの追加
```python
def _load_ambulance_static_info(self):
    """
    全救急車の静的情報を読み込み
    EMSEnvironmentと同じフィルタリング処理を適用
    """
```

**処理内容**:
1. `data/tokyo/import/amb_place_master.csv`から救急車データを読み込み
2. EMSEnvironmentと同じフィルタリングを適用:
   - special_flag == 1
   - 「救急隊なし」を除外
   - デイタイム救急を除外（amb != 0）
   - 東京23区のみ
3. 各救急車の静的情報（所属署のH3インデックス、チーム名）を保存
4. 192台に満たない場合はダミーデータで補完

**補助メソッド**:
- `_create_dummy_static_info()`: ダミーの静的情報を作成
- `_fill_missing_static_info()`: 不足している静的情報をダミーで埋める

### 3. 状態辞書構築メソッドの修正
```python
def _build_state_dict(self, request, available_ambulances, context):
    """
    ValidationSimulatorの状態を学習環境形式に変換（Phase 3修正版）
    常に192台分の完全な状態辞書を構築
    """
```

**修正内容**:
1. **初期化**: まず192台分のデフォルト状態を作成
   - 全ての救急車を「unavailable」として初期化
   - 各救急車の所属署H3インデックスを静的情報から設定
   
2. **上書き**: ValidationSimulatorから渡された実際の状態で上書き
   - ID対応表を使用して正しい整数インデックスに変換
   - 利用可能な救急車の実際の状態を反映
   
3. **検証**: 状態辞書が必ず192台分あることを確認
   ```python
   assert len(ambulances) == 192, f"救急車数が不正: {len(ambulances)}"
   ```

### 4. アクションマスク作成の改善
```python
def _create_action_mask(self, available_ambulances):
    """利用可能な救急車のマスクを作成（Phase 3修正版）"""
```

**追加機能**:
- マスクの妥当性チェック
- 利用可能な救急車が1台もない場合のフォールバック処理

## 動作フロー

```
初期化時:
1. ID対応表を読み込み (id_mapping_proposal.json)
2. 静的情報を読み込み (amb_place_master.csv)
   └─> 192台分の所属署情報を保持

ディスパッチ時:
1. 192台分のデフォルト状態を作成（全員unavailable）
2. ValidationSimulatorの実際の状態で上書き
3. 状態エンコーディング（StateEncoder）
4. PPOエージェントで行動選択
5. 選択された行動を救急車にマッピング
```

## メリット

### 1. 堅牢性の向上
- ValidationSimulatorから不完全な情報が渡されても動作
- 192台未満の情報しか受け取らない場合でも対応可能

### 2. 学習環境との整合性
- 学習時と同じ192台分の状態表現を維持
- StateEncoderが期待する入力形式を保証

### 3. デバッグの容易化
- 情報の更新状況をログ出力
- 状態辞書のサイズを検証

## 使用方法

既存のコードと互換性があるため、使用方法の変更はありません。

```python
# 既存の使い方と同じ
strategy = PPOStrategy()
strategy.initialize({
    'model_path': 'models/hybrid_ppo_20251023_193126.pth',
    'config_path': 'reinforcement_learning/experiments/config_tokyo23_hybrid.yaml',
    'hybrid_mode': True,
    'severe_conditions': ['重症', '重篤', '死亡'],
    'mild_conditions': ['軽症', '中等症']
})
```

初期化時に以下のメッセージが表示されます：
```
PPO戦略を初期化中（Phase 3修正版）...
  ✓ ID対応表読み込み完了: 192件
  全救急車の静的情報を読み込み中...
  フィルタリング後: 192台の救急車
  ✓ 静的情報読み込み完了: 192台
  ...
```

## 依存関係

- `pandas`: 救急車データの読み込みとフィルタリングに使用
- `h3`: 緯度経度からH3インデックスへの変換に使用
- `id_mapping_proposal.json`: ID対応表（Phase 1で生成）
- `data/tokyo/import/amb_place_master.csv`: 救急車の静的データ

## トラブルシューティング

### 静的情報が読み込まれない
**症状**: "⚠️ 救急署データが見つかりません"

**対処**: 
- `data/tokyo/import/amb_place_master.csv`が存在するか確認
- ファイルパスが正しいか確認

### 192台に満たない
**症状**: "⚠️ 警告: 静的情報が192台に満たない"

**影響**: 
- 自動的にダミーデータで補完されるため、動作に問題なし
- ただし、ID対応表と救急車データの整合性を確認することを推奨

### デバッグ情報
以下のメッセージで状態を確認できます：
- `[INFO] ValidationSimulatorから X/192 台の情報のみ受信`
- `[INFO] X/192 台の救急車情報を更新`

## テスト

修正後の動作確認は以下のコマンドで実行できます：

```bash
python validation_simulation.py --strategy ppo --config config.yaml
```

## 今後の改善案

1. **静的情報のキャッシュ化**: 毎回CSVを読み込むのではなく、一度読み込んだ情報をキャッシュ
2. **動的な救急車台数への対応**: 192台以外の台数にも対応
3. **より詳細なログ**: どの救急車の情報が更新されたかを詳細に記録

## 変更履歴

- **2025-10-24**: Phase 3修正版を実装
  - 全救急車の静的情報保持機能を追加
  - 常に192台分の完全な状態辞書を構築するように修正
  - マスクの妥当性チェックを追加

