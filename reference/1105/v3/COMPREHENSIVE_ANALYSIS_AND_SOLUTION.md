# 救急車復帰問題 - 包括的分析と解決策

## 📊 問題の全体像

### 現象
```
ステップ723で全車出動（0/192台）
平均応答時間: 20分以上
6分達成率: 2-3%（目標40%）
教師あり学習が機能していない
```

### 問題の3層構造

#### 第1層: 表層の症状
- ステップ723（約12時間）で全救急車が出動中
- 利用可能台数が徐々に減少（71台→0台）
- 学習が全く進まない

#### 第2層: 直接的な原因
1. **復帰イベント処理の不備** ✅ 修正済み
   - `completion_time`がクリアされない
   - 復帰した救急車が「利用可能」にならない

2. **教師あり学習が機能していない** 🔴 未解決
   - 90%の確率で最近接隊を選ぶはずが機能せず
   - ログ: `教師一致: False`

3. **遠い救急車を選択**
   - 応答時間25分（本来は5-10分程度）
   - PPOがランダムに選択している

#### 第3層: 根本原因
**`get_optimal_action()`が`None`を返している**

```python
# trainer.pyの処理フロー
optimal_action = self.env.get_optimal_action() if teacher_prob > 0 else None
use_teacher = optimal_action is not None and np.random.random() < teacher_prob

# optimal_action = None の場合:
# → use_teacher = False
# → PPOがランダムに行動選択
# → 遠い救急車を選択する可能性
```

---

## 🔍 根本原因の特定

### なぜ`get_optimal_action()`が`None`を返すのか？

**仮説1: H3インデックスの不一致**
```python
# get_optimal_action()内の処理
for amb_id, amb_state in self.ambulance_states.items():
    if amb_state['status'] != 'available':
        continue
    
    travel_time = self._calculate_travel_time(
        amb_state['current_h3'],    # 救急車のH3
        self.pending_call['h3_index']  # 事案のH3
    )
```

```python
# _calculate_travel_time()の実装
from_idx = self.grid_mapping.get(from_h3)  # → Noneの可能性
to_idx = self.grid_mapping.get(to_h3)      # → Noneの可能性

if from_idx is None or to_idx is None:
    return 600.0  # デフォルト10分
```

**問題点**:
- H3インデックスが`grid_mapping`に存在しない
- デフォルト値（600秒）が返される
- しかし、これだけでは例外は発生しない

**仮説2: 全救急車で例外が発生**
```python
try:
    travel_time = self._calculate_travel_time(...)
    if travel_time < min_travel_time:
        min_travel_time = travel_time
        best_action = amb_id
except Exception as e:
    continue  # ← 例外が発生すると、この救急車はスキップ
```

もし**全ての救急車**で例外が発生すると：
- `best_action`が更新されない
- 最終的に`None`が返される

**仮説3: travel_time_matricesのロード失敗**
```python
current_travel_time_matrix = self.travel_time_matrices.get(phase)

if current_travel_time_matrix is None:
    return 600.0  # デフォルト10分

travel_time = current_travel_time_matrix[from_idx, to_idx]  # ← ここで例外？
```

---

## 💊 解決策

### 修正1: `_handle_ambulance_return_event`（既に修正済み）

```python
def _handle_ambulance_return_event(self, event: Event):
    """救急車復帰イベントの処理"""
    amb_id = event.data['ambulance_id']
    station_h3 = event.data['station_h3']
    
    if amb_id in self.ambulance_states:
        self.ambulance_states[amb_id]['status'] = 'available'
        self.ambulance_states[amb_id]['current_h3'] = station_h3
        self.ambulance_states[amb_id]['completion_time'] = 0.0  # ★追加★
```

**効果**: 復帰した救急車が次回配車に使用可能になる

---

### 修正2: `get_optimal_action`のデバッグ強化（新規）

```python
def get_optimal_action(self) -> Optional[int]:
    # ... 既存のコード ...
    
    # ★★★ Noneを返す場合の詳細情報を出力 ★★★
    if best_action is None and available_count > 0:
        print(f"\n[CRITICAL] get_optimal_actionがNoneを返しました")
        print(f"available救急車数: {available_count}台")
        print(f"エラー発生数: {error_count}件")
        print(f"pending_call.h3_index: {self.pending_call.get('h3_index')}")
        
        # H3インデックスの検証
        call_h3 = self.pending_call.get('h3_index')
        in_mapping = call_h3 in self.grid_mapping
        print(f"事案H3がgrid_mappingに存在: {in_mapping}")
        
        # 救急車のH3も確認
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_state['status'] == 'available':
                amb_h3 = amb_state.get('current_h3')
                in_mapping = amb_h3 in self.grid_mapping
                print(f"救急車{amb_id} H3がgrid_mappingに存在: {in_mapping}")
                break
    
    return best_action
```

**効果**: 
- なぜ`None`が返されるのか、詳細な原因を特定
- H3インデックスの不一致を検出
- travel_time計算のエラーを記録

---

### 修正3: データロード時のH3検証（新規）

```python
def _load_episode_calls(self):
    # ... 既存のコード ...
    
    # ★★★ H3インデックスの検証 ★★★
    invalid_h3_count = 0
    for call in episode_calls:
        if call['h3_index'] not in self.grid_mapping:
            invalid_h3_count += 1
            if invalid_h3_count <= 5:
                print(f"[WARN] 事案{call['id']}のH3が不正: {call['h3_index']}")
    
    if invalid_h3_count > 0:
        total = len(episode_calls)
        print(f"[WARN] {invalid_h3_count}/{total}件の事案が不正なH3を持っています")
        print(f"  → これらの事案では教師あり学習が機能しません")
```

**効果**: 
- データロード時に問題を早期発見
- 不正なH3インデックスを持つ事案を特定

---

### 修正4: 救急車初期化時のH3検証（新規）

```python
def _initialize_ambulances_realistic(self):
    # ... 既存のコード ...
    
    # ★★★ H3検証を追加 ★★★
    invalid_amb_h3_count = 0
    for amb_id, amb_state in self.ambulance_states.items():
        station_h3 = amb_state.get('station_h3')
        if station_h3 not in self.grid_mapping:
            invalid_amb_h3_count += 1
            if invalid_amb_h3_count <= 5:
                print(f"[WARN] 救急車{amb_id}のH3が不正: {station_h3}")
    
    if invalid_amb_h3_count > 0:
        total = len(self.ambulance_states)
        print(f"[WARN] {invalid_amb_h3_count}/{total}台のH3が不正です")
```

**効果**: 
- 救急車の配置が正しいか確認
- 不正なH3を持つ救急車を特定

---

## 🎯 期待される結果

### 修正後の動作

1. **デバッグログの出力**
   ```
   [CRITICAL] get_optimal_actionがNoneを返しました
   available救急車数: 71台
   エラー発生数: 71件
   pending_call.h3_index: 89283082c2fffff
   事案H3がgrid_mappingに存在: False
   ```
   → **原因が明確に判明！**

2. **H3検証の結果**
   ```
   H3インデックス検証結果:
     有効な事案: 1200/1526 (78.6%)
     無効な事案: 326/1526 (21.4%)
     → [WARNING] 無効な事案では教師あり学習が機能しません
   ```
   → **問題の範囲を特定！**

3. **教師あり学習の正常化**
   ```
   [配車] Ep1-0: 軽症 → 最寄り救急(実車) 8.5分
   教師一致: True
   ```
   → **期待される動作！**

---

## 🔧 次のステップ

### 優先度1: デバッグ実行
```bash
# 修正版で学習を実行
python train_ppo.py --config config_hybrid_continuous.yaml --debug

# ログを確認
# - [CRITICAL]メッセージが出力されるか？
# - H3検証で何件の事案/救急車が無効か？
```

### 優先度2: データ修正（必要に応じて）

**ケース1: grid_mappingが古い**
```python
# 最新のgrid_mapping_res9.jsonを使用
# または、事案データのH3を再計算
```

**ケース2: 解像度の不一致**
```python
# H3解像度を確認（resolution 9を使用しているか？）
h3_index = h3.latlng_to_cell(lat, lng, 9)  # resolution=9
```

**ケース3: 事案データの座標が不正**
```python
# 座標の範囲チェックを強化
if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
    # 不正な座標をスキップまたは修正
```

### 優先度3: 根本対策

**対策A: フォールバック戦略**
```python
def get_optimal_action(self) -> Optional[int]:
    best_action = None
    # ... 既存のロジック ...
    
    # ★フォールバック: Noneの場合、利用可能な最初の救急車を返す★
    if best_action is None and available_count > 0:
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_state['status'] == 'available':
                best_action = amb_id
                break
        
        if best_action is not None:
            print(f"[FALLBACK] フォールバック: 救急車{best_action}を選択")
    
    return best_action
```

**対策B: H3インデックスの自動修正**
```python
def _normalize_h3_index(self, h3_index: str) -> str:
    """
    H3インデックスを正規化
    - 解像度を統一
    - grid_mappingに存在しない場合は最近接のH3に変換
    """
    # 実装の詳細は省略
```

---

## 📝 まとめ

### 修正の優先順位

| 修正 | 優先度 | 効果 | 状態 |
|------|--------|------|------|
| **completion_timeクリア** | 🔴最高 | 復帰処理 | ✅完了 |
| **get_optimal_actionデバッグ** | 🔴最高 | 原因特定 | 📝要実装 |
| **H3検証** | 🟠高 | 問題範囲特定 | 📝要実装 |
| **フォールバック戦略** | 🟡中 | 安定性向上 | 🔄検討中 |

### 期待される改善

```
修正前:
- 教師一致: False（90%の確率なのに機能せず）
- 応答時間: 25分（遠い救急車を選択）
- ステップ723: 全車出動

修正後:
- 教師一致: True（90%の確率で最近接隊）
- 応答時間: 8-10分（適切な救急車を選択）
- 救急車が継続的に復帰（枯渇しない）
```

---

## ✅ 実装チェックリスト

- [x] 修正1: `_handle_ambulance_return_event`に`completion_time`クリアを追加
- [ ] 修正2: `get_optimal_action`にデバッグログを追加
- [ ] 修正3: `_load_episode_calls`にH3検証を追加
- [ ] 修正4: `_initialize_ambulances_realistic`にH3検証を追加
- [ ] デバッグ実行してログを確認
- [ ] 根本原因を特定
- [ ] データまたはコードを修正
- [ ] 学習が正常に進むことを確認

---

**作成日**: 2025年11月5日  
**対象**: ems_environment_fixed.py  
**問題**: 救急車復帰問題と教師あり学習の不具合  
**解決策**: 4段階の修正とデバッグ強化
