# 🚑 救急車復帰バグ修正 - README

## 🐛 問題の症状

```
[trainer.py デバッグ] action_mask全てFalse:
  ステップ: 510, available救急車: 0/192
  現在時刻: 30600.0秒 (8.5時間)
```

**症状**: 8.5時間経過時点で全救急車が出動中、復帰しない

---

## 🔍 根本原因

### 原因1: イベント処理ループのバグ
```python
# ❌ 修正前
if self.pending_call is not None:
    if event_type == AMBULANCE_AVAILABLE:
        process_event()
    else:
        break  # ← ここで即座にループ終了！後ろの復帰イベントが処理されない
```

### 原因2: 復帰時刻の二重計算
```python
# ❌ 修正前
return_time = dispatch_time + completion_time_seconds
# completion_time_secondsは既に絶対時刻なのに、さらにdispatch_timeを足している
```

---

## ✅ 修正内容

### 修正1: 復帰イベントを最優先で処理
```python
# ✅ 修正後
while event_queue and event_queue[0].time <= end_time:
    event = event_queue[0]
    
    # 復帰イベントは常に処理
    if event.event_type == EventType.AMBULANCE_AVAILABLE:
        process_event()
        continue  # ← ループを継続
    
    # NEW_CALLは1件のみ
    if event.event_type == EventType.NEW_CALL:
        if self.pending_call is not None:
            break
        process_event()
```

### 修正2: 復帰時刻の直接使用
```python
# ✅ 修正後
return_time = dispatch_result.get('completion_time_seconds')
# completion_time_secondsを直接使用（二重計算を回避）
```

---

## 📦 納品ファイル

| ファイル | 説明 |
|---------|------|
| **ems_environment.py** | ⭐ 修正済み環境ファイル |
| **BUG_FIX_REPORT.md** | 詳細なバグ修正レポート |
| **CODE_COMPARISON.md** | 修正前後のコード比較 |
| **diagnose_ambulance_return.py** | 診断スクリプト |

---

## 🚀 使用方法

### 1. ファイルを配置
```bash
cp ems_environment.py reinforcement_learning/environment/
```

### 2. 動作確認（オプション）
```bash
python diagnose_ambulance_return.py
```

### 3. 学習を実行
```bash
python train_ppo.py --config config.yaml --debug
```

### 期待される結果
```
ステップ0: 利用可能=192台
ステップ10: 利用可能=150台
ステップ67: 利用可能=151台 ← 1台復帰！✓
ステップ100: 利用可能=130台
...継続的に復帰と配車が繰り返される
```

---

## 📊 修正の効果

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| 復帰イベント処理 | ❌ スキップされる | ✅ 正しく処理 |
| 復帰時刻 | ❌ 二重計算で遅延 | ✅ 正確に計算 |
| 利用可能台数 | ❌ 0台になる | ✅ 適切に維持 |
| 学習の進行 | ❌ 停止する | ✅ 正常に進行 |

---

## 🔬 検証方法

### デバッグ情報の確認
```python
# 各ステップで利用可能台数を確認
available_count = sum(1 for amb in env.ambulance_states.values() 
                     if amb['status'] == 'available')
print(f"ステップ{env.episode_step}: 利用可能={available_count}台")
```

### イベントキューの確認
```python
print(f"イベントキュー数: {len(env.event_queue)}")
for event in env.event_queue[:5]:
    print(f"  時刻={event.time:.1f}秒, タイプ={event.event_type}")
```

---

## 📝 技術詳細

### 修正箇所
1. `step()` メソッド
   - イベント処理ループ（line 878-897）
   - 復帰イベントのスケジュール（line 929-940）

### 変更なしで動作
- `_calculate_ambulance_completion_time()` ✓
- `_dispatch_ambulance()` ✓
- `_handle_ambulance_return_event()` ✓

---

## ⚠️ 重要な注意事項

### 修正の本質
この修正は**時間管理システムの一部**です。以下の変更と**セット**で使用してください：

1. ✅ 固定時間ステップ制（1ステップ=60秒）
2. ✅ イベント駆動型の処理
3. ✅ 復帰イベントの適切なスケジュール

### 既存機能との互換性
- ✅ 既存の`config.yaml`をそのまま使用可能
- ✅ サービス時間生成、病院選択など既存機能は維持
- ✅ ハイブリッドモードも正常に動作

---

## 📚 関連ドキュメント

1. **BUG_FIX_REPORT.md** - 詳細な原因分析と修正内容
2. **CODE_COMPARISON.md** - 修正前後のコード比較
3. **IMPLEMENTATION_GUIDE.md** - 時間管理システム全体の実装ガイド

---

## ✅ チェックリスト

修正の確認:
- [ ] ems_environment.pyを配置
- [ ] 学習を実行
- [ ] 利用可能台数が0にならないことを確認
- [ ] 救急車が適切に復帰することを確認

---

## 💡 まとめ

### Before（修正前）
```
ステップ510: 全救急車が出動中（0/192台）
→ 復帰イベントが処理されない
→ 学習が進まない ❌
```

### After（修正後）
```
ステップ67: 救急車が復帰（151/192台）
→ 復帰イベントが正しく処理される
→ 学習が正常に進行 ✅
```

---

**修正日**: 2025年11月5日  
**対象**: `ems_environment.py`  
**バージョン**: 修正版 v1.1  
**テスト状況**: 診断スクリプトで検証済み ✅
