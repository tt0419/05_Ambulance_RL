# 報酬が0になる問題の修正

## 🔥 問題の症状

**全てのエピソードで報酬が0.0**
- 学習が全く進まない
- モデルが改善しない

---

## 🔍 考えられる原因

### 原因1: hybrid_modeが誤って有効になっている

```python
# reinforcement_learning/environment/ems_environment.py: 773-787行
if self.hybrid_mode and current_incident:
    severity = current_incident.get('severity', '')
    if severity in self.severe_conditions:
        # 報酬は0（学習対象外）
        reward = 0.0
```

**症状:**
- 全事案が重症系として処理される
- または、hybrid_modeが true になっている

---

## 🚀 修正方法（3つの選択肢）

### **修正1: EMSEnvironmentの初期化を強制的に修正** ⭐⭐⭐

**ファイル:** `reinforcement_learning/environment/ems_environment.py`

```python
# 239行目を修正
# 修正前:
self.hybrid_mode = self.config.get('hybrid_mode', {}).get('enabled', False)

# 修正後:
hybrid_config = self.config.get('hybrid_mode', {})
self.hybrid_mode = hybrid_config.get('enabled', False)

# デバッグ出力を追加
print(f"[EMS環境] hybrid_mode: {self.hybrid_mode}")
if self.hybrid_mode:
    print(f"  severe_conditions: {self.severe_conditions}")
```

---

### **修正2: 設定ファイルを明示的に確認**

`config_tokyo23_simple.yaml` の hybrid_mode セクションを確認：

```yaml
hybrid_mode:
  enabled: false  # ← falseになっているか確認
```

もし `enabled: true` になっていたら、`false` に変更。

---

### **修正3: 強制的に通常モードで実行（最も確実）**

**ファイル:** `reinforcement_learning/environment/ems_environment.py`

```python
# 773行目のif文を無効化
# 修正前:
if self.hybrid_mode and current_incident:
    # ...

# 修正後:
if False and self.hybrid_mode and current_incident:  # ← Falseを追加
    # ...
```

これで、hybrid_modeの設定に関わらず、常に通常モードで実行されます。

---

## 🛠️ 推奨修正手順

### Step 1: デバッグ実行

```bash
python debug_reward_issue.py
```

**出力を確認:**
- `hybrid_mode: True` → 問題あり
- `hybrid_mode: False` だが報酬が0 → 別の問題

### Step 2: 修正1を適用

```bash
# ems_environment.pyを編集
# 239行目にデバッグ出力を追加
```

### Step 3: 再学習

```bash
python train_ppo.py --config reinforcement_learning/experiments/config_tokyo23_simple.yaml
```

---

## 📊 確認方法

### 学習開始時の出力を確認

```
[EMS環境] hybrid_mode: False  ← これが表示されるべき
[Simple報酬] 傷病度: 軽症, 時間: 8.5分, 報酬: -2.50  ← 報酬が0以外
```

### training_stats.json を確認

```json
{
  "episode_rewards": [
    -15.3,  ← 0以外の値
    -12.8,
    -18.5,
    ...
  ]
}
```

---

## 🎯 修正後の期待結果

### 学習曲線

```
エピソード10:  平均報酬: -150, 平均応答時間: 15分
エピソード50:  平均報酬: -80,  平均応答時間: 12分
エピソード100: 平均報酬: -50,  平均応答時間: 10分
エピソード200: 平均報酬: -30,  平均応答時間: 8-9分
```

---

## 🔧 緊急回避策

**もし修正してもダメな場合:**

```python
# reinforcement_learning/environment/ems_environment.py
# 239行目を以下に完全に置き換え

self.hybrid_mode = False  # 強制的にFalse
print("[EMS環境] hybrid_mode を強制的に無効化しました")
```

これで100%通常モードで実行されます。

---

## 📝 次のステップ

1. **今すぐ**: `debug_reward_issue.py` を実行
2. **問題特定**: 出力を確認
3. **修正適用**: 修正1または修正3を適用
4. **再学習**: 200エピソード実行
5. **確認**: 報酬が0以外になっていることを確認

**実行してください！**

