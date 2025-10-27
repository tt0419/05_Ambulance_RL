# 次のステップ：PPO改善の実行計画

## 📊 **現状の問題まとめ**

### 発見された3つの根本原因

1. **報酬関数の構造的欠陥**
   ```
   time_penalty_per_minute: -0.1  ← 小さすぎる
   coverage_bonus: +10.0          ← 大きすぎる
   
   結果：10分遅れても、カバレッジ維持で報酬が良くなる
   ```

2. **学習時のハイブリッドモード有効**
   ```
   hybrid_mode: enabled: true
   
   結果：重症系は直近隊、軽症系のみPPO → データの偏り
   ```

3. **模倣学習が無効**
   ```
   teacher: enabled: false
   
   結果：直近隊の良い行動を学習できていない
   ```

---

## 🚀 **推奨アクション（優先順位順）**

### ✅ **Option 1: シンプル報酬 + 模倣学習（最速・最確実）** ⭐⭐⭐⭐⭐

**実行時間:** 1-2時間  
**成功確率:** 90%以上  
**期待される改善:** 14.33分 → 8-9分

#### 実行コマンド

```bash
python train_ppo.py --config reinforcement_learning/experiments/config_tokyo23_simple.yaml
```

#### 主な変更点

- 応答時間ペナルティ: -0.1 → **-1.0**（10倍）
- カバレッジ報酬: 無視（ボーナス0）
- 模倣学習: 90% → 30%に減衰
- ハイブリッドモード: 学習時は無効

#### 期待される結果

```
エピソード50: 平均12分
エピソード100: 平均10分
エピソード150: 平均8.5分
エピソード200: 平均8-9分（目標達成）
```

---

### Option 2: バランス型報酬（より慎重）

**実行時間:** 2-4時間  
**成功確率:** 80%  
**期待される改善:** 14.33分 → 7.5-8.5分

```bash
python train_ppo.py --config reinforcement_learning/experiments/config_tokyo23_hybrid_fixed_v2.yaml
```

---

### Option 3: RewardDesigner の実装修正（根本的）

**実行時間:** 1日  
**成功確率:** 100%（確実だが時間がかかる）

`reinforcement_learning/environment/reward_designer.py` を修正：

```python
def _calculate_hybrid_reward(self, ...):
    # 修正1: time_penalty を強化
    time_reward = params['time_penalty_per_minute'] * response_time_minutes
    # -0.1 * 10分 = -1.0 → -1.0 * 10分 = -10.0
    
    # 修正2: カバレッジ報酬を削減
    if coverage_after >= 0.8:
        coverage_reward = 2.0  # 10.0 → 2.0
```

---

## 📋 **実行チェックリスト**

### Phase 1: 準備（5分）

- [ ] 設定ファイルの確認：`config_tokyo23_simple.yaml`
- [ ] GPUが利用可能か確認：`nvidia-smi`
- [ ] ディスク容量の確認（5GB以上）

### Phase 2: 学習実行（1-2時間）

```bash
# バックグラウンド実行
nohup python train_ppo.py --config reinforcement_learning/experiments/config_tokyo23_simple.yaml > training_simple.log 2>&1 &

# 進捗確認（別ターミナル）
tail -f training_simple.log | grep "Episode"
```

### Phase 3: 結果確認（10分）

```bash
# 新しいモデルで検証
python baseline_comparison.py \
  --model reinforcement_learning/experiments/ppo_training/ppo_YYYYMMDD_HHMMSS/final_model.pth \
  --config reinforcement_learning/experiments/ppo_training/ppo_YYYYMMDD_HHMMSS/configs/config.json
```

### Phase 4: 評価（5分）

#### 成功基準

| メトリック | 現状 | 目標 | 判定 |
|----------|------|------|------|
| 軽症平均RT | 14.33分 | 9分以下 | ✅ / ❌ |
| 中等症平均RT | 14.81分 | 9分以下 | ✅ / ❌ |
| 全体平均RT | - | 8分以下 | ✅ / ❌ |

#### 目標未達の場合

- **8-10分:** 学習を延長（500エピソードまで）
- **10-12分:** バランス型報酬に切り替え
- **12分以上:** RewardDesigner の実装修正が必要

---

## 🎯 **今すぐ実行**

```bash
# 推奨：Option 1（シンプル報酬）
python train_ppo.py --config reinforcement_learning/experiments/config_tokyo23_simple.yaml
```

**学習中にすること:**
1. ログを監視（`tail -f training_simple.log`）
2. WandBでメトリクスを確認
3. 50エピソードごとに応答時間の改善を確認

**2時間後:**
- 新しいモデルで検証実験
- 結果を報告
- 必要に応じて次の手を打つ

---

## 💡 **トラブルシューティング**

### 問題1: 学習が進まない（応答時間が改善しない）

**対処法:**
- 模倣確率を上げる（0.9 → 0.95）
- 学習率を下げる（0.0003 → 0.0001）
- エピソード数を増やす（200 → 500）

### 問題2: 過学習（検証データで性能が悪い）

**対処法:**
- Early Stopping を有効活用
- ドロップアウトを追加
- 学習データを増やす（1週間 → 2週間）

### 問題3: それでも改善しない

**対処法:**
- RewardDesigner の実装を直接修正
- または、直近隊戦略を使用

---

**まずは Option 1 を実行してください！成功確率が最も高いです。** 💪

