# 報酬関数の再設計提案

## 現在の問題

```python
# 現在の設計（バランスが悪い）
time_penalty = -0.1 per minute      # 小さすぎる
coverage_bonus = +10.0              # 大きすぎる
coverage_penalty = -10.0            # 大きすぎる

# 結果：カバレッジの影響が支配的
```

---

## 提案1: シンプルな報酬設計（最速）

### コンセプト
**「応答時間のみを最小化」**

```python
def calculate_reward_simple(response_time_minutes, severity):
    """シンプルな報酬：応答時間のみ"""
    
    # 基本ペナルティ（強化）
    reward = -response_time_minutes  # 1分 = -1ポイント
    
    # ボーナス
    if response_time_minutes <= 6:
        reward += 10.0
    elif response_time_minutes <= 13:
        reward += 5.0
    
    # ペナルティ
    if response_time_minutes > 20:
        reward += -50.0
    
    return reward

# 例:
# 5分: -5 + 10 = +5点   ← 良い
# 10分: -10 + 5 = -5点  ← 普通
# 15分: -15 = -15点     ← 悪い
# 25分: -25 + (-50) = -75点  ← とても悪い
```

### 実装

```yaml
# config_tokyo23_simple.yaml
reward:
  core:
    mode: "simple"
    simple_params:
      time_penalty_per_minute: -1.0    # 強化（-0.1 → -1.0）
      under_6min_bonus: 10.0
      under_13min_bonus: 5.0
      over_20min_penalty: -50.0

hybrid_mode:
  enabled: false  # 学習時は無効
```

---

## 提案2: バランス型報酬（推奨）

### コンセプト
**「応答時間優先、カバレッジは補助」**

```python
def calculate_reward_balanced(response_time_minutes, coverage_after):
    """バランス型：応答時間メイン、カバレッジはわずかに考慮"""
    
    # A. 応答時間報酬（90%）
    time_reward = -response_time_minutes
    if response_time_minutes <= 6:
        time_reward += 15.0
    elif response_time_minutes <= 13:
        time_reward += 8.0
    if response_time_minutes > 20:
        time_reward += -100.0
    
    # B. カバレッジ報酬（10%）- 大幅削減
    coverage_reward = 0.0
    if coverage_after >= 0.8:
        coverage_reward = 2.0    # +10.0 → +2.0（1/5）
    elif coverage_after >= 0.6:
        coverage_reward = 1.0
    else:
        coverage_reward = -1.0   # -10.0 → -1.0（1/10）
    
    # 合計
    total = time_reward * 0.9 + coverage_reward * 0.1
    return total

# 例:
# 8分、カバレッジ悪化:
#   time: -8 + 8 = 0
#   coverage: -1
#   total: 0 * 0.9 + (-1) * 0.1 = -0.1点
#
# 15分、カバレッジ維持:
#   time: -15
#   coverage: +2
#   total: -15 * 0.9 + 2 * 0.1 = -13.5 + 0.2 = -13.3点
#
# → 近い隊（8分）の方が圧倒的に良い！
```

### 実装

```yaml
# config_tokyo23_balanced.yaml
reward:
  core:
    mode: "hybrid"
    hybrid_params:
      time_penalty_per_minute: -1.0        # -0.1 → -1.0
      mild_under_13min_bonus: 8.0          # 5.0 → 8.0
      under_6min_bonus: 15.0               # 新規追加
      over_13min_penalty: -10.0            # -5.0 → -10.0
      over_20min_penalty: -100.0           # -50.0 → -100.0
      
      good_coverage_bonus: 2.0             # 10.0 → 2.0
      coverage_maintenance_bonus: 1.0      # 5.0 → 1.0
      poor_coverage_penalty: -1.0          # -10.0 → -1.0

hybrid_mode:
  reward_weights:
    response_time: 0.9    # 70% → 90%
    coverage: 0.1         # 20% → 10%
```

---

## 提案3: 模倣学習（最も確実）

### コンセプト
**「直近隊を模倣して学習」**

```yaml
teacher:
  enabled: true
  strategy: "closest"
  initial_prob: 0.8      # 初期80%模倣
  final_prob: 0.2        # 最終20%模倣
  decay_episodes: 300
  apply_to: ["軽症", "中等症", "重症", "重篤", "死亡"]

  # 模倣ボーナス
  imitation_bonus: 5.0   # 教師と同じ選択をしたらボーナス
```

---

## 推奨アクション

### 段階1: シンプル報酬でテスト（2時間）
```bash
python train_ppo.py --config reinforcement_learning/experiments/config_tokyo23_simple.yaml --episodes 200
```

**期待結果:** 直近隊の80-90%の性能

### 段階2: 模倣学習を追加（4時間）
```bash
python train_ppo.py --config reinforcement_learning/experiments/config_tokyo23_balanced.yaml --episodes 500
```

**期待結果:** 直近隊の95-100%の性能

### 段階3: ファインチューニング（オプション）
- カバレッジも考慮した最適化
- 複数日での学習

---

## 比較表

| 方式 | 応答時間 | カバレッジ | 学習時間 | 成功率 |
|------|---------|----------|---------|--------|
| 現状 | 14.33分 | ？ | 2-4時間 | ❌ 失敗 |
| シンプル | 8-9分（予測） | 低下 | 2時間 | ⭐⭐⭐ 高 |
| バランス | 7.5-8.5分（予測） | 維持 | 4時間 | ⭐⭐⭐⭐ 高 |
| 模倣学習 | 7.0-7.5分（予測） | 維持 | 4-6時間 | ⭐⭐⭐⭐⭐ 最高 |

---

## 次のステップ

1. **今すぐ**: シンプル報酬で200エピソード学習
2. **結果確認**: 8-9分に改善されているか
3. **段階2へ**: 模倣学習を追加して最適化

**最速で結果を出すなら「シンプル報酬」から始めましょう！**

