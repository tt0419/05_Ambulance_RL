# PPOディスパッチ戦略 実行ガイド

## 🚀 実行方法

### 比較実験の実行

```bash
python baseline_comparison.py
```

📈 直近隊運用 vs PPO運用の比較実験  
📁 結果は `data/tokyo/experiments/` に保存  
🖼️ グラフは `strategy_comparison.png`  
📊 wandbに自動アップロード

---

## 📝 設定の確認

### baseline_comparison.py

現在の設定を確認：

```python
# 行54-60
EXPERIMENT_CONFIG = {
    'strategies': ['closest', 'ppo_agent'],  # ← これを確認
    ...
}

# 行103-108
'ppo_agent': {
    'model_path': 'models/normal_ppo_20250926_010459.pth',  # ← モデルパス確認
    'hybrid_mode': False  # ← モード確認
}
```

---

## ⚙️ モード設定

### 🟢 通常モード（PPOのみ）

```python
'ppo_agent': {
    'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/final_model.pth',
    'config_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/configs/config.yaml',
    'hybrid_mode': False
}
```

**重要**: `model_path` と `config_path` の両方を指定してください。  
モデルファイルだけでは学習時の設定が不足している場合があります。

### 🔵 ハイブリッドモード（重症系=直近隊、軽症系=PPO）

```python
'ppo_agent': {
    'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/final_model.pth',
    'config_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/configs/config.yaml',
    'hybrid_mode': True,
    'severe_conditions': ['重症', '重篤', '死亡'],
    'mild_conditions': ['軽症', '中等症']
}
```

---

## 🔧 トラブルシューティング

### ❌ モデルファイルが見つからない

```bash
# 利用可能なモデルを確認
ls reinforcement_learning/experiments/ppo_training/

# 最新のモデルディレクトリを確認
ls reinforcement_learning/experiments/ppo_training/ppo_20250925_134035/

# baseline_comparison.py の model_path と config_path を変更
'model_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/final_model.pth',
'config_path': 'reinforcement_learning/experiments/ppo_training/ppo_XXXXXXXX_XXXXXX/configs/config.yaml'
```

### ❌ メモリエラー

実験期間を短縮：

```python
EXPERIMENT_PARAMS = {
    'duration_hours': 24,  # 720 → 24に短縮
    'num_runs': 1,  # 5 → 1に短縮
}
```

---

## 📊 結果の見方

### グラフ（strategy_comparison.png）

1. **全体平均応答時間** → 低いほど良い
2. **重症系平均応答時間** → 低いほど良い（最重要）
3. **軽症系平均応答時間** → 低いほど良い
4. **6分以内達成率** → 高いほど良い
5. **13分以内達成率** → 高いほど良い
6. **重症系6分以内達成率** → 高いほど良い（最重要）

### レポート（comparison_summary.txt）

```
【直近隊運用】
1. 平均応答時間
   全体: X.XX ± Y.YY 分
   重症系: X.XX ± Y.YY 分

【PPOエージェント運用】
1. 平均応答時間
   全体: X.XX ± Y.YY 分
   重症系: X.XX ± Y.YY 分

統計的比較結果:
  直近隊運用 vs PPOエージェント運用: t=X.XXX, p=0.XXXX
```

---

## 🎯 期待される結果

### 通常モードPPO

- **全体応答時間**: 直近隊とほぼ同等〜やや改善
- **重症系応答時間**: 直近隊とほぼ同等
- **軽症系応答時間**: 改善の可能性
- **カバレッジ**: 改善の可能性

### ハイブリッドモードPPO

- **重症系応答時間**: 直近隊と同等（直近隊ロジック使用）
- **軽症系応答時間**: 改善（PPO最適化）
- **全体バランス**: 最適

---

## 📚 詳細ドキュメント

より詳しい情報は `PPO_DISPATCH_GUIDE.md` を参照

---

## ✨ 実験結果の例

実験が成功すると、以下のような結果が得られます：

### wandb出力例
```
charts/response_time_severe_mean: 12.66分
charts/response_time_mild_mean: 20.66分
charts/response_time_severe_under_6min_rate: 18.9%
```

### 保存されるファイル
- `data/tokyo/experiments/strategy_comparison.png` - 比較グラフ
- `data/tokyo/experiments/comparison_summary.txt` - 統計レポート
- 各実行のシミュレーション詳細レポート
