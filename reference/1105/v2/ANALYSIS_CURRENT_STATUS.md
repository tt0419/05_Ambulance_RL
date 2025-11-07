# 救急車復帰問題の現状分析レポート

## 📊 現状

### 改善された点 ✅
1. **復帰イベントは正常に機能している**
   - ステップ150: 118台 → 122台（4台復帰）
   - ステップ180: 122台 → 129台（7台復帰）
   - 復帰イベントの処理ロジックは正しい

2. **全車出動の発生が遅延**
   - 修正前: ステップ510で全車出動
   - 修正後: ステップ722で全車出動
   - 約200ステップ（3.5時間）改善

### 残存する問題 ⚠️
1. **依然として全車出動が発生**
   - Episode 1: ステップ722（12時間時点）
   - Episode 2以降: 最低3台程度は維持されるが、非常に少ない

2. **ピーク時間帯の救急車不足**
   - ステップ500以降、急激に減少
   - 81台 → 73台 → ... → 0台

## 🔍 根本原因の特定

### 原因1: 事案の時間分布が不均一

```
実際のデータ（2023年6月15日）:
- 24時間で1526件
- 平均: 1.06件/分 = 63.6件/時間

しかし、これは「平均」であり、実際は:
- 昼間（8-20時）: 80-100件/時間（ピーク）
- 夜間（0-8時）:  30-40件/時間（閑散）
```

### 原因2: 1ステップ=1事案ではなく、1ステップ=1分

**重要な発見**:
```python
# 現在の実装
self.max_steps_per_episode = 1526  # 事案数
self.episode_step = 0-1440  # 実際のログを見ると1440

# しかし、時間管理は:
self.current_time_seconds = 0-86400  # 24時間 = 86400秒
self.time_per_step = 60  # 1ステップ = 60秒
```

**問題**: `max_steps_per_episode`が事案数（1526）に設定されているが、実際のステップは時間（1440分）で進んでいる！

## 🐛 発見したバグ

### バグ: `max_steps_per_episode`の設定が間違っている

```python
# ems_environment.py line 622-630
config_max_steps = self.config.get('data', {}).get('max_steps_per_episode') or \
                  self.config.get('max_steps_per_episode')

if config_max_steps:
    # configで指定されている場合、事案数との小さい方を使用
    self.max_steps_per_episode = min(config_max_steps, len(self.current_episode_calls))
else:
    # configで指定されていない場合、事案数を使用
    self.max_steps_per_episode = len(self.current_episode_calls)  # ← これが問題！
```

**問題点**:
- `max_steps_per_episode = 1526`（事案数）
- しかし、実際には1440ステップ（24時間）しか進まない
- つまり、86件の事案が未処理のまま終了する

### 実際のログでの確認

```
Episode 1/5000
  長さ: 1440  ← 1440ステップ（24時間）で終了
  平均応答時間: 20.44分
```

**1440ステップ = 1440分 = 24時間** で終了している。

しかし、1526件の事案を処理するには**1526ステップ必要**（1ステップで1事案を処理）。

## 📈 理論値との比較

### Little's Law による必要救急車数

```
L = λ × W
L = 必要救急車数
λ = 1.06 件/分（配車レート）
W = 79分（平均活動時間）

L = 1.06 × 79 = 83.7台

実際の救急車数: 192台
稼働率: 43.6%（理論値）
```

**結論**: 理論的には192台で十分足りる。

### しかし実際は？

```
定常状態の利用可能台数: 約66台（シミュレーション）
ピーク時の利用可能台数: 0台（実測）
```

**なぜ？**
→ **事案がピーク時間帯に集中しているため**

## 🎯 解決策

### 解決策1: `max_steps_per_episode`を修正 ✅

```python
# 修正前
self.max_steps_per_episode = len(self.current_episode_calls)  # 1526

# 修正後
episode_duration_hours = self.config['data']['episode_duration_hours']  # 24
self.max_steps_per_episode = episode_duration_hours * 60  # 1440分
```

**効果**: エピソードが正しい時間（24時間=1440分）で終了する。

### 解決策2: `_is_episode_done`を修正 ✅

```python
def _is_episode_done(self) -> bool:
    # イベントキューが空で、pending_callもない場合は終了
    if not self.event_queue and self.pending_call is None:
        return True
    
    # 時間制限（最優先）
    episode_hours = self.config['data'].get('episode_duration_hours', 24)
    max_time_seconds = episode_hours * 3600.0
    if self.current_time_seconds >= max_time_seconds:
        return True
    
    return False
```

### 解決策3: リアルタイムの事案生成（今後の課題）

現在の実装では、全事案を最初にロードしているため：
- メモリ効率が悪い
- 事案の時刻をステップに変換する必要がある

**改善案**: 
```python
# resetで全事案をイベントキューに追加（現在の実装）
# ↓
# リアルタイムで事案を生成（将来の改善）
```

## 📝 まとめ

### 現状の問題

1. ✅ **復帰イベントは正常に機能** - 修正完了
2. ⚠️ **`max_steps_per_episode`の設定ミス** - 要修正
3. ⚠️ **ピーク時間帯の救急車不足** - 根本的な問題（データ起因）

### 優先度

| 問題 | 優先度 | 対応 |
|------|--------|------|
| 復帰イベントの処理 | 高 | ✅ 完了 |
| max_steps設定 | 高 | ⚠️ 要修正 |
| ピーク時の不足 | 中 | 📊 データ起因 |

### 次のステップ

1. **即座に修正**: `max_steps_per_episode`の計算
2. **検証**: 修正後に全車出動が発生しないか確認
3. **分析**: ピーク時間帯の事案分布を詳細に調査

---

**作成日**: 2025年11月5日  
**ステータス**: 部分的に改善、追加修正が必要
