"""
measure_agent_performance.py

エージェント自身の性能を測定（教師なし）

目的: 学習済みエージェントが実際にどの程度の性能を持っているかを測定
"""

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reinforcement_learning.environment.ems_environment import EMSEnvironment
from reinforcement_learning.agents.ppo_agent import PPOAgent
from reinforcement_learning.config_utils import load_config_with_inheritance


def evaluate_agent_performance(agent, env, n_episodes=5):
    """
    エージェント自身の性能を測定（教師なし）
    """
    print("\n" + "="*70)
    print("エージェント性能測定（教師なし）")
    print("="*70)
    
    episode_stats = []
    
    for ep in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_rts = []
        
        print(f"\nエピソード {ep+1}/{n_episodes} 実行中...")
        
        while True:
            # 時間を進める
            env.advance_time()
            
            # 行動選択（教師なし: teacher_prob=0.0）
            action_mask = env.get_action_mask()
            
            # ★★★ 重要: teacher_prob=0.0（教師を使わない） ★★★
            action, log_prob, value = agent.select_action_with_teacher(
                state,
                action_mask,
                optimal_action=None,  # 教師は使わない
                teacher_prob=0.0,     # 0%教師あり = エージェントのみ
                deterministic=True    # 決定的選択
            )
            
            # ステップ実行
            step_result = env.step(action)
            
            episode_reward += step_result.reward
            state = step_result.observation
            episode_steps += 1
            
            if step_result.done:
                break
        
        # エピソード統計を取得
        stats = env.get_episode_statistics()
        summary = stats.get('summary', {})
        
        mean_rt = summary.get('mean_response_time', 0)
        episode_rts.append(mean_rt)
        
        print(f"  報酬: {episode_reward:.2f}")
        print(f"  平均応答時間: {mean_rt:.2f}分")
        print(f"  ステップ数: {episode_steps}")
        
        episode_stats.append({
            'reward': episode_reward,
            'mean_rt': mean_rt,
            'steps': episode_steps,
            'summary': summary
        })
    
    # 全体統計
    avg_reward = np.mean([s['reward'] for s in episode_stats])
    avg_rt = np.mean([s['mean_rt'] for s in episode_stats])
    
    print("\n" + "="*70)
    print("測定結果サマリー")
    print("="*70)
    print(f"\n平均報酬: {avg_reward:.2f}")
    print(f"平均応答時間: {avg_rt:.2f}分")
    
    # 6分達成率など
    if episode_stats[0]['summary']:
        total_6min = sum([s['summary'].get('achieved_6min', 0) for s in episode_stats])
        total_calls = sum([s['summary'].get('total_calls', 0) for s in episode_stats])
        if total_calls > 0:
            rate_6min = total_6min / total_calls * 100
            print(f"6分達成率: {rate_6min:.1f}%")
    
    return avg_reward, avg_rt


def main():
    print("\n" + "="*70)
    print("エージェント性能測定ツール")
    print("="*70)
    
    # 設定ファイル
    config_path = "reinforcement_learning/experiments/config_continuous.yaml"
    print(f"\n設定ファイル: {config_path}")
    
    # 設定の読み込み
    config = load_config_with_inheritance(config_path)
    
    # 環境の初期化
    print("\n環境を初期化中...")
    env = EMSEnvironment(config_path, mode="eval")
    
    # 状態・行動次元
    state_dim = config['data']['area_restriction'].get('state_dim', env.state_dim)
    action_dim = config['data']['area_restriction'].get('action_dim', env.action_dim)
    
    # エージェントの初期化
    print("\nエージェントを初期化中...")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config['ppo']
    )
    
    # モデルの読み込み（オプション）
    model_path = input("\nモデルファイルのパスを入力（空でランダム初期化）: ").strip()
    
    if model_path and os.path.exists(model_path):
        print(f"モデルを読み込み中: {model_path}")
        agent.load(model_path)
        print("✓ モデル読み込み完了")
    else:
        print("ランダム初期化のエージェントで測定します")
    
    # 性能測定
    n_episodes = int(input("\n測定エピソード数（推奨: 5）: ") or "5")
    
    avg_reward, avg_rt = evaluate_agent_performance(agent, env, n_episodes)
    
    # 結果の解釈
    print("\n" + "="*70)
    print("結果の解釈")
    print("="*70)
    
    if avg_rt > 18:
        print(f"\n❌ 平均応答時間 {avg_rt:.2f}分 は悪い結果です")
        print("   → エージェントは十分に学習できていません")
        print("   → より多くのエピソードで学習が必要です")
    elif avg_rt > 12:
        print(f"\n⚠️  平均応答時間 {avg_rt:.2f}分 は改善の余地があります")
        print("   → 学習は進んでいますが、まだ不十分です")
    else:
        print(f"\n✅ 平均応答時間 {avg_rt:.2f}分 は良い結果です")
        print("   → エージェントは学習できています")
    
    print("\n【参考】")
    print("  - 教師（直近隊運用）: 約7-8分")
    print("  - ランダム: 約20分")
    print("  - 目標: 10分以下")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
