"""
simple_diagnosis.py

最小限の診断スクリプト - 既存コード構造に完全適合

使用方法:
    python simple_diagnosis.py
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# プロジェクトパスの設定（train_ppo.pyと同じ）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reinforcement_learning.environment.ems_environment import EMSEnvironment
from reinforcement_learning.agents.ppo_agent import PPOAgent
from reinforcement_learning.config_utils import load_config_with_inheritance


def main():
    print("\n" + "="*70)
    print("PPO学習診断ツール（シンプル版）")
    print("="*70)
    
    # 設定ファイルのパス（実際に使用しているもの）
    config_path = "reinforcement_learning/experiments/config_continuous.yaml"
    
    print(f"\n設定ファイル: {config_path}")
    
    # 設定の読み込み
    config = load_config_with_inheritance(config_path)
    
    # 環境の初期化（train_ppo.pyと同じ方法）
    print("\n環境を初期化中...")
    env = EMSEnvironment(config_path, mode="train")
    
    # 状態・行動次元の取得（train_ppo.pyと同じ方法）
    state_dim = config['data']['area_restriction'].get('state_dim', env.state_dim)
    action_dim = config['data']['area_restriction'].get('action_dim', env.action_dim)
    
    print(f"状態次元: {state_dim}")
    print(f"行動次元: {action_dim}")
    
    # エージェントの初期化（train_ppo.pyと同じ方法）
    print("\nエージェントを初期化中...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config['ppo']
    )
    
    print(f"デバイス: {agent.device}")
    print(f"バッチサイズ: {agent.batch_size}")
    print(f"初期バッファサイズ: {len(agent.buffer)}")
    
    # ====================================================================
    # 診断1: 1エピソード実行してバッファサイズを確認
    # ====================================================================
    print("\n" + "="*70)
    print("【診断1】1エピソード実行してバッファサイズを確認")
    print("="*70)
    
    state = env.reset()
    episode_steps = 0
    
    # 教師確率を1.0に設定（100%教師あり）
    teacher_prob = 1.0
    
    print(f"\nエピソード実行中（teacher_prob={teacher_prob}）...")
    
    while True:
        # 時間を進める（trainer.pyと同じ）
        env.advance_time()
        
        # 行動選択
        action_mask = env.get_action_mask()
        optimal_action = env.get_optimal_action()
        
        # 教師ありで行動選択
        action, log_prob, value = agent.select_action_with_teacher(
            state,
            action_mask,
            optimal_action,
            teacher_prob,
            deterministic=False
        )
        
        # ステップ実行
        step_result = env.step(action)
        
        # バッファに保存
        agent.store_transition(
            state=state,
            action=action,
            reward=step_result.reward,
            next_state=step_result.observation,
            done=step_result.done,
            log_prob=log_prob,
            value=value,
            action_mask=action_mask
        )
        
        state = step_result.observation
        episode_steps += 1
        
        if step_result.done:
            break
    
    final_buffer_size = len(agent.buffer)
    batch_size = agent.batch_size
    
    print(f"\nエピソード完了:")
    print(f"  総ステップ数: {episode_steps}")
    print(f"  最終バッファサイズ: {final_buffer_size}")
    print(f"  必要なバッチサイズ: {batch_size}")
    
    if final_buffer_size < batch_size:
        print(f"\n❌ 【問題発見】バッファサイズ（{final_buffer_size}）< バッチサイズ（{batch_size}）")
        print(f"   → update()は実行されません")
        print(f"   → これが学習が進まない根本原因です")
        print(f"\n【解決策】")
        print(f"   config.yamlのbatch_sizeを {final_buffer_size//2} 程度に変更してください")
    else:
        print(f"\n✅ バッファサイズは十分です")
    
    # ====================================================================
    # 診断2: update()を実行してパラメータが変化するか確認
    # ====================================================================
    print("\n" + "="*70)
    print("【診断2】update()を実行してパラメータ変化を確認")
    print("="*70)
    
    if final_buffer_size < batch_size:
        print("\nバッファサイズ不足のため、update()をスキップします")
        print("（これが実際の学習でも起きている問題です）")
    else:
        # 初期パラメータを保存
        initial_params = [p.data.clone().cpu() for p in agent.actor.parameters()]
        
        print("\nupdate()を実行中...")
        update_stats = agent.update()
        
        if update_stats:
            print(f"  actor_loss: {update_stats.get('actor_loss', 'N/A'):.6f}")
            print(f"  critic_loss: {update_stats.get('critic_loss', 'N/A'):.6f}")
            
            # パラメータの変化を確認
            final_params = [p.data.clone().cpu() for p in agent.actor.parameters()]
            
            total_change = 0.0
            for i, (init_p, final_p) in enumerate(zip(initial_params, final_params)):
                change = torch.abs(final_p - init_p).sum().item()
                total_change += change
                if i == 0:  # 最初の層のみ表示
                    print(f"\n  Actor層0のパラメータ変化量: {change:.8f}")
            
            if total_change > 1e-6:
                print(f"\n✅ パラメータは変化しました（総変化量: {total_change:.8f}）")
                print("   → update()は正常に機能しています")
            else:
                print(f"\n❌ パラメータが変化していません")
                print("   → 学習率や勾配計算に問題がある可能性があります")
        else:
            print("❌ update()が統計を返しませんでした")
    
    # ====================================================================
    # 診断結果サマリー
    # ====================================================================
    print("\n" + "="*70)
    print("診断結果サマリー")
    print("="*70)
    
    print(f"\n1エピソードのステップ数: {episode_steps}")
    print(f"バッファサイズ: {final_buffer_size}/{batch_size}")
    
    if final_buffer_size < batch_size:
        print(f"\n【結論】バッファサイズ不足が原因")
        print(f"  - 1エピソードで{final_buffer_size}ステップ分のデータしか集まらない")
        print(f"  - batch_size={batch_size}に到達しないため、update()が実行されない")
        print(f"  - パラメータが更新されず、モデルは未学習のまま")
        print(f"  - 学習時のログ（7.46分）は教師の行動による環境の結果")
        print(f"  - テスト時に未学習エージェントを使用 → 20分台")
        
        print(f"\n【推奨される対策】")
        print(f"  1. config.yamlの以下を変更:")
        print(f"     batch_size: {final_buffer_size//2}  # 現在: {batch_size}")
        print(f"  2. 再学習を実行:")
        print(f"     python train_ppo.py --config reinforcement_learning/experiments/config_continuous.yaml")
    else:
        print(f"\n【結論】バッファサイズは問題なし")
        print(f"  他の原因を調査する必要があります")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        print("\nエラーの詳細を報告してください")
