"""
debug_training_process.py

学習プロセスの詳細診断スクリプト

このスクリプトは、PPO学習が実際に機能しているかを診断します。
以下の3つの重要なポイントを検証します：

1. バッファにデータが正しく蓄積されているか
2. PPOの更新が実際に実行されているか
3. モデルのパラメータが学習によって変化しているか

使用方法:
    python debug_training_process.py --config config.yaml --episodes 5
"""

import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
import copy

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from reinforcement_learning.agents.ppo_agent import PPOAgent
from reinforcement_learning.environment.ems_environment import EMSEnvironment
from reinforcement_learning.training.trainer import PPOTrainer


def check_buffer_accumulation(agent, env, config, n_steps=10):
    """
    バッファにデータが正しく蓄積されるかをチェック
    """
    print("\n" + "=" * 70)
    print("【検証1】バッファへのデータ蓄積")
    print("=" * 70)
    
    # バッファをクリア
    agent.buffer.clear()
    print(f"\n初期バッファサイズ: {len(agent.buffer)}")
    
    # 環境をリセット
    state = env.reset()
    
    # 教師確率を1.0に設定（100%教師あり）
    teacher_prob = 1.0
    
    print(f"\n{n_steps}ステップ実行してバッファにデータを蓄積...")
    
    for step in range(n_steps):
        # 時間を進める
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
            next_state=step_result.next_state,
            done=step_result.done,
            log_prob=log_prob,
            value=value,
            action_mask=action_mask
        )
        
        # デバッグ情報
        if step < 3:  # 最初の3ステップのみ詳細表示
            print(f"\nステップ{step}:")
            print(f"  optimal_action: {optimal_action}")
            print(f"  selected_action: {action}")
            print(f"  matched: {action == optimal_action}")
            print(f"  log_prob: {log_prob:.4f}")
            print(f"  value: {value:.4f}")
            print(f"  reward: {step_result.reward:.4f}")
            print(f"  buffer_size: {len(agent.buffer)}")
        
        state = step_result.next_state
        
        if step_result.done:
            print(f"\nエピソード終了（ステップ{step}）")
            break
    
    final_buffer_size = len(agent.buffer)
    print(f"\n最終バッファサイズ: {final_buffer_size}")
    
    if final_buffer_size == 0:
        print("❌ 【問題】バッファにデータが蓄積されていません！")
        return False
    elif final_buffer_size < agent.batch_size:
        print(f"⚠️  【警告】バッファサイズ（{final_buffer_size}）がバッチサイズ（{agent.batch_size}）未満です")
        print("    → update()は実行されません")
        return False
    else:
        print(f"✅ バッファに十分なデータが蓄積されています（{final_buffer_size} >= {agent.batch_size}）")
        return True


def check_ppo_update(agent):
    """
    PPOの更新が実際に実行されるかをチェック
    """
    print("\n" + "=" * 70)
    print("【検証2】PPOの更新実行")
    print("=" * 70)
    
    buffer_size = len(agent.buffer)
    batch_size = agent.batch_size
    
    print(f"\n現在のバッファサイズ: {buffer_size}")
    print(f"必要なバッチサイズ: {batch_size}")
    
    if buffer_size < batch_size:
        print(f"❌ 【問題】バッファサイズが不足しているため、update()は実行されません")
        print(f"    バッファサイズ（{buffer_size}）< バッチサイズ（{batch_size}）")
        return False
    
    print("\nPPO更新を実行中...")
    
    try:
        update_stats = agent.update()
        
        if not update_stats:
            print("❌ 【問題】update()は実行されましたが、統計が返されませんでした")
            return False
        
        print("✅ PPO更新が正常に実行されました")
        print(f"\n更新統計:")
        print(f"  actor_loss: {update_stats.get('actor_loss', 'N/A'):.6f}")
        print(f"  critic_loss: {update_stats.get('critic_loss', 'N/A'):.6f}")
        print(f"  entropy: {update_stats.get('entropy', 'N/A'):.6f}")
        print(f"  kl_divergence: {update_stats.get('kl_divergence', 'N/A'):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 【エラー】update()の実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_parameter_change(agent, env, config, n_episodes=3):
    """
    モデルのパラメータが学習によって変化するかをチェック
    """
    print("\n" + "=" * 70)
    print("【検証3】モデルパラメータの変化")
    print("=" * 70)
    
    # 初期パラメータを保存
    initial_actor_params = copy.deepcopy([p.data.clone() for p in agent.actor.parameters()])
    initial_critic_params = copy.deepcopy([p.data.clone() for p in agent.critic.parameters()])
    
    print(f"\n初期Actorパラメータ（先頭層の先頭5要素）:")
    print(f"  {initial_actor_params[0].flatten()[:5]}")
    
    print(f"\n{n_episodes}エピソード実行して学習...")
    
    teacher_prob = 1.0  # 100%教師あり
    
    for ep in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        step_count = 0
        
        while True:
            # 時間を進める
            env.advance_time()
            
            # 行動選択
            action_mask = env.get_action_mask()
            optimal_action = env.get_optimal_action()
            
            action, log_prob, value = agent.select_action_with_teacher(
                state,
                action_mask,
                optimal_action,
                teacher_prob,
                deterministic=False
            )
            
            # ステップ実行
            step_result = env.step(action)
            episode_reward += step_result.reward
            
            # バッファに保存
            agent.store_transition(
                state=state,
                action=action,
                reward=step_result.reward,
                next_state=step_result.next_state,
                done=step_result.done,
                log_prob=log_prob,
                value=value,
                action_mask=action_mask
            )
            
            state = step_result.next_state
            step_count += 1
            
            if step_result.done:
                break
        
        print(f"\nエピソード{ep+1}: 報酬={episode_reward:.2f}, ステップ数={step_count}")
        
        # バッファが十分なサイズになったら更新
        if len(agent.buffer) >= agent.batch_size:
            print(f"  バッファサイズ: {len(agent.buffer)} → PPO更新を実行")
            update_stats = agent.update()
            print(f"  actor_loss: {update_stats['actor_loss']:.6f}")
        else:
            print(f"  バッファサイズ: {len(agent.buffer)} → 更新なし")
    
    # 最終パラメータを取得
    final_actor_params = [p.data.clone() for p in agent.actor.parameters()]
    final_critic_params = [p.data.clone() for p in agent.critic.parameters()]
    
    print(f"\n最終Actorパラメータ（先頭層の先頭5要素）:")
    print(f"  {final_actor_params[0].flatten()[:5]}")
    
    # パラメータの変化を計算
    print("\nパラメータ変化の検証:")
    
    actor_changed = False
    critic_changed = False
    
    for i, (init_p, final_p) in enumerate(zip(initial_actor_params, final_actor_params)):
        diff = torch.abs(final_p - init_p).sum().item()
        if diff > 1e-6:
            actor_changed = True
            if i == 0:  # 最初の層のみ詳細表示
                print(f"  Actor層{i}: パラメータ変化量 = {diff:.6f}")
    
    for i, (init_p, final_p) in enumerate(zip(initial_critic_params, final_critic_params)):
        diff = torch.abs(final_p - init_p).sum().item()
        if diff > 1e-6:
            critic_changed = True
            if i == 0:
                print(f"  Critic層{i}: パラメータ変化量 = {diff:.6f}")
    
    if actor_changed and critic_changed:
        print("\n✅ ActorとCriticの両方のパラメータが変化しました")
        print("   → 学習が正常に機能しています")
        return True
    elif actor_changed:
        print("\n⚠️  Actorのみが変化しました（Criticは変化なし）")
        return False
    elif critic_changed:
        print("\n⚠️  Criticのみが変化しました（Actorは変化なし）")
        return False
    else:
        print("\n❌ 【問題】ActorもCriticもパラメータが変化していません！")
        print("   → 学習が全く機能していません")
        return False


def check_model_save_load(agent, temp_path="/tmp/test_model.pth"):
    """
    モデルの保存・読み込みが正しく機能するかをチェック
    """
    print("\n" + "=" * 70)
    print("【検証4】モデルの保存・読み込み")
    print("=" * 70)
    
    # 現在のパラメータを保存
    original_params = [p.data.clone() for p in agent.actor.parameters()]
    print(f"\n元のActorパラメータ（先頭5要素）:")
    print(f"  {original_params[0].flatten()[:5]}")
    
    # モデルを保存
    print(f"\nモデルを保存中: {temp_path}")
    agent.save(temp_path)
    
    # パラメータをランダムに変更
    print("\nパラメータをランダムに変更...")
    for p in agent.actor.parameters():
        p.data = torch.randn_like(p.data)
    
    modified_params = [p.data.clone() for p in agent.actor.parameters()]
    print(f"変更後のActorパラメータ（先頭5要素）:")
    print(f"  {modified_params[0].flatten()[:5]}")
    
    # モデルを読み込み
    print(f"\nモデルを読み込み中: {temp_path}")
    agent.load(temp_path)
    
    # 読み込み後のパラメータを確認
    loaded_params = [p.data.clone() for p in agent.actor.parameters()]
    print(f"読み込み後のActorパラメータ（先頭5要素）:")
    print(f"  {loaded_params[0].flatten()[:5]}")
    
    # 元のパラメータと一致するか確認
    match = True
    for orig_p, load_p in zip(original_params, loaded_params):
        if not torch.allclose(orig_p, load_p):
            match = False
            break
    
    if match:
        print("\n✅ モデルの保存・読み込みが正しく機能しています")
        return True
    else:
        print("\n❌ 【問題】読み込まれたパラメータが元のパラメータと一致しません！")
        return False


def main():
    parser = argparse.ArgumentParser(description='PPO学習プロセスの診断')
    parser.add_argument('--config', type=str, required=True,
                       help='設定ファイルのパス')
    parser.add_argument('--episodes', type=int, default=3,
                       help='検証用エピソード数')
    
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "=" * 70)
    print("PPO学習プロセス診断ツール")
    print("=" * 70)
    print(f"\n設定ファイル: {args.config}")
    
    # 環境の初期化
    print("\n環境を初期化中...")
    env = EMSEnvironment(config)
    env.set_mode("train")
    
    # エージェントの初期化
    print("エージェントを初期化中...")
    state_dim = config['state_representation'].get('state_dim', env.state_dim)
    action_dim = config['state_representation'].get('action_dim', env.action_dim)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config['ppo'],
        device=device
    )
    
    print(f"  状態次元: {state_dim}")
    print(f"  行動次元: {action_dim}")
    print(f"  バッチサイズ: {agent.batch_size}")
    print(f"  デバイス: {device}")
    
    # 診断実行
    results = {}
    
    # 検証1: バッファへのデータ蓄積
    results['buffer_accumulation'] = check_buffer_accumulation(agent, env, config, n_steps=50)
    
    # 検証2: PPOの更新実行
    results['ppo_update'] = check_ppo_update(agent)
    
    # 検証3: パラメータの変化
    results['parameter_change'] = check_parameter_change(agent, env, config, n_episodes=args.episodes)
    
    # 検証4: モデルの保存・読み込み
    results['model_save_load'] = check_model_save_load(agent)
    
    # 総合判定
    print("\n" + "=" * 70)
    print("診断結果サマリー")
    print("=" * 70)
    
    print("\n✅ = 正常, ⚠️ = 警告, ❌ = 問題あり\n")
    
    status_icon = lambda x: "✅" if x else "❌"
    
    print(f"{status_icon(results['buffer_accumulation'])} 【検証1】バッファへのデータ蓄積")
    print(f"{status_icon(results['ppo_update'])} 【検証2】PPOの更新実行")
    print(f"{status_icon(results['parameter_change'])} 【検証3】モデルパラメータの変化")
    print(f"{status_icon(results['model_save_load'])} 【検証4】モデルの保存・読み込み")
    
    all_ok = all(results.values())
    
    if all_ok:
        print("\n" + "=" * 70)
        print("✅ すべての検証に合格しました")
        print("学習プロセスは正常に機能しています")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ 一部の検証に失敗しました")
        print("=" * 70)
        
        if not results['buffer_accumulation']:
            print("\n【対策】バッファへのデータ蓄積の問題:")
            print("  - store_transition()が正しく呼ばれているか確認")
            print("  - バッファのサイズ制限を確認")
        
        if not results['ppo_update']:
            print("\n【対策】PPO更新の問題:")
            print("  - batch_sizeの設定を確認（現在: {})".format(agent.batch_size))
            print("  - エピソード長が短すぎないか確認")
        
        if not results['parameter_change']:
            print("\n【対策】パラメータ変化の問題:")
            print("  - 学習率が小さすぎないか確認")
            print("  - 勾配が正しく計算されているか確認")
            print("  - update()が実際に呼ばれているか確認")
        
        if not results['model_save_load']:
            print("\n【対策】モデル保存・読み込みの問題:")
            print("  - ファイルパスが正しいか確認")
            print("  - 書き込み権限があるか確認")


if __name__ == '__main__':
    main()


"""
【実行例】

python debug_training_process.py --config config.yaml --episodes 5

【期待される出力】

正常な場合:
  ✅ 【検証1】バッファへのデータ蓄積
  ✅ 【検証2】PPOの更新実行
  ✅ 【検証3】モデルパラメータの変化
  ✅ 【検証4】モデルの保存・読み込み

問題がある場合:
  ❌ 【検証3】モデルパラメータの変化
  → 学習が全く機能していない
"""
