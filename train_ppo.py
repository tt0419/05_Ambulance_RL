"""
train_ppo.py
PPO学習の本格実行スクリプト
"""

import argparse
import yaml
import os
import sys
from datetime import datetime
import torch
import numpy as np
from pathlib import Path
import shutil
import json

# プロジェクトパスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reinforcement_learning.environment.ems_environment import EMSEnvironment
from reinforcement_learning.agents.ppo_agent import PPOAgent
from reinforcement_learning.training.trainer import PPOTrainer
from reinforcement_learning.config_utils import load_config_with_inheritance


def setup_directories(experiment_name: str) -> Path:
    """実験用ディレクトリの設定"""
    # 基本ディレクトリ（reinforcement_learning/experiments以下に配置）
    base_dir = Path("reinforcement_learning") / "experiments" / "ppo_training" / experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ
    subdirs = ["checkpoints", "logs", "configs", "visualizations"]
    for subdir in subdirs:
        (base_dir / subdir).mkdir(exist_ok=True)
    
    return base_dir


def save_config(config: dict, output_dir: Path):
    """設定ファイルを保存"""
    config_path = output_dir / "configs" / "config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # JSON形式でも保存（解析しやすい）
    json_path = output_dir / "configs" / "config.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"設定ファイル保存: {config_path}")


def print_training_info(config: dict, experiment_name: str, output_dir: Path):
    """学習情報の表示"""
    print("\n" + "=" * 70)
    print("PPO学習設定")
    print("=" * 70)
    print(f"実験名: {experiment_name}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"デバイス: {config['experiment']['device']}")
    print(f"シード値: {config['experiment']['seed']}")
    
    print("\n【学習設定】")
    print(f"総エピソード数: {config['ppo']['n_episodes']:,}")
    print(f"バッチサイズ: {config['ppo']['batch_size']}")
    print(f"学習率 (Actor): {config['ppo']['learning_rate']['actor']}")
    print(f"学習率 (Critic): {config['ppo']['learning_rate']['critic']}")
    
    print("\n【データ設定】")
    print("学習期間:")
    for period in config['data']['train_periods']:
        print(f"  - {period['start_date']} ～ {period['end_date']}")
    print(f"エピソード長: {config['data']['episode_duration_hours']}時間")
    
    print("\n【傷病度設定】")
    for category, info in config['severity']['categories'].items():
        conditions = ', '.join(info['conditions'])
        print(f"  {category}: {conditions} (重み: {info['reward_weight']})")
    
    print("\n【評価設定】")
    print(f"評価間隔: {config['evaluation']['interval']}エピソードごと")
    print(f"評価エピソード数: {config['evaluation']['n_eval_episodes']}")
    
    print("=" * 70 + "\n")


def set_random_seed(seed: int):
    """乱数シードの設定"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"乱数シード設定: {seed}")


def check_gpu():
    """GPU情報の確認"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU検出: {gpu_name}")
        print(f"GPUメモリ: {gpu_memory:.1f}GB")
        return "cuda"
    else:
        print("GPUが利用できません。CPUで実行します。")
        return "cpu"


def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='PPO学習実行')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='設定ファイルのパス')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='実験名（省略時は自動生成）')
    parser.add_argument('--resume', type=str, default=None,
                       help='学習再開用チェックポイントパス')
    parser.add_argument('--device', type=str, default=None,
                       help='使用デバイス (cpu/cuda)')
    parser.add_argument('--debug', action='store_true',
                       help='デバッグモード')
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    print(f"\n設定ファイル読み込み: {args.config}")
    
    # パスの解決
    config_path = args.config
    if not os.path.exists(config_path):
        # reinforcement_learning/ サブディレクトリを含むパスを試す
        alt_config_path = f"reinforcement_learning/{config_path}"
        if os.path.exists(alt_config_path):
            config_path = alt_config_path
            print(f"設定ファイルパス修正: {config_path}")
        else:
            print(f"❌ 設定ファイルが見つかりません:")
            print(f"   試行1: {args.config}")
            print(f"   試行2: {alt_config_path}")
            print(f"   現在のディレクトリ: {os.getcwd()}")
            print(f"   利用可能な設定ファイル:")
            
            # 利用可能な設定ファイルを探して表示
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.yaml') and 'config' in file:
                        print(f"     {os.path.join(root, file)}")
            
            sys.exit(1)
    
    # 設定ファイルの読み込みと継承処理
    config = load_config_with_inheritance(config_path)
    
    # デバイスの設定
    if args.device:
        config['experiment']['device'] = args.device
    else:
        detected_device = check_gpu()
        if config['experiment']['device'] == 'cuda' and detected_device == 'cpu':
            print("警告: 設定ではCUDAが指定されていますが、GPUが利用できません。")
            config['experiment']['device'] = 'cpu'
    
    # デバッグモードの設定
    if args.debug:
        print("\n【デバッグモード】")
        config['ppo']['n_episodes'] = 10
        config['evaluation']['interval'] = 5
        config['training']['checkpoint_interval'] = 5
        print("エピソード数を10に制限")
    
    # 実験名の設定
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        # 自動生成: ppo_YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"ppo_{timestamp}"
    
    # ディレクトリの準備
    output_dir = setup_directories(experiment_name)
    
    # 設定の保存
    save_config(config, output_dir)
    
    # 学習情報の表示
    print_training_info(config, experiment_name, output_dir)
    
    # ユーザー確認
    if not args.debug:
        response = input("この設定で学習を開始しますか？ (y/n): ")
        if response.lower() != 'y':
            print("学習をキャンセルしました。")
            return
    
    # 乱数シードの設定
    set_random_seed(config['experiment']['seed'])
    
    try:
        # 環境の初期化
        print("\n環境を初期化中...")
        env = EMSEnvironment(config_path, mode="train")
        
        # 状態次元の取得
        initial_state = env.reset()
        state_dim = len(initial_state)
        action_dim = 192  # 救急車台数
        
        print(f"状態空間次元: {state_dim}")
        print(f"行動空間次元: {action_dim}")
        
        # エージェントの初期化
        print("\nPPOエージェントを初期化中...")
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config['ppo'],
            device=config['experiment']['device']
        )
        
        # トレーナーの初期化
        print("\nトレーナーを初期化中...")
        trainer = PPOTrainer(
            agent=agent,
            env=env,
            config=config,
            output_dir=output_dir
        )
        
        # チェックポイントからの再開
        if args.resume:
            print(f"\nチェックポイントから再開: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # 学習開始
        print("\n" + "=" * 70)
        print("学習開始")
        print("=" * 70)
        
        trainer.train()
        
        print("\n" + "=" * 70)
        print("学習完了！")
        print(f"結果保存先: {output_dir}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n学習が中断されました。")
        print(f"途中結果: {output_dir}")
        
    except Exception as e:
        print(f"\n\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n途中結果: {output_dir}")


if __name__ == "__main__":
    main()