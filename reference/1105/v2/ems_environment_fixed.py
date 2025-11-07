"""
ems_environment.py
救急隊ディスパッチのための強化学習環境
"""

import numpy as np
import torch
import yaml
import json
import h3
import pandas as pd
import pickle
import random
import heapq  # イベントキュー管理に使用
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from data_cache import get_emergency_data_cache
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 統一された傷病度定数をインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from constants import SEVERITY_GROUPS, is_severe_condition

from validation_simulation import (
    ValidationSimulator,
    EventType,
    AmbulanceStatus,
    EmergencyCall,
    Event,
    ServiceTimeGenerator
)

# 設定ユーティリティのインポート
try:
    from ..config_utils import load_config_with_inheritance
except ImportError:
    # スタンドアロン実行時のフォールバック
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_utils import load_config_with_inheritance

@dataclass
class StepResult:
    """ステップ実行結果"""
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]

class EMSEnvironment:
    """
    PPO学習用のEMS環境
    OpenAI Gym形式のインターフェースを提供
    """
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "train"):
        """
        Args:
            config_path: 設定ファイルのパス
            mode: "train" or "eval"
        """
        # 設定読み込み（継承機能付き）
        self.config = load_config_with_inheritance(config_path)
        
        self.mode = mode
        self.current_period_idx = 0
        
        # ログ制御フラグ
        self._first_period_logged = False
        self._episode_count = 0
        
        print("=" * 60)
        print(f"EMS環境初期化 (モード: {mode})")
        print(f"設定ファイル: {config_path}")
        print("=" * 60)
        
        # データキャッシュの初期化
        print("データキャッシュを初期化中...")
        self.data_cache = get_emergency_data_cache()
        
        # 初回データ読み込み（起動時に一度だけ）
        print("初期データ読み込み中...")
        self.data_cache.load_data()
        print("データキャッシュ準備完了")
        

        
        # 傷病度設定の初期化
        self._setup_severity_mapping()
        
        # データパスの設定
        self.base_dir = Path("data/tokyo")
        self._load_base_data()
        
        # 移動時間行列の読み込み（ValidationSimulatorと同じ方法）
        self.travel_time_matrices = {}
        self.travel_distance_matrices = {}
        
        calibration_dir = self.base_dir / "calibration2"
        travel_time_stats_path = calibration_dir / 'travel_time_statistics_all_phases.json'
        
        if travel_time_stats_path.exists():
            with open(travel_time_stats_path, 'r', encoding='utf-8') as f:
                phase_stats_data = json.load(f)
            
            # ValidationSimulatorと同じロジックで行列を読み込み
            for phase in ['response', 'transport', 'return']:
                matrix_filename = None
                
                if phase in phase_stats_data and 'calibrated' in phase_stats_data[phase]:
                    model_type = phase_stats_data[phase]['calibrated'].get('model_type')
                    
                    if model_type == "uncalibrated":
                        matrix_filename = f"uncalibrated_travel_time_{phase}.npy"
                    elif model_type in ['linear', 'log']:
                        matrix_filename = f"{model_type}_calibrated_{phase}.npy"
                    
                    if matrix_filename:
                        matrix_path = calibration_dir / matrix_filename
                        if matrix_path.exists():
                            self.travel_time_matrices[phase] = np.load(matrix_path)
                            print(f"  移動時間行列読み込み: {phase} ({model_type})")
        
        # 距離行列も同様に読み込み
        distance_matrix_path = self.base_dir / "processed/travel_distance_matrix_res9.npy"
        if distance_matrix_path.exists():
            travel_distance_matrix = np.load(distance_matrix_path)
            # ValidationSimulatorと同じ形式に変換
            self.travel_distance_matrices = {
                'dispatch_to_scene': travel_distance_matrix,
                'scene_to_hospital': travel_distance_matrix,
                'hospital_to_station': travel_distance_matrix
            }
        
        # シミュレータの初期化は reset() で行う
        self.simulator = None
        self.current_episode_calls = []
        self.pending_call = None
        self.episode_step = 0
        self.max_steps_per_episode = None
        
        # ★★★【時間管理の追加】★★★
        # 固定時間ステップ制: 1ステップ = 1分 = 60秒
        self.time_per_step = 60.0  # 秒
        self.current_time_seconds = 0.0  # エピソード開始からの経過秒数
        self.event_queue = []  # イベント優先度付きキュー（heapq使用）
        # ★★★【追加ここまで】★★★
        
        # デバッグ用のverbose_logging属性を初期化
        self.verbose_logging = False
        
        # 教師一致情報の初期化
        self.current_matched_teacher = False
        
        # 状態・行動空間の次元
        self.action_dim = len(self.ambulance_data)  # 実際の救急車数
        
        # ★★★【修正提案】★★★
        # StateEncoderの初期化をここで行い、インスタンスをクラス変数として保持する
        response_matrix = self.travel_time_matrices.get('response', None)
        if response_matrix is None:
            print("警告: responseフェーズの移動時間行列が見つかりません。")

        # StateEncoderを初期化して、self.state_encoderとして保持
        from .state_encoder import StateEncoder
        self.state_encoder = StateEncoder(
            config=self.config,
            max_ambulances=self.action_dim,
            travel_time_matrix=response_matrix,
            grid_mapping=self.grid_mapping
        )
        
        # StateEncoderインスタンスから状態次元を取得する
        self.state_dim = self.state_encoder.state_dim
        # ★★★【修正ここまで】★★★
        
        print(f"状態空間次元: {self.state_dim}")
        print(f"行動空間次元: {self.action_dim}")
        
        # 統計情報の初期化
        self.statistics = {}  # 病院選択統計用（_init_hospital_selectionで使用）
        self.episode_stats = self._init_episode_stats()
        
        # RewardDesignerを一度だけ初期化
        from .reward_designer import RewardDesigner
        self.reward_designer = RewardDesigner(self.config)
        
        # DispatchLoggerの初期化
        from .dispatch_logger import DispatchLogger
        self.dispatch_logger = DispatchLogger(enabled=True)
        
        # ServiceTimeGeneratorの初期化
        self._init_service_time_generator()
        
        # 病院選択の初期化（validation_simulation互換）
        self._init_hospital_selection()
        
        # ハイブリッドモード設定
        self.hybrid_mode = self.config.get('hybrid_mode', {}).get('enabled', False)
        if self.hybrid_mode:
            self.severe_conditions = ['重症', '重篤', '死亡']
            self.mild_conditions = ['軽症', '中等症']
            self.direct_dispatch_count = 0  # 直近隊運用の回数
            self.ppo_dispatch_count = 0     # PPO運用の回数
            print("ハイブリッドモード有効: 重症系は直近隊、軽症系はPPO学習")
        
    def _row_is_virtual(self, row: pd.Series) -> bool:
        """DataFrame行から仮想フラグを安全に判定（NaNはFalse）。"""
        try:
            value = row.get('is_virtual', False)
        except Exception:
            return False
        # NaN対策
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        # 真偽/文字列の両対応
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ['true', '1', 'yes', 'y']
        if isinstance(value, (int, float)):
            return int(value) == 1
        return False

    def _setup_severity_mapping(self):
        """傷病度マッピングの設定"""
        self.severity_to_category = {}
        self.severity_weights = {}
        
        for category, info in self.config['severity']['categories'].items():
            for condition in info['conditions']:
                self.severity_to_category[condition] = category
                self.severity_weights[condition] = info['reward_weight']
        
        print("傷病度設定:")
        for category, info in self.config['severity']['categories'].items():
            conditions = ', '.join(info['conditions'])
            weight = info['reward_weight']
            print(f"  {category}: {conditions} (重み: {weight})")
    
    def _init_service_time_generator(self):
        """ServiceTimeGeneratorの初期化（validation_simulation互換）"""
        # v1の単純版パラメータファイルを使用（validation_simulationと同一）
        possible_params_paths = [
            self.base_dir / "service_time_analysis/v1/lognormal_parameters.json",
            "data/tokyo/service_time_analysis/v1/lognormal_parameters.json"
        ]
        
        params_file = None
        for path in possible_params_paths:
            if Path(path).exists():
                params_file = str(path)
                print(f"  サービス時間パラメータ読み込み: {params_file}")
                break
        
        if params_file:
            try:
                # validation_simulationと同じServiceTimeGeneratorを使用
                self.service_time_generator = ServiceTimeGenerator(params_file)
                print("  ✓ ServiceTimeGenerator初期化成功（validation_simulation互換）")
            except Exception as e:
                print(f"  ❌ ServiceTimeGenerator初期化失敗: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"ServiceTimeGeneratorの初期化に失敗しました: {e}")
        else:
            raise FileNotFoundError(
                "サービス時間パラメータファイルが見つかりません。"
                "data/tokyo/service_time_analysis/v1/lognormal_parameters.json が必要です。"
            )
    
    def _init_hospital_selection(self):
        """病院選択の初期化（validation_simulation互換）"""
        print("\n病院選択モデル初期化中...")
        
        # hospital_dataが読み込まれていることを確認
        if not hasattr(self, 'hospital_data') or self.hospital_data is None:
            raise RuntimeError(
                "_init_hospital_selection()はhospital_dataの読み込み後に呼ばれる必要があります。"
                "_load_base_data()を先に呼び出してください。"
            )
        
        # デバッグ: カラム名を確認
        print(f"  hospital_dataのカラム: {self.hospital_data.columns.tolist()}")
        print(f"  hospital_data件数: {len(self.hospital_data)}")
        
        # 病院H3インデックスのリストを作成
        self.hospital_h3_indices = []
        skipped_count = 0
        error_count = 0
        
        for idx, hospital in self.hospital_data.iterrows():
            try:
                if pd.notna(hospital['latitude']) and pd.notna(hospital['longitude']):
                    h3_idx = h3.latlng_to_cell(hospital['latitude'], hospital['longitude'], 9)
                    if h3_idx in self.grid_mapping:
                        self.hospital_h3_indices.append(h3_idx)
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                error_count += 1
                if error_count <= 3:  # 最初の3件のエラーのみ表示
                    print(f"  エラー(病院{idx}): {e}")
                continue
        
        print(f"  病院H3インデックス数: {len(self.hospital_h3_indices)}")
        if skipped_count > 0:
            print(f"  スキップ: {skipped_count}件 (座標なし or grid外)")
        if error_count > 0:
            print(f"  エラー: {error_count}件")
        
        # 統計情報の初期化（_classify_hospitalsより前に実行）
        if 'hospital_selection_stats' not in self.statistics:
            self.statistics['hospital_selection_stats'] = {
                'tertiary_selections': 0,
                'secondary_primary_selections': 0,
                'no_hospital_found': 0,
                'by_severity': {}
            }
        
        # 病院分類の初期化
        self._classify_hospitals()
        
        # 確率的病院選択モデルの読み込み
        self.use_probabilistic_selection = True  # デフォルトで有効
        self._load_hospital_selection_model()
        
        print("  ✓ 病院選択モデル初期化完了")
    
    def _classify_hospitals(self):
        """病院を3次救急とそれ以外に分類（validation_simulation互換）"""
        if self.hospital_data is None:
            print("警告: 病院データが提供されていません。全ての病院を2次以下として扱います。")
            self.secondary_primary_hospitals = set(self.hospital_h3_indices)
            self.tertiary_hospitals = set()
            return
        
        print("  病院を救急医療機関レベル別に分類中...")
        
        # H3インデックスを計算して病院データに追加
        self.hospital_data = self.hospital_data.copy()
        self.hospital_data['h3_index'] = self.hospital_data.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], 9)
            if pd.notna(row['latitude']) and pd.notna(row['longitude']) else None,
            axis=1
        )
        
        # genre_codeに基づいて分類
        if 'genre_code' in self.hospital_data.columns:
            tertiary_hospitals_df = self.hospital_data[
                (self.hospital_data['genre_code'] == 1) & 
                (self.hospital_data['h3_index'].notna())
            ]
            
            secondary_primary_hospitals_df = self.hospital_data[
                (self.hospital_data['genre_code'] == 2) & 
                (self.hospital_data['h3_index'].notna())
            ]
            
            # H3インデックスをセットに変換
            self.tertiary_hospitals = set(tertiary_hospitals_df['h3_index'].tolist())
            self.secondary_primary_hospitals = set(secondary_primary_hospitals_df['h3_index'].tolist())
            
            # grid_mappingに存在しない病院を除外
            self.tertiary_hospitals = {h3_idx for h3_idx in self.tertiary_hospitals if h3_idx in self.grid_mapping}
            self.secondary_primary_hospitals = {h3_idx for h3_idx in self.secondary_primary_hospitals if h3_idx in self.grid_mapping}
            
            print(f"    3次救急医療機関: {len(self.tertiary_hospitals)}件")
            print(f"    2次以下医療機関: {len(self.secondary_primary_hospitals)}件")
            
            # 分類されなかった病院を2次以下に追加
            unclassified = set(self.hospital_h3_indices) - self.tertiary_hospitals - self.secondary_primary_hospitals
            if unclassified:
                print(f"    未分類病院（2次以下に追加）: {len(unclassified)}件")
                self.secondary_primary_hospitals.update(unclassified)
        else:
            print("警告: hospital_dataに'genre_code'カラムが見つかりません。全ての病院を2次以下として扱います。")
            self.tertiary_hospitals = set()
            self.secondary_primary_hospitals = set(self.hospital_h3_indices)
        
        # 統計初期化
        for severity in ['軽症', '中等症', '重症', '重篤', '死亡', 'その他']:
            self.statistics['hospital_selection_stats']['by_severity'][severity] = {
                'tertiary': 0,
                'secondary_primary': 0,
                'default': 0,
                'probabilistic_success': 0,
                'deterministic_fallback': 0,
                'static_fallback_used': 0,
                'error_fallback': 0
            }
    
    def _load_hospital_selection_model(self):
        """確率的病院選択モデルを読み込む（validation_simulation互換）"""
        model_path = self.base_dir / 'processed/hospital_selection_model_revised.pkl'
        
        try:
            with open(model_path, 'rb') as f:
                main_model = pickle.load(f)
                self.hospital_selection_model = main_model['selection_probabilities']
                
                # 静的フォールバックモデルを読み込む
                self.static_fallback_model = main_model.get('static_fallback_model', {}) 
                
                self.model_hospital_master = pd.DataFrame(main_model['hospital_master'])
                self.model_h3_centers = main_model['h3_centers']
            
            print(f"  確率的病院選択モデルを読み込みました:")
            print(f"    実績ベースの条件数: {len(self.hospital_selection_model)}")
            if self.static_fallback_model:
                print(f"    静的フォールバックモデルの傷病度: {list(self.static_fallback_model.keys())}")
            
        except FileNotFoundError as e:
            print(f"  警告: 確率モデルファイルが見つかりません: {e}")
            print("  デフォルトの最寄り病院選択を使用します。")
            self.use_probabilistic_selection = False
    
    def _load_base_data(self):
        """基本データの読み込み（修正版：ValidationSimulatorと同じフィルタリング）"""
        print("\n基本データ読み込み中...")
        
        # 救急署データ
        firestation_path = self.base_dir / "import/amb_place_master.csv"
        ambulance_data_full = pd.read_csv(firestation_path, encoding='utf-8')
        ambulance_data_full = ambulance_data_full[ambulance_data_full['special_flag'] == 1]
        
        print(f"  元データ: {len(ambulance_data_full)}台")
        
        # ★★★ 修正1: 常に「救急隊なし」を除外 ★★★
        if 'team_name' in ambulance_data_full.columns:
            before_exclusion = len(ambulance_data_full)
            
            # ValidationSimulatorと同じフィルタリング
            team_mask = (ambulance_data_full['team_name'] != '救急隊なし')
            ambulance_data_full = ambulance_data_full[team_mask].copy()
            
            excluded_count = before_exclusion - len(ambulance_data_full)
            print(f"  「救急隊なし」除外: {before_exclusion}台 → {len(ambulance_data_full)}台 (除外: {excluded_count}台)")
        
        # ★★★ 修正2: デイタイム救急の除外（オプション）★★★
        # config.yamlで制御できるようにする
        exclude_daytime = self.config.get('data', {}).get('exclude_daytime_ambulances', True)
        
        if exclude_daytime and 'team_name' in ambulance_data_full.columns:
            before_daytime = len(ambulance_data_full)
            
            daytime_mask = ~ambulance_data_full['team_name'].str.contains('デイタイム', na=False)
            ambulance_data_full = ambulance_data_full[daytime_mask].copy()
            
            excluded_daytime = before_daytime - len(ambulance_data_full)
            print(f"  「デイタイム救急」除外: {before_daytime}台 → {len(ambulance_data_full)}台 (除外: {excluded_daytime}台)")
        
        print(f"  フィルタリング後: {len(ambulance_data_full)}台")
        
        # エリア制限フィルタリングの設定確認
        area_restriction = self.config.get('data', {}).get('area_restriction', {})
        
        if area_restriction.get('enabled', False):
            section_code = area_restriction.get('section_code')
            area_name = area_restriction.get('area_name', '指定エリア')
            
            # section_codeがnullまたはNoneの場合は全方面を使用（東京23区全域など）
            if section_code is None or section_code == 'null':
                print(f"  {area_name}（全方面）を使用")
                self.ambulance_data = ambulance_data_full
                
            elif section_code in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                # 指定方面の救急隊に限定
                before_filter = len(ambulance_data_full)
                section_filtered = ambulance_data_full[ambulance_data_full['section'] == section_code].copy()
                
                self.ambulance_data = section_filtered
                print(f"  {area_name}フィルタ適用: {before_filter}台 → {len(self.ambulance_data)}台")
                
                if len(self.ambulance_data) == 0:
                    print(f"  警告: {area_name}の救急車が見つかりません。全体を使用します。")
                    self.ambulance_data = ambulance_data_full
            else:
                # その他の場合は全体を使用
                self.ambulance_data = ambulance_data_full
        else:
            # ★★★ 修正3: エリア制限なしでもフィルタリング済みデータを使用 ★★★
            self.ambulance_data = ambulance_data_full
        
        print(f"  最終救急車数: {len(self.ambulance_data)}台")
        
        # ★★★ 修正4: ValidationSimulatorとの一致確認 ★★★
        print(f"\n  ✓ ValidationSimulatorとの一致確認:")
        print(f"    - 「救急隊なし」: 除外済み")
        if exclude_daytime:
            print(f"    - 「デイタイム救急」: 除外済み")
        else:
            print(f"    - 「デイタイム救急」: 含む")
        print(f"    - 最終台数: {len(self.ambulance_data)}台")
        print(f"    - この台数がValidationSimulatorと一致する必要があります")
        
        # ★★★ 仮想救急車の作成（学習モード時）★★★
        if self.mode == 'train':
            self.ambulance_data = self._create_virtual_ambulances_if_needed(self.ambulance_data)
            print(f"  最終救急車数（仮想含む）: {len(self.ambulance_data)}台")
        
        # 病院データ（方面に関係なく全体を使用）
        hospital_path = self.base_dir / "import/hospital_master.csv"
        self.hospital_data = pd.read_csv(hospital_path, encoding='utf-8')
        # validation_simulationと同じカラム名リネーム処理
        self.hospital_data = self.hospital_data.rename(columns={
            'hospital_latitude': 'latitude', 
            'hospital_longitude': 'longitude'
        })
        print(f"  病院数: {len(self.hospital_data)}")
        
        # グリッドマッピング
        grid_mapping_path = self.base_dir / "processed/grid_mapping_res9.json"
        with open(grid_mapping_path, 'r', encoding='utf-8') as f:
            self.grid_mapping = json.load(f)
        print(f"  H3グリッド数: {len(self.grid_mapping)}")
        
        # 移動時間行列（軽量版 - 学習用）
        self.travel_time_matrices = {}
        calibration_dir = self.base_dir / "calibration2"
        for phase in ['response', 'transport', 'return']:
            matrix_path = calibration_dir / f"linear_calibrated_{phase}.npy"
            if matrix_path.exists():
                self.travel_time_matrices[phase] = np.load(matrix_path)
        
        # 距離行列
        distance_matrix_path = self.base_dir / "processed/travel_distance_matrix_res9.npy"
        self.travel_distance_matrix = np.load(distance_matrix_path)
        

    def set_mode(self, mode: str):
        """
        環境のモードを切り替える（トレーニング/評価）
        
        Args:
            mode: "train" または "eval"
            
        Raises:
            ValueError: 無効なモードが指定された場合
        """
        if mode not in ["train", "eval"]:
            raise ValueError(f"無効なモード: {mode}. 'train' または 'eval' を指定してください。")
        
        old_mode = self.mode
        self.mode = mode
        
        # モード切り替え時はログフラグをリセット（期間情報を再表示するため）
        if old_mode != mode:
            self._first_period_logged = False
            print(f"環境モード切り替え: {old_mode} → {mode}")

    
    def reset(self, period_index: Optional[int] = None) -> np.ndarray:
        """
        環境のリセット
        
        Returns:
            初期観測
        """
        # ★★★【時間管理のリセット】★★★
        self.current_time_seconds = 0.0
        self.event_queue = []
        self.episode_step = 0
        # ★★★【リセットここまで】★★★
        
        # 期間の選択
        if self.mode == "train":
            periods = self.config['data']['train_periods']
        else:
            periods = self.config['data']['eval_periods']
        
        if period_index is None:
            period_index = np.random.randint(len(periods))
        
        self.current_period_idx = period_index
        period = periods[period_index]
        
        # エピソード開始情報は最初の期間のみ表示
        if not self._first_period_logged:
            print(f"\nエピソード開始: {period['start_date']} - {period['end_date']}")
            self._first_period_logged = True
        
        # エピソードカウンタをインクリメント
        self._episode_count += 1
        
        # シミュレータの初期化
        self._init_simulator_for_period(period)
        
        # エピソード統計のリセット
        self.episode_stats = self._init_episode_stats()
        
        # 対応不能事案管理の初期化
        self.unhandled_calls = []  # 対応不能になった事案のリスト
        self.call_start_times = {}  # 事案の発生時刻記録
        
        # ★★★【イベントキューへの事案追加】★★★
        # 全ての事案をNEW_CALLイベントとしてキューに追加
        
        # ★★★【修正箇所】★★★
        # episode_start_time が _prepare_episode_calls で設定されない場合のフォールバック
        if not hasattr(self, 'episode_start_time') or self.episode_start_time is None:
            if len(self.current_episode_calls) > 0:
                self.episode_start_time = self.current_episode_calls[0]['datetime']
            else:
                self.episode_start_time = datetime.now()  # フォールバック
        # ★★★【修正ここまで】★★★
            
        for call in self.current_episode_calls:
            # 事案の発生時刻を計算（エピソード開始からの経過秒数）
            call_time_delta = (call['datetime'] - self.episode_start_time).total_seconds()
            
            event = Event(
                time=call_time_delta,
                event_type=EventType.NEW_CALL,
                data={'call': call}
            )
            self._schedule_event(event)
        
        # ★★★【イベントキュー追加ここまで】★★★
        
        # 最初の事案を設定（pending_callはstep内で設定）
        self.pending_call = None
        
        # 初期観測を返す
        return self._get_observation()
    
    def _init_simulator_for_period(self, period: Dict):
        """指定期間用のシミュレータを初期化"""
        # 救急事案データの読み込み
        calls_df = self._load_calls_for_period(period)
        
        # エピソード用の事案を準備
        self.current_episode_calls = self._prepare_episode_calls(calls_df)
        
        # ★★★【重要な修正】max_steps_per_episodeを時間ベースに設定★★★
        # 1ステップ = 1分なので、エピソード時間（時間）× 60 = ステップ数
        episode_duration_hours = self.config['data'].get('episode_duration_hours', 24)
        time_based_max_steps = episode_duration_hours * 60  # 時間ベースのステップ数
        
        config_max_steps = self.config.get('data', {}).get('max_steps_per_episode') or \
                          self.config.get('max_steps_per_episode')
        
        if config_max_steps:
            # configで指定されている場合、それを使用
            self.max_steps_per_episode = config_max_steps
        else:
            # configで指定されていない場合、時間ベースの値を使用
            self.max_steps_per_episode = time_based_max_steps
        
        print(f"読み込まれた事案数: {len(self.current_episode_calls)}")
        print(f"エピソード時間: {episode_duration_hours}時間 = {time_based_max_steps}分")
        print(f"最大ステップ数: {self.max_steps_per_episode} ({'config設定' if config_max_steps else '時間ベース'})")
        # ★★★【修正ここまで】★★★
        
        # ★★★【修正箇所】★★★
        # 救急車状態の初期化を、古いものから現実的なものに変更
        self._initialize_ambulances_realistic()
        # ★★★【修正ここまで】★★★
        
        # エピソードカウンタ初期化（重要！）
        self.episode_step = 0
        self.pending_call = None
        
    def _load_calls_for_period(self, period: Dict) -> pd.DataFrame:
        """
        指定期間の救急事案データを読み込み（最適化版）
        キャッシュからデータを取得するため高速
        """
        start_date = str(period['start_date'])
        end_date = str(period['end_date'])
        
        # エリア制限の設定確認
        area_restriction = self.config.get('data', {}).get('area_restriction', {})
        area_filter = None
        if area_restriction.get('enabled', False):
            area_filter = area_restriction.get('districts', [])
        
        # 最初の期間のみ詳細情報を表示
        if not self._first_period_logged:
            area_name = area_restriction.get('area_name', 'エリア制限')
            area_info = f" ({area_name}: {', '.join(area_filter)})" if area_filter else ""
            print(f"期間データ取得中: {start_date} - {end_date}{area_info}")
        
        # キャッシュから高速取得（エリアフィルタ付き）
        filtered_df = self.data_cache.get_period_data(start_date, end_date, area_filter)
        
        if not self._first_period_logged:
            print(f"期間内の事案数: {len(filtered_df)}件")
        
        # 必要なカラムの存在確認
        required_columns = ['救急事案番号キー', 'Y_CODE', 'X_CODE', '収容所見程度', '出場年月日時分']
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        if missing_columns:
            print(f"警告: 必要なカラムが不足: {missing_columns}")
            return pd.DataFrame()
        
        if not self._first_period_logged:
            print(f"最終的な事案数: {len(filtered_df)}件")
            
            if len(filtered_df) > 0:
                # 傷病度の分布を表示
                severity_counts = filtered_df['収容所見程度'].value_counts()
                print("傷病度分布:")
                for severity, count in severity_counts.head().items():
                    print(f"  {severity}: {count}件")
            print(f"エピソード長: {self.config['data']['episode_duration_hours']}時間")
        
        return filtered_df
    
    def _prepare_episode_calls(self, calls_df: pd.DataFrame) -> List[Dict]:
        """エピソード用の事案リストを準備"""
        import h3
        import numpy as np
        import pandas as pd
        
        if len(calls_df) == 0:
            print("警告: 事案データが空です")
            # ★★★【修正箇所】★★★
            # episode_start_time の設定を追加
            self.episode_start_time = datetime.now()
            # ★★★【修正ここまで】★★★
            return []
        
        episode_calls = []
        
        # エピソード長の設定（時間）
        episode_hours = self.config['data']['episode_duration_hours']
        print(f"エピソード長: {episode_hours}時間")
        
        # 時刻でソート
        calls_df = calls_df.sort_values('出場年月日時分')
        
        # エピソードの開始時刻をランダムに選択
        start_time = calls_df['出場年月日時分'].iloc[0]
        end_time = calls_df['出場年月日時分'].iloc[-1]
        
        # エピソード期間内のデータを選択できる開始時刻の範囲
        max_start_time = end_time - pd.Timedelta(hours=episode_hours)
        
        if start_time >= max_start_time:
            # データが短すぎる場合は全体を使用
            episode_start = start_time
            episode_end = end_time
            print(f"警告: データ期間が短いため、全期間を使用")
        else:
            # ランダムな開始時刻を選択
            time_range = (max_start_time - start_time).total_seconds()
            random_offset = np.random.uniform(0, time_range)
            episode_start = start_time + pd.Timedelta(seconds=random_offset)
            episode_end = episode_start + pd.Timedelta(hours=episode_hours)
        
        # ★★★【修正箇所】★★★
        # エピソードの開始時刻をクラス変数に保存
        self.episode_start_time = episode_start
        # ★★★【修正ここまで】★★★
        
        # エピソード期間内の事案を抽出
        mask = (calls_df['出場年月日時分'] >= episode_start) & (calls_df['出場年月日時分'] <= episode_end)
        episode_df = calls_df[mask].copy()
        
        # 毎回表示する情報（簡潔版）
        print(f"エピソード期間: {episode_start.strftime('%Y-%m-%d %H:%M')} ～ {episode_end.strftime('%Y-%m-%d %H:%M')}")
        print(f"エピソード内事案数: {len(episode_df)}件")
        
        for _, row in episode_df.iterrows():
            # H3インデックスの計算
            try:
                # 座標の有効性チェック
                lat = float(row['Y_CODE'])
                lng = float(row['X_CODE'])
                
                if -90 <= lat <= 90 and -180 <= lng <= 180:
                    h3_index = h3.latlng_to_cell(lat, lng, 9)
                else:
                    continue  # 無効な座標はスキップ
            except Exception as e:
                continue  # 変換エラーはスキップ
            
            call_info = {
                'id': str(row['救急事案番号キー']),
                'h3_index': h3_index,
                'severity': row.get('収容所見程度', 'その他'),
                'datetime': row['出場年月日時分'],
                'location': (lat, lng)
            }
            episode_calls.append(call_info)
        
        # 時間順にソート
        episode_calls.sort(key=lambda x: x['datetime'])
        
        print(f"有効な事案数: {len(episode_calls)}件")
        
        return episode_calls
    
    def _initialize_ambulances_realistic(self):
        """
        現実的な救急車初期化処理（VALIDATION_INTEGRATION_PLAN.md準拠）
        
        エピソード開始時に、一部の救急車が活動中の状態を再現し、
        それらの復帰イベントをイベントキューにスケジュールします。
        """
        self.ambulance_states = {}
        print(f"  救急車データから現実的初期化開始: {len(self.ambulance_data)}台")
        
        for amb_id, (_, row) in enumerate(self.ambulance_data.iterrows()):
            if amb_id >= self.action_dim:
                break
            
            try:
                # 座標の検証とH3インデックスの計算
                lat = float(row['latitude'])
                lng = float(row['longitude'])
                if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                    continue
                station_h3 = h3.latlng_to_cell(lat, lng, 9)
                
                # 50-70%の確率で初期活動中とする
                is_busy = np.random.uniform() < np.random.uniform(0.5, 0.7)
                
                # 復帰時刻（秒単位）
                # (注: self.current_time_seconds は reset() で 0 になっています)
                completion_time = self.current_time_seconds + np.random.uniform(0, 1800) if is_busy else 0.0  # 0-30分
                
                # 状態辞書を作成
                self.ambulance_states[amb_id] = {
                    'status': 'dispatched' if is_busy else 'available',
                    'completion_time': completion_time,
                    'current_h3': station_h3,
                    'station_h3': station_h3,
                    'calls_today': 1 if is_busy else 0
                    # 'name' や 'id' など、他のメソッドで使われていないキーは
                    # 状態ベクトル生成に影響しないため、ここでは省略
                }

                # ★新規追加: 初期活動中の救急車の復帰イベントをスケジュール★
                if is_busy:
                    return_event = Event(
                        time=self.ambulance_states[amb_id]['completion_time'],
                        event_type=EventType.AMBULANCE_AVAILABLE,
                        data={
                            'ambulance_id': amb_id,
                            'station_h3': station_h3
                        }
                    )
                    self._schedule_event(return_event)
            
            except Exception as e:
                print(f"    ❌ 救急車{amb_id}の初期化でエラー: {e}")
                continue

        available_count = sum(1 for st in self.ambulance_states.values() if st['status'] == 'available')
        print(f"  救急車初期化完了: {len(self.ambulance_states)}台 (初期利用可能: {available_count}台)")
    
    def step(self, action: int) -> StepResult:
        """
        環境のステップ実行（固定時間ステップ制: 1ステップ=1分=60秒）
        
        Args:
            action: 選択された救急車のインデックス
            
        Returns:
            StepResult: 観測、報酬、終了フラグ、追加情報
        """
        try:
            # 1ステップ = 60秒を進める
            start_time = self.current_time_seconds
            end_time = start_time + self.time_per_step
            
            reward = 0.0
            info = {}
            
            # --- 1. この1分間に発生する全イベントを処理 ---
            # ★ 重要: 復帰イベントは必ず処理、NEW_CALLは1件のみ処理 ★
            while self.event_queue and self.event_queue[0].time <= end_time:
                event = self.event_queue[0]  # 次のイベントを確認（まだpopしない）
                
                # 復帰イベントは常に処理
                if event.event_type == EventType.AMBULANCE_AVAILABLE:
                    self._process_next_event()
                    continue
                
                # NEW_CALLイベントの処理
                if event.event_type == EventType.NEW_CALL:
                    # 既にpending_callがある場合、新しい事案は次のステップで処理
                    if self.pending_call is not None:
                        break
                    # pending_callが空なら、この事案を処理
                    self._process_next_event()
                    # (self.pending_callがセットされる)
                else:
                    # その他のイベント（将来の拡張用）
                    self._process_next_event()
            
            # --- 2. エージェントの行動（配車）を処理 ---
            if self.pending_call is not None:
                current_incident = self.pending_call
                
                # (ここから下のロジックは既存のものを流用)
                # ハイブリッドモード：重症系は直近隊運用を強制
                if self.hybrid_mode:
                    severity = current_incident.get('severity', '')
                    
                    if severity in self.severe_conditions:
                        self.direct_dispatch_count += 1
                        closest_action = self._get_closest_ambulance_action(current_incident)
                        dispatch_result = self._dispatch_ambulance(closest_action)
                        reward = 0.0  # 学習対象外
                        info = {'dispatch_type': 'direct_closest', 'skipped_learning': True}
                    else:
                        # 軽症系：PPOで学習
                        self.ppo_dispatch_count += 1
                        dispatch_result = self._dispatch_ambulance(action)
                        reward = self._calculate_reward(dispatch_result)
                        info = {'dispatch_type': 'ppo_learning'}
                else:
                    # 通常モード
                    dispatch_result = self._dispatch_ambulance(action)
                    reward = self._calculate_reward(dispatch_result)
                    info = {'dispatch_type': 'ppo_normal'}
                
                # ★ 修正点2: 配車が成功した場合のみ、事案をクリアする ★
                if dispatch_result and dispatch_result['success']:
                    # 配車成功時の処理
                    self._log_dispatch_action(dispatch_result, self.ambulance_states[dispatch_result['ambulance_id']])
                    self._update_statistics(dispatch_result)
                    
                    # 復帰イベントをスケジュール
                    amb_id = dispatch_result['ambulance_id']
                    # ★ completion_time_secondsは既に絶対時刻（current_time + 活動時間）★
                    return_time = dispatch_result.get('completion_time_seconds', 
                                                      self.current_time_seconds + 4000)  # フォールバック
                    
                    return_event = Event(
                        time=return_time,
                        event_type=EventType.AMBULANCE_AVAILABLE,
                        data={
                            'ambulance_id': amb_id,
                            'station_h3': self.ambulance_states[amb_id]['station_h3']
                        }
                    )
                    self._schedule_event(return_event)
                    
                    # 成功したので事案をクリア
                    self.pending_call = None
                    
                else:
                    # 配車失敗時の処理（例: 利用可能台数0）
                    # ★★★ 事案をクリアせず、次のステップで再試行する ★★★
                    # 失敗の理由に応じてペナルティを与える
                    if dispatch_result:
                        reason = dispatch_result.get('reason', 'unknown')
                        if reason == 'ambulance_busy':
                            reward = self.reward_designer.get_failure_penalty('no_available')
                        else:
                            reward = self.reward_designer.get_failure_penalty('dispatch')
                    else:
                        reward = self.reward_designer.get_failure_penalty('dispatch')
                    
                    info.update({
                        'dispatch_failed': True,
                        'reason': dispatch_result.get('reason', 'unknown') if dispatch_result else 'no_result',
                        'retry_next_step': True
                    })
            
            # --- 3. 時間を60秒の終わりまで進める ---
            # (注: ループで時間が進んでいるため、end_timeで上書きする)
            self.current_time_seconds = end_time
            self.episode_step += 1
            
            # エピソード終了判定
            done = self._is_episode_done()
            
            # 次の観測を取得
            observation = self._get_observation()
            
            info.update({
                'episode_stats': self.episode_stats.copy(),
                'step': self.episode_step
            })
            
            return StepResult(
                observation=observation,
                reward=reward,
                done=done,
                info=info
            )
        except Exception as e:
            print(f"❌ step()メソッドでエラー発生: {e}")
            import traceback
            traceback.print_exc()
            # エラー時も安全に終了させる
            return StepResult(
                observation=np.zeros(self.state_dim),
                reward=0.0,
                done=True,
                info={'error': str(e)}
            )

    def get_optimal_action(self) -> Optional[int]:
        """
        現在の事案に対して最適な救急車を選択（最近接）
        ValidationSimulatorのfind_closest_available_ambulanceと同じロジック
        
        Returns:
            最適な救急車のID、または None
        """
        if self.pending_call is None:
            return None
        
        best_action = None
        min_travel_time = float('inf')
        
        # 全ての救急車をチェック
        for amb_id, amb_state in self.ambulance_states.items():
            # 利用可能な救急車のみ対象
            if amb_state['status'] != 'available':
                continue
            
            try:
                # 現在位置から事案発生地点への移動時間を計算
                travel_time = self._calculate_travel_time(
                    amb_state['current_h3'],
                    self.pending_call['h3_index']
                )
                
                # より近い救急車を発見
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    best_action = amb_id
                    
            except Exception as e:
                # エラーが発生した場合はスキップ
                continue
        
        # デバッグ情報の出力（verboseモード時）
        if best_action is not None and hasattr(self, 'verbose_logging') and self.verbose_logging:
            print(f"[最適選択] 救急車{best_action}を選択 (移動時間: {min_travel_time/60:.1f}分)")
        
        return best_action

    def _get_closest_ambulance_action(self, incident):
        """最寄りの救急車を選択するアクション番号を取得"""
        available_ambulances = self.get_action_mask()
        if not available_ambulances.any():
            return 0
        
        min_distance = float('inf')
        closest_idx = 0
        
        for idx, is_available in enumerate(available_ambulances):
            if is_available and idx < len(self.ambulance_states):
                amb_state = self.ambulance_states[idx]
                distance = self._calculate_travel_time(
                    amb_state['current_h3'], 
                    incident['h3_index']
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = idx
        
        return closest_idx

    def _calculate_coverage_info(self):
        """カバレッジ情報を計算"""
        # 各地域の空き救急車までの平均距離を計算
        coverage_scores = []
        high_risk_scores = []
        
        # 簡易的なカバレッジ計算（利用可能救急車の割合）
        available_count = sum(1 for amb in self.ambulance_states.values() 
                             if amb['status'] == 'available')
        total_count = len(self.ambulance_states)
        
        if total_count > 0:
            overall_coverage = available_count / total_count
            coverage_scores.append(overall_coverage)
            
            # 高リスク地域の判定（簡易版：重症系事案が多い地域を想定）
            # 実際の実装では、過去の重症系事案データから高リスク地域を特定
            high_risk_coverage = overall_coverage  # 簡略化
        
        return {
            'overall_coverage': np.mean(coverage_scores) if coverage_scores else 0.0,
            'high_risk_area_coverage': high_risk_coverage if 'high_risk_coverage' in locals() else 0.0,
            'min_coverage': min(coverage_scores) if coverage_scores else 0.0
        }

    def is_high_risk_area(self, area):
        """高リスク地域の判定（簡易版）"""
        # 実際の実装では、過去の重症系事案データから判定
        # ここでは簡易的にランダムで判定
        return np.random.random() < 0.3  # 30%の確率で高リスク地域

    def get_available_count_in_area(self, area):
        """指定エリアの利用可能救急車数を取得（簡易版）"""
        # 実際の実装では、エリア内の救急車をカウント
        # ここでは全体の利用可能数を返す
        return sum(1 for amb in self.ambulance_states.values() 
                  if amb['status'] == 'available')

    def get_total_count_in_area(self, area):
        """指定エリアの総救急車数を取得（簡易版）"""
        # 実際の実装では、エリア内の救急車をカウント
        # ここでは全体の総数を返す
        return len(self.ambulance_states)

    @property
    def areas(self):
        """エリアリストを取得（簡易版）"""
        # 実際の実装では、地理的エリアのリストを返す
        # ここでは簡易的に1つのエリアを返す
        return ['default_area']

    
    def _dispatch_ambulance(self, action: int) -> Dict:
        """救急車を配車"""
        if self.pending_call is None:
            return {'success': False, 'reason': 'no_pending_call'}
        
        # 行動の妥当性チェック
        if action >= len(self.ambulance_states):
            return {'success': False, 'reason': 'invalid_action'}
        
        amb_state = self.ambulance_states[action]
        
        # 利用可能性チェック
        if amb_state['status'] != 'available':
            return {'success': False, 'reason': 'ambulance_busy'}
        
        # 移動時間の計算（秒単位）
        travel_time = self._calculate_travel_time(
            amb_state['current_h3'],
            self.pending_call['h3_index']
        )
        
        # 配車実行
        amb_state['status'] = 'dispatched'
        amb_state['calls_today'] += 1
        amb_state['last_dispatch_time'] = self.current_time_seconds  # 秒単位に変更
        amb_state['current_severity'] = self.pending_call['severity']  # 傷病度を記録
        
        # ValidationSimulatorと同じ活動時間計算（秒単位）
        completion_time_seconds = self._calculate_ambulance_completion_time(
            action, self.pending_call, travel_time
        )
        amb_state['call_completion_time'] = completion_time_seconds
        
        result = {
            'success': True,
            'ambulance_id': action,
            'call_id': self.pending_call['id'],
            'severity': self.pending_call['severity'],
            'response_time': travel_time,
            'response_time_minutes': travel_time / 60.0,
            'completion_time_seconds': completion_time_seconds,  # 秒単位の完了時刻を追加
            'estimated_completion_time': completion_time_seconds / 60.0,  # 互換性のため分単位も残す
            'matched_teacher': self.current_matched_teacher
        }
        
        return result
    
    def _log_dispatch_action(self, dispatch_result: Dict, ambulance_state: Dict):
        """配車アクションのログを記録"""
        if not hasattr(self, 'dispatch_logger') or not self.dispatch_logger.enabled:
            return
        
        # 最適救急車を取得
        optimal_ambulance_id = self.get_optimal_action()
        optimal_response_time = None
        if optimal_ambulance_id is not None:
            optimal_response_time = self._calculate_travel_time(
                self.ambulance_states[optimal_ambulance_id]['current_h3'],
                self.pending_call['h3_index']
            ) / 60.0
        
        # 利用可能救急車数とアクションマスク情報
        available_count = sum(1 for amb in self.ambulance_states.values() 
                            if amb['status'] == 'available')
        total_count = len(self.ambulance_states)
        action_mask = self.get_action_mask()
        valid_action_count = action_mask.sum()
        
        # エピソード平均報酬
        episode_reward_avg = 0.0
        if hasattr(self, 'episode_stats') and self.episode_stats['total_dispatches'] > 0:
            # 簡易的な平均報酬計算
            episode_reward_avg = sum(self.episode_stats.get('rewards', [0])) / max(1, len(self.episode_stats.get('rewards', [1])))
        
        # 救急車情報を準備
        ambulance_id = dispatch_result['ambulance_id']
        
        # シンプルかつ安全な判定: DataFrameから直接確認（NaNはFalse）
        is_virtual = False
        try:
            if ambulance_id < len(self.ambulance_data):
                is_virtual = self._row_is_virtual(self.ambulance_data.iloc[ambulance_id])
        except Exception:
            is_virtual = False
        
        # 表示名（デフォルトは状態キャッシュのname、なければデータフレーム由来）
        display_name = ambulance_state.get('name')
        try:
            if not display_name and ambulance_id < len(self.ambulance_data):
                row = self.ambulance_data.iloc[ambulance_id]
                # 仮想は一律virtual_team、実隊はteam_name
                if self._row_is_virtual(row):
                    display_name = f"virtual_team_{ambulance_id}"
                else:
                    display_name = row.get('team_name') or row.get('name') or f"救急車{ambulance_id}"
        except Exception:
            display_name = f"救急車{ambulance_id}"

        ambulance_info = {
            'station_h3': ambulance_state.get('station_h3', 'unknown'),
            'is_virtual': is_virtual,
            'response_time_minutes': dispatch_result['response_time_minutes'],
            'name': display_name
        }

        # この時点で報酬を計算
        if dispatch_result['success']:
            reward = self._calculate_reward(dispatch_result)
        else:
            reward = 0.0
        
        # ログを記録
        self.dispatch_logger.log_dispatch(
            episode=self._episode_count,
            step=self.episode_step,
            call_info=self.pending_call,
            selected_ambulance_id=dispatch_result['ambulance_id'],
            ambulance_info=ambulance_info,
            response_time_minutes=dispatch_result['response_time_minutes'],
            available_count=available_count,
            total_count=total_count,
            action_mask_valid_count=valid_action_count,
            optimal_ambulance_id=optimal_ambulance_id,
            optimal_response_time=optimal_response_time,
            teacher_match=dispatch_result.get('matched_teacher', False),
            reward=reward,  # 報酬は後で計算される
            episode_reward_avg=episode_reward_avg
        )
    
    def _calculate_ambulance_completion_time(self, ambulance_id: int, call: Dict, response_time: float) -> float:
        """救急車の活動完了時間を計算（秒単位、ValidationSimulator互換）"""
        current_time = self.current_time_seconds  # 現在時刻（秒単位）
        severity = call['severity']
        
        # 1. 現場到着時刻 = 現在時刻 + 応答時間（秒）
        arrive_scene_time = current_time + response_time
        
        # 2. 現場活動時間（ServiceTimeGeneratorを使用、分単位で返される）
        on_scene_time_minutes = self.service_time_generator.generate_time(severity, 'on_scene_time')
        on_scene_time = on_scene_time_minutes * 60.0  # 秒に変換
        
        # 3. 現場出発時刻
        depart_scene_time = arrive_scene_time + on_scene_time
        
        # 4. 病院選択と搬送時間
        hospital_h3 = self.select_hospital(call['h3_index'], severity)
        if hospital_h3 is None:
            hospital_h3 = call['h3_index']
        transport_time = self._calculate_travel_time_for_phase(call['h3_index'], hospital_h3, phase='transport')
        
        # 5. 病院到着時刻
        arrive_hospital_time = depart_scene_time + transport_time
        
        # 6. 病院滞在時間（分単位で返されるので秒に変換）
        hospital_time_minutes = self.service_time_generator.generate_time(severity, 'hospital_time')
        hospital_time = hospital_time_minutes * 60.0  # 秒に変換
        
        # 7. 病院出発時刻
        depart_hospital_time = arrive_hospital_time + hospital_time
        
        # 8. 帰署時間
        amb_state = self.ambulance_states[ambulance_id]
        return_time = self._calculate_travel_time_for_phase(hospital_h3, amb_state['station_h3'], phase='return')
        
        # 9. 最終完了時刻（秒単位）
        completion_time = depart_hospital_time + return_time
        
        if self.verbose_logging:
            print(f"救急車{ambulance_id}活動時間計算（秒単位）:")
            print(f"  応答: {response_time:.1f}秒, 現場: {on_scene_time:.1f}秒")
            print(f"  搬送: {transport_time:.1f}秒, 病院: {hospital_time:.1f}秒, 帰署: {return_time:.1f}秒")
            print(f"  総活動時間: {(completion_time - current_time)/60:.1f}分")
        
        return completion_time
    
    def select_hospital(self, incident_h3: str, severity: str) -> Optional[str]:
        """傷病度に応じた病院選択（validation_simulation互換）"""
        severe_conditions = ['重症', '重篤']
        
        # 重症・重篤の案件は決定論的選択
        if severity in severe_conditions:
            return self._select_hospital_deterministic(incident_h3, severity)
        
        # 軽症・中等症・死亡：確率的選択
        if not self.use_probabilistic_selection:
            return self._select_hospital_deterministic(incident_h3, severity)

        # 時間情報とキーの作成
        current_hour = int((self.episode_step / 60) % 24)  # ステップ数を時間に変換
        time_slot = current_hour // 4
        days_elapsed = int((self.episode_step / 60) / 24)
        day_of_week = days_elapsed % 7
        day_type = 'weekend' if day_of_week >= 5 else 'weekday'
        key = (time_slot, day_type, severity, incident_h3)

        # 1. 実績ベースの事前計算モデルから検索
        hospital_probs = self.hospital_selection_model.get(key)

        if hospital_probs:
            if self.verbose_logging:
                print(f"[INFO] 事前計算モデル（実績ベース）を使用: {len(hospital_probs)}候補")
        else:
            # 2. 静的フォールバックモデルから検索
            if hasattr(self, 'static_fallback_model'):
                hospital_probs = self.static_fallback_model.get(severity, {}).get(incident_h3)
                if hospital_probs:
                     if self.verbose_logging:
                        print(f"[INFO] ★静的フォールバックモデル★を使用: {severity}, 候補数: {len(hospital_probs)}")
                else:
                    # 静的フォールバックにもない場合は、最終手段として決定論的選択
                    if self.verbose_logging:
                        print(f"[INFO] フォールバックモデルにも候補なし、決定論的選択にフォールバック")
                    return self._select_hospital_deterministic(incident_h3, severity)
            else:
                 # 完全にフォールバック
                if self.verbose_logging:
                    print(f"[INFO] 確率モデルなし、決定論的選択にフォールバック")
                return self._select_hospital_deterministic(incident_h3, severity)

        # 確率的選択の実行
        selected_hospital = self._probabilistic_selection(hospital_probs)
        
        # 統計の更新（選択方法も記録）
        if selected_hospital:
            # 選択方法を判定
            if key in self.hospital_selection_model:
                selection_method = 'probabilistic_success'
            elif hasattr(self, 'static_fallback_model') and self.static_fallback_model.get(severity, {}).get(incident_h3):
                selection_method = 'static_fallback_used'
            else:
                selection_method = 'deterministic_fallback'
            
            self._update_selection_statistics(selected_hospital, severity, selection_method)

        return selected_hospital
    
    def _select_hospital_deterministic(self, incident_h3: str, severity: str) -> Optional[str]:
        """決定論的な病院選択（validation_simulation互換）"""
        severe_conditions = ['重症', '重篤']
        
        if severity in severe_conditions:
            # 重症・重篤: 3次救急から選択
            if self.tertiary_hospitals:
                # 距離15km以内の3次救急から選択
                candidates = []
                inc_lat, inc_lon = h3.cell_to_latlng(incident_h3)
                
                for hospital_h3 in self.tertiary_hospitals:
                    try:
                        hosp_lat, hosp_lon = h3.cell_to_latlng(hospital_h3)
                        distance = self._calculate_distance(inc_lat, inc_lon, hosp_lat, hosp_lon)
                        if distance <= 15.0:  # 15km以内
                            candidates.append((hospital_h3, distance))
                    except:
                        continue
                
                if candidates:
                    # 上位3候補からランダム選択
                    candidates.sort(key=lambda x: x[1])
                    top_candidates = candidates[:3]
                    selected = random.choice(top_candidates)[0]
                    
                    # 統計情報を更新する
                    self.statistics['hospital_selection_stats']['tertiary_selections'] += 1
                    self._update_hospital_selection_stats(severity, 'tertiary', 'deterministic_fallback')
                    if self.verbose_logging:
                        print(f"[INFO] {severity}: 3次救急を選択 {selected}")
                        
                    return selected
        
        # 軽症・中等症・死亡の場合、または3次救急が見つからない重症・重篤ケース：2次以下から探す
        if self.secondary_primary_hospitals:
            nearest_secondary = self._find_nearest_hospital(incident_h3, self.secondary_primary_hospitals)
            if nearest_secondary:
                self.statistics['hospital_selection_stats']['secondary_primary_selections'] += 1
                self._update_hospital_selection_stats(severity, 'secondary_primary', 'deterministic_fallback')
                if self.verbose_logging:
                    selection_reason = "2次以下優先" if severity not in severe_conditions else "3次救急見つからず2次以下で代用"
                    print(f"[INFO] {severity}: {selection_reason} {nearest_secondary}")
                return nearest_secondary

        # それでも見つからない場合：軽症・中等症・死亡なら3次から探す
        if severity not in severe_conditions and self.tertiary_hospitals:
            nearest_tertiary = self._find_nearest_hospital(incident_h3, self.tertiary_hospitals)
            if nearest_tertiary:
                self.statistics['hospital_selection_stats']['tertiary_selections'] += 1
                self._update_hospital_selection_stats(severity, 'tertiary', 'deterministic_fallback')
                if self.verbose_logging:
                    print(f"[INFO] {severity}: 2次以下見つからず3次救急で代用 {nearest_tertiary}")
                return nearest_tertiary
                
        # 全ての候補を探しても見つからない場合
        self.statistics['hospital_selection_stats']['no_hospital_found'] += 1
        self._update_hospital_selection_stats(severity, 'no_hospital_found', 'error_fallback')
        if self.verbose_logging:
            print(f"[WARN] {severity}: 病院が見つかりませんでした")
        return None
    
    def _probabilistic_selection(self, hospital_probs: Dict[str, float]) -> str:
        """確率分布に基づいて病院を選択（validation_simulation互換）"""
        
        if not hospital_probs:
            return None
        
        # NumPyの確率的選択を使用
        hospitals = list(hospital_probs.keys())
        probabilities = list(hospital_probs.values())
        
        # 確率値の型を修正
        try:
            probabilities = [float(p) for p in probabilities]
        except (ValueError, TypeError) as e:
            print(f"[ERROR] 確率値の変換エラー: {e}")
            print(f"[ERROR] 問題のある確率値: {[(h, type(p), p) for h, p in hospital_probs.items()]}")
            return None
        
        # 正規化
        prob_sum = sum(probabilities)
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]
        else:
            # 全て同じ確率
            probabilities = [1.0 / len(hospitals)] * len(hospitals)
        
        # 確率的選択
        selected_hospital = np.random.choice(hospitals, p=probabilities)
        
        return selected_hospital
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """2点間の距離をhaversine公式で計算（km単位、validation_simulation互換）"""
        R = 6371  # 地球の半径（km）
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance
    
    def _update_selection_statistics(self, hospital_h3: str, severity: str, method: str = 'unknown'):
        """選択統計の更新（validation_simulation互換）"""
        
        # 病院種別の判定
        if hospital_h3 in self.tertiary_hospitals:
            selection_type = 'tertiary'
            self.statistics['hospital_selection_stats']['tertiary_selections'] += 1
        elif hospital_h3 in self.secondary_primary_hospitals:
            selection_type = 'secondary_primary'
            self.statistics['hospital_selection_stats']['secondary_primary_selections'] += 1
        else:
            selection_type = 'default'
        
        # 詳細統計の更新
        self._update_hospital_selection_stats(severity, selection_type, method)
    
    def _find_nearest_hospital(self, incident_h3: str, hospital_candidates: set) -> str:
        """指定された病院候補群から最寄りの病院を検索（validation_simulation互換）"""
        if not hospital_candidates:
            return incident_h3
        
        min_time = float('inf')
        nearest_hospital = list(hospital_candidates)[0]
        
        for hospital_h3 in hospital_candidates:
            travel_time = self._calculate_travel_time_for_phase(incident_h3, hospital_h3, phase='transport')
            if travel_time < min_time:
                min_time = travel_time
                nearest_hospital = hospital_h3
        
        return nearest_hospital
    
    def _update_hospital_selection_stats(self, severity: str, selection_type: str, method: str = 'unknown'):
        """統計情報の詳細化（validation_simulation互換）"""
        if severity not in self.statistics['hospital_selection_stats']['by_severity']:
            self.statistics['hospital_selection_stats']['by_severity'][severity] = {
                'tertiary': 0,
                'secondary_primary': 0,
                'default': 0,
                'probabilistic_success': 0,
                'deterministic_fallback': 0,
                'static_fallback_used': 0,
                'error_fallback': 0
            }
        
        # 病院種別の統計更新
        if selection_type == 'tertiary':
            self.statistics['hospital_selection_stats']['by_severity'][severity]['tertiary'] += 1
        elif selection_type == 'secondary_primary':
            self.statistics['hospital_selection_stats']['by_severity'][severity]['secondary_primary'] += 1
        elif selection_type == 'default':
            self.statistics['hospital_selection_stats']['by_severity'][severity]['default'] += 1
        
        # 選択方法の統計更新
        if method in self.statistics['hospital_selection_stats']['by_severity'][severity]:
            self.statistics['hospital_selection_stats']['by_severity'][severity][method] += 1
    
    def _calculate_travel_time(self, from_h3: str, to_h3: str) -> float:
        """
        移動時間を計算（秒単位）
        ValidationSimulatorのget_travel_timeと同じロジックを使用
        デフォルトでresponseフェーズを使用
        """
        return self._calculate_travel_time_for_phase(from_h3, to_h3, phase='response')
    
    def _calculate_travel_time_for_phase(self, from_h3: str, to_h3: str, phase: str = 'response') -> float:
        """
        指定されたフェーズの移動時間を計算（秒単位）
        ValidationSimulatorのget_travel_timeと同じロジック
        """
        from_idx = self.grid_mapping.get(from_h3)
        to_idx = self.grid_mapping.get(to_h3)
        
        if from_idx is None or to_idx is None:
            # グリッドマッピングにない場合のフォールバック
            return 600.0  # デフォルト10分
        
        # 移動時間行列から取得
        current_travel_time_matrix = self.travel_time_matrices.get(phase)
        
        if current_travel_time_matrix is None:
            # 指定フェーズの行列がない場合
            return 600.0  # デフォルト10分
        
        try:
            travel_time = current_travel_time_matrix[from_idx, to_idx]
            
            # 異常値チェック（ValidationSimulatorにはないが、安全のため）
            if travel_time <= 0 or travel_time > 3600:  # 1時間以上は異常
                return 600.0  # デフォルト10分
            
            return travel_time
        except:
            return 600.0  # エラー時のデフォルト
    

    
    def _calculate_reward(self, dispatch_result: Dict) -> float:
        """報酬を計算（RewardDesignerに完全委譲）"""
        if not dispatch_result['success']:
            # 失敗の種類に応じてペナルティを取得
            if dispatch_result.get('reason') == 'no_pending_call':
                return 0.0  # 事案なしは報酬なし
            elif dispatch_result.get('reason') == 'ambulance_busy':
                return self.reward_designer.get_failure_penalty('no_available')
            else:
                return self.reward_designer.get_failure_penalty('dispatch')
        
        # 成功時の報酬計算
        severity = dispatch_result['severity']
        response_time = dispatch_result['response_time']
        
        # カバレッジ影響の計算（簡易版）
        coverage_impact = self._calculate_coverage_impact(dispatch_result.get('ambulance_id'))
        
        # 追加情報（教師との一致など）
        additional_info = {
            'matched_teacher': dispatch_result.get('matched_teacher', False),
            'distance_rank': dispatch_result.get('distance_rank', None)
        }
        
        # RewardDesignerで報酬計算
        reward = self.reward_designer.calculate_step_reward(
            severity=severity,
            response_time=response_time,
            coverage_impact=coverage_impact,
            additional_info=additional_info
        )
        
        # デバッグ用ログ（最初の数回のみ）
        if hasattr(self, '_debug_reward_count'):
            self._debug_reward_count += 1
        else:
            self._debug_reward_count = 1
            
        if self._debug_reward_count <= 5:
            print(f"[報酬デバッグ] 傷病度: {severity}, 応答時間: {response_time/60:.1f}分, 報酬: {reward:.2f}")
            print(f"  - カバレッジ影響: {coverage_impact:.3f}")
            print(f"  - 教師一致: {additional_info.get('matched_teacher', False)}")
        
        return reward
    
    def _update_statistics(self, dispatch_result: Dict):
        """統計情報を更新（拡張版）"""
        if not dispatch_result['success']:
            self.episode_stats['failed_dispatches'] += 1
            return
        
        self.episode_stats['total_dispatches'] += 1
        
        # 基本的な応答時間統計
        rt_minutes = dispatch_result['response_time_minutes']
        self.episode_stats['response_times'].append(rt_minutes)
        
        # 傷病度別統計
        severity = dispatch_result['severity']
        if severity not in self.episode_stats['response_times_by_severity']:
            self.episode_stats['response_times_by_severity'][severity] = []
        self.episode_stats['response_times_by_severity'][severity].append(rt_minutes)
        
        # 閾値達成率
        if rt_minutes <= 6.0:
            self.episode_stats['achieved_6min'] += 1
        if rt_minutes <= 13.0:
            self.episode_stats['achieved_13min'] += 1
        
        # 重症系の6分達成率
        if is_severe_condition(severity):
            self.episode_stats['critical_total'] += 1
            if rt_minutes <= 6.0:
                self.episode_stats['critical_6min'] += 1
        
        # 拡張統計の更新
        self._update_extended_statistics(dispatch_result)
    
    def _update_extended_statistics(self, dispatch_result: Dict):
        """拡張統計情報の更新"""
        try:
            ambulance_id = dispatch_result['ambulance_id']
            severity = dispatch_result['severity']
            rt_minutes = dispatch_result['response_time_minutes']
            
            # 救急車稼働統計
            if ambulance_id not in self.episode_stats['ambulance_utilization']['total_dispatches_by_ambulance']:
                self.episode_stats['ambulance_utilization']['total_dispatches_by_ambulance'][ambulance_id] = 0
            self.episode_stats['ambulance_utilization']['total_dispatches_by_ambulance'][ambulance_id] += 1
            
            # 時間別統計
            if self.pending_call and 'datetime' in self.pending_call:
                hour = self.pending_call['datetime'].hour
                self.episode_stats['temporal_patterns']['hourly_call_counts'][hour] += 1
                self.episode_stats['temporal_patterns']['hourly_response_times'][hour].append(rt_minutes)
                self.episode_stats['ambulance_utilization']['hourly_counts'][hour] += 1
            
            # 空間統計
            if self.pending_call and 'h3_index' in self.pending_call:
                h3_area = self.pending_call['h3_index']
                self.episode_stats['spatial_coverage']['areas_served'].add(h3_area)
                
                if h3_area not in self.episode_stats['spatial_coverage']['response_time_by_area']:
                    self.episode_stats['spatial_coverage']['response_time_by_area'][h3_area] = []
                    self.episode_stats['spatial_coverage']['call_density_by_area'][h3_area] = 0
                
                self.episode_stats['spatial_coverage']['response_time_by_area'][h3_area].append(rt_minutes)
                self.episode_stats['spatial_coverage']['call_density_by_area'][h3_area] += 1
            
            # 傷病度別詳細統計
            severity_category = self._get_severity_category(severity)
            if severity_category in self.episode_stats['severity_detailed_stats']:
                stats = self.episode_stats['severity_detailed_stats'][severity_category]
                stats['count'] += 1
                stats['response_times'].append(rt_minutes)
                if rt_minutes <= 6.0:
                    stats['under_6min'] += 1
                if rt_minutes <= 13.0:
                    stats['under_13min'] += 1
            
            # 移動距離の推定（簡易版）
            if hasattr(self, 'ambulance_states') and ambulance_id in self.ambulance_states:
                amb_state = self.ambulance_states[ambulance_id]
                if self.pending_call and 'h3_index' in self.pending_call:
                    # 距離行列から移動距離を取得（可能な場合）
                    estimated_distance = self._estimate_travel_distance(
                        amb_state['current_h3'], 
                        self.pending_call['h3_index']
                    )
                    self.episode_stats['efficiency_metrics']['total_distance'] += estimated_distance
                    
        except Exception as e:
            # 統計更新エラーは致命的ではないため、警告のみ出力
            print(f"統計更新でエラー: {e}")
    
    def _get_severity_category(self, severity: str) -> str:
        """傷病度から標準カテゴリに変換"""
        if severity in ['重篤', '重症', '死亡']:
            return 'critical'
        elif severity in ['中等症']:
            return 'moderate'
        elif severity in ['軽症']:
            return 'mild'
        else:
            return 'mild'  # デフォルト
    
    def _estimate_travel_distance(self, from_h3: str, to_h3: str) -> float:
        """移動距離の推定（km）"""
        try:
            from_idx = self.grid_mapping.get(from_h3)
            to_idx = self.grid_mapping.get(to_h3)
            
            if from_idx is not None and to_idx is not None and hasattr(self, 'travel_distance_matrix'):
                distance = self.travel_distance_matrix[from_idx, to_idx]
                return distance / 1000.0  # メートルからキロメートルに変換
            else:
                # フォールバック: 移動時間から距離を推定（平均時速30km/h）
                travel_time_seconds = self._calculate_travel_time(from_h3, to_h3)
                travel_time_hours = travel_time_seconds / 3600.0
                return travel_time_hours * 30.0  # 30km/h
        except:
            return 5.0  # デフォルト5km
    
    def get_episode_statistics(self) -> Dict:
        """エピソード統計を取得（RewardDesignerと連携、ハイブリッドモード対応）"""
        stats = self.episode_stats.copy()
        
        # 集計値の計算
        if stats['response_times']:
            total_calls = len(stats['response_times'])
            stats['summary'] = {
                'total_calls': total_calls,
                'mean_response_time': np.mean(stats['response_times']),
                'median_response_time': np.median(stats['response_times']),
                '95th_percentile_response_time': np.percentile(stats['response_times'], 95),
                '6min_achievement_rate': stats['achieved_6min'] / total_calls,
                '13min_achievement_rate': stats['achieved_13min'] / total_calls,
            }
            
            # 重症系達成率
            if stats['critical_total'] > 0:
                stats['summary']['critical_6min_rate'] = stats['critical_6min'] / stats['critical_total']
            else:
                stats['summary']['critical_6min_rate'] = 0.0
        
        # 救急車稼働率の計算
        if stats['ambulance_utilization']['total_dispatches_by_ambulance']:
            dispatches = list(stats['ambulance_utilization']['total_dispatches_by_ambulance'].values())
            stats['ambulance_utilization']['mean'] = np.mean(dispatches)
            stats['ambulance_utilization']['max'] = np.max(dispatches)
            stats['ambulance_utilization']['std'] = np.std(dispatches)
        
        # エリアカバレッジ
        stats['spatial_coverage']['areas_served'] = len(stats['spatial_coverage']['areas_served'])
        
        # 効率性メトリクス
        if stats['total_dispatches'] > 0:
            stats['efficiency_metrics']['distance_per_call'] = (
                stats['efficiency_metrics']['total_distance'] / stats['total_dispatches']
            )
        
        # ハイブリッドモード統計の追加
        if self.hybrid_mode:
            stats['hybrid_stats'] = {
                'direct_dispatch_count': self.direct_dispatch_count,
                'ppo_dispatch_count': self.ppo_dispatch_count,
                'direct_ratio': self.direct_dispatch_count / max(1, self.direct_dispatch_count + self.ppo_dispatch_count)
            }
        
        # エピソード報酬を計算
        if self.reward_designer:
            stats['episode_reward'] = self.reward_designer.calculate_episode_reward(stats)
        
        return stats
    
    # ★★★【削除】_advance_to_next_callメソッド★★★
    # 固定時間ステップ制では不要（step()内のイベント処理に統合）
    
    # ★★★【削除】_update_ambulance_availabilityメソッド★★★
    # 固定時間ステップ制では不要（イベントキューの復帰イベントで処理）
    
    def _get_max_wait_time(self, severity: str) -> int:
        """傷病度に応じた最大待機時間（分）- 現実的な救急システム"""
        if severity in ['重篤', '重症']:
            return 10  # 重症は10分で他地域から緊急応援
        elif severity == '中等症':
            return 20  # 中等症は20分で他地域応援
        else:  # 軽症
            return 45  # 軽症は45分で他地域応援（または搬送見送り）
    
    def _handle_unresponsive_call(self, call: Dict, wait_time: int):
        """対応不能事案の処理 - 現実的な救急システム"""
        severity = call['severity']
        
        # 重症度別の対応決定
        if severity in ['重篤', '重症']:
            response_type = 'emergency_support'  # 緊急応援（高速応答）
            support_time = 15 + wait_time  # 応援隊の到着時間（分）
            print(f"🚨 重症緊急応援: {severity} ({wait_time}分待機) → 他地域緊急隊が{support_time}分で対応")
        elif severity == '中等症':
            response_type = 'standard_support'  # 標準応援
            support_time = 25 + wait_time
            print(f"⚡ 中等症応援: {severity} ({wait_time}分待機) → 他地域隊が{support_time}分で対応")
        else:  # 軽症
            # 軽症は状況に応じて対応を分岐
            if wait_time > 60:
                response_type = 'transport_cancel'  # 搬送見送り
                support_time = None
                print(f"📋 軽症搬送見送り: {severity} ({wait_time}分待機) → 患者自力搬送または待機")
            else:
                response_type = 'delayed_support'  # 遅延応援
                support_time = 40 + wait_time
                print(f"🕐 軽症遅延応援: {severity} ({wait_time}分待機) → 他地域隊が{support_time}分で対応")
        
        # 対応不能事案として記録
        unhandled_call = {
            'call_id': call['id'],
            'severity': call['severity'],
            'wait_time': wait_time,
            'location': call.get('location', None),
            'handled_by': response_type,
            'support_time': support_time,
            'total_time': support_time if support_time else wait_time
        }
        self.unhandled_calls.append(unhandled_call)
        
        # 重症度別統計の更新
        self._update_unhandled_statistics(unhandled_call)
        
        # 重症度別ペナルティ（RewardDesignerに委譲）
        if self.reward_designer:
            penalty = self.reward_designer.calculate_unhandled_penalty(call['severity'], wait_time, response_type)
            if not hasattr(self, 'unhandled_penalty_total'):
                self.unhandled_penalty_total = 0
            self.unhandled_penalty_total += penalty
    
    def _update_unhandled_statistics(self, unhandled_call: Dict):
        """対応不能事案の詳細統計更新"""
        severity = unhandled_call['severity']
        response_type = unhandled_call['handled_by']
        
        # 重症度別統計
        if severity in ['重篤', '重症']:
            self.episode_stats['critical_unhandled'] = getattr(self.episode_stats, 'critical_unhandled', 0) + 1
            if response_type == 'emergency_support':
                self.episode_stats['critical_emergency_support'] = getattr(self.episode_stats, 'critical_emergency_support', 0) + 1
        elif severity == '中等症':
            self.episode_stats['moderate_unhandled'] = getattr(self.episode_stats, 'moderate_unhandled', 0) + 1
            if response_type == 'standard_support':
                self.episode_stats['moderate_standard_support'] = getattr(self.episode_stats, 'moderate_standard_support', 0) + 1
        else:  # 軽症
            self.episode_stats['mild_unhandled'] = getattr(self.episode_stats, 'mild_unhandled', 0) + 1
            if response_type == 'transport_cancel':
                self.episode_stats['mild_transport_cancel'] = getattr(self.episode_stats, 'mild_transport_cancel', 0) + 1
            elif response_type == 'delayed_support':
                self.episode_stats['mild_delayed_support'] = getattr(self.episode_stats, 'mild_delayed_support', 0) + 1
        
        # 全体統計
        self.episode_stats['unhandled_calls'] = getattr(self.episode_stats, 'unhandled_calls', 0) + 1
        self.episode_stats['total_support_time'] = getattr(self.episode_stats, 'total_support_time', 0) + unhandled_call.get('total_time', 0)
    
    def _calculate_coverage_impact(self, ambulance_id: Optional[int]) -> float:
        """
        カバレッジへの影響を簡易計算
        
        Returns:
            0.0-1.0の範囲（0=影響なし、1=大きな影響）
        """
        if ambulance_id is None:
            return 0.0
        
        # 利用可能な救急車の割合から簡易計算
        available_count = sum(1 for amb in self.ambulance_states.values() 
                             if amb['status'] == 'available')
        total_count = len(self.ambulance_states)
        
        if total_count == 0:
            return 0.0
        
        utilization_rate = 1.0 - (available_count / total_count)
        
        # 稼働率が高いほど、1台の出動の影響が大きい
        if utilization_rate > 0.8:
            return 0.8
        elif utilization_rate > 0.6:
            return 0.5
        elif utilization_rate > 0.4:
            return 0.3
        else:
            return 0.1
    

    
    def _is_episode_done(self) -> bool:
        """エピソード終了判定（時間ベースと事案数ベース）"""
        # イベントキューが空で、pending_callもない場合は終了
        if not self.event_queue and self.pending_call is None:
            return True
        
        # エピソード時間制限（時間ベース）
        episode_hours = self.config['data'].get('episode_duration_hours', 24)
        max_time_seconds = episode_hours * 3600.0
        if self.current_time_seconds >= max_time_seconds:
            return True
        
        # 最大ステップ数制限（ステップ数ベース）
        max_steps = self.config.get('data', {}).get('max_steps_per_episode') or \
                    self.config.get('max_steps_per_episode') or \
                    10000  # デフォルト値（1分×10000 = 約7日）
        if self.episode_step >= max_steps:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """
        現在の観測を取得
        
        時間管理の統一:
        - current_time（秒単位）のみを使用
        - episode_stepは削除（混乱を避けるため）
        """
        state_dict = {
            'ambulances': self.ambulance_states,
            'pending_call': self.pending_call,
            'current_time': self.current_time_seconds,  # 経過秒数（唯一の時間ソース）
            'time_of_day': self._get_time_of_day()
        }
        
        # 初期化時に作成したインスタンスをそのまま使用する
        observation = self.state_encoder.encode_state(state_dict)
        
        return observation
    
    def _get_time_of_day(self) -> int:
        """現在の時刻を取得（0-23）"""
        if self.pending_call and 'datetime' in self.pending_call:
            return self.pending_call['datetime'].hour
        return 12  # デフォルト
    
    def _init_episode_stats(self) -> Dict:
        """エピソード統計の初期化（拡張版）"""
        return {
            # 基本統計
            'total_dispatches': 0,
            'failed_dispatches': 0,
            'response_times': [],
            'response_times_by_severity': {},
            'achieved_6min': 0,
            'achieved_13min': 0,
            'critical_total': 0,
            'critical_6min': 0,
            
            # 対応不能事案統計（詳細版）
            'unhandled_calls': 0,
            'critical_unhandled': 0,
            'moderate_unhandled': 0,
            'mild_unhandled': 0,
            'unhandled_penalty_total': 0.0,
            
            # 他地域応援統計
            'critical_emergency_support': 0,    # 重症緊急応援
            'moderate_standard_support': 0,     # 中等症標準応援
            'mild_delayed_support': 0,          # 軽症遅延応援
            'mild_transport_cancel': 0,         # 軽症搬送見送り
            'total_support_time': 0,            # 総応援対応時間
            
            # 救急車稼働統計
            'ambulance_utilization': {
                'hourly_counts': [0] * 24,  # 時間別出動回数
                'total_dispatches_by_ambulance': {},  # 救急車別出動回数
                'busy_time_by_ambulance': {},  # 救急車別稼働時間
            },
            
            # 空間統計
            'spatial_coverage': {
                'areas_served': set(),  # サービス提供エリア
                'response_time_by_area': {},  # エリア別応答時間
                'call_density_by_area': {},  # エリア別事案密度
            },
            
            # 時間パターン
            'temporal_patterns': {
                'hourly_call_counts': [0] * 24,  # 時間別事案数
                'hourly_response_times': {i: [] for i in range(24)},  # 時間別応答時間
            },
            
            # 効率性メトリクス
            'efficiency_metrics': {
                'total_distance': 0.0,  # 総移動距離
                'distance_per_call': 0.0,  # 事案あたり移動距離
                'travel_time_accuracy': [],  # 移動時間予測精度
            },
            
            # 傷病度別詳細統計
            'severity_detailed_stats': {
                'critical': {'count': 0, 'under_6min': 0, 'under_13min': 0, 'response_times': []},
                'moderate': {'count': 0, 'under_6min': 0, 'under_13min': 0, 'response_times': []},
                'mild': {'count': 0, 'under_6min': 0, 'under_13min': 0, 'response_times': []},
            }
        }
    
    def get_action_mask(self) -> np.ndarray:
        """利用可能な行動のマスクを取得"""
        mask = np.zeros(self.action_dim, dtype=bool)
        
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_id < self.action_dim and amb_state['status'] == 'available':
                mask[amb_id] = True
        
        return mask
 
    def get_best_action_for_call(self) -> Optional[int]:
        """
        現在の事案に対して最適な救急車（行動）を選択
        学習初期はこれを教師として使用できる
        """
        if self.pending_call is None:
            return None
        
        best_action = None
        min_travel_time = float('inf')
        
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_state['status'] != 'available':
                continue
            
            # 移動時間を計算
            travel_time = self._calculate_travel_time(
                amb_state['current_h3'],
                self.pending_call['h3_index']
            )
            
            if travel_time < min_travel_time:
                min_travel_time = travel_time
                best_action = amb_id
        
        return best_action
    
    def render(self, mode: str = 'human'):
        """環境の可視化（オプション）"""
        if mode == 'human':
            print(f"\nStep {self.episode_step}")
            if self.pending_call:
                print(f"  事案: {self.pending_call['severity']} at {self.pending_call['h3_index']}")
            
            available_count = sum(1 for a in self.ambulance_states.values() if a['status'] == 'available')
            print(f"  利用可能救急車: {available_count}/{len(self.ambulance_states)}")
            
            if self.episode_stats['total_dispatches'] > 0:
                avg_rt = np.mean(self.episode_stats['response_times'])
                rate_6min = self.episode_stats['achieved_6min'] / self.episode_stats['total_dispatches'] * 100
                print(f"  平均応答時間: {avg_rt:.1f}分")
                print(f"  6分達成率: {rate_6min:.1f}%")
    
    def _create_virtual_ambulances_if_needed(self, actual_ambulances: pd.DataFrame) -> pd.DataFrame:
        """
        必要に応じて仮想救急車を作成
        
        Args:
            actual_ambulances: 実際の救急車データ
            
        Returns:
            仮想救急車を含む救急車データ
        """
        # 設定から仮想救急車パラメータを取得
        data_config = self.config.get('data', {})
        virtual_count = data_config.get('virtual_ambulances', None)
        multiplier = data_config.get('ambulance_multiplier', 1.0)
        
        if virtual_count and virtual_count > 0:
            # 仮想救急車を追加（既存の救急車は保持）
            target_count = len(actual_ambulances) + virtual_count
            print(f"  仮想救急車追加: {len(actual_ambulances)}台 → {target_count}台 (追加: {virtual_count}台)")
            return self._create_virtual_ambulances(actual_ambulances, target_count)
        elif multiplier > 1.0:
            # 既存の救急車を複製
            target_count = int(len(actual_ambulances) * multiplier)
            print(f"  救急車複製: {len(actual_ambulances)}台 → {target_count}台 (倍率: {multiplier})")
            return self._duplicate_ambulances(actual_ambulances, target_count)
        else:
            # 仮想救急車は作成しない
            return actual_ambulances
    
    def _create_virtual_ambulances(self, actual_ambulances: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """
        既存の救急署から均等に仮想救急車を追加
        
        Args:
            actual_ambulances: 実際の救急車データ
            target_count: 目標救急車数
            
        Returns:
            仮想救急車を含む救急車データ
        """
        result_ambulances = actual_ambulances.copy()
        
        # 救急署ごとにグループ化（H3インデックスで）
        stations = {}
        for _, amb in actual_ambulances.iterrows():
            try:
                # H3インデックスを計算
                lat = float(amb['latitude'])
                lng = float(amb['longitude'])
                station_h3 = h3.latlng_to_cell(lat, lng, 9)
                
                if station_h3 not in stations:
                    stations[station_h3] = []
                stations[station_h3].append(amb)
            except Exception as e:
                print(f"警告: 救急車のH3計算エラー: {e}")
                continue
        
        print(f"  救急署数: {len(stations)}署")
        print(f"  実際の救急車数: {len(actual_ambulances)}台")
        
        # 各署に仮想救急車を均等に追加
        virtual_id_counter = len(actual_ambulances)
        
        while len(result_ambulances) < target_count:
            for station_h3, station_ambs in stations.items():
                if len(result_ambulances) >= target_count:
                    break
                
                # この署に仮想救急車を1台追加
                base_ambulance = station_ambs[0]  # 代表的な救急車をベースにする
                virtual_ambulance = base_ambulance.copy()
                
                # 仮想救急車の識別情報を更新
                virtual_ambulance['id'] = f"virtual_{virtual_id_counter}"
                virtual_ambulance['name'] = f"virtual_team_{virtual_id_counter}"
                virtual_ambulance['team_name'] = f"virtual_team_{virtual_id_counter}"
                virtual_ambulance['is_virtual'] = True
                
                # 同じ署の位置を使用（位置は変更しない）
                # 必要に応じて微細な位置調整も可能
                lat_offset = np.random.uniform(-0.001, 0.001)  # 約100m以内
                lng_offset = np.random.uniform(-0.001, 0.001)  # 約100m以内
                
                virtual_ambulance['latitude'] = float(virtual_ambulance['latitude']) + lat_offset
                virtual_ambulance['longitude'] = float(virtual_ambulance['longitude']) + lng_offset
                
                # 座標の有効性チェック
                if (-90 <= virtual_ambulance['latitude'] <= 90 and 
                    -180 <= virtual_ambulance['longitude'] <= 180):
                    
                    # 仮想救急車を追加
                    result_ambulances = pd.concat([result_ambulances, virtual_ambulance.to_frame().T], 
                                                ignore_index=True)
                    virtual_id_counter += 1
                    
                    print(f"  仮想救急車{virtual_id_counter-1}を署{station_h3}に追加")
        
        return result_ambulances
    
    def _duplicate_ambulances(self, actual_ambulances: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """
        既存の救急車を複製
        
        Args:
            actual_ambulances: 実際の救急車データ
            target_count: 目標救急車数
            
        Returns:
            複製された救急車データ
        """
        result_ambulances = actual_ambulances.copy()
        
        # 複製カウンタ
        duplicate_counter = 0
        
        while len(result_ambulances) < target_count:
            for _, base_ambulance in actual_ambulances.iterrows():
                if len(result_ambulances) >= target_count:
                    break
                
                # 既存の救急車を複製
                duplicate_ambulance = base_ambulance.copy()
                
                # 複製救急車の識別情報を更新
                duplicate_ambulance['id'] = f"duplicate_{duplicate_counter}"
                duplicate_ambulance['name'] = f"複製救急車{duplicate_counter}"
                
                # 位置を少しずらす（半径300m以内のランダムな位置）
                lat_offset = np.random.uniform(-0.0027, 0.0027)  # 約300m
                lng_offset = np.random.uniform(-0.0027, 0.0027)  # 約300m
                
                duplicate_ambulance['latitude'] = float(duplicate_ambulance['latitude']) + lat_offset
                duplicate_ambulance['longitude'] = float(duplicate_ambulance['longitude']) + lng_offset
                
                # 座標の有効性チェック
                if (-90 <= duplicate_ambulance['latitude'] <= 90 and 
                    -180 <= duplicate_ambulance['longitude'] <= 180):
                    
                    # 複製救急車を追加
                    result_ambulances = pd.concat([result_ambulances, duplicate_ambulance.to_frame().T], 
                                                ignore_index=True)
                    duplicate_counter += 1
                    
                    if len(result_ambulances) >= target_count:
                        break
        
        return result_ambulances
    
    # ★★★【イベント処理メソッドの追加】★★★
    def _schedule_event(self, event: Event):
        """イベントをキューに追加"""
        heapq.heappush(self.event_queue, event)
    
    def _process_next_event(self) -> Optional[Event]:
        """次のイベントを処理して返す"""
        if not self.event_queue:
            return None
        
        event = heapq.heappop(self.event_queue)
        
        # ★★★【重要】★★★
        # シミュレーションの現在時刻を、処理するイベントの時刻まで進める
        self.current_time_seconds = event.time
        
        # イベントタイプに応じた処理
        if event.event_type == EventType.NEW_CALL:
            self._handle_new_call_event(event)
        elif event.event_type == EventType.AMBULANCE_AVAILABLE:
            self._handle_ambulance_return_event(event)
        
        return event
    
    def _handle_new_call_event(self, event: Event):
        """NEW_CALLイベントの処理"""
        call = event.data['call']
        self.pending_call = call
        # 事案の発生時刻を記録（復帰時刻計算に使用）
        self.call_start_times[call['id']] = self.current_time_seconds
    
    def _handle_ambulance_return_event(self, event: Event):
        """救急車復帰イベントの処理"""
        amb_id = event.data['ambulance_id']
        station_h3 = event.data['station_h3']
        
        if amb_id in self.ambulance_states:
            self.ambulance_states[amb_id]['status'] = 'available'
            self.ambulance_states[amb_id]['current_h3'] = station_h3
    # ★★★【イベント処理メソッドここまで】★★★