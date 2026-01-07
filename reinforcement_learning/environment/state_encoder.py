"""
state_encoder.py
複雑な環境状態を固定長ベクトルに変換
移動時間行列を統合し、実際の道路網での移動時間を特徴量として使用
"""

import numpy as np
import h3
from typing import Dict, List, Optional
import torch
import sys
import os

# 統一された傷病度定数をインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from constants import SEVERITY_INDICES

class StateEncoder:
    """
    EMS環境の状態をニューラルネットワーク用のベクトルに変換
    実際の移動時間行列を使用して空間的関係を正確に表現
    """
    
    def __init__(self, config: Dict, max_ambulances: int = 192,
                 travel_time_matrix: Optional[np.ndarray] = None,
                 grid_mapping: Optional[Dict] = None):
        """
        Args:
            config: 設定辞書
            max_ambulances: 最大救急車数
            travel_time_matrix: responseフェーズの移動時間行列
            grid_mapping: H3インデックスから行列インデックスへのマッピング
        """
        self.config = config
        
        # 移動時間行列とグリッドマッピング
        self.travel_time_matrix = travel_time_matrix
        self.grid_mapping = grid_mapping
        
        # 特徴量の次元設定（動的に設定可能）
        self.max_ambulances = max_ambulances
        # 救急車特徴量を4→5次元に拡張（移動時間を追加）
        self.ambulance_features = 5  # 位置x, 位置y, 状態, 出動回数, 事案現場までの移動時間
        self.incident_features = 10  # 位置、傷病度など
        self.temporal_features = 8  # 時間関連
        # ★★★【修正箇所①】★★★
        # 空間特徴量の次元を1つ追加（カバレッジ率）
        self.spatial_features = 21  # 空間統計（改良版）+ カバレッジ率
        
        # 傷病度のone-hotエンコーディング用
        self.severity_indices = SEVERITY_INDICES
        
        # ★★★【修正箇所】★★★
        # コンフィグからカバレッジの時間閾値を読み込む
        coverage_config = config.get('coverage_params', {})
        self.coverage_time_threshold = coverage_config.get('time_threshold_seconds', 600)
        
    def encode_state(self, state_dict: Dict, grid_mapping: Dict = None) -> np.ndarray:
        """
        状態辞書を固定長ベクトルに変換
        
        Args:
            state_dict: 環境の状態情報
            grid_mapping: H3インデックスとグリッドIDのマッピング（後方互換性のため残す）
            
        Returns:
            状態ベクトル
        """
        # grid_mappingが引数で渡された場合はそれを使用（後方互換性）
        if grid_mapping is None:
            grid_mapping = self.grid_mapping
            
        features = []
        
        # 1. 救急車の特徴量（移動時間を含む拡張版）
        ambulance_features = self._encode_ambulances_with_travel_time(
            state_dict['ambulances'], 
            state_dict.get('pending_call'),
            grid_mapping
        )
        features.append(ambulance_features)
        
        # 2. 事案の特徴量
        incident_features = self._encode_incident(
            state_dict.get('pending_call'), grid_mapping
        )
        features.append(incident_features)
        
        # 3. 時間的特徴量
        temporal_features = self._encode_temporal(
            state_dict.get('episode_step', 0),
            state_dict.get('time_of_day', 12)
        )
        features.append(temporal_features)
        
        # ★★★【修正箇所②】★★★
        # 4. 空間的特徴量（カバレッジ率を追加）
        spatial_features = self._encode_spatial_with_coverage(
            state_dict['ambulances'],
            state_dict.get('pending_call'),
            grid_mapping
        )
        features.append(spatial_features)
        
        # 全特徴量を結合
        state_vector = np.concatenate(features)
        
        # NaN値のチェックと修正
        if np.any(np.isnan(state_vector)):
            print(f"警告: StateEncoderでNaN値を検出しました")
            state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1.0, neginf=0.0)
        
        return state_vector.astype(np.float32)
    
    def _encode_ambulances_with_travel_time(self, ambulances: Dict, 
                                           incident: Optional[Dict],
                                           grid_mapping: Dict) -> np.ndarray:
        """救急車情報をエンコード（移動時間を含む）"""
        # 動的に設定された台数を使用
        features = np.zeros(self.max_ambulances * self.ambulance_features)
        
        # 事案がある場合、その位置のグリッドインデックスを取得
        incident_grid_idx = None
        if incident is not None and self.travel_time_matrix is not None and grid_mapping:
            try:
                incident_h3 = incident.get('h3_index')
                if incident_h3 and incident_h3 in grid_mapping:
                    incident_grid_idx = grid_mapping[incident_h3]
            except Exception as e:
                print(f"警告: 事案位置のグリッドインデックス取得失敗: {e}")
        
        for amb_id, amb_state in ambulances.items():
            if amb_id >= self.max_ambulances:
                break
            
            idx = amb_id * self.ambulance_features
            
            # H3インデックスを座標に変換
            try:
                lat, lng = h3.cell_to_latlng(amb_state['current_h3'])
            except:
                lat, lng = 35.6762, 139.6503  # デフォルト（東京）
            
            # 基本特徴量の設定（安全な正規化）
            features[idx] = (lat + 90.0) / 180.0  # 緯度を[0, 1]に正規化
            features[idx + 1] = (lng + 180.0) / 360.0  # 経度を[0, 1]に正規化
            features[idx + 2] = 1.0 if amb_state['status'] == 'available' else 0.0
            features[idx + 3] = min(amb_state.get('calls_today', 0) / 20.0, 1.0)  # 出動回数を正規化
            
            # 新規追加：事案現場までの実際の移動時間
            travel_time_minutes = 0.0
            if incident_grid_idx is not None and self.travel_time_matrix is not None:
                try:
                    amb_h3 = amb_state.get('current_h3')
                    if amb_h3 and amb_h3 in grid_mapping:
                        amb_grid_idx = grid_mapping[amb_h3]
                        # 移動時間行列から実際の移動時間を取得（秒）
                        travel_time_seconds = self.travel_time_matrix[amb_grid_idx, incident_grid_idx]
                        # 分に変換して正規化（0-30分を0-1にマッピング）
                        travel_time_minutes = min(travel_time_seconds / 60.0 / 30.0, 1.0)
                except Exception as e:
                    # エラー時はデフォルト値を使用
                    travel_time_minutes = 0.5
            
            features[idx + 4] = travel_time_minutes
        
        return features
    
    def _encode_incident(self, incident: Optional[Dict], grid_mapping: Dict) -> np.ndarray:
        """事案情報をエンコード"""
        features = np.zeros(self.incident_features)
        
        if incident is None:
            return features
        
        # 位置情報
        try:
            lat, lng = h3.cell_to_latlng(incident['h3_index'])
            features[0] = (lat + 90.0) / 180.0
            features[1] = (lng + 180.0) / 360.0
        except:
            features[0] = 0.5
            features[1] = 0.5
        
        # 傷病度（one-hot encoding）
        severity = incident.get('severity', '軽症')
        if severity in self.severity_indices:
            severity_idx = self.severity_indices[severity]
            if 2 + severity_idx < len(features):
                features[2 + severity_idx] = 1.0
        
        # 待機時間（正規化）
        wait_time = incident.get('wait_time', 0)
        features[8] = min(wait_time / 600.0, 1.0)  # 10分を1.0とする
        
        # 優先度スコア
        priority = incident.get('priority', 0.5)
        features[9] = priority
        
        return features
    
    def _encode_temporal(self, episode_step: int, time_of_day: float) -> np.ndarray:
        """時間的特徴量をエンコード"""
        features = np.zeros(self.temporal_features)
        
        # エピソード進行度
        max_steps = self.config.get('data', {}).get('episode_duration_hours', 24) * 60
        features[0] = min(episode_step / max_steps, 1.0)
        
        # 時刻（周期的エンコーディング）
        hour = time_of_day % 24
        features[1] = np.sin(2 * np.pi * hour / 24)
        features[2] = np.cos(2 * np.pi * hour / 24)
        
        # 時間帯カテゴリ（朝、昼、夕、夜）
        if 6 <= hour < 10:
            features[3] = 1.0  # 朝
        elif 10 <= hour < 17:
            features[4] = 1.0  # 昼
        elif 17 <= hour < 21:
            features[5] = 1.0  # 夕
        else:
            features[6] = 1.0  # 夜
        
        # 曜日情報（仮定：平日）
        features[7] = 1.0  # 平日フラグ
        
        return features
    
    def _encode_spatial_with_travel_time(self, ambulances: Dict, 
                                        incident: Optional[Dict],
                                        grid_mapping: Dict) -> np.ndarray:
        """
        空間的特徴量をエンコード（移動時間行列を使用した改良版）
        実際の道路網での移動時間統計を計算
        """
        features = np.zeros(20)  # 元の20次元の特徴量
        
        if incident is None or self.travel_time_matrix is None or grid_mapping is None:
            # 移動時間行列が利用できない場合は従来の方法にフォールバック
            return self._encode_spatial_fallback(ambulances, incident)
        
        # 事案位置のグリッドインデックスを取得
        try:
            incident_h3 = incident.get('h3_index')
            if not incident_h3 or incident_h3 not in grid_mapping:
                return self._encode_spatial_fallback(ambulances, incident)
            
            incident_grid_idx = grid_mapping[incident_h3]
        except Exception as e:
            print(f"警告: 空間特徴量計算でエラー: {e}")
            return self._encode_spatial_fallback(ambulances, incident)
        
        # 利用可能な救急車の移動時間を収集
        available_times = []
        all_times = []
        
        for amb_id, amb_state in ambulances.items():
            try:
                amb_h3 = amb_state.get('current_h3')
                if amb_h3 and amb_h3 in grid_mapping:
                    amb_grid_idx = grid_mapping[amb_h3]
                    travel_time_seconds = self.travel_time_matrix[amb_grid_idx, incident_grid_idx]
                    travel_time_minutes = travel_time_seconds / 60.0
                    
                    all_times.append(travel_time_minutes)
                    
                    if amb_state['status'] == 'available':
                        available_times.append(travel_time_minutes)
            except:
                continue
        
        # 統計量を計算
        if available_times:
            # 利用可能な救急車の統計
            features[0] = min(available_times) / 30.0  # 最短時間（30分で正規化）
            features[1] = np.mean(available_times) / 30.0  # 平均時間
            features[2] = np.median(available_times) / 30.0  # 中央値
            features[3] = np.std(available_times) / 10.0 if len(available_times) > 1 else 0  # 標準偏差
            features[4] = len(available_times) / max(len(ambulances), 1)  # 利用可能率
            
            # 時間帯別カウント（5分、10分、15分、20分以内）
            features[5] = sum(1 for t in available_times if t <= 5) / max(len(available_times), 1)
            features[6] = sum(1 for t in available_times if t <= 10) / max(len(available_times), 1)
            features[7] = sum(1 for t in available_times if t <= 15) / max(len(available_times), 1)
            features[8] = sum(1 for t in available_times if t <= 20) / max(len(available_times), 1)
        
        if all_times:
            # 全救急車の統計
            features[9] = min(all_times) / 30.0  # 全体最短時間
            features[10] = np.mean(all_times) / 30.0  # 全体平均時間
            features[11] = np.median(all_times) / 30.0  # 全体中央値
            features[12] = max(all_times) / 60.0  # 最長時間（60分で正規化）
            
            # 分位数
            features[13] = np.percentile(all_times, 25) / 30.0  # 第1四分位
            features[14] = np.percentile(all_times, 75) / 30.0  # 第3四分位
            
        # 救急車の稼働状況
        total_ambulances = len(ambulances)
        if total_ambulances > 0:
            available_count = sum(1 for a in ambulances.values() if a['status'] == 'available')
            busy_count = total_ambulances - available_count
            
            features[15] = available_count / total_ambulances  # 利用可能率
            features[16] = busy_count / total_ambulances  # 稼働率
            features[17] = available_count / 20.0  # 絶対数（20台で正規化）
            features[18] = min(available_count / 5.0, 1.0)  # 5台以上で飽和
            features[19] = 1.0 if available_count > 0 else 0.0  # 利用可能フラグ
        
        return features
    
    def _encode_spatial_with_coverage(self, ambulances: Dict, 
                                    incident: Optional[Dict],
                                    grid_mapping: Dict) -> np.ndarray:
        """
        空間的特徴量をエンコード。最後にカバレッジ率を追加する。
        """
        # 既存の空間特徴量（20次元）を計算
        features = np.zeros(self.spatial_features)  # 21次元の配列を初期化
        
        # 既存の空間特徴量計算を呼び出して最初の20次元を埋める
        existing_features = self._encode_spatial_with_travel_time(
            ambulances, incident, grid_mapping
        )
        features[:20] = existing_features
        
        # --- 新しいカバレッジ特徴量の計算 ---
        # 1. 利用可能な救急隊のH3インデックスを取得
        available_amb_h3s = [
            amb_state['current_h3'] 
            for amb_state in ambulances.values() 
            if amb_state['status'] == 'available'
        ]

        # 2. カバレッジを計算
        coverage_ratio = 0.0
        if available_amb_h3s and self.travel_time_matrix is not None and grid_mapping:
            total_grids = len(grid_mapping)
            covered_grids = set()
            
            # 各利用可能隊から10分以内のグリッドを調べる
            for h3_index in available_amb_h3s:
                amb_grid_idx = grid_mapping.get(h3_index)
                if amb_grid_idx is None:
                    continue

                # 移動時間行列から、この救急隊からの移動時間リストを取得
                travel_times_from_amb = self.travel_time_matrix[amb_grid_idx, :]
                
                # ★★★【修正箇所】★★★
                # ハードコーディングされた600を、コンフィグから読み込んだ変数に置き換える
                covered_indices = np.where(travel_times_from_amb <= self.coverage_time_threshold)[0]
                
                # setに追加して重複を除外
                covered_grids.update(covered_indices)
            
            # 全グリッド数に対するカバーされたグリッド数の割合を計算
            if total_grids > 0:
                coverage_ratio = len(covered_grids) / total_grids
        
        # 計算したカバレッジ率を最後の特徴量として追加
        features[20] = coverage_ratio
        
        return features
    
    def _encode_spatial_fallback(self, ambulances: Dict, 
                                incident: Optional[Dict]) -> np.ndarray:
        """
        空間的特徴量をエンコード（フォールバック版）
        移動時間行列が利用できない場合の従来の実装
        """
        features = np.zeros(20)  # 元の20次元の特徴量
        
        if incident is None:
            return features
        
        # 事案位置
        try:
            incident_lat, incident_lng = h3.cell_to_latlng(incident['h3_index'])
        except:
            return features
        
        # 各救急車との距離を計算
        distances = []
        available_distances = []
        
        for amb_state in ambulances.values():
            try:
                lat, lng = h3.cell_to_latlng(amb_state['current_h3'])
                # Haversine距離（km）
                dist = self._haversine_distance(incident_lat, incident_lng, lat, lng)
                distances.append(dist)
                
                if amb_state['status'] == 'available':
                    available_distances.append(dist)
            except:
                continue
        
        # 統計量を計算
        if available_distances:
            features[0] = min(available_distances) / 10.0  # 最短距離
            features[1] = np.mean(available_distances) / 10.0
            features[2] = np.std(available_distances) / 5.0 if len(available_distances) > 1 else 0
            features[3] = len(available_distances) / 10.0  # 利用可能な救急車数
        
        if distances:
            features[4] = min(distances) / 10.0
            features[5] = np.mean(distances) / 10.0
        
        return features
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """2点間のHaversine距離を計算（km）"""
        R = 6371  # 地球の半径（km）
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    @property
    def state_dim(self) -> int:
        """状態ベクトルの次元数を返す"""
        return (self.max_ambulances * self.ambulance_features + 
                self.incident_features + 
                self.temporal_features + 
                self.spatial_features)


# ============================================================
# CompactStateEncoder - コンパクトな状態エンコーダー（37次元）
# ============================================================

class CompactStateEncoder:
    """
    コンパクトな状態エンコーダー（37次元）
    
    状態ベクトル構造:
    [0]     is_severe: 重症系フラグ（重症/重篤/死亡 = 1.0）
    [1]     is_mild: 軽症系フラグ（軽症/中等症 = 1.0）
    [2-4]   Top-1救急車: travel_time, coverage_loss, station_distance
    [5-7]   Top-2救急車: travel_time, coverage_loss, station_distance
    ...
    [29-31] Top-10救急車: travel_time, coverage_loss, station_distance
    [32]    available_count_normalized: 利用可能救急車数 / 192
    [33]    coverage_rate: 現在のカバレッジ率（0-1）
    [34]    time_of_day_normalized: 時刻 / 24
    [35]    within_6min_ratio: 6分以内到達可能な救急車の割合
    [36]    avg_travel_time_normalized: Top-10の平均移動時間 / 30分
    """
    
    def __init__(self, 
                 config: Dict,
                 top_k: int = 10,
                 travel_time_matrix: Optional[np.ndarray] = None,
                 grid_mapping: Optional[Dict] = None):
        """
        Args:
            config: 設定辞書
            top_k: 考慮する上位救急車数
            travel_time_matrix: responseフェーズの移動時間行列
            grid_mapping: H3インデックス→行列インデックスのマッピング
        """
        self.config = config
        self.top_k = top_k
        self.travel_time_matrix = travel_time_matrix
        self.grid_mapping = grid_mapping
        
        # 特徴量の次元設定
        self.severity_features = 2       # is_severe, is_mild
        self.features_per_ambulance = 3  # travel_time, coverage_loss, station_distance
        self.global_features = 5         # 5つのグローバル統計
        
        # 正規化パラメータ
        encoding_config = config.get('state_encoding', {}).get('normalization', {})
        self.max_travel_time_minutes = encoding_config.get('max_travel_time_minutes', 30)
        self.max_station_distance_km = encoding_config.get('max_station_distance_km', 10)
        
        # カバレッジ計算の時間閾値（秒）
        coverage_config = config.get('coverage_params', {})
        self.coverage_time_threshold = coverage_config.get('time_threshold_seconds', 600)
        
        # ★★★ カバレッジ考慮型ソート設定（解決策1）★★★
        encoding_config = config.get('state_encoding', {})
        sorting_config = encoding_config.get('coverage_aware_sorting', {})
        self.coverage_aware_sorting = sorting_config.get('enabled', False)
        # デフォルト値を設定（無効時も使用される可能性があるため）
        self.sorting_time_weight = sorting_config.get('time_weight', 0.6)
        self.sorting_coverage_weight = sorting_config.get('coverage_weight', 0.4)
        self.coverage_sample_size = sorting_config.get('sample_size', 20)
        self.coverage_sample_radius = sorting_config.get('sample_radius', 2)
        # ★★★ 最適化設定 ★★★
        self.pre_filter_size = sorting_config.get('pre_filter_size', 30)  # 事前フィルタリング数（案1）
        self.use_adaptive_sampling = sorting_config.get('use_adaptive_sampling', True)  # 動的サンプリング（案2）
        self.max_candidates_for_advanced = sorting_config.get('max_candidates_for_advanced', 50)  # 高度計算の最大候補数（案3）
        if self.coverage_aware_sorting:
            print(f"  カバレッジ考慮型ソート有効: time_weight={self.sorting_time_weight}, coverage_weight={self.sorting_coverage_weight}")
            print(f"  最適化設定: pre_filter={self.pre_filter_size}, adaptive_sampling={self.use_adaptive_sampling}, max_advanced={self.max_candidates_for_advanced}")
        
        # 傷病度判定用の定数をインポート
        self.severe_conditions = ['重症', '重篤', '死亡']
        self.mild_conditions = ['軽症', '中等症']
        
        print(f"CompactStateEncoder初期化: top_k={top_k}, state_dim={self.state_dim}")
    
    @property
    def state_dim(self) -> int:
        """状態ベクトルの次元数"""
        return self.severity_features + (self.top_k * self.features_per_ambulance) + self.global_features
    
    def encode_state(self, state_dict: Dict, grid_mapping: Dict = None) -> np.ndarray:
        """
        状態辞書を37次元ベクトルに変換
        
        Args:
            state_dict: 環境の状態情報
                - 'ambulances': {amb_id: {'current_h3': str, 'status': str, 'station_h3': str, ...}}
                - 'pending_call': {'h3_index': str, 'severity': str, ...} or None
                - 'time_of_day': float (0-24)
            grid_mapping: H3→インデックスのマッピング（省略時はself.grid_mappingを使用）
        
        Returns:
            37次元のnumpy配列 (float32)
        """
        if grid_mapping is None:
            grid_mapping = self.grid_mapping
        
        features = np.zeros(self.state_dim, dtype=np.float32)
        
        incident = state_dict.get('pending_call')
        ambulances = state_dict.get('ambulances', {})
        time_of_day = state_dict.get('time_of_day', 12.0)
        
        # ========== 1. 傷病度（2次元）==========
        if incident:
            severity = incident.get('severity', '')
            features[0] = 1.0 if severity in self.severe_conditions else 0.0  # is_severe
            features[1] = 1.0 if severity in self.mild_conditions else 0.0    # is_mild
        
        # ========== 2. Top-K救急車（30次元）==========
        top_k_list = self._get_top_k_ambulances(ambulances, incident, grid_mapping)
        
        for i, amb_info in enumerate(top_k_list):
            base_idx = self.severity_features + i * self.features_per_ambulance
            features[base_idx + 0] = amb_info['travel_time_normalized']
            features[base_idx + 1] = amb_info['coverage_loss']
            features[base_idx + 2] = amb_info['station_distance_normalized']
        
        # ========== 3. グローバル統計（5次元）==========
        global_idx = self.severity_features + self.top_k * self.features_per_ambulance
        
        # 利用可能救急車数
        available_count = sum(1 for a in ambulances.values() if a.get('status') == 'available')
        total_ambulances = len(ambulances) if len(ambulances) > 0 else 192
        features[global_idx + 0] = available_count / total_ambulances
        
        # カバレッジ率
        features[global_idx + 1] = self._calculate_coverage_rate(ambulances, grid_mapping)
        
        # 時刻
        features[global_idx + 2] = time_of_day / 24.0
        
        # 6分以内到達可能な救急車の割合（Top-K内）
        within_6min_count = sum(1 for a in top_k_list if a['travel_time_minutes'] <= 6)
        features[global_idx + 3] = within_6min_count / self.top_k
        
        # 平均移動時間
        valid_times = [a['travel_time_minutes'] for a in top_k_list if a['amb_id'] >= 0]
        if valid_times:
            avg_travel_time = np.mean(valid_times)
        else:
            avg_travel_time = self.max_travel_time_minutes
        features[global_idx + 4] = min(avg_travel_time / self.max_travel_time_minutes, 1.0)
        
        # NaNチェック
        if np.any(np.isnan(features)):
            print("警告: CompactStateEncoderでNaN値を検出")
            features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def _get_top_k_ambulances(self, 
                              ambulances: Dict, 
                              incident: Optional[Dict],
                              grid_mapping: Dict) -> List[Dict]:
        """
        移動時間順にTop-K救急車の情報を取得
        
        Returns:
            List[Dict]: 各要素は以下のキーを持つ
                - amb_id: int
                - travel_time_seconds: float
                - travel_time_minutes: float
                - travel_time_normalized: float (0-1)
                - coverage_loss: float (0-1)
                - station_distance_km: float
                - station_distance_normalized: float (0-1)
        """
        # 事案がない場合はダミーデータを返す
        if incident is None:
            return self._get_dummy_top_k()
        
        incident_h3 = incident.get('h3_index')
        if not incident_h3 or not grid_mapping or incident_h3 not in grid_mapping:
            return self._get_dummy_top_k()
        
        incident_grid_idx = grid_mapping[incident_h3]
        
        # 利用可能な救急車を収集
        candidates = []
        available_amb_ids = []
        
        for amb_id, amb_state in ambulances.items():
            if amb_state.get('status') != 'available':
                continue
            
            available_amb_ids.append(amb_id)
            
            # 移動時間を計算
            amb_h3 = amb_state.get('current_h3')
            if amb_h3 and grid_mapping and amb_h3 in grid_mapping and self.travel_time_matrix is not None:
                amb_grid_idx = grid_mapping[amb_h3]
                travel_time_seconds = self.travel_time_matrix[amb_grid_idx, incident_grid_idx]
            else:
                travel_time_seconds = 1800  # 30分（デフォルト）
            
            travel_time_minutes = travel_time_seconds / 60.0
            
            # 署からの距離を計算
            station_distance_km = self._calculate_station_distance(amb_state)
            
            candidates.append({
                'amb_id': amb_id,
                'travel_time_seconds': travel_time_seconds,
                'travel_time_minutes': travel_time_minutes,
                'travel_time_normalized': min(travel_time_minutes / self.max_travel_time_minutes, 1.0),
                'coverage_loss': 0.0,  # 後で計算
                'station_distance_km': station_distance_km,
                'station_distance_normalized': min(station_distance_km / self.max_station_distance_km, 1.0)
            })
        
        # ★★★ カバレッジ考慮型ソート（解決策1 + 最適化）★★★
        if self.coverage_aware_sorting:
            # ★★★ 最適化案1: 段階的計算 ★★★
            # まず移動時間で上位候補を絞り込み
            candidates.sort(key=lambda x: x['travel_time_seconds'])
            pre_filtered = candidates[:min(self.pre_filter_size, len(candidates))]
            
            # ★★★ 最適化案3: 候補数が多い場合は簡易計算にフォールバック ★★★
            use_advanced_calculation = len(candidates) <= self.max_candidates_for_advanced
            
            # 絞り込んだ候補に対してカバレッジ損失を計算
            for cand in pre_filtered:
                if use_advanced_calculation:
                    # ★★★ 最適化案2: 動的サンプリング ★★★
                    # 候補数に応じてサンプル数を調整
                    if self.use_adaptive_sampling:
                        if len(candidates) > 100:
                            sample_size = 5
                        elif len(candidates) > 50:
                            sample_size = 10
                        else:
                            sample_size = self.coverage_sample_size
                    else:
                        sample_size = self.coverage_sample_size
                    
                    cand['coverage_loss'] = self._calculate_coverage_loss_advanced(
                        cand['amb_id'],
                        ambulances,
                        available_amb_ids,
                        grid_mapping,
                        sample_size=sample_size
                    )
                else:
                    # 候補数が多い場合は簡易計算を使用
                    cand['coverage_loss'] = self._calculate_coverage_loss_simple(
                        cand['amb_id'],
                        ambulances,
                        available_amb_ids
                    )
            
            # 移動時間とカバレッジ損失の複合スコアでソート
            for cand in pre_filtered:
                # スコア = 移動時間正規化値 × time_weight + カバレッジ損失 × coverage_weight
                # スコアが小さいほど良い（移動時間が短く、カバレッジ損失が小さい）
                cand['sort_score'] = (
                    cand['travel_time_normalized'] * self.sorting_time_weight +
                    cand['coverage_loss'] * self.sorting_coverage_weight
                )
            
            pre_filtered.sort(key=lambda x: x['sort_score'])
            
            # ソート済みの候補を先頭に、残りを移動時間順で結合
            remaining = candidates[self.pre_filter_size:]
            candidates = pre_filtered + remaining
        else:
            # 従来の移動時間順ソート
            candidates.sort(key=lambda x: x['travel_time_seconds'])
        
        # Top-Kを取得
        top_k_candidates = candidates[:self.top_k]
        
        # カバレッジ損失を計算（カバレッジ考慮型ソートでは既に計算済み）
        for cand in top_k_candidates:
            if 'coverage_loss' not in cand or cand.get('coverage_loss', 0.0) == 0.0:
                cand['coverage_loss'] = self._calculate_coverage_loss_simple(
                    cand['amb_id'], 
                    ambulances, 
                    available_amb_ids
                )
        
        # Top-Kに満たない場合はダミーで埋める
        while len(top_k_candidates) < self.top_k:
            top_k_candidates.append(self._get_dummy_ambulance_info())
        
        return top_k_candidates
    
    def _get_dummy_top_k(self) -> List[Dict]:
        """ダミーのTop-Kリストを返す"""
        return [self._get_dummy_ambulance_info() for _ in range(self.top_k)]
    
    def _get_dummy_ambulance_info(self) -> Dict:
        """ダミーの救急車情報を返す"""
        return {
            'amb_id': -1,
            'travel_time_seconds': 1800,
            'travel_time_minutes': 30.0,
            'travel_time_normalized': 1.0,
            'coverage_loss': 0.5,
            'station_distance_km': 5.0,
            'station_distance_normalized': 0.5
        }
    
    def _calculate_station_distance(self, amb_state: Dict) -> float:
        """署からの距離を計算（km）"""
        try:
            current_h3 = amb_state.get('current_h3')
            station_h3 = amb_state.get('station_h3')
            
            if current_h3 and station_h3:
                # H3インデックスから座標を取得
                current_lat, current_lng = h3.cell_to_latlng(current_h3)
                station_lat, station_lng = h3.cell_to_latlng(station_h3)
                
                # Haversine距離を計算
                return self._haversine_distance(current_lat, current_lng, station_lat, station_lng)
            return 0.0
        except:
            return 0.0
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """2点間のHaversine距離を計算（km）"""
        R = 6371  # 地球の半径（km）
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _calculate_coverage_loss_simple(self, 
                                        amb_id: int, 
                                        ambulances: Dict,
                                        available_amb_ids: List[int]) -> float:
        """
        簡易版カバレッジ損失計算
        
        考え方: この救急車を出動させた場合、近隣の利用可能救急車数が減る
        近隣に他の救急車が多いほど、損失は小さい
        """
        try:
            amb_state = ambulances.get(amb_id)
            if not amb_state:
                return 0.5
            
            amb_h3 = amb_state.get('current_h3')
            if not amb_h3 or not self.grid_mapping or amb_h3 not in self.grid_mapping:
                return 0.5
            
            if self.travel_time_matrix is None:
                return 0.5
            
            amb_grid_idx = self.grid_mapping[amb_h3]
            
            # 10分以内に到達可能な他の利用可能救急車をカウント
            nearby_count = 0
            for other_id in available_amb_ids:
                if other_id == amb_id:
                    continue
                
                other_state = ambulances.get(other_id)
                if not other_state:
                    continue
                
                other_h3 = other_state.get('current_h3')
                if not other_h3 or other_h3 not in self.grid_mapping:
                    continue
                
                other_grid_idx = self.grid_mapping[other_h3]
                
                # 2台の救急車間の移動時間を確認
                travel_time = self.travel_time_matrix[amb_grid_idx, other_grid_idx]
                if travel_time <= self.coverage_time_threshold:
                    nearby_count += 1
            
            # 近隣救急車が多いほど損失は小さい
            # nearby_count = 0 → loss = 1.0
            # nearby_count = 5 → loss ≈ 0.17
            return 1.0 / (nearby_count + 1)
        
        except Exception as e:
            return 0.5
    
    def _calculate_coverage_loss_advanced(self,
                                         amb_id: int,
                                         ambulances: Dict,
                                         available_amb_ids: List[int],
                                         grid_mapping: Dict,
                                         sample_size: Optional[int] = None) -> float:
        """
        高度なカバレッジ損失計算（SeverityBasedStrategyと同様のロジック）
        
        6分カバレッジと13分カバレッジの変化を計算し、重み付け合成
        
        Args:
            amb_id: 対象の救急車ID
            ambulances: 全救急車の状態辞書
            available_amb_ids: 利用可能な救急車IDリスト
            grid_mapping: H3→インデックスのマッピング
        
        Returns:
            float: 0-1の範囲のカバレッジ損失（高いほど損失大）
        """
        try:
            amb_state = ambulances.get(amb_id)
            if not amb_state:
                return 0.5
            
            # ステーション位置を取得（なければ現在位置を使用）
            station_h3 = amb_state.get('station_h3') or amb_state.get('current_h3')
            if not station_h3 or not grid_mapping or station_h3 not in grid_mapping:
                return 0.5
            
            if self.travel_time_matrix is None:
                return 0.5
            
            # その救急車を除いた利用可能な救急車リスト
            remaining_amb_ids = [aid for aid in available_amb_ids if aid != amb_id]
            if not remaining_amb_ids:
                return 1.0  # 他に救急車がない場合は最大損失
            
            # サンプルポイントを取得（動的サンプリング対応）
            actual_sample_size = sample_size if sample_size is not None else self.coverage_sample_size
            sample_points = self._get_coverage_sample_points(station_h3, grid_mapping, actual_sample_size)
            
            if not sample_points:
                # サンプルポイントが取得できない場合は簡易計算にフォールバック
                return self._calculate_coverage_loss_simple(amb_id, ambulances, available_amb_ids)
            
            # 6分・13分カバレッジへの影響を計算
            coverage_6min_before = 0
            coverage_13min_before = 0
            coverage_6min_after = 0
            coverage_13min_after = 0
            
            time_threshold_6min = 360  # 6分
            time_threshold_13min = 780  # 13分
            
            for point_h3 in sample_points:
                if point_h3 not in grid_mapping:
                    continue
                
                point_grid_idx = grid_mapping[point_h3]
                
                # 現在の状態でのカバレッジ（全利用可能救急車）
                min_time_before = self._get_min_response_time_for_coverage(
                    point_grid_idx, available_amb_ids, ambulances, grid_mapping
                )
                if min_time_before <= time_threshold_6min:
                    coverage_6min_before += 1
                if min_time_before <= time_threshold_13min:
                    coverage_13min_before += 1
                
                # 救急車が出動した後のカバレッジ
                min_time_after = self._get_min_response_time_for_coverage(
                    point_grid_idx, remaining_amb_ids, ambulances, grid_mapping
                )
                if min_time_after <= time_threshold_6min:
                    coverage_6min_after += 1
                if min_time_after <= time_threshold_13min:
                    coverage_13min_after += 1
            
            # カバレッジ率の変化を計算
            total_points = len(sample_points)
            if total_points == 0:
                return 0.5  # デフォルト値
            
            # 6分カバレッジと13分カバレッジの損失を重み付け合成
            loss_6min = (coverage_6min_before - coverage_6min_after) / total_points
            loss_13min = (coverage_13min_before - coverage_13min_after) / total_points
            
            # 6分カバレッジと13分カバレッジの損失を等価に重み付け（SeverityBasedStrategyと同じ）
            combined_loss = loss_6min * 0.5 + loss_13min * 0.5
            
            # 0-1の範囲にクリップ
            return max(0.0, min(1.0, combined_loss))
        
        except Exception as e:
            # エラー時は簡易計算にフォールバック
            return self._calculate_coverage_loss_simple(amb_id, ambulances, available_amb_ids)
    
    def _get_coverage_sample_points(self,
                                   center_h3: str,
                                   grid_mapping: Dict,
                                   sample_size: int = 20) -> List[str]:
        """
        カバレッジ計算用のサンプルポイントを取得
        
        SeverityBasedStrategyと同様のロジック
        """
        try:
            import h3
            # 中心から2リング以内のグリッドを取得
            nearby_grids = h3.grid_disk(center_h3, self.coverage_sample_radius)
            
            # grid_mappingに存在するグリッドのみを使用
            valid_grids = [g for g in nearby_grids if g in grid_mapping]
            
            # サンプルサイズを調整
            if len(valid_grids) <= sample_size:
                return valid_grids
            
            # ランダムサンプリング
            import random
            return random.sample(valid_grids, sample_size)
            
        except Exception:
            # エラーの場合は空リストを返す
            return []
    
    def _get_min_response_time_for_coverage(self,
                                           target_grid_idx: int,
                                           ambulance_ids: List[int],
                                           ambulances: Dict,
                                           grid_mapping: Dict) -> float:
        """
        指定地点への最小応答時間を取得（カバレッジ計算用、最適化版）
        
        Args:
            target_grid_idx: 目標地点のグリッドインデックス
            ambulance_ids: 考慮する救急車IDリスト
            ambulances: 全救急車の状態辞書
            grid_mapping: H3→インデックスのマッピング
        
        Returns:
            float: 最小応答時間（秒）
        """
        if not ambulance_ids or self.travel_time_matrix is None:
            return float('inf')
        
        # ★★★ 最適化: 救急車のグリッドインデックスを事前に取得 ★★★
        amb_grid_indices = []
        for amb_id in ambulance_ids:
            amb_state = ambulances.get(amb_id)
            if not amb_state:
                continue
            
            amb_h3 = amb_state.get('current_h3')
            if amb_h3 and amb_h3 in grid_mapping:
                amb_grid_indices.append(grid_mapping[amb_h3])
        
        if not amb_grid_indices:
            return 1800  # デフォルト30分
        
        try:
            # ★★★ 最適化: numpy配列演算で一括計算 ★★★
            # 全救急車から目標地点への移動時間を一括取得
            travel_times = self.travel_time_matrix[np.array(amb_grid_indices), target_grid_idx]
            min_time = np.min(travel_times)
            return float(min_time) if min_time < float('inf') else 1800
        except (IndexError, KeyError, ValueError):
            # フォールバック: 個別計算
            min_time = float('inf')
            for amb_grid_idx in amb_grid_indices:
                try:
                    travel_time = self.travel_time_matrix[amb_grid_idx, target_grid_idx]
                    if travel_time < min_time:
                        min_time = travel_time
                except (IndexError, KeyError):
                    continue
            return min_time if min_time != float('inf') else 1800
    
    def _calculate_coverage_rate(self, ambulances: Dict, grid_mapping: Dict) -> float:
        """現在のカバレッジ率を計算"""
        if not grid_mapping or self.travel_time_matrix is None:
            return 0.5
        
        try:
            # 利用可能な救急車のグリッドインデックスを取得
            available_indices = []
            for amb_state in ambulances.values():
                if amb_state.get('status') == 'available':
                    amb_h3 = amb_state.get('current_h3')
                    if amb_h3 and amb_h3 in grid_mapping:
                        available_indices.append(grid_mapping[amb_h3])
            
            if not available_indices:
                return 0.0
            
            # カバーされているグリッド数をカウント
            total_grids = len(grid_mapping)
            covered_grids = set()
            
            for amb_idx in available_indices:
                # この救急車から閾値時間以内のグリッドを取得
                travel_times = self.travel_time_matrix[amb_idx, :]
                covered_indices = np.where(travel_times <= self.coverage_time_threshold)[0]
                covered_grids.update(covered_indices)
            
            return len(covered_grids) / total_grids if total_grids > 0 else 0.0
        
        except Exception as e:
            return 0.5
    
    def get_top_k_ambulance_ids(self, ambulances: Dict, incident: Optional[Dict]) -> List[int]:
        """
        Top-K救急車のIDリストを返す
        
        ems_environment.pyでactionを実際の救急車IDに変換するために使用
        
        Returns:
            List[int]: Top-K救急車のIDリスト（移動時間順）
        """
        top_k_list = self._get_top_k_ambulances(ambulances, incident, self.grid_mapping)
        return [amb['amb_id'] for amb in top_k_list if amb['amb_id'] >= 0]


# ============================================================
# ファクトリ関数
# ============================================================

def create_state_encoder(config: Dict, **kwargs):
    """
    設定に応じてStateEncoderを作成するファクトリ関数
    
    Args:
        config: 設定辞書
        **kwargs: travel_time_matrix, grid_mapping など
    
    Returns:
        StateEncoder または CompactStateEncoder
    """
    encoding_config = config.get('state_encoding', {})
    mode = encoding_config.get('mode', 'full')
    
    if mode == 'compact':
        top_k = encoding_config.get('top_k', 10)
        return CompactStateEncoder(config, top_k=top_k, **kwargs)
    else:
        # 既存のStateEncoderを使用
        max_ambulances = kwargs.pop('max_ambulances', 192)
        return StateEncoder(config, max_ambulances=max_ambulances, **kwargs)