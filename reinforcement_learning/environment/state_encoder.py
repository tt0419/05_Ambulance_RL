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
        # ★ Stage 3: current_timeを渡す（存在する場合）
        temporal_features = self._encode_temporal(
            state_dict.get('episode_step', 0),
            state_dict.get('time_of_day', 12),
            current_time=state_dict.get('current_time')  # ← 追加
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
    
    def _encode_temporal(self, episode_step: int, time_of_day: float, current_time: float = None) -> np.ndarray:
        """
        時間的特徴量をエンコード
        
        【Stage 3改修】current_time（連続時間）対応：
        - current_timeが渡された場合は、それを使用してエピソード進行度を計算
        - 後方互換性のため、episode_stepも引き続きサポート
        
        Args:
            episode_step: エピソードのステップ数（後方互換性用）
            time_of_day: 時刻（時間単位）
            current_time: 現在時刻（秒単位、オプション）
        """
        features = np.zeros(self.temporal_features)
        
        # ★ Stage 3: エピソード進行度の計算
        if current_time is not None:
            # current_timeが渡された場合は、秒単位で進行度を計算
            episode_duration_seconds = self.config.get('data', {}).get('episode_duration_hours', 24) * 3600
            features[0] = min(current_time / episode_duration_seconds, 1.0)
        else:
            # 後方互換性：episode_stepベースの計算
            max_steps = self.config.get('data', {}).get('episode_duration_hours', 24) * 60
            features[0] = min(episode_step / max_steps, 1.0) if max_steps > 0 else 0.0
        
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