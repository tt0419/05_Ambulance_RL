"""
state_encoder.py
複雑な環境状態を固定長ベクトルに変換
"""

import numpy as np
import h3
from typing import Dict, List, Optional
import torch

class StateEncoder:
    """
    EMS環境の状態をニューラルネットワーク用のベクトルに変換
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 特徴量の次元設定
        self.max_ambulances = 192
        self.ambulance_features = 4  # 位置x, 位置y, 状態, 出動回数
        self.incident_features = 10  # 位置、傷病度など
        self.temporal_features = 8  # 時間関連
        self.spatial_features = 20  # 空間統計
        
        # 傷病度のone-hotエンコーディング用
        self.severity_indices = {
            '重篤': 0, '重症': 1, '死亡': 2,
            '中等症': 3, '軽症': 4, 'その他': 5
        }
        
    def encode_state(self, state_dict: Dict, grid_mapping: Dict) -> np.ndarray:
        """
        状態辞書を固定長ベクトルに変換
        
        Args:
            state_dict: 環境の状態情報
            grid_mapping: H3インデックスとグリッドIDのマッピング
            
        Returns:
            状態ベクトル
        """
        features = []
        
        # 1. 救急車の特徴量
        ambulance_features = self._encode_ambulances(
            state_dict['ambulances'], grid_mapping
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
        
        # 4. 空間的特徴量
        spatial_features = self._encode_spatial(
            state_dict['ambulances'],
            state_dict.get('pending_call')
        )
        features.append(spatial_features)
        
        # 全特徴量を結合
        state_vector = np.concatenate(features)
        
        return state_vector.astype(np.float32)
    
    def _encode_ambulances(self, ambulances: Dict, grid_mapping: Dict) -> np.ndarray:
        """救急車情報をエンコード"""
        # 192台固定
        max_ambulances = 192
        features = np.zeros(max_ambulances * self.ambulance_features)
        
        for amb_id, amb_state in ambulances.items():
            if amb_id >= max_ambulances:
                break
            
            idx = amb_id * self.ambulance_features
            
            # H3インデックスを座標に変換
            try:
                lat, lng = h3.cell_to_latlng(amb_state['current_h3'])
            except:
                lat, lng = 35.6762, 139.6503  # デフォルト（東京）
            
            # 特徴量の設定
            features[idx] = lat / 90.0  # 緯度を正規化
            features[idx + 1] = lng / 180.0  # 経度を正規化
            features[idx + 2] = 1.0 if amb_state['status'] == 'available' else 0.0
            features[idx + 3] = min(amb_state['calls_today'] / 20.0, 1.0)  # 出動回数を正規化
        
        return features
    
    def _encode_incident(self, incident: Optional[Dict], grid_mapping: Dict) -> np.ndarray:
        """事案情報をエンコード"""
        features = np.zeros(self.incident_features)
        
        if incident is None:
            return features
        
        # 位置情報
        lat, lng = h3.cell_to_latlng(incident['h3_index'])
        features[0] = lat / 90.0
        features[1] = lng / 180.0
        
        # 傷病度（one-hot）
        severity = incident.get('severity', 'その他')
        severity_idx = self.severity_indices.get(severity, 5)
        features[2 + severity_idx] = 1.0
        
        # 傷病度の重み（連続値）
        severity_weights = {
            '重篤': 1.0, '重症': 1.0, '死亡': 1.0,  # 同じ重み
            '中等症': 0.4, '軽症': 0.2, 'その他': 0.1
        }
        features[8] = severity_weights.get(severity, 0.1)
        
        # 事案の存在フラグ
        features[9] = 1.0
        
        return features
    
    def _encode_temporal(self, episode_step: int, time_of_day: int) -> np.ndarray:
        """時間的特徴量をエンコード"""
        features = np.zeros(self.temporal_features)
        
        # 時刻の周期的エンコーディング
        hour_rad = 2 * np.pi * time_of_day / 24
        features[0] = np.sin(hour_rad)
        features[1] = np.cos(hour_rad)
        
        # 時間帯カテゴリ（朝、昼、夜、深夜）
        if 6 <= time_of_day < 12:
            features[2] = 1.0  # 朝
        elif 12 <= time_of_day < 18:
            features[3] = 1.0  # 昼
        elif 18 <= time_of_day < 24:
            features[4] = 1.0  # 夜
        else:
            features[5] = 1.0  # 深夜
        
        # エピソード進行度
        max_steps = 1000  # 仮定値
        features[6] = min(episode_step / max_steps, 1.0)
        
        # 繁忙期フラグ（12月）
        # 実際の実装では日付から判定
        features[7] = 0.0  # デフォルトは通常期
        
        return features
    
    def _encode_spatial(self, ambulances: Dict, incident: Optional[Dict]) -> np.ndarray:
        """空間的特徴量をエンコード"""
        features = np.zeros(self.spatial_features)
        
        if incident is None:
            return features
        
        incident_lat, incident_lng = h3.cell_to_latlng(incident['h3_index'])
        
        # 利用可能な救急車の統計
        available_distances = []
        busy_distances = []
        
        for amb_state in ambulances.values():
            amb_lat, amb_lng = h3.cell_to_latlng(amb_state['current_h3'])
            
            # ハバーシン距離（簡易版）
            distance = self._haversine_distance(
                incident_lat, incident_lng, amb_lat, amb_lng
            )
            
            if amb_state['status'] == 'available':
                available_distances.append(distance)
            else:
                busy_distances.append(distance)
        
        # 統計量の計算
        if available_distances:
            features[0] = np.min(available_distances) / 50.0  # 最近接距離（km）
            features[1] = np.mean(available_distances) / 50.0  # 平均距離
            features[2] = np.std(available_distances) / 50.0  # 標準偏差
            features[3] = len(available_distances) / self.max_ambulances  # 利用可能率
            
            # 距離閾値内の救急車数
            features[4] = sum(1 for d in available_distances if d <= 3.0) / 10.0  # 3km以内
            features[5] = sum(1 for d in available_distances if d <= 5.0) / 20.0  # 5km以内
            features[6] = sum(1 for d in available_distances if d <= 10.0) / 30.0  # 10km以内
        
        # 稼働率
        total_ambulances = len(ambulances)
        if total_ambulances > 0:
            features[7] = len(busy_distances) / total_ambulances
        
        # 地域の混雑度（簡易版）
        # 実際の実装では、過去の事案密度等を使用
        features[8:20] = 0.5  # デフォルト値
        
        return features
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """ハバーシン距離を計算（km）"""
        R = 6371  # 地球の半径（km）
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_state_dim(self) -> int:
        """状態ベクトルの次元を返す"""
        return (self.max_ambulances * self.ambulance_features + 
                self.incident_features + 
                self.temporal_features + 
                self.spatial_features)