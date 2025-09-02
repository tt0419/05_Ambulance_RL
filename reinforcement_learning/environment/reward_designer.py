"""
reward_designer.py
階層型報酬設計システム
"""

import numpy as np
from typing import Dict, Optional
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from constants import SEVERITY_GROUPS, is_severe_condition

def severity_to_category(severity: str) -> str:
    """傷病度から標準カテゴリに変換"""
    if severity in ['重篤', '重症', '死亡']:
        return 'critical'
    elif severity in ['中等症']:
        return 'moderate'
    elif severity in ['軽症']:
        return 'mild'
    else:
        return 'mild'  # デフォルト


class RewardDesigner:
    """
    階層型報酬設計クラス
    設定ファイルから報酬パラメータを読み込み、適切な報酬計算を行う
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書（reward セクションを含む）
        """
        # 設定の読み込み
        self.reward_config = config.get('reward', {})
        
        # システムレベル設定
        self.system_config = self.reward_config.get('system', {})
        self.dispatch_failure_penalty = self.system_config.get('dispatch_failure', -10.0)
        self.no_available_penalty = self.system_config.get('no_available_ambulance', -5.0)
        self.unhandled_penalty_base = self.system_config.get('unhandled_call_penalty', -50.0)
        
        # コアレベル設定
        self.core_config = self.reward_config.get('core', {})
        self.mode = self.core_config.get('mode', 'continuous')
        self.coverage_weight = self.core_config.get('coverage_impact_weight', 0.5)
        
        # モード別パラメータの初期化
        self._initialize_mode_params()
        
        # エピソードレベル設定
        self.episode_config = self.reward_config.get('episode', {})
        
        print(f"RewardDesigner初期化完了: モード={self.mode}")
    
    def _initialize_mode_params(self):
        """モード別パラメータの初期化"""
        if self.mode == 'continuous':
            self.continuous_params = self.core_config.get('continuous_params', {})
            if not self.continuous_params:
                # デフォルト値
                self.continuous_params = {
                    'critical': {'target': 6, 'max_bonus': 50.0, 'penalty_scale': 5.0, 'weight': 5.0},
                    'moderate': {'target': 13, 'max_bonus': 20.0, 'penalty_scale': 2.0, 'weight': 2.0},
                    'mild': {'target': 13, 'max_bonus': 10.0, 'penalty_scale': 0.5, 'weight': 1.0}
                }
                
        elif self.mode == 'discrete':
            self.discrete_params = self.core_config.get('discrete_params', {})
            if not self.discrete_params:
                # デフォルト値
                self.discrete_params = {
                    'weights': {'response_time': 2.0, 'severity_bonus': 3.0, 'coverage_preservation': 0.5},
                    'penalties': {'over_6min': -10.0, 'over_13min': -15.0, 'per_minute_over': -2.0}
                }
                
        elif self.mode == 'simple':
            self.simple_params = self.core_config.get('simple_params', {})
            if not self.simple_params:
                # デフォルト値
                self.simple_params = {
                    'time_penalty_per_minute': -0.25,
                    'critical_under_6min_bonus': 10.0,
                    'moderate_under_13min_bonus': 5.0,
                    'mild_under_13min_bonus': 2.0,
                    'over_13min_penalty': -2.0,
                    'over_20min_penalty': -5.0,
                    'imitation_bonus': 0.5
                }
    
    def calculate_step_reward(self,
                             severity: str,
                             response_time: float,
                             coverage_impact: float = 0.0,
                             additional_info: Optional[Dict] = None) -> float:
        """
        ステップ報酬を計算（メインインターフェース）
        
        Args:
            severity: 傷病度
            response_time: 応答時間（秒）
            coverage_impact: カバレッジへの影響（0-1）
            additional_info: 追加情報
            
        Returns:
            報酬値
        """
        response_time_minutes = response_time / 60.0
        
        if self.mode == 'continuous':
            reward = self._calculate_continuous_reward(severity, response_time_minutes)
        elif self.mode == 'discrete':
            reward = self._calculate_discrete_reward(severity, response_time_minutes)
        elif self.mode == 'simple':
            reward = self._calculate_simple_reward(severity, response_time_minutes, additional_info)
        else:
            raise ValueError(f"Unknown reward mode: {self.mode}")
        
        # カバレッジペナルティ
        if coverage_impact > 0 and self.coverage_weight > 0:
            coverage_penalty = -coverage_impact * self.coverage_weight * 10.0
            reward += coverage_penalty
        
        # クリッピング
        return np.clip(reward, -100.0, 100.0)
    
    def _calculate_continuous_reward(self, severity: str, response_time_minutes: float) -> float:
        """連続報酬モードの計算"""
        category = severity_to_category(severity)
        params = self.continuous_params.get(category, self.continuous_params.get('mild'))
        
        target = params['target']
        max_bonus = params['max_bonus']
        penalty_scale = params['penalty_scale']
        weight = params.get('weight', 1.0)
        
        if category == 'critical':
            # 重症：指数関数的な変化
            if response_time_minutes <= target:
                lambda_param = 0.115
                reward = max_bonus * np.exp(-lambda_param * response_time_minutes)
            else:
                overtime = response_time_minutes - target
                reward = -penalty_scale * (overtime ** 1.5)
        else:
            # 中等症・軽症：線形的な変化
            if response_time_minutes <= target:
                reward = max_bonus * (1 - response_time_minutes / target)
            else:
                overtime = response_time_minutes - target
                reward = -penalty_scale * overtime
        
        return reward * weight
    
    def _calculate_discrete_reward(self, severity: str, response_time_minutes: float) -> float:
        """離散報酬モードの計算"""
        weights = self.discrete_params['weights']
        penalties = self.discrete_params['penalties']
        
        # 基本時間報酬
        time_reward = (6.0 - response_time_minutes) * weights['response_time']
        
        # 傷病度ボーナス
        category = severity_to_category(severity)
        severity_weight = {'critical': 5.0, 'moderate': 2.0, 'mild': 1.0}.get(category, 1.0)
        severity_bonus = 0.0
        
        if category == 'critical' and response_time_minutes <= 6:
            severity_bonus = severity_weight * weights['severity_bonus']
        elif response_time_minutes <= 13:
            severity_bonus = severity_weight * weights['severity_bonus'] * 0.5
        
        # 閾値ペナルティ
        threshold_penalty = 0.0
        if response_time_minutes > 6:
            threshold_penalty += penalties['over_6min']
            if is_severe_condition(severity):
                threshold_penalty *= 2.0
        if response_time_minutes > 13:
            threshold_penalty += penalties['over_13min']
            overtime = response_time_minutes - 13
            threshold_penalty += penalties['per_minute_over'] * overtime
        
        return time_reward + severity_bonus + threshold_penalty
    
    def _calculate_simple_reward(self, severity: str, response_time_minutes: float, 
                                 additional_info: Optional[Dict] = None) -> float:
        """シンプル報酬モードの計算"""
        params = self.simple_params
        
        # 基本時間ペナルティ
        reward = params['time_penalty_per_minute'] * response_time_minutes
        
        # 達成ボーナス
        category = severity_to_category(severity)
        if category == 'critical' and response_time_minutes <= 6:
            reward += params['critical_under_6min_bonus']
        elif category == 'moderate' and response_time_minutes <= 13:
            reward += params['moderate_under_13min_bonus']
        elif category == 'mild' and response_time_minutes <= 13:
            reward += params['mild_under_13min_bonus']
        
        # 閾値超過ペナルティ
        if response_time_minutes > 13:
            reward += params['over_13min_penalty']
        if response_time_minutes > 20:
            reward += params['over_20min_penalty']
        
        # 教師模倣ボーナス
        if additional_info and additional_info.get('matched_teacher', False):
            reward += params.get('imitation_bonus', 0.0)
        
        return reward
    
    def get_failure_penalty(self, failure_type: str) -> float:
        """
        失敗ペナルティを取得
        
        Args:
            failure_type: 失敗の種類
        
        Returns:
            ペナルティ値（負の値）
        """
        if failure_type == 'dispatch':
            return self.dispatch_failure_penalty
        elif failure_type == 'no_available':
            return self.no_available_penalty
        elif failure_type == 'unhandled':
            return self.unhandled_penalty_base
        else:
            return -10.0  # デフォルト
    
    def calculate_episode_reward(self, episode_stats: Dict) -> float:
        """
        エピソード全体の評価報酬
        
        Args:
            episode_stats: エピソード統計
            
        Returns:
            エピソード報酬
        """
        if episode_stats.get('total_dispatches', 0) == 0:
            return -100.0
        
        # 基本報酬（平均応答時間）
        avg_response_time = np.mean(episode_stats.get('response_times', [10.0]))
        base_reward = self.episode_config.get('base_penalty_per_minute', -1.0) * avg_response_time
        
        # 達成率ボーナス
        bonuses = self.episode_config.get('achievement_bonuses', {})
        total_dispatches = episode_stats['total_dispatches']
        
        bonus_6min = 0.0
        if 'achieved_6min' in episode_stats:
            rate_6min = episode_stats['achieved_6min'] / total_dispatches
            bonus_6min = rate_6min * bonuses.get('rate_6min', 50.0)
        
        bonus_13min = 0.0
        if 'achieved_13min' in episode_stats:
            rate_13min = episode_stats['achieved_13min'] / total_dispatches
            bonus_13min = rate_13min * bonuses.get('rate_13min', 30.0)
        
        bonus_critical = 0.0
        if episode_stats.get('critical_total', 0) > 0:
            critical_rate = episode_stats.get('critical_6min', 0) / episode_stats['critical_total']
            bonus_critical = critical_rate * bonuses.get('critical_6min_rate', 100.0)
        
        # 配車失敗ペナルティ
        failure_penalty = episode_stats.get('failed_dispatches', 0) * \
                         self.episode_config.get('failure_penalty_per_incident', -5.0)
        
        total_reward = base_reward + bonus_6min + bonus_13min + bonus_critical + failure_penalty
        
        return total_reward
    
    def calculate_unhandled_penalty(self, severity: str, wait_time: int, response_type: str) -> float:
        """
        対応不能事案のペナルティ計算
        
        Args:
            severity: 傷病度
            wait_time: 待機時間（分）
            response_type: 対応タイプ
            
        Returns:
            ペナルティ値（負の値）
        """
        # 基本ペナルティ（重症度に応じて）
        if severity in ['重篤', '重症']:
            base_penalty = -150.0  # 重症対応不能は深刻
        elif severity == '中等症':
            base_penalty = -75.0   # 中等症対応不能も問題
        else:  # 軽症
            base_penalty = -25.0   # 軽症対応不能は相対的に軽微
        
        # 対応タイプ別の調整
        if response_type == 'transport_cancel':
            # 搬送見送りは最も軽いペナルティ
            type_multiplier = 0.3
        elif response_type == 'emergency_support':
            # 緊急応援は迅速対応なので中程度のペナルティ
            type_multiplier = 0.6
        elif response_type == 'standard_support':
            # 標準応援は通常のペナルティ
            type_multiplier = 0.8
        elif response_type == 'delayed_support':
            # 遅延応援は重いペナルティ
            type_multiplier = 1.2
        else:
            type_multiplier = 1.0
        
        # 待機時間による追加ペナルティ
        time_penalty = -min(wait_time * 2, 120)  # 最大120分ペナルティ
        
        total_penalty = (base_penalty * type_multiplier) + time_penalty
        
        return total_penalty
    
    def get_info(self) -> Dict:
        """現在の報酬設定情報を返す"""
        return {
            'mode': self.mode,
            'system_config': self.system_config,
            'core_config': self.core_config,
            'episode_config': self.episode_config
        }