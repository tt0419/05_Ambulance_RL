"""
reward_designer.py
階層型報酬設計システム（整理版）
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
    報酬設計クラス（整理版）
    設定ファイルから報酬パラメータを読み込み、適切な報酬計算を行う
    """
    
    def __init__(self, config: Dict):
        """
        報酬設計の初期化
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # ===== システムレベル設定 =====
        system_config = config.get('reward', {}).get('system', {})
        self.dispatch_failure_penalty = system_config.get('dispatch_failure', -1.0)
        self.no_available_penalty = system_config.get('no_available_ambulance', 0.0)
        self.unhandled_penalty_base = system_config.get('unhandled_call_penalty', -1.0)
        
        # ===== 報酬モード設定 =====
        # 新しい設定構造に対応
        reward_mode_config = config.get('reward_mode', {})
        if reward_mode_config:
            # config_tokyo23.yamlの新しい構造
            self.mode = reward_mode_config.get('mode', 'simple')
            self._load_mode_params_from_reward_mode(reward_mode_config)
        else:
            # 古い構造（config.yaml）にフォールバック
            reward_config = config.get('reward', {})
            core_config = reward_config.get('core', {})
            self.mode = core_config.get('mode', 'simple')
            self._load_mode_params_from_core(core_config)
        
        # hybrid modeの場合、reward.core.hybrid_paramsを読み込み
        if self.mode == 'hybrid':
            reward_config = config.get('reward', {})
            core_config = reward_config.get('core', {})
            self.hybrid_params = core_config.get('hybrid_params', {
                'time_penalty_per_minute': -0.3,
                'mild_under_13min_bonus': 5.0,
                'moderate_under_13min_bonus': 10.0,
                'over_13min_penalty': -5.0,
                'over_20min_penalty': -50.0,
                'good_coverage_bonus': 10.0,
                'coverage_maintenance_bonus': 5.0,
                'poor_coverage_penalty': -10.0,
                'balanced_workload_bonus': 2.0,
                'overloaded_penalty': -5.0
            })
        
        # ===== カバレッジ設定 =====
        # reward.core.coverage_impact_weightから読み込み（元の設計）
        reward_config = config.get('reward', {})
        core_config = reward_config.get('core', {})
        self.coverage_impact_weight = core_config.get('coverage_impact_weight', 0.0)
        
        # カバレッジが有効かどうか（0より大きい場合は有効）
        self.coverage_enabled = (self.coverage_impact_weight > 0)
        
        # カバレッジパラメータ（オプション）
        optional_features = config.get('optional_features', {})
        coverage_config = optional_features.get('coverage', {})
        self.coverage_drop_threshold = coverage_config.get('drop_threshold', 0.05)
        self.coverage_drop_weight = coverage_config.get('drop_weight', -20.0)
        
        # ハイブリッドモード設定（config.yamlのトップレベルから読み込み）
        hybrid_config = config.get('hybrid_mode', {})
        self.hybrid_enabled = (self.mode == 'hybrid' and 
                              hybrid_config.get('enabled', False))
        
        if self.hybrid_enabled:
            self._init_hybrid_mode(hybrid_config)
        
        # ===== エピソードレベル設定 =====
        self.episode_config = config.get('reward', {}).get('episode', {})
        
        # デバッグカウンタ
        self._debug_count = 0
        self._max_debug_logs = 5
        
        print(f"RewardDesigner初期化完了:")
        print(f"  モード: {self.mode}")
        print(f"  カバレッジ: {'有効' if self.coverage_enabled else '無効'}")
        print(f"  ハイブリッド: {'有効' if self.hybrid_enabled else '無効'}")
    
    def _load_mode_params_from_reward_mode(self, reward_mode_config):
        """新しい設定構造から報酬パラメータを読み込み"""
        if self.mode == 'simple':
            params = reward_mode_config.get('simple', {})
            self.simple_params = {
                'time_penalty_per_minute': params.get('time_penalty_per_minute', -0.5),
                'critical_under_6min_bonus': params.get('critical_under_6min_bonus', 30.0),
                'moderate_under_13min_bonus': params.get('moderate_under_13min_bonus', 10.0),
                'mild_under_13min_bonus': params.get('mild_under_13min_bonus', 5.0),
                'over_13min_penalty': params.get('over_13min_penalty', -5.0),
                'over_20min_penalty': params.get('over_20min_penalty', -20.0)
            }
        
        elif self.mode == 'continuous':
            params = reward_mode_config.get('continuous', {})
            self.continuous_params = {
                'critical': params.get('critical', {
                    'target': 6, 'max_bonus': 50.0, 'penalty_scale': 5.0, 'weight': 5.0
                }),
                'moderate': params.get('moderate', {
                    'target': 13, 'max_bonus': 20.0, 'penalty_scale': 2.0, 'weight': 2.0
                }),
                'mild': params.get('mild', {
                    'target': 13, 'max_bonus': 10.0, 'penalty_scale': 0.5, 'weight': 1.0
                })
            }
        
        elif self.mode == 'discrete':
            params = reward_mode_config.get('discrete', {})
            self.discrete_params = {
                'weights': params.get('weights', {
                    'response_time': 2.0, 'severity_bonus': 3.0, 'coverage_preservation': 0.5
                }),
                'penalties': params.get('penalties', {
                    'over_6min': -10.0, 'over_13min': -15.0, 'per_minute_over': -2.0
                })
            }
    
    def _load_mode_params_from_core(self, core_config):
        """古い設定構造から報酬パラメータを読み込み（後方互換性）"""
        if self.mode == 'simple':
            params = core_config.get('simple_params', {})
            self.simple_params = params if params else {
                'time_penalty_per_minute': -0.5,
                'critical_under_6min_bonus': 30.0,
                'moderate_under_13min_bonus': 10.0,
                'mild_under_13min_bonus': 5.0,
                'over_13min_penalty': -5.0,
                'over_20min_penalty': -20.0
            }
        
        elif self.mode == 'continuous':
            self.continuous_params = core_config.get('continuous_params', {})
        
        elif self.mode == 'discrete':
            self.discrete_params = core_config.get('discrete_params', {})
    
    def _init_hybrid_mode(self, hybrid_config):
        """ハイブリッドモードの初期化"""
        # severity_classificationから読み込み（config.yamlの構造に合わせる）
        severity_class = hybrid_config.get('severity_classification', {})
        self.severe_conditions = severity_class.get('severe_conditions', 
                                                     ['重症', '重篤', '死亡'])
        self.mild_conditions = severity_class.get('mild_conditions', 
                                                   ['軽症', '中等症'])
        
        # reward_weightsから読み込み
        weights = hybrid_config.get('reward_weights', {})
        self.weight_rt = weights.get('response_time', 0.4)
        self.weight_coverage = weights.get('coverage', 0.5)
        self.weight_workload = weights.get('workload_balance', 0.1)
        
        print(f"  ハイブリッドモード設定:")
        print(f"    重症系: {self.severe_conditions}")
        print(f"    軽症系: {self.mild_conditions}")
    
    def calculate_step_reward(self,
                             severity: str,
                             response_time: float,
                             coverage_impact: float = 0.0,
                             coverage_before: float = 0.0,
                             coverage_after: float = 0.0,
                             additional_info: Optional[Dict] = None) -> float:
        """
        ステップ報酬を計算
        
        Args:
            severity: 傷病度
            response_time: 応答時間（秒）
            coverage_impact: カバレッジへの影響
            coverage_before: 行動前のカバレッジ率
            coverage_after: 行動後のカバレッジ率
            additional_info: 追加情報
            
        Returns:
            報酬値
        """
        response_time_minutes = response_time / 60.0
        
        # モード別の報酬計算
        if self.mode == 'simple':
            reward = self._calculate_simple_reward(severity, response_time_minutes)
        
        elif self.mode == 'continuous':
            reward = self._calculate_continuous_reward(severity, response_time_minutes)
        
        elif self.mode == 'discrete':
            reward = self._calculate_discrete_reward(severity, response_time_minutes)
        
        elif self.mode == 'hybrid' and self.hybrid_enabled:
            reward = self._calculate_hybrid_reward(severity, response_time_minutes, 
                                                  coverage_after, additional_info)
        else:
            raise ValueError(f"Unknown reward mode: {self.mode}")
        
        # カバレッジペナルティ（simpleモード以外で有効）
        if self.coverage_enabled and self.mode != 'simple':
            # カバレッジ影響ペナルティ
            if coverage_impact > 0 and self.coverage_impact_weight > 0:
                coverage_penalty = -coverage_impact * self.coverage_impact_weight * 10.0
                reward += coverage_penalty
            
            # カバレッジ低下ペナルティ
            coverage_drop = coverage_before - coverage_after
            if coverage_drop > self.coverage_drop_threshold:
                drop_penalty = self.coverage_drop_weight * (coverage_drop ** 2)
                reward += drop_penalty
        
        # クリッピング
        return np.clip(reward, -100.0, 100.0)
    
    def _calculate_simple_reward(self, severity: str, response_time_minutes: float) -> float:
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
        
        # デバッグ出力（最初の数回のみ）
        if self._debug_count < self._max_debug_logs:
            self._debug_count += 1
            print(f"[Simple報酬] 傷病度: {severity}, 時間: {response_time_minutes:.1f}分, 報酬: {reward:.2f}")
        
        return reward
    
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
    
    def _calculate_hybrid_reward(self, severity: str, response_time_minutes: float,
                                coverage_after: float, additional_info: Optional[Dict]) -> float:
        """ハイブリッドモードの報酬計算
        
        重症系: 0（直近隊運用、学習対象外）
        軽症系: RT最小化 + カバレッジ維持 + 稼働バランス
        """
        # 重症系は報酬なし（直近隊運用）
        if severity in self.severe_conditions:
            return 0.0
        
        # === A: 応答時間報酬（40%） ===
        params = self.hybrid_params
        time_reward = params['time_penalty_per_minute'] * response_time_minutes
        
        # 傷病度別ボーナス
        if severity == '中等症' and response_time_minutes <= 13:
            time_reward += params['moderate_under_13min_bonus']
        elif severity == '軽症' and response_time_minutes <= 13:
            time_reward += params['mild_under_13min_bonus']
        
        # 閾値ペナルティ
        if response_time_minutes > 13:
            time_reward += params['over_13min_penalty']
        if response_time_minutes > 20:
            time_reward += params['over_20min_penalty']
        
        # === B: カバレッジ報酬（50%） ===
        coverage_reward = 0.0
        if coverage_after >= 0.8:
            coverage_reward = params['good_coverage_bonus']
        elif coverage_after >= 0.6:
            coverage_reward = params['coverage_maintenance_bonus']
        else:
            coverage_reward = params['poor_coverage_penalty']
        
        # === C: 稼働バランス報酬（10%） ===
        workload_reward = 0.0
        if additional_info:
            avg_calls = additional_info.get('avg_calls_per_ambulance', 0)
            if avg_calls > 0:
                if avg_calls < 5:  # バランスが良い
                    workload_reward = params['balanced_workload_bonus']
                elif avg_calls > 10:  # 過負荷
                    workload_reward = params['overloaded_penalty']
        
        # 重み付け合計
        total_reward = (
            time_reward * self.weight_rt +
            coverage_reward * self.weight_coverage +
            workload_reward * self.weight_workload
        )
        
        return total_reward
    
    def get_failure_penalty(self, failure_type: str) -> float:
        """失敗ペナルティを取得"""
        if failure_type == 'dispatch':
            return self.dispatch_failure_penalty
        elif failure_type == 'no_available':
            return self.no_available_penalty
        elif failure_type == 'unhandled':
            return self.unhandled_penalty_base
        else:
            return -10.0
    
    def calculate_episode_reward(self, episode_stats: Dict) -> float:
        """エピソード全体の評価報酬"""
        if episode_stats.get('total_dispatches', 0) == 0:
            return -100.0
        
        # 基本報酬
        avg_response_time = np.mean(episode_stats.get('response_times', [10.0]))
        base_reward = self.episode_config.get('base_penalty_per_minute', -0.5) * avg_response_time
        
        # 達成率ボーナス
        bonuses = self.episode_config.get('achievement_bonuses', {})
        total_dispatches = episode_stats['total_dispatches']
        
        total_bonus = 0.0
        if 'achieved_6min' in episode_stats:
            rate_6min = episode_stats['achieved_6min'] / total_dispatches
            total_bonus += rate_6min * bonuses.get('rate_6min', 20.0)
        
        if 'achieved_13min' in episode_stats:
            rate_13min = episode_stats['achieved_13min'] / total_dispatches
            total_bonus += rate_13min * bonuses.get('rate_13min', 10.0)
        
        # 配車失敗ペナルティ
        failure_penalty = episode_stats.get('failed_dispatches', 0) * \
                         self.episode_config.get('failure_penalty_per_incident', -1.0)
        
        return base_reward + total_bonus + failure_penalty
    
    def get_info(self) -> Dict:
        """現在の報酬設定情報を返す"""
        return {
            'mode': self.mode,
            'coverage_enabled': self.coverage_enabled,
            'hybrid_enabled': self.hybrid_enabled,
            'params': self._get_current_params()
        }
    
    def _get_current_params(self) -> Dict:
        """現在のモードのパラメータを取得"""
        if self.mode == 'simple':
            return self.simple_params
        elif self.mode == 'continuous':
            return self.continuous_params
        elif self.mode == 'discrete':
            return self.discrete_params
        elif self.mode == 'hybrid':
            return {
                'severe_conditions': self.severe_conditions,
                'mild_conditions': self.mild_conditions,
                'weights': {
                    'response_time': self.weight_rt,
                    'coverage': self.weight_coverage,
                    'workload': self.weight_workload
                }
            }
        return {}