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

# ============= 追加部分 =============
# ファイルの既存のインポート部分の後に追加

class HybridModeRewardCalculator:
    """ハイブリッドモード（重症系直近隊、軽症系PPO）用の報酬計算"""
    
    def __init__(self, config):
        self.config = config
        
        # 報酬バランス設定
        self.weight_rt = 0.4  # A: 軽症系RT最小化
        self.weight_coverage = 0.5  # B: カバレッジ維持
        self.weight_workload = 0.1  # C: 稼働バランス
        
        # 時間閾値設定（分）
        self.threshold_good = 13  # 良好
        self.threshold_warning = 20  # 警告
        
        # ペナルティ設定
        self.penalty_over_warning = -50.0  # 20分超過時の重いペナルティ
        
        # 重症系・軽症系の分類
        self.severe_conditions = ['重症', '重篤', '死亡']
        self.mild_conditions = ['軽症', '中等症']
    
    def calculate_reward(self, state, action, outcome, coverage_info=None):
        """
        ハイブリッドモード用の報酬計算
        
        Args:
            state: 現在の状態
            action: 選択した行動
            outcome: 結果（response_time, severity等を含む）
            coverage_info: カバレッジ情報（オプション）
        
        Returns:
            float: 計算された報酬値
        """
        # 重症系事案の場合は報酬計算をスキップ（環境側で処理）
        if outcome.get('severity') in self.severe_conditions:
            return 0.0
        
        reward = 0.0
        
        # A. 軽症系RTの評価（40%）
        rt_minutes = outcome.get('response_time', 0) / 60.0
        rt_reward = self._calculate_rt_reward(rt_minutes)
        reward += self.weight_rt * rt_reward
        
        # B. カバレッジ維持の評価（50%）
        if coverage_info:
            coverage_reward = self._calculate_coverage_reward(coverage_info)
            reward += self.weight_coverage * coverage_reward
        
        # C. 稼働バランスの評価（10%）
        workload_reward = self._calculate_workload_reward(state, action)
        reward += self.weight_workload * workload_reward
        
        return reward
    
    def _calculate_rt_reward(self, rt_minutes):
        """応答時間に基づく報酬計算"""
        if rt_minutes <= self.threshold_good:
            # 13分以内：良好なボーナス
            return 10.0 * (1.0 - rt_minutes / self.threshold_good)
        elif rt_minutes <= self.threshold_warning:
            # 13-20分：線形減少
            return -5.0 * (rt_minutes - self.threshold_good) / (self.threshold_warning - self.threshold_good)
        else:
            # 20分超過：重いペナルティ
            return self.penalty_over_warning + (-2.0 * (rt_minutes - self.threshold_warning))
    
    def _calculate_coverage_reward(self, coverage_info):
        """カバレッジ維持に基づく報酬計算"""
        # 重症系事案発生確率が高い地域のカバレッジを重視
        high_risk_coverage = coverage_info.get('high_risk_area_coverage', 1.0)
        overall_coverage = coverage_info.get('overall_coverage', 1.0)
        
        # 重み付き平均（高リスク地域を重視）
        weighted_coverage = 0.7 * high_risk_coverage + 0.3 * overall_coverage
        
        # カバレッジが良好な場合はボーナス、悪化した場合はペナルティ
        if weighted_coverage >= 0.8:
            return 10.0 * weighted_coverage
        elif weighted_coverage >= 0.6:
            return 5.0 * weighted_coverage
        else:
            return -10.0 * (1.0 - weighted_coverage)
    
    def _calculate_workload_reward(self, state, action):
        """稼働バランスに基づく報酬計算"""
        # 選択された救急車の稼働状況を評価
        if hasattr(state, 'ambulance_workloads'):
            workloads = state.ambulance_workloads
            if len(workloads) > 0:
                # 稼働率の標準偏差が小さいほど良い
                workload_std = np.std(workloads)
                max_workload = np.max(workloads)
                
                # バランスの良さを評価
                if workload_std < 0.1:
                    return 5.0
                elif max_workload > 0.9:  # 過負荷の救急車がある
                    return -10.0
                else:
                    return 2.0 * (1.0 - workload_std)
        return 0.0

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
        報酬設計の初期化（修正版）
        Args:
            config: 設定辞書（reward セクションを含む）
        """
        # 設定の読み込み
        self.config = config
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
        
        # コアモードの判定
        self.core_mode = self.reward_config.get('core', {}).get('mode', 'simple')
        
        # モード別パラメータの初期化
        self._initialize_mode_params()
        
        # エピソードレベル設定
        self.episode_config = self.reward_config.get('episode', {})
        
        # ★★★【修正箇所】★★★
        # コンフィグからカバレッジのペナルティ閾値を読み込む
        coverage_config = config.get('coverage_params', {})
        self.coverage_drop_threshold = coverage_config.get('drop_penalty_threshold', 0.05)
        self.coverage_drop_weight = coverage_config.get('drop_penalty_weight', -20.0)
        
        # ハイブリッドモードの初期化
        if self.core_mode == 'hybrid':
            self._init_hybrid_mode()
        
        # 既存のハイブリッドモード（後方互換性のため）
        self.hybrid_mode = config.get('hybrid_mode', {}).get('enabled', False)
        if self.hybrid_mode:
            self.hybrid_calculator = HybridModeRewardCalculator(config)
        
        print(f"RewardDesigner初期化完了: モード={self.mode}, コアモード={self.core_mode}, ハイブリッドモード={self.hybrid_mode}")
    
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
                # デフォルト値（より明確な報酬設計）
                self.simple_params = {
                    'time_penalty_per_minute': -1.0,  # 時間ペナルティを強化
                    'critical_under_6min_bonus': 20.0,  # 重症6分達成ボーナスを強化
                    'moderate_under_13min_bonus': 10.0,  # 中等症13分達成ボーナスを強化
                    'mild_under_13min_bonus': 5.0,  # 軽症13分達成ボーナスを強化
                    'over_13min_penalty': -5.0,  # 13分超過ペナルティを強化
                    'over_20min_penalty': -10.0,  # 20分超過ペナルティを強化
                    'imitation_bonus': 2.0  # 教師模倣ボーナスを強化
                }
            
            # デバッグ用：simpleモードのパラメータを表示
            print(f"Simpleモード報酬パラメータ:")
            for key, value in self.simple_params.items():
                print(f"  {key}: {value}")
    
    def _init_hybrid_mode(self):
        """ハイブリッドモードの初期化"""
        hybrid_config = self.config.get('hybrid_mode', {})
        self.hybrid_enabled = True
        
        # 傷病度分類
        self.severe_conditions = hybrid_config.get('severity_classification', {}).get(
            'severe_conditions', ['重症', '重篤', '死亡']
        )
        self.mild_conditions = hybrid_config.get('severity_classification', {}).get(
            'mild_conditions', ['軽症', '中等症']
        )
        
        # 報酬パラメータ
        hybrid_params = self.reward_config.get('core', {}).get('hybrid_params', {})
        
        # 時間関連
        self.time_penalty_per_minute = hybrid_params.get('time_penalty_per_minute', -0.3)
        self.mild_under_13min_bonus = hybrid_params.get('mild_under_13min_bonus', 5.0)
        self.moderate_under_13min_bonus = hybrid_params.get('moderate_under_13min_bonus', 10.0)
        self.over_13min_penalty = hybrid_params.get('over_13min_penalty', -5.0)
        self.over_20min_penalty = hybrid_params.get('over_20min_penalty', -50.0)
        
        # カバレッジ関連
        self.good_coverage_bonus = hybrid_params.get('good_coverage_bonus', 10.0)
        self.coverage_maintenance_bonus = hybrid_params.get('coverage_maintenance_bonus', 5.0)
        self.poor_coverage_penalty = hybrid_params.get('poor_coverage_penalty', -10.0)
        
        # 稼働バランス関連
        self.balanced_workload_bonus = hybrid_params.get('balanced_workload_bonus', 2.0)
        self.overloaded_penalty = hybrid_params.get('overloaded_penalty', -5.0)
        
        # 重み設定（A:40%, B:50%, C:10%）
        weights = hybrid_config.get('reward_weights', {})
        self.weight_rt = weights.get('response_time', 0.4)
        self.weight_coverage = weights.get('coverage', 0.5)
        self.weight_workload = weights.get('workload_balance', 0.1)
        
        print("ハイブリッドモード初期化完了:")
        print(f"  重症系条件: {self.severe_conditions}")
        print(f"  軽症系条件: {self.mild_conditions}")
        print(f"  重み設定: RT={self.weight_rt}, カバレッジ={self.weight_coverage}, 稼働バランス={self.weight_workload}")
    
    def calculate_step_reward(self,
                             severity: str,
                             response_time: float,
                             coverage_impact: float = 0.0,
                             coverage_before: float = 0.0,
                             coverage_after: float = 0.0,
                             additional_info: Optional[Dict] = None) -> float:
        """
        ステップ報酬を計算（メインインターフェース）
        
        Args:
            severity: 傷病度
            response_time: 応答時間（秒）
            coverage_impact: カバレッジへの影響（0-1）
            coverage_before: 行動前のカバレッジ率
            coverage_after: 行動後のカバレッジ率
            additional_info: 追加情報
            
        Returns:
            報酬値
        """
        # 新しいハイブリッドモードの場合
        if self.core_mode == 'hybrid':
            info = {
                'severity': severity,
                'response_time': response_time,
                'coverage_info': {
                    'overall_coverage': coverage_after,
                    'high_risk_area_coverage': coverage_after  # 簡略化
                },
                'workload_info': additional_info.get('workload_info', {}) if additional_info else {}
            }
            return self._calculate_hybrid_reward(None, None, None, info)
        
        # 既存のハイブリッドモードの場合（後方互換性）
        elif self.hybrid_mode:
            outcome = {
                'severity': severity,
                'response_time': response_time
            }
            coverage_info = {
                'overall_coverage': coverage_after,
                'high_risk_area_coverage': coverage_after  # 簡略化
            }
            return self.hybrid_calculator.calculate_reward(
                None, None, outcome, coverage_info
            )
        
        # シンプルモードの場合
        elif self.core_mode == 'simple':
            return self._calculate_simple_reward(severity, response_time, additional_info)
        
        # その他のモード（既存の実装）
        else:
            response_time_minutes = response_time / 60.0
            
            if self.mode == 'continuous':
                reward = self._calculate_continuous_reward(severity, response_time_minutes)
            elif self.mode == 'discrete':
                reward = self._calculate_discrete_reward(severity, response_time_minutes)
            else:
                raise ValueError(f"Unknown reward mode: {self.mode}")
            
            # カバレッジペナルティ
            if coverage_impact > 0 and self.coverage_weight > 0:
                coverage_penalty = -coverage_impact * self.coverage_weight * 10.0
                reward += coverage_penalty
            
            # ★★★【修正箇所④】★★★
            # --- 新しいカバレッジ低下ペナルティ ---
            coverage_drop_penalty = 0.0
            coverage_drop_ratio = coverage_before - coverage_after
            
            # ★★★【修正箇所】★★★
            # ハードコーディングされた値を、コンフィグから読み込んだ変数に置き換える
            if coverage_drop_ratio > self.coverage_drop_threshold:
                coverage_drop_penalty = self.coverage_drop_weight * (coverage_drop_ratio ** 2)
            
            reward += coverage_drop_penalty
            
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
        time_penalty = params['time_penalty_per_minute'] * response_time_minutes
        reward = time_penalty
        
        # 達成ボーナス
        category = severity_to_category(severity)
        achievement_bonus = 0.0
        if category == 'critical' and response_time_minutes <= 6:
            achievement_bonus = params['critical_under_6min_bonus']
            reward += achievement_bonus
        elif category == 'moderate' and response_time_minutes <= 13:
            achievement_bonus = params['moderate_under_13min_bonus']
            reward += achievement_bonus
        elif category == 'mild' and response_time_minutes <= 13:
            achievement_bonus = params['mild_under_13min_bonus']
            reward += achievement_bonus
        
        # 閾値超過ペナルティ
        threshold_penalty = 0.0
        if response_time_minutes > 13:
            threshold_penalty += params['over_13min_penalty']
        if response_time_minutes > 20:
            threshold_penalty += params['over_20min_penalty']
        reward += threshold_penalty
        
        # 教師模倣ボーナス
        imitation_bonus = 0.0
        if additional_info and additional_info.get('matched_teacher', False):
            imitation_bonus = params.get('imitation_bonus', 0.0)
            reward += imitation_bonus
        
        # デバッグ用ログ（最初の数回のみ）
        if not hasattr(self, '_debug_simple_count'):
            self._debug_simple_count = 0
        self._debug_simple_count += 1
        
        if self._debug_simple_count <= 3:
            print(f"[Simple報酬詳細] 傷病度: {severity} ({category})")
            print(f"  応答時間: {response_time_minutes:.1f}分")
            print(f"  時間ペナルティ: {time_penalty:.2f}")
            print(f"  達成ボーナス: {achievement_bonus:.2f}")
            print(f"  閾値ペナルティ: {threshold_penalty:.2f}")
            print(f"  模倣ボーナス: {imitation_bonus:.2f}")
            print(f"  最終報酬: {reward:.2f}")
        
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
            'core_mode': self.core_mode,
            'system_config': self.system_config,
            'core_config': self.core_config,
            'episode_config': self.episode_config
        }

    def _calculate_hybrid_reward(self, state, action, next_state, info):
        """ハイブリッドモード用の報酬計算"""
        
        # 基本情報の取得
        severity = info.get('severity', '')
        response_time = info.get('response_time', 0)  # 秒単位
        dispatch_type = info.get('dispatch_type', '')
        
        # 重症系の場合は報酬なし（直近隊運用）
        if severity in self.severe_conditions or dispatch_type == 'direct_closest':
            return 0.0
        
        # 軽症系の報酬計算
        rt_minutes = response_time / 60.0
        reward_components = {}
        
        # A. 応答時間報酬（40%）
        rt_reward = self._calc_hybrid_rt_reward(rt_minutes, severity)
        reward_components['rt'] = self.weight_rt * rt_reward
        
        # B. カバレッジ報酬（50%）
        coverage_info = info.get('coverage_info', {})
        coverage_reward = self._calc_hybrid_coverage_reward(coverage_info)
        reward_components['coverage'] = self.weight_coverage * coverage_reward
        
        # C. 稼働バランス報酬（10%）
        workload_info = info.get('workload_info', {})
        workload_reward = self._calc_hybrid_workload_reward(workload_info)
        reward_components['workload'] = self.weight_workload * workload_reward
        
        # 合計報酬
        total_reward = sum(reward_components.values())
        
        # デバッグ情報
        if info.get('debug', False):
            print(f"Hybrid Reward Components: {reward_components}")
            print(f"Total Reward: {total_reward}")
        
        return total_reward
    
    def _calc_hybrid_rt_reward(self, rt_minutes, severity):
        """ハイブリッドモード：応答時間報酬の計算"""
        
        # 基本時間ペナルティ
        base_penalty = self.time_penalty_per_minute * rt_minutes
        
        # 13分以内ボーナス
        if rt_minutes <= 13:
            if severity == '軽症':
                bonus = self.mild_under_13min_bonus
            elif severity == '中等症':
                bonus = self.moderate_under_13min_bonus
            else:
                bonus = 0
            return base_penalty + bonus
        
        # 13-20分：追加ペナルティ
        elif rt_minutes <= 20:
            return base_penalty + self.over_13min_penalty
        
        # 20分超過：重いペナルティ
        else:
            over_minutes = rt_minutes - 20
            return base_penalty + self.over_20min_penalty + (self.time_penalty_per_minute * 2 * over_minutes)
    
    def _calc_hybrid_coverage_reward(self, coverage_info):
        """ハイブリッドモード：カバレッジ報酬の計算"""
        
        if not coverage_info:
            return 0.0
        
        # 高リスク地域のカバレッジを重視
        high_risk_coverage = coverage_info.get('high_risk_area_coverage', 0.7)
        overall_coverage = coverage_info.get('overall_coverage', 0.7)
        
        # 重み付き平均（高リスク地域70%、通常30%）
        weighted_coverage = 0.7 * high_risk_coverage + 0.3 * overall_coverage
        
        # カバレッジレベルに応じた報酬
        if weighted_coverage >= 0.8:
            return self.good_coverage_bonus
        elif weighted_coverage >= 0.6:
            return self.coverage_maintenance_bonus * weighted_coverage
        else:
            return self.poor_coverage_penalty * (1.0 - weighted_coverage)
    
    def _calc_hybrid_workload_reward(self, workload_info):
        """ハイブリッドモード：稼働バランス報酬の計算"""
        
        if not workload_info:
            return 0.0
        
        workload_std = workload_info.get('workload_std', 0.0)
        max_workload = workload_info.get('max_workload', 0.0)
        
        # 過負荷チェック
        if max_workload > 0.9:
            return self.overloaded_penalty
        
        # バランスの良さを評価
        if workload_std < 0.15:
            return self.balanced_workload_bonus
        else:
            return -workload_std * 5.0  # 標準偏差に応じたペナルティ