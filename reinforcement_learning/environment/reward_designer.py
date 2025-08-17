"""
reward_designer.py
PPO学習用の報酬関数設計
重症・重篤・死亡を同じ重みで扱う
"""

import numpy as np
from typing import Dict, Optional

class RewardDesigner:
    """
    多目的最適化を考慮した報酬設計
    """
    
    # reward_designer.py
    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書（config.yamlから読み込み）
        """
        self.config = config
        self.severity_config = config['severity']
        self.reward_weights = config['reward']['weights']
        self.penalties = config['reward']['penalties']
        
        # 傷病度カテゴリのマッピング
        self.severity_to_category = {}
        for category, info in self.severity_config['categories'].items():
            for condition in info['conditions']:
                self.severity_to_category[condition] = category
        
        # 初期化メッセージは1回だけ（verboseフラグで制御）
        if config.get('verbose', False):
            print("報酬設計初期化完了")
            print(f"重症系（同一重み）: {self.severity_config['categories']['critical']['conditions']}")
        
    def calculate_reward(self,
                        severity: str,
                        response_time: float,
                        coverage_impact: float = 0.0,
                        additional_info: Optional[Dict] = None) -> float:
        """
        総合報酬を計算
        
        Args:
            severity: 傷病度
            response_time: 応答時間（秒）
            coverage_impact: カバレッジへの影響（0-1）
            additional_info: 追加情報（オプション）
            
        Returns:
            総合報酬値
        """
        # 各報酬成分を計算
        time_reward = self._calculate_time_reward(response_time)
        severity_reward = self._calculate_severity_reward(severity, response_time)
        threshold_penalty = self._calculate_threshold_penalty(severity, response_time)
        coverage_reward = self._calculate_coverage_reward(coverage_impact)
        
        # 追加報酬（オプション）
        bonus = 0.0
        if additional_info:
            bonus = self._calculate_additional_rewards(additional_info)
        
        # 総合報酬
        total_reward = (
            time_reward + 
            severity_reward + 
            threshold_penalty + 
            coverage_reward + 
            bonus
        )
        
        # 報酬のクリッピング（-100 ～ 100）
        total_reward = np.clip(total_reward, -100.0, 100.0)
        
        return total_reward
    
    def _calculate_time_reward(self, response_time: float) -> float:
        """応答時間に基づく報酬"""
        # 応答時間が短いほど高い報酬
        # 基準：6分（360秒）で0、それより早ければプラス、遅ければマイナス
        time_minutes = response_time / 60.0
        base_time = 6.0  # 基準時間（分）
        
        # 線形報酬
        time_reward = (base_time - time_minutes) * self.reward_weights['response_time']
        
        return time_reward
    
    def _calculate_severity_reward(self, severity: str, response_time: float) -> float:
        """傷病度に応じた報酬"""
        category = self.severity_to_category.get(severity, 'mild')
        severity_weight = self.severity_config['categories'][category]['reward_weight']
        
        # 重症・重篤・死亡（critical）は同じ重み（5.0）
        # 中等症（moderate）は2.0
        # 軽症（mild）は1.0
        
        # 目標時間内に到着した場合のボーナス
        target_time = self.severity_config['categories'][category]['time_limit_seconds']
        
        if response_time <= target_time:
            # 目標達成ボーナス
            achievement_bonus = severity_weight * self.reward_weights['severity_bonus']
            
            # 目標時間よりどれだけ早いかに応じた追加ボーナス
            time_margin = (target_time - response_time) / target_time
            achievement_bonus *= (1 + time_margin * 0.5)  # 最大1.5倍
            
            return achievement_bonus
        else:
            # 目標未達成でも部分的な報酬
            overtime_ratio = response_time / target_time
            if overtime_ratio < 2.0:  # 目標の2倍以内
                partial_reward = severity_weight * (2.0 - overtime_ratio) * 0.5
                return partial_reward
            else:
                return 0.0
    
    def _calculate_threshold_penalty(self, severity: str, response_time: float) -> float:
        """閾値超過に対するペナルティ"""
        penalty = 0.0
        
        # 6分閾値（重症系で特に重要）
        golden_time = self.severity_config['thresholds']['golden_time']
        if response_time > golden_time:
            over_minutes = (response_time - golden_time) / 60.0
            
            # 重症系の場合は厳しいペナルティ
            if severity in self.severity_config['categories']['critical']['conditions']:
                penalty += self.penalties['over_6min'] * 2.0  # 2倍のペナルティ
                penalty += self.penalties['per_minute_over'] * over_minutes * 1.5
            else:
                penalty += self.penalties['over_6min']
                penalty += self.penalties['per_minute_over'] * over_minutes
        
        # 13分閾値（全体目標）
        standard_time = self.severity_config['thresholds']['standard_time']
        if response_time > standard_time:
            over_minutes = (response_time - standard_time) / 60.0
            penalty += self.penalties['over_13min']
            penalty += self.penalties['per_minute_over'] * over_minutes * 2.0  # 追加ペナルティ
        
        return penalty
    
    def _calculate_coverage_reward(self, coverage_impact: float) -> float:
        """カバレッジ維持に対する報酬"""
        # coverage_impact: 0（影響なし）～1（大きな影響）
        # 影響が小さいほど高い報酬
        coverage_reward = self.reward_weights['coverage_preservation'] * (1.0 - coverage_impact)
        
        return coverage_reward
    
    def _calculate_additional_rewards(self, info: Dict) -> float:
        """追加報酬の計算"""
        bonus = 0.0
        
        # 効率的な配車（近い救急車を選択）
        if 'distance_rank' in info:
            if info['distance_rank'] == 1:  # 最も近い
                bonus += 2.0
            elif info['distance_rank'] <= 3:  # 上位3台
                bonus += 1.0
        
        # 稼働バランス（出動回数の少ない救急車を選択）
        if 'workload_balance' in info:
            bonus += info['workload_balance'] * 1.0
        
        return bonus
    
    def get_episode_reward(self, episode_stats: Dict) -> float:
        """
        エピソード全体の評価報酬
        
        Args:
            episode_stats: エピソード統計
            
        Returns:
            エピソード報酬
        """
        if episode_stats['total_dispatches'] == 0:
            return -100.0
        
        # 平均応答時間
        avg_response_time = np.mean(episode_stats['response_times'])
        base_reward = -avg_response_time
        
        # 6分達成率ボーナス
        rate_6min = episode_stats['achieved_6min'] / episode_stats['total_dispatches']
        bonus_6min = rate_6min * 50.0
        
        # 13分達成率ボーナス
        rate_13min = episode_stats['achieved_13min'] / episode_stats['total_dispatches']
        bonus_13min = rate_13min * 30.0
        
        # 重症系6分達成率ボーナス（最重要）
        if episode_stats['critical_total'] > 0:
            critical_rate = episode_stats['critical_6min'] / episode_stats['critical_total']
            bonus_critical = critical_rate * 100.0
        else:
            bonus_critical = 0.0
        
        # 配車失敗ペナルティ
        failure_penalty = episode_stats['failed_dispatches'] * -5.0
        
        total_reward = base_reward + bonus_6min + bonus_13min + bonus_critical + failure_penalty
        
        return total_reward