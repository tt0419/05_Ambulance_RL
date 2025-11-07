# ============================================================
# 根本的な解決策: step()の完全な修正
# ============================================================
# 
# 【適用ファイル】
# reinforcement_learning/environment/ems_environment.py
# 
# 【修正箇所】
# step()メソッド全体を置き換え
# 
# ============================================================

def step(self, action: int) -> StepResult:
    """
    環境のステップ実行（根本的修正版）
    
    正しい処理順序:
    1. イベント処理（NEW_CALL設定を含む）
    2. 配車処理（actionを実行）
    3. 時間を進める
    
    重要な修正:
    - タイムアウト処理の追加
    - イベント処理を配車の前に実行
    - pending_callの状態管理を明確化
    """
    try:
        end_time = self.current_time_seconds + self.time_per_step
        reward = 0.0
        info = {}
        
        # デバッグ（最初の3ステップのみ）
        if self.episode_step <= 3:
            print(f"\n[STEP DEBUG] ステップ{self.episode_step}開始")
            print(f"  current_time: {self.current_time_seconds}秒")
            print(f"  pending_call(前): {'あり' if self.pending_call else 'なし'}")
            if self.pending_call:
                wait_time = self.current_time_seconds - self.call_start_times.get(
                    self.pending_call['id'], self.current_time_seconds
                )
                print(f"  待機時間: {wait_time:.1f}秒")
        
        # ===== Phase 1: イベント処理 =====
        events_processed = 0
        new_calls_processed = 0
        
        while self.event_queue and self.event_queue[0].time <= end_time:
            event = self.event_queue[0]
            
            # 復帰イベントは常に処理
            if event.event_type == EventType.AMBULANCE_AVAILABLE:
                self._process_next_event()
                events_processed += 1
                continue
            
            # NEW_CALLイベント
            if event.event_type == EventType.NEW_CALL:
                if self.pending_call is None:
                    # pending_callが空なら、新しい事案を設定
                    self._process_next_event()
                    events_processed += 1
                    new_calls_processed += 1
                else:
                    # ★タイムアウトチェック★
                    wait_time_seconds = self.current_time_seconds - self.call_start_times.get(
                        self.pending_call['id'], self.current_time_seconds
                    )
                    max_wait_seconds = self._get_max_wait_time(self.pending_call['severity']) * 60
                    
                    if wait_time_seconds > max_wait_seconds:
                        # タイムアウト: 古い事案を放棄して新しい事案を処理
                        print(f"[TIMEOUT] 事案{self.pending_call['id']}をタイムアウト（{wait_time_seconds/60:.1f}分待機）")
                        self._handle_unresponsive_call(self.pending_call, wait_time_seconds / 60)
                        
                        # 古い事案をクリアして新しい事案を処理
                        self.pending_call = None
                        self._process_next_event()
                        events_processed += 1
                        new_calls_processed += 1
                    else:
                        # まだタイムアウトしていない: 次のステップで処理
                        break
            else:
                # その他のイベント
                self._process_next_event()
                events_processed += 1
        
        # デバッグ
        if self.episode_step <= 3:
            print(f"  イベント処理数: {events_processed}（NEW_CALL: {new_calls_processed}件）")
            print(f"  pending_call(後): {'あり' if self.pending_call else 'なし'}")
            if self.pending_call:
                print(f"    事案ID: {self.pending_call.get('id')}")
        
        # ===== Phase 2: 配車処理 =====
        if self.pending_call is not None:
            current_incident = self.pending_call
            
            # ハイブリッドモード処理
            if self.hybrid_mode:
                severity = current_incident.get('severity', '')
                if severity in self.severe_conditions:
                    self.direct_dispatch_count += 1
                    closest_action = self._get_closest_ambulance_action(current_incident)
                    dispatch_result = self._dispatch_ambulance(closest_action)
                    reward = 0.0
                    info = {'dispatch_type': 'direct_closest', 'skipped_learning': True}
                else:
                    self.ppo_dispatch_count += 1
                    dispatch_result = self._dispatch_ambulance(action)
                    reward = self._calculate_reward(dispatch_result)
                    info = {'dispatch_type': 'ppo_learning'}
            else:
                dispatch_result = self._dispatch_ambulance(action)
                reward = self._calculate_reward(dispatch_result)
                info = {'dispatch_type': 'ppo_normal'}
            
            # 配車結果の処理
            if dispatch_result and dispatch_result['success']:
                self._log_dispatch_action(dispatch_result, self.ambulance_states[dispatch_result['ambulance_id']])
                self._update_statistics(dispatch_result)
                
                # 復帰イベントをスケジュール
                amb_id = dispatch_result['ambulance_id']
                return_time = dispatch_result.get('completion_time_seconds', 
                                                 self.current_time_seconds + 4000)
                
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
                # 配車失敗: pending_call保持（再試行）
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
        
        # ===== Phase 3: 時間を進める =====
        self.current_time_seconds = end_time
        self.episode_step += 1
        
        # 終了判定
        done = self._is_episode_done()
        
        # 観測を取得
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
        return StepResult(
            observation=np.zeros(self.state_dim),
            reward=0.0,
            done=True,
            info={'error': str(e)}
        )

# ============================================================
# 適用手順
# ============================================================
# 
# 1. ems_environment.pyのstep()メソッドを上記のコードで置き換え
# 2. 学習を再実行
# 3. デバッグログで確認:
#    - イベント処理数が増加しているか
#    - NEW_CALLが処理されているか
#    - total_callsが1526件に近づくか
#    - タイムアウトが発生しているか
# 
# 期待される結果:
# - total_calls: 1526件（100%）
# - [TIMEOUT]ログが数回出る（全隊出場中の場合）
# - すべての事案が処理される
# 
# ============================================================

