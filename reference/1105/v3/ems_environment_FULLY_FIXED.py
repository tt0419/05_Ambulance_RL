# ============================================================
# 救急車復帰問題 - 完全修正版
# ============================================================
# 
# 【修正内容】
# 1. _handle_ambulance_return_event: completion_timeのクリア追加
# 2. get_optimal_action: デバッグログ強化、None返却時の詳細情報出力
# 3. _load_episode_calls: H3インデックス検証の追加
#
# 【使用方法】
# このファイルの修正部分を元のems_environment_fixed.pyに適用してください
#
# ============================================================

# ------------------------------------------------
# 修正1: _handle_ambulance_return_event (完全版)
# ------------------------------------------------
# 場所: line 2241-2249

def _handle_ambulance_return_event(self, event: Event):
    """救急車復帰イベントの処理（完全修正版）"""
    amb_id = event.data['ambulance_id']
    station_h3 = event.data['station_h3']
    
    if amb_id in self.ambulance_states:
        # ★★★ 重要: 全ての状態をクリア ★★★
        self.ambulance_states[amb_id]['status'] = 'available'
        self.ambulance_states[amb_id]['current_h3'] = station_h3
        self.ambulance_states[amb_id]['completion_time'] = 0.0  # ★追加: これがないと次回配車できない★
        
        # デバッグログ（verbose時のみ）
        if hasattr(self, 'verbose_logging') and self.verbose_logging:
            available_count = sum(1 for amb in self.ambulance_states.values() 
                                if amb['status'] == 'available')
            print(f"[復帰] 救急車{amb_id}が復帰 (利用可能: {available_count}台)")


# ------------------------------------------------
# 修正2: get_optimal_action (デバッグ強化版)
# ------------------------------------------------
# 場所: line 1010-1050

def get_optimal_action(self) -> Optional[int]:
    """
    現在の事案に対して最適な救急車を選択（最近接）
    デバッグ強化版: Noneを返す原因を特定
    """
    if self.pending_call is None:
        if hasattr(self, 'verbose_logging') and self.verbose_logging:
            print("[WARN] get_optimal_action: pending_callがNone")
        return None
    
    best_action = None
    min_travel_time = float('inf')
    available_count = 0
    error_count = 0
    error_details = []
    
    # 全ての救急車をチェック
    for amb_id, amb_state in self.ambulance_states.items():
        # 利用可能な救急車のみ対象
        if amb_state['status'] != 'available':
            continue
        
        available_count += 1
        
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
            error_count += 1
            if error_count <= 3:  # 最初の3件のみ記録
                error_details.append({
                    'amb_id': amb_id,
                    'amb_h3': amb_state.get('current_h3'),
                    'error': str(e)
                })
            continue
    
    # ★★★ デバッグ: Noneを返す場合の詳細情報 ★★★
    if best_action is None and available_count > 0:
        print(f"\n{'='*60}")
        print(f"[CRITICAL] get_optimal_actionがNoneを返しました")
        print(f"{'='*60}")
        print(f"available救急車数: {available_count}台")
        print(f"エラー発生数: {error_count}件")
        print(f"pending_call.h3_index: {self.pending_call.get('h3_index')}")
        
        # H3インデックスの検証
        call_h3 = self.pending_call.get('h3_index')
        if call_h3:
            in_mapping = call_h3 in self.grid_mapping
            print(f"事案H3がgrid_mappingに存在: {in_mapping}")
            if not in_mapping:
                print(f"  → [ERROR] 事案のH3インデックスが不正: {call_h3}")
        
        # 救急車のH3も確認（最初の3台）
        checked_count = 0
        for amb_id, amb_state in self.ambulance_states.items():
            if amb_state['status'] == 'available' and checked_count < 3:
                amb_h3 = amb_state.get('current_h3')
                in_mapping = amb_h3 in self.grid_mapping if amb_h3 else False
                print(f"救急車{amb_id} H3={amb_h3}, grid_mappingに存在={in_mapping}")
                checked_count += 1
        
        # エラー詳細
        if error_details:
            print(f"\nエラー詳細（最初の3件）:")
            for detail in error_details:
                print(f"  救急車{detail['amb_id']}: {detail['error']}")
        
        print(f"{'='*60}\n")
    
    return best_action


# ------------------------------------------------
# 修正3: H3インデックス検証の追加
# ------------------------------------------------
# 場所: _load_episode_callsメソッドの最後に追加

def _load_episode_calls(self, calls_df: pd.DataFrame, episode_start: datetime, episode_end: datetime):
    """
    エピソード事案をロード（H3検証追加版）
    """
    episode_calls = []
    
    # ... 既存のコード（事案のロード） ...
    
    # ★★★ 追加: H3インデックスの検証 ★★★
    invalid_h3_count = 0
    valid_h3_count = 0
    
    for call in episode_calls:
        call_h3 = call.get('h3_index')
        if call_h3 and call_h3 in self.grid_mapping:
            valid_h3_count += 1
        else:
            invalid_h3_count += 1
            # 最初の5件のみ詳細を表示
            if invalid_h3_count <= 5:
                print(f"[WARN] 事案{call.get('id')}のH3インデックスが不正またはgrid_mappingに存在しません: {call_h3}")
    
    # 検証結果のサマリー
    total_calls = len(episode_calls)
    if total_calls > 0:
        valid_rate = (valid_h3_count / total_calls) * 100
        print(f"\nH3インデックス検証結果:")
        print(f"  有効な事案: {valid_h3_count}/{total_calls} ({valid_rate:.1f}%)")
        if invalid_h3_count > 0:
            print(f"  無効な事案: {invalid_h3_count}/{total_calls} ({(invalid_h3_count/total_calls)*100:.1f}%)")
            print(f"  → [WARNING] 無効な事案では教師あり学習が機能しません")
    
    return episode_calls


# ------------------------------------------------
# 修正4: 救急車初期化時のH3検証
# ------------------------------------------------
# 場所: _initialize_ambulances_realisticメソッド内に追加

def _initialize_ambulances_realistic(self):
    """
    現実的な救急車初期化処理（H3検証追加版）
    """
    self.ambulance_states = {}
    invalid_amb_h3_count = 0
    
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
            
            # ★★★ H3検証を追加 ★★★
            if station_h3 not in self.grid_mapping:
                invalid_amb_h3_count += 1
                if invalid_amb_h3_count <= 5:
                    print(f"  [WARN] 救急車{amb_id}のH3がgrid_mappingに存在しません: {station_h3}")
            
            # ... 既存のコード（状態辞書の作成、復帰イベントのスケジュール） ...
            
        except Exception as e:
            print(f"    ❌ 救急車{amb_id}の初期化でエラー: {e}")
            continue
    
    # H3検証結果
    if invalid_amb_h3_count > 0:
        total_amb = len(self.ambulance_states)
        print(f"  [WARN] {invalid_amb_h3_count}台の救急車のH3がgrid_mappingに存在しません")
        print(f"    → これらの救急車では移動時間計算が不正確になる可能性があります")
    
    available_count = sum(1 for st in self.ambulance_states.values() if st['status'] == 'available')
    print(f"  救急車初期化完了: {len(self.ambulance_states)}台 (初期利用可能: {available_count}台)")


# ============================================================
# 適用手順
# ============================================================
# 
# 1. ems_environment_fixed.pyをバックアップ
# 2. 上記の4つの修正を対応する行に適用
# 3. 学習を再実行
# 4. デバッグログで問題の根本原因を特定
#
# 期待される結果:
# - [CRITICAL]ログで、なぜoptimal_actionがNoneなのかが判明
# - H3検証で、どの事案/救急車が問題なのかが判明
# - 復帰イベントが正常に処理され、救急車が枯渇しない
#
# ============================================================
