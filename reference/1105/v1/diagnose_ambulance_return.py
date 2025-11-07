"""
診断スクリプト: 救急車の復帰イベントが正しく処理されているか確認
"""

import numpy as np
from datetime import datetime

def diagnose_ambulance_return():
    """救急車復帰の診断"""
    print("=" * 60)
    print("救急車復帰イベント診断")
    print("=" * 60)
    
    # シミュレーション例
    print("\n【シナリオ】")
    print("- 時刻0秒: 事案A発生、救急車0を配車")
    print("- 活動時間: 4000秒（約67分）")
    print("- 期待復帰時刻: 4000秒")
    
    current_time = 0
    dispatch_time = 0
    activity_duration = 4000  # 67分
    return_time = dispatch_time + activity_duration
    
    print(f"\n配車時刻: {dispatch_time}秒 ({dispatch_time/60:.1f}分)")
    print(f"復帰時刻: {return_time}秒 ({return_time/60:.1f}分)")
    
    # ステップ進行のシミュレーション
    print("\n【ステップ進行】")
    time_per_step = 60  # 1ステップ = 60秒
    
    for step in range(0, 70):
        current_time = step * time_per_step
        
        # 復帰イベントが発生するタイミング
        if current_time <= return_time < current_time + time_per_step:
            print(f"ステップ{step}: {current_time}秒 - 救急車0が復帰！")
            break
        elif step % 10 == 0:
            print(f"ステップ{step}: {current_time}秒 - 救急車0は活動中")
    
    print("\n" + "=" * 60)
    print("診断完了")
    print("=" * 60)


def test_event_queue_processing():
    """イベントキュー処理のテスト"""
    print("\n" + "=" * 60)
    print("イベントキュー処理テスト")
    print("=" * 60)
    
    import heapq
    from validation_simulation import Event, EventType
    
    # イベントキューの作成
    event_queue = []
    
    # 事案と復帰イベントを混在させる
    heapq.heappush(event_queue, Event(time=0, event_type=EventType.NEW_CALL, data={'call_id': 'A'}))
    heapq.heappush(event_queue, Event(time=120, event_type=EventType.NEW_CALL, data={'call_id': 'B'}))
    heapq.heappush(event_queue, Event(time=4000, event_type=EventType.AMBULANCE_AVAILABLE, data={'ambulance_id': 0}))
    heapq.heappush(event_queue, Event(time=5000, event_type=EventType.AMBULANCE_AVAILABLE, data={'ambulance_id': 1}))
    heapq.heappush(event_queue, Event(time=180, event_type=EventType.NEW_CALL, data={'call_id': 'C'}))
    
    print("\nイベントキュー（時刻順）:")
    temp_queue = event_queue.copy()
    while temp_queue:
        event = heapq.heappop(temp_queue)
        print(f"  時刻={event.time}秒: {event.event_type.value}")
    
    # 1ステップずつ処理
    print("\n【ステップごとの処理】")
    current_time = 0
    time_per_step = 60
    pending_call = None
    
    for step in range(100):
        start_time = current_time
        end_time = start_time + time_per_step
        
        events_in_step = []
        
        # このステップで処理されるイベントを確認
        while event_queue and event_queue[0].time <= end_time:
            event = event_queue[0]
            
            # 復帰イベントは常に処理
            if event.event_type == EventType.AMBULANCE_AVAILABLE:
                event = heapq.heappop(event_queue)
                events_in_step.append(event)
                continue
            
            # NEW_CALLイベント
            if event.event_type == EventType.NEW_CALL:
                if pending_call is not None:
                    # 既に事案がある場合は処理しない
                    break
                event = heapq.heappop(event_queue)
                events_in_step.append(event)
                pending_call = event.data
        
        if events_in_step:
            print(f"\nステップ{step} ({current_time}秒):")
            for event in events_in_step:
                if event.event_type == EventType.NEW_CALL:
                    print(f"  - NEW_CALL: {event.data['call_id']}")
                elif event.event_type == EventType.AMBULANCE_AVAILABLE:
                    print(f"  - AMBULANCE_AVAILABLE: 救急車{event.data['ambulance_id']}")
        
        # 事案を処理したらクリア
        if pending_call:
            # print(f"  配車: 事案{pending_call['call_id']}")
            pending_call = None
        
        current_time = end_time
        
        if not event_queue:
            print(f"\nステップ{step}: 全イベント処理完了")
            break
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


def check_completion_time_calculation():
    """活動完了時刻の計算確認"""
    print("\n" + "=" * 60)
    print("活動完了時刻の計算確認")
    print("=" * 60)
    
    # サンプル値
    current_time = 1000  # 配車時刻
    response_time = 480  # 8分 = 480秒
    on_scene_time = 15 * 60  # 15分 = 900秒
    transport_time = 12 * 60  # 12分 = 720秒
    hospital_time = 20 * 60  # 20分 = 1200秒
    return_time = 12 * 60  # 12分 = 720秒
    
    # 段階的に計算
    arrive_scene = current_time + response_time
    depart_scene = arrive_scene + on_scene_time
    arrive_hospital = depart_scene + transport_time
    depart_hospital = arrive_hospital + hospital_time
    completion = depart_hospital + return_time
    
    print(f"\n配車時刻: {current_time}秒 ({current_time/60:.1f}分)")
    print(f"現場到着: {arrive_scene}秒 ({arrive_scene/60:.1f}分)")
    print(f"現場出発: {depart_scene}秒 ({depart_scene/60:.1f}分)")
    print(f"病院到着: {arrive_hospital}秒 ({arrive_hospital/60:.1f}分)")
    print(f"病院出発: {depart_hospital}秒 ({depart_hospital/60:.1f}分)")
    print(f"帰署完了: {completion}秒 ({completion/60:.1f}分)")
    
    total_activity_time = completion - current_time
    print(f"\n総活動時間: {total_activity_time}秒 ({total_activity_time/60:.1f}分)")
    
    # 復帰イベントのスケジュール
    print(f"\n復帰イベント時刻: {completion}秒")
    print(f"これは配車から約{total_activity_time/60:.0f}分後です")
    
    print("\n" + "=" * 60)
    print("計算確認完了")
    print("=" * 60)


if __name__ == "__main__":
    print("\n救急車復帰イベントの診断を開始します\n")
    
    try:
        diagnose_ambulance_return()
        test_event_queue_processing()
        check_completion_time_calculation()
        
        print("\n" + "=" * 60)
        print("全診断完了！")
        print("=" * 60)
        print("\n主な発見:")
        print("1. 復帰イベントは配車から67分後に発生")
        print("2. イベントキューは時刻順に処理")
        print("3. 復帰イベントは常に処理される（pending_callの有無に関係なく）")
        print("4. NEW_CALLは1ステップに1件のみ処理")
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
