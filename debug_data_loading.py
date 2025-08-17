"""
debug_data_loading.py
データ読み込みの詳細デバッグ
"""

import pandas as pd
import os
from datetime import datetime

def debug_data_loading():
    """データ読み込みの詳細確認"""
    
    # パスのリスト（優先順位順）
    possible_paths = [
        "C:/Users/tetsu/OneDrive - Yokohama City University/30_データカタログ/tfd_data/hanso_special_wards.csv",
        "C:/Users/hp/OneDrive - Yokohama City University/30_データカタログ/tfd_data/hanso_special_wards.csv",
    ]
    
    # 有効なパスを探す
    calls_path = None
    for path in possible_paths:
        if os.path.exists(path):
            calls_path = path
            print(f"✓ データファイル発見: {path}")
            break
    
    if calls_path is None:
        print("❌ データファイルが見つかりません")
        return
    
    # データ読み込み
    print("\nデータ読み込み中...")
    calls_df = pd.read_csv(calls_path, encoding='utf-8')
    print(f"読み込み完了: {len(calls_df)}行")
    
    # カラム確認
    print("\nカラム一覧:")
    for col in calls_df.columns:
        print(f"  - {col}")
    
    # 日付カラムの確認
    print("\n日付カラムの最初の5件:")
    print(calls_df['出場年月日時分'].head())
    
    # 日付変換テスト
    print("\n日付変換テスト...")
    calls_df['出場年月日時分'] = pd.to_datetime(calls_df['出場年月日時分'], errors='coerce')
    
    # NaT（Not a Time）の確認
    nat_count = calls_df['出場年月日時分'].isna().sum()
    print(f"変換失敗（NaT）: {nat_count}件")
    
    # 有効なデータのみ
    valid_df = calls_df.dropna(subset=['出場年月日時分'])
    print(f"有効なデータ: {len(valid_df)}件")
    
    # 日付範囲の確認
    if len(valid_df) > 0:
        min_date = valid_df['出場年月日時分'].min()
        max_date = valid_df['出場年月日時分'].max()
        print(f"\n日付範囲: {min_date} ～ {max_date}")
        
        # 2023年4月のデータ確認
        april_2023_start = pd.to_datetime('2023-04-01')
        april_2023_end = pd.to_datetime('2023-04-30 23:59:59')
        
        april_mask = (valid_df['出場年月日時分'] >= april_2023_start) & (valid_df['出場年月日時分'] <= april_2023_end)
        april_data = valid_df[april_mask]
        
        print(f"\n2023年4月のデータ: {len(april_data)}件")
        
        if len(april_data) > 0:
            # 日ごとの件数
            april_data['date'] = april_data['出場年月日時分'].dt.date
            daily_counts = april_data['date'].value_counts().sort_index()
            
            print("\n日ごとの件数（最初の7日）:")
            for date, count in daily_counts.head(7).items():
                print(f"  {date}: {count}件")
            
            # 座標の確認
            print("\n座標データの確認:")
            print(f"  Y_CODE（緯度）のNaN: {april_data['Y_CODE'].isna().sum()}件")
            print(f"  X_CODE（経度）のNaN: {april_data['X_CODE'].isna().sum()}件")
            
            # サンプルデータ表示
            print("\nサンプルデータ（最初の3件）:")
            sample_cols = ['救急事案番号キー', '出場年月日時分', 'Y_CODE', 'X_CODE', '収容所見程度']
            print(april_data[sample_cols].head(3))

if __name__ == "__main__":
    debug_data_loading()