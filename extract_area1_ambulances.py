#!/usr/bin/env python3
import pandas as pd

# CSVファイルを読み込み
df = pd.read_csv('data/tokyo/import/amb_place_master.csv')

# 第1方面の救急車を抽出（section=1で救急車がある（team_nameに「救急」が含まれ、「救急隊なし」ではない））
area1_ambulances = df[
    (df['section'] == 1) & 
    (df['team_name'].str.contains('救急', na=False)) & 
    (~df['team_name'].str.contains('救急隊なし', na=False))
]

print(f"第1方面の救急車数: {len(area1_ambulances)}台")
print("\n第1方面の救急車一覧:")
for i, (_, row) in enumerate(area1_ambulances.iterrows()):
    print(f"  {i:2d}: {row['team_name']}")

# 仮想救急車の名前例も生成
print(f"\n仮想救急車名の例（48台）:")
for i in range(48):
    print(f"  {16+i:2d}: 仮想救急車{i+1:02d}号")
