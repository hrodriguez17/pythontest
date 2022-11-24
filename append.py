import pandas as pd

import csv

game_data = pd.read_csv('all_games_test.csv', encoding="utf-8")
print(game_data)
new_data = game_data.drop_duplicates(subset=['name'])
new_data.drop(index=new_data.index[0])
# new_data.loc[-1] = ['id', 'name', 'platform', 'release_date', 'summary', 'meta_score', 'user_review']
# new_data.index = new_data + 1
# new_data = new_data.sort_index()
filename2 = 'all_games_test1.csv'
filename = 'all_games_test.csv'
print(new_data)
new_data.to_csv(filename2, index=False)
game_data2 = pd.read_csv('all_games_test1.csv', encoding="utf-8")
game_data2.to_csv('all_games_test2.csv')
