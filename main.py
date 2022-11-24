import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

game_data = pd.read_csv('file2.csv', encoding="utf-8", sep=';')

columns = ['Name', 'Summary']

cm = CountVectorizer().fit_transform(game_data['Summary'].values.astype('U'))

cs = cosine_similarity(cm)

game = game_data['Name'][2]
# print(game)

game_id = game_data[game_data.Name == game]['id'].values[0]
print(game_id)

scores = list(enumerate(cs[game_id]))
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
sorted_scores = sorted_scores[1:]
j = 0
print("recommend for " + game)
for item in sorted_scores:
    game_title = game_data[game_data.id == item[0]]['Name'].values[0]
    print(j + 1, game_title)
    j = j + 1
    if j >= 5:
        break
