import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import time


def get_game(name, df):
    print('get game running')
    dfy = df.loc[df['Name'] == name]
    game_id = dfy['id'].iloc[0]
    return game_id


def run_rec(game_id):
    with st.spinner('Wait for it...'):
        cm = CountVectorizer().fit_transform(game_data['Summary'].values.astype('U'))

        cs = cosine_similarity(cm)

        game = game_data['Name'][game_id]

        game_id = game_data[game_data.Name == game]['id'].values[0]

        scores = list(enumerate(cs[game_id]))

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        sorted_scores = sorted_scores[1:]
        j = 0
        st.write("Here are your recommendations for: " + str(game))
        for item in sorted_scores:
            game_title = game_data[game_data.id == item[0]]['Name'].values[0]
            x = str(j + 1) + ': ' + str(game_title)
            st.write(x)
            st.write('  Similarity: ' + str(item[1]))
            j = j + 1
            if j >= 5:
                break


game_data = pd.read_csv('file2.csv', encoding="utf-8", sep=';')

col_one_list = game_data['Name'].tolist()
sb_01 = st.selectbox('Select', col_one_list)
idf = get_game(sb_01, game_data)
print('printing', idf)

if st.button("Recommend Me!"):
    run_rec(get_game(sb_01, game_data))

