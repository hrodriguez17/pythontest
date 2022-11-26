import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def get_game(name, df):
    dfy = df.loc[df['Name'] == name]
    game_id = dfy['id'].iloc[0]
    return game_id


def run_rec(game_id):
    with st.spinner('Wait for it...'):
        cm = CountVectorizer(lowercase=True, max_features=10000, max_df=0.1, min_df=1, ngram_range=(1, 2),
                             stop_words="english",).fit_transform(game_data['Summary'].values.astype('U'))

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


def get_map():
    with st.spinner('Visuals...'):

        df = game_data

        vectorizer = TfidfVectorizer(lowercase=True, max_features=1000, max_df=0.1, min_df=1,
                                     ngram_range=(1, 2), stop_words="english")

        vectors = vectorizer.fit_transform(df['Summary'])
        feature_names = vectorizer.get_feature_names_out()

        dense = vectors.todense()
        denseList = dense.tolist()
        all_keywords = []
        docFull = df
        names = df["Name"]
        for summaries in denseList:
            x = 0
            keywords = []
            for word in summaries:
                if word > 0:
                    keywords.append(feature_names[x])
                x = x + 1
            all_keywords.append(keywords)

        true_k = 6
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
                  "#17becf", "#F0F8FF", "#FAEBD7", "#00FFFF", "#7FFFD4", "#F0FFFF", "#F5F5DC", "#FFE4C4"]
        model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)

        model.fit(vectors)

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()

        with open("results.txt", "w", encoding="utf-8") as f:
            for i in range(true_k):
                f.write(f"Cluster {i}")
                f.write("\n")
                for ind in order_centroids[i, :10]:
                    f.write(" %s" % terms[ind], )
                    f.write("\n")
                f.write("\n")
                f.write("\n")

        cosine_similarities = linear_kernel(vectors, vectors)
        results = {}
        print("done with txt")

        df2 = game_data
        # Map
        docFull = df2
        names = df2["Name"]
        kmean_indices = model.fit_predict(vectors)
        pca = PCA(n_components=2)
        scatter_plot_points = pca.fit_transform(vectors.toarray())

        x_axis = [o[0] for o in scatter_plot_points]
        y_axis = [o[1] for o in scatter_plot_points]

        fig, ax = plt.subplots(figsize=(40, 40))
        ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

        for i, txt in enumerate(names):
            ax.annotate(txt[0:0], (x_axis[i], y_axis[i]))
        st.pyplot(fig)
        plt.hist(model.labels_, bins=true_k)
        st.pyplot(plt)


game_data = pd.read_csv('file2.csv', encoding="utf-8", sep=';')
icols = game_data.select_dtypes('integer').columns
game_data[icols] = game_data[icols].apply(pd.to_numeric, downcast='integer')
print(game_data.dtypes)

st.title("Game Recommendation System")

col_one_list = game_data['Name'].tolist()
sb_01 = st.selectbox('Select', col_one_list)
idf = get_game(sb_01, game_data)

if st.button("Recommend Me!"):
    run_rec(get_game(sb_01, game_data))
    get_map()
