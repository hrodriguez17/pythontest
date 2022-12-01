import streamlit as st
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


def get_game(name, df):
    dfy = df.loc[df['Name'] == name]
    game_id = dfy['id'].iloc[0]
    return game_id


def pop_bar(chart):
    chart = chart.set_index("Name")
    st.bar_chart(chart)


def run_rec(game_id):
    with st.spinner('Wait for it...'):
        game = game_data['Name'][game_id]
        game_id = game_data[game_data.Name == game]['id'].values[0]

        st.write("Here are your recommendations for: " + str(game))
        column = ["Name", "Positive Reviews"]
        chart = pd.DataFrame(columns=column)

        tfidf = TfidfVectorizer(lowercase=True, max_features=10000, max_df=.2, min_df=50,
                                ngram_range=(1, 1), stop_words="english")
        tfidf_matrix = tfidf.fit_transform([str(i) for i in game_data['Summary']])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(game_data.index, index=game_data['Name'])
        idx = indices[game_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        j = 0

        for item in sim_scores:
            game_title = game_data[game_data.id == item[0]]['Name'].values[0]
            game_reviews = game_data[game_data.id == item[0]]['Positive Reviews'].values[0]
            x = str(j + 1) + ': ' + str(game_title)
            st.write(x)
            st.write('  Similarity: ' + str(item[1]))
            new_row = {"Name": game_title, "Positive Reviews": game_reviews}
            chart = chart.append(new_row, ignore_index=True)
            j = j + 1

        st.caption("This is a bar graph showing these top 5 games and their amounts of positive reviews.")
        pop_bar(chart)


def get_map():
    with st.spinner('Visuals...'):

        df = game_data

        vectorizer = TfidfVectorizer(lowercase=True, max_features=9000, max_df=.12, min_df=10,
                                     ngram_range=(1, 1), stop_words="english")

        vectors = vectorizer.fit_transform(df['Summary'])
        feature_names = vectorizer.get_feature_names_out()

        dense = vectors.todense()
        denseList = dense.tolist()
        all_keywords = []

        for summaries in denseList:
            x = 0
            keywords = []
            for word in summaries:
                if word > 0:
                    keywords.append(feature_names[x])
                x = x + 1
            all_keywords.append(keywords)
        # with open("words.txt", "w", encoding="utf-8") as f:
        #     for i in all_keywords:
        #         for ind in i:
        #             f.write(ind)
        #             f.write("\n")
        # print('words done')
        true_k = 6

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)

        model.fit(vectors)

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()

        with open("results.txt", "w", encoding="utf-8") as f:
            for i in range(true_k):
                f.write(f"Cluster {i - 1}")
                f.write("\n")
                for ind in order_centroids[i, :10]:
                    f.write(" %s" % terms[ind], )
                    f.write("\n")
                f.write("\n")
                f.write("\n")

        arr = []

        for i in range(true_k):
            col = [(f"Cluster {i}")]
            for ind in order_centroids[i, :10]:
                col.append(terms[ind])
            arr.append(col)

        df = pd.DataFrame(arr)
        df = df.transpose()
        new_header = df.iloc[0]  # grab the first row for the header
        df = df[1:]  # take the data less the header row
        df.columns = new_header

        image = Image.open('wordcloud.png')
        st.caption("This is a WordCloud of all the keywords used for prediction.")
        st.image(image)
        st.caption("This is a table with the keywords organized into clusters for the prediction model.")
        st.table(df)

        kmean_indices = model.fit_predict(vectors)
        pca = PCA(n_components=2)
        scatter_plot_points = pca.fit_transform(vectors.toarray())

        x_axis = [o[0] for o in scatter_plot_points]
        y_axis = [o[1] for o in scatter_plot_points]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i),
                                  markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]
        plt.legend(handles=legend_elements, loc='upper right')
        st.caption("This is a cluster map to provide visualization of how clusters are used to predict certain games.")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 10))
        plt.hist(model.labels_, bins=true_k)
        rects = ax.patches
        labels = ["Cluster%d" % i for i in range(len(rects))]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
                    ha='center', va='bottom')
        st.caption("This is a histogram showing how the keywords are clumped into each cluster.")
        st.pyplot(plt)


game_data = pd.read_csv('file2.csv', encoding="utf-8", sep=';')
icols = game_data.select_dtypes('integer').columns
game_data[icols] = game_data[icols].apply(pd.to_numeric, downcast='integer')


# text_file = open("words.txt", "r")
# text = text_file.read()
# text_file.close()
# wordcloud = WordCloud().generate(text)
# plt.figure(figsize=(10, 5), facecolor='k')
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show()

st.title("Game Recommendation System")
st.text("Please select a game to help us find recommendations based off your interests.")

col_one_list = game_data['Name'].tolist()
sb_01 = st.selectbox('Select', col_one_list)
idf = get_game(sb_01, game_data)

if st.button("Recommend Me!"):
    run_rec(get_game(sb_01, game_data))
    get_map()
