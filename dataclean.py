import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import linear_kernel


def remove_stops(text, stops):
    text = re.sub("new", "", text)
    text = re.sub("access", "", text)
    text = re.sub("ACCESS", "", text)
    text = re.sub("free", "", text)
    text = re.sub("early", "", text)
    text = re.sub("Early", "", text)
    text = re.sub("EARLY", "", text)
    text = re.sub("Access", "", text)
    text = re.sub("Free", "", text)
    text = re.sub("FREE", "", text)
    text = re.sub("Play ", "", text)
    text = re.sub("play ", "", text)
    text = re.sub("Indie ", "", text)
    text = re.sub(" iii", "three", text)
    text = re.sub(" ii", "two", text)
    text = re.sub(" III", "three", text)
    text = re.sub(" II", "two", text)
    text = re.sub("beat em", "beatem", text)
    text = re.sub("shoot em", "shootem", text)
    text = re.sub("Shoot Em", "shootem", text)
    text = re.sub("Beat Em", "beatem", text)
    text = re.sub("Shoot em", "shootem", text)
    text = re.sub("Beat em", "beatem", text)
    text = re.sub("Shoot Em Up", "shootemup", text)
    text = re.sub("Beat Em Up", "beatemup", text)
    text = re.sub("shootem up", "shootemup", text)
    text = re.sub("beatem up", "beatemup", text)
    text = re.sub("content ", "", text)
    text = re.sub("Content ", "", text)
    text = re.sub("controller ", "", text)
    text = re.sub("Controller ", "", text)
    text = re.sub("point click ", "pointclick", text)
    text = re.sub("Point Click ", "pointclick", text)
    text = re.sub("Point click ", "pointclick", text)
    text = re.sub("point Click ", "pointclick", text)
    text = re.sub("Point & Click", "pointclick", text)

    things = text.split()
    final = []
    for thing in things:
        if thing not in stops:
            if thing not in final:
                final.append(thing)
    final = " ".join(final)
    final = final.translate(str.maketrans("", "", string.punctuation))
    final = "".join([i for i in final if not i.isdigit()])
    while "  " in final:
        final = final.replace("  ", " ")
    return final


def clean_docs(docs):
    stops = stopwords.words("english")
    final = []
    for doc in docs:
        clean_doc = remove_stops(doc, stops)
        final.append(clean_doc)
    return final


columns = ['Name', 'Short Description']


def combine_features(data):
    features = []
    for i in range(0, data.shape[0]):
        features.append(data['Name'][i] + ' ' + data['Short Description'][i] + ' ' + data['Developer'][i] + ' '
                        + data['Publisher'][i] + ' ' + data['Genre'][i] + ' ' + data['Tags'][i] + ' ' + data['name'][i])
    return features


pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

stemmer = PorterStemmer()
stop_words = stopwords.words('english')

df = pd.read_csv('steam_games.csv', encoding="utf-8", sep=';', low_memory=False)
df = df.drop(["Languages", "Platforms", "Owners", "Price", "Initial Price", "Discount", "CCU", "Release Date",
              "Required Age", "Website", "Header Image", "App ID"], axis=1)
df = df.sort_values(by=['Positive Reviews'], ascending=False)
df.to_csv('file.csv', sep=';', index=False)

df = pd.read_csv('file.csv', encoding="utf-8", sep=';', nrows=10000, index_col=False)

column = df.columns.tolist()

features = []

for i in range(0, df.shape[0]):
    features.append(str(df['Developer'][i]) + ' '
                    + str(df['Publisher'][i]) + ' ' + str(df['Genre'][i]) + ' ' + str(df['Tags'][i]))

df["Summary"] = features
desc = []
for item in df["Summary"]:
    desc.append(remove_stops(item, stop_words))
df["Summary"] = desc

desc2 = []
char_to_replace = {"[^\x00-\x7F]+": ':'}
for item in df["Name"]:
    for key, value in char_to_replace.items():
        words = re.sub(key, value, item)
    desc2.append(words)
df["Name"] = desc2
desc3 = []
for item in df["Name"]:
    for key, value in char_to_replace.items():
        words = re.sub(" The ", " ", item)
    desc3.append(words)
df["Name"] = desc3

df.to_csv('file.csv', sep=';', index=True, index_label='id')
df = pd.read_csv('file.csv', encoding="utf-8", sep=';')
df2 = df[["Name", "Summary"]]

df2 = df2.drop_duplicates(subset=["Name"])
vectorizer = TfidfVectorizer(lowercase=True, max_features=10000, max_df=0.2, min_df=500,
                             ngram_range=(1, 1), stop_words="english")

vectors = vectorizer.fit_transform(df2['Summary'])
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

true_k = 36
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
          "#17becf",
          "#F0F8FF", "#FAEBD7", "#00FFFF", "#7FFFD4", "#F0FFFF", "#F5F5DC", "#FFE4C4", "#000000", "#FFEBCD",
          "#0000FF",
          "#8A2BE2", "#A52A2A", "#DEB887", "#5F9EA0", "#7e1e9c", "#7FFF00", "#D2691E", "#FFF8DC", "#DC143C",
          "#00008B",
          "#B8860B", "#A9A9A9", "#006400", "#BDB76B", "#8B008B", "#556B2F"]

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


def item(id):
    return docFull.loc[docFull['id'] == id]['description'].tolist()[0].split(' - ')[0]


df2.to_csv('file2.csv', sep=';', index=False)
df2 = pd.read_csv('file2.csv', encoding="utf-8", sep=';')
df2.to_csv('file2.csv', sep=';', index_label='id')
