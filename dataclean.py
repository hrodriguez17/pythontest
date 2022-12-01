import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.cluster import KMeans


def remove_stops(text, stops):
    text = re.sub("new", "", text)
    text = re.sub("New", "", text)
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
    text = re.sub("Massively Multiplayer", "mmo", text)
    text = re.sub("Studio", "", text)
    text = re.sub("Studio ", "", text)
    text = re.sub("studio ", "", text)
    text = re.sub("STUDIO", "", text)
    text = re.sub("studios", "", text)
    text = re.sub("Studios", "", text)
    text = re.sub("ReValue", "", text)
    text = re.sub("matter", "", text)
    text = re.sub("Matter", "", text)
    text = re.sub("choose", "", text)
    text = re.sub("Choose", "", text)
    text = re.sub("choices", "", text)
    text = re.sub("Choices", "", text)
    text = re.sub("crafting", "craft", text)
    text = re.sub("Crafting", "craft", text)
    text = re.sub("character", "", text)
    text = re.sub("Character", "", text)
    text = re.sub("Character", "", text)
    text = re.sub("local", "", text)
    text = re.sub("Local", "", text)
    text = re.sub("family friendly", "familyfriendly", text)
    text = re.sub("Family friendly", "familyfriendly", text)
    text = re.sub("family Friendly", "familyfriendly", text)
    text = re.sub("Family Friendly", "familyfriendly", text)
    text = re.sub(" friendly", "", text)
    text = re.sub(" Friendly", "", text)
    text = re.sub("friends", "", text)
    text = re.sub("Friends", "", text)
    text = re.sub("llc", "", text)
    text = re.sub("LLC", "", text)
    text = re.sub("digital", "", text)
    text = re.sub("different", "", text)
    text = re.sub(" amp", "", text)
    text = re.sub("amp ", "", text)
    text = re.sub(" make ", "", text)
    text = re.sub(" Make ", "", text)
    text = re.sub(" year ", "", text)
    text = re.sub(" Year ", "", text)
    text = re.sub("year ", "", text)
    text = re.sub("Year ", "", text)
    text = re.sub("Player", "", text)
    text = re.sub("player", "", text)
    text = re.sub("Players", "", text)
    text = re.sub("players", "", text)
    text = re.sub("content", "", text)
    text = re.sub("Content", "", text)
    text = re.sub(" building", "build", text)
    text = re.sub(" Building", "build", text)
    text = re.sub("builder", "build", text)
    text = re.sub("Builder", "build", text)
    text = re.sub("builderbuilder", "build", text)
    text = re.sub("amp", "", text)
    text = re.sub("Amp", "", text)
    text = re.sub("bring", "", text)
    text = re.sub("Bring", "", text)
    text = re.sub("place", "", text)
    text = re.sub("Place", "", text)
    text = re.sub("join", "", text)
    text = re.sub("Join", "", text)
    text = re.sub("come", "", text)
    text = re.sub("Come", "", text)
    text = re.sub("alternate", "", text)
    text = re.sub("Alternate", "", text)
    text = re.sub("pc", "", text)
    text = re.sub("Pc", "", text)
    text = re.sub("PC", "", text)
    text = re.sub("mode", "", text)
    text = re.sub("Mode", "", text)
    text = re.sub("dy", "", text)
    text = re.sub("Dy", "", text)
    text = re.sub("perma death", "permadeath", text)
    text = re.sub("Perma death", "permadeath", text)
    text = re.sub("perma Death", "permadeath", text)
    text = re.sub("Perma Death", "permadeath", text)
    text = re.sub("popular", " ", text)
    text = re.sub("Popular", " ", text)
    text = re.sub("genre", "", text)
    text = re.sub("Genre", "", text)
    text = re.sub("tactics", "tactical", text)
    text = re.sub("Tactics", "tactical", text)
    text = re.sub("returns", "", text)
    text = re.sub("Returns", "", text)
    text = re.sub(" way", "", text)
    text = re.sub("Way", "", text)
    text = re.sub("endings", "", text)
    text = re.sub("Endings", "", text)
    text = re.sub("sexual", "mature", text)
    text = re.sub("Sexual", "mature", text)
    text = re.sub("nudity", "mature", text)
    text = re.sub("Nudity", "mature", text)
    text = re.sub("Hentai", "mature", text)
    text = re.sub("hentai", "mature", text)
    text = re.sub("Multiple", "", text)
    text = re.sub("multiple", "", text)
    text = re.sub(" game", "", text)
    text = re.sub(" Game", "", text)
    text = re.sub(" games", "", text)
    text = re.sub(" Games", "", text)
    text = re.sub("simulation", "sim", text)
    text = re.sub("Simulation", "sim", text)
    text = re.sub("Single", "", text)
    text = re.sub("single", "", text)
    text = re.sub("Adventure", "", text)
    text = re.sub("adventure", "", text)



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

features = []

for i in range(0, df.shape[0]):
    features.append(str(df['Name'][i]) + str(df['Short Description'][i]) + str(df['Developer'][i]) + ' '
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
df2 = df[["Name", "Summary", "Positive Reviews"]]

df2 = df2.drop_duplicates(subset=["Name"])

vectorizer = TfidfVectorizer(lowercase=True, max_features=10000, max_df=0.8, min_df=1,
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

true_k = 6

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

df2.to_csv('file2.csv', sep=';', index=False)
df2 = pd.read_csv('file2.csv', encoding="utf-8", sep=';')
df2.to_csv('file2.csv', sep=';', index_label='id')
# print(df2)
