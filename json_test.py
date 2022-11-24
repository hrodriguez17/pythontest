import nltk

import csv

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import linear_kernel
import string
from nltk.corpus import stopwords
import json
import glob
import re


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return (data)


def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def remove_stops(text, stops):
    words = text.split()
    final = []
    for word in words:
        if word not in stops:
            final.append(word)
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


summary = load_data("games_data.json")["summary"]
name = load_data("games_data.json")["name"]
docFull = load_data("games_data.json")
print(summary[0])
cleaned_docs = clean_docs(summary)
print(cleaned_docs[0])
vectorizer = TfidfVectorizer(lowercase=True, max_features=100, max_df=0.8, min_df=5,
                             ngram_range=(1, 3), stop_words="english")

vectors = vectorizer.fit_transform(cleaned_docs)

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

print(all_keywords[0])

true_k = 20

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
for idx, row in docFull:
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], docFull['id'][i]) for i in similar_indices]
    results[row['id']] = similar_items[1:]


def item(id):
    return docFull.loc[docFull['id'] == id]['description'].tolist()[0].split(' - ')[0]


# Just reads the results out of the dictionary.def

def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")


recommend(item_id=11, num=5)
