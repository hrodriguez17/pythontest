import numpy as np
import pandas as pd
import nltk.corpus
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string


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


columns = ['Name', 'Short Description']


def combine_features(data):
    features = []
    for i in range(0, data.shape[0]):
        print(data)
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
df = df.drop(["Languages", "Platforms", "Type", "Owners", "Price", "Initial Price", "Discount", "CCU", "Release Date",
              "Required Age", "Website", "Header Image", "App ID"], axis=1)
df = df.sort_values(by=['Positive Reviews'], ascending=False)
df.to_csv('file.csv', sep=';', index=False)

df = pd.read_csv('file.csv', encoding="utf-8", sep=';', nrows=10000, index_col=False)
print(df.keys())

column = df.columns.tolist()
print(column)
features = []
print('hi', df['Name'][0])

for i in range(0, df.shape[0]):
    features.append(str(df['Short Description'][i]) + ' ' + str(df['Developer'][i]) + ' '
                    + str(df['Publisher'][i]) + ' ' + str(df['Genre'][i]) + ' ' + str(df['Tags'][i]))

df["Summary"] = features
desc = []
for item in df["Summary"]:
    desc.append(remove_stops(item, stop_words))
df["Summary"] = desc
df.to_csv('file.csv', sep=';', index=True, index_label='id')
df = pd.read_csv('file.csv', encoding="utf-8", sep=';')
print(df)

df2 = df.drop(["Short Description", "Developer", "Publisher", "Genre", "Tags", "Categories"], axis=1)
df2 = df2[["Name", "Summary", "Positive Reviews", "Negative Reviews"]]
print(df2)
df2 = df2.drop_duplicates()
df2.to_csv('file2.csv', sep=';', index_label='id')
# for i in range(0, df.shape[0]):
#     features.append(df['Name'][i] + " " + df['Short Description'][i] + ' ' + df['Developer'][i] + ' '
#                     + df['Publisher'][i] + ' ' + df['Genre'][i] + ' ' + df['Tags'][i])