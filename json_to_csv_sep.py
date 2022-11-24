import pandas as pd


df = pd.read_csv('file.csv', encoding="utf-8", sep=';', nrows=10000, index_col=False)
ti_col = ['Name', 'Short Description', 'Developer', 'Publisher', 'Genre',
          'Tags', 'Categories', 'Positive Reviews', 'Negative Reviews']
df = df.iloc[0]
dff = df.join(pd.DataFrame(columns=ti_col))