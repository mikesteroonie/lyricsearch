import cohere
import numpy as np
import re
import pandas as pd
import json
from tqdm import tqdm
from datasets import load_dataset
import umap
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings
import time
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

# Load JSON file
with open('lyrics-data/drake.json') as f:
    data = json.load(f)
 # Convert each lyrics to a list of lines
for d in data:
    if d["lyrics"] is not None:  
        lines = d["lyrics"].split('\n')[::2]
        titles = [d["lyrics_title"]] * len(lines)
        d["lyrics"] = list(zip(titles, lines))
    else:
        d["lyrics"] = []

num_rows = 10000  
df = pd.DataFrame(data).head(num_rows)
df = df.explode('lyrics')
df[['lyrics_title', 'lyrics']] = pd.DataFrame(df['lyrics'].tolist(), index=df.index)
df.drop_duplicates(subset=['lyrics_title', 'lyrics'], inplace=True)
#print first 10 rows just to see if data is in properly

#print(df.head(10))

api_key ='OvRr5AjLI98AuCFtlK3KjPE8bT25NVL9BxxK6w3z'
co = cohere.Client(api_key)

chunk_size = 50  
chunks = [df['lyrics'][i:i + chunk_size] for i in range(0, df['lyrics'].shape[0], chunk_size)]
embeds = []

for i, chunk in enumerate(chunks):
    chunk = chunk.dropna()  
    chunk = chunk.replace([np.inf, -np.inf], '')  

    chunk_embeds = co.embed(texts=list(chunk), model='embed-english-v2.0').embeddings
    embeds.extend(chunk_embeds)

    #doing this because of cohere api limit of 100 calls per minute
    if i % 100 == 99 and i != len(chunks) - 1:
        time.sleep(60)  

search_index = AnnoyIndex(np.array(embeds).shape[1], 'angular')

for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10)  
search_index.save('test.ann')

query = "you are hot and I want to take you out"
query_embed = co.embed(texts=[query], model="embed-english-v2.0").embeddings
similar_item_ids = search_index.get_nns_by_vector(query_embed[0], 10, include_distances=True)

for idx, dist in zip(similar_item_ids[0], similar_item_ids[1]):
    similarity = 1 - (dist / 2) + 0.3
    print(f"Similarity: {similarity:.3f} Lyrics: {df['lyrics'].iloc[idx]} Title: {df['lyrics_title'].iloc[idx]}")