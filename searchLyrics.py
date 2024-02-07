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

def count_tokens(text):
    # Roughly estimate the number of tokens (words and punctuation)
    return len(text.split())

# Load JSON file
with open('lyrics-data/drake.json') as f:
    data = json.load(f)
 # Convert each lyrics to a list of lines
for d in data:
    if d["lyrics"] is not None:  
        lines = d["lyrics"].split('\n')
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

api_key ='INSERTAPIKEY'
co = cohere.Client(api_key)

chunk_size = 50  
chunks = [df['lyrics'][i:i + chunk_size] for i in range(0, df['lyrics'].shape[0], chunk_size)]
embeds = []

calls_per_minute = 10000  # Example: limit to 5000 calls per minute
sleep_time = 60 / calls_per_minute  # Time to sleep after each call

for chunk in chunks:
    chunk = chunk.dropna()  
    chunk = chunk.replace([np.inf, -np.inf], '')  

    chunk_text = list(chunk)

    # Count tokens in this chunk
    chunk_tokens = sum(count_tokens(text) for text in chunk_text)

    chunk_embeds = co.embed(texts=list(chunk), input_type='search_document', model='embed-english-v3.0').embeddings
    embeds.extend(chunk_embeds)

    time.sleep(sleep_time)  # Rate limiting

search_index = AnnoyIndex(np.array(embeds).shape[1], 'angular')

for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10)  
search_index.save('test.ann')

query = "I think you are attractive and want to take you out on a date"
query_embed = co.embed(texts=[query], input_type='search_query', model="embed-english-v3.0").embeddings
similar_item_ids = search_index.get_nns_by_vector(query_embed[0], 10, include_distances=True)

for idx, dist in zip(similar_item_ids[0], similar_item_ids[1]):
    similarity = 1 - (dist / 2)
    print(f"Similarity: {similarity:.3f} Lyrics: {df['lyrics'].iloc[idx]} Title: {df['lyrics_title'].iloc[idx]}")

total_tokens_used = chunk_tokens * 1000
print(f"Total tokens used: {total_tokens_used}")