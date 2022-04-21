from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

embedder = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('./../datapoc/jan_2021_match_id.csv', sep=';', encoding='iso-8859-1')
df.head()

df_nielsen=df[(df['SOURCE'] == 'nielsen')]
df_nielsen.head()

df_acara=df[(df['SOURCE'] == 'GENSCTV')]
df_acara.head()

############bert#########################
corpus = df_nielsen['PRODUCT_NAME'].values.tolist()
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

def search_matching (query, corpus):
    top_k = min(1, len(corpus))
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    
    for score, idx in zip(top_results[0], top_results[1]):
        print (corpus[idx])
        print ("{:.4f}".format(score))
        return ((corpus[idx]),("{:.4f}".format(score)))

df_matching = df_acara
df_matching ['product matching'] = df_matching['product'].apply(lambda x : search_matching(x,corpus)[0])
df_matching ['score matching'] = df_matching['product'].apply(lambda x : search_matching(x,corpus)[1])
df_matching.head()

df.to_csv('./matching_bert', sep=';', encoding= 'iso=8859-1')

#######################fuzzy matcher##########################################
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
from fuzzymatcher import link_table, fuzzy_left_join
import fuzzymatcher
from datetime import datetime

start_time = datetime.now()
df3 = fuzzymatcher.link_table(df_acara, df_nielsen, 'product', 'product')
end_time = datetime.now()
print ("duration", (end_time - start_time))

df3.head()
df.to_csv('./matching_fuzzy', sep=';', encoding= 'iso=8859-1')
