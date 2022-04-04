import pandas as pd
# df = pd.read_csv('data/data_gab.csv')
df = pd.read_csv("jan_2021_match_id.csv")
df.head()

interactions_train = pd.DataFrame(
    df[["MATCHED_ID", "PRODUCT_NAME"]].dropna().groupby(["MATCHED_ID", "PRODUCT_NAME"]).size(),
    columns=["count"],
).reset_index()
interactions_train.head()


#matrix using count
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# create zero-based index position <-> user/item ID mappings
index_to_user = pd.Series(np.sort(np.unique(interactions_train['MATCHED_ID'])))
index_to_item = pd.Series(np.sort(np.unique(interactions_train['PRODUCT_NAME'])))

# create reverse mappings from user/item ID to index positions
user_to_index = pd.Series(data=index_to_user.index, index=index_to_user.values)
item_to_index = pd.Series(data=index_to_item.index, index=index_to_item.values)

# convert user/item identifiers to index positions
interactions_train_imp = interactions_train.copy()
interactions_train_imp['user_id'] = interactions_train['MATCHED_ID'].map(user_to_index)
interactions_train_imp['product_name_id'] = interactions_train['PRODUCT_NAME'].map(item_to_index)

# prepare the data for CSR creation
# data = sample_weight_train
data = interactions_train_imp ['count']
rows = interactions_train_imp['user_id']
cols = interactions_train_imp['product_name_id']

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import re

#############################################################################################################
def ngrams(string, n=3):

    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

room_types = df['PRODUCT_NAME']
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(room_types)

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))


matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, 0.8)

##################################################################################################

n_users_train = len(user_to_index)
n_items_train = len (item_to_index)

# create the required user-item and item-user CSR matrices
user_items_imp = csr_matrix((data, (rows, cols)), shape=(n_users_train, n_items_train))
item_users_imp = user_items_imp.T.tocsr()

from implicit.als import AlternatingLeastSquares
imp_model = AlternatingLeastSquares(factors=50)
imp_model.fit(item_users_imp)


#reverse
def recommend_users(als_model, plays_matrix, traid, n, tracks_mapping, users_mapping):
    """Recommends users based on a song id from the last.fm dataset

    Args:
        als_model (AlternatingLeastSquares): [description]
        plays_matrix (sparse matrix): A sparse matrix of shape
            (n_user n_items)
        traid ([type]): [description]
        n (int): Number of desired recommendations
        tracks_mapping (dict): {traid: track_idx} dict mapping the
            track ids to their idxs in the ALS factors arrays
        users_mapping (dict): {user_idx: userid} dict mapping the
            user idx in the ALS factors arrays to their userid

    Returns:
        [DataFrame]: pandas DataFrame conataining the user recommendations
            and their latent factors
    """
    item_id = tracks_mapping[traid]
#     recommendations = als_model.recommend(userid=item_id, user_items=plays_matrix, N=n)
    recommendations = als_model.recommend(item_id,item_users_imp[item_id],5)
#     print (recommendations)
#     print (len(recommendations))
#     recommendations = [
#         (users_mapping[x[0]], x[1], als_model.user_factors[x[0]])
#         for x in recommendations
#     ]
    user,acc = recommendations
#     print (user, acc)
    df_new = pd.DataFrame(list(zip(user, acc)), columns =['user', 'score'])
    return df_new


interactions_train_imp['product_name_id']= interactions_train_imp['product_name_id'].astype("category")
interactions_train_imp['user_id'] = interactions_train_imp['user_id'].astype("category")

# mapping the userid and traid to the userd index in thet ALS factors
usercode_userids = (
    pd.concat([interactions_train_imp['user_id'], interactions_train_imp['user_id'].cat.codes], axis=1)
    .drop_duplicates()
    .set_index(0)["user_id"]
).to_dict()
# mapping the musicid to the music index in the ALS factors
traid_tracode = (
    pd.concat([interactions_train_imp['product_name_id'], interactions_train_imp['product_name_id'].cat.codes], axis=1)
    .drop_duplicates()
    .set_index('product_name_id')[0]
).to_dict()

alpha = 1
recommendations = recommend_users(
    als_model=imp_model,
#     plays_matrix=alpha * plays.tocsr(),
    plays_matrix=item_users_imp,
    traid=27,
    n=10,
    tracks_mapping=traid_tracode,
    users_mapping=usercode_userids,
)
print (recommendations)

import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
def save_als_model(als_model, save_dir):
    """Saves the als model on disk as a .pickle

    Args:
        als_model ([type]): Implicit ALS model
        trial (Trial): Optuna trial object
        mapk (float): Map@k for this model
        save_dir (str): Directory to save into
    """
    Path(save_dir).mkdir(exist_ok=True)
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    name ='model'
    file_name = f"{name}-{date_time}.pickle"
    file_path = Path(save_dir) / file_name
    with open(file_path, "wb") as fout:
        pickle.dump(als_model, fout)


SAVE_DIR = "./models/"

save_als_model(als_model=imp_model, save_dir=SAVE_DIR)