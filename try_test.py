import pandas as pd
df = pd.read_csv('./../datapoc/jan_2021_match_id.csv', sep=';', encoding='iso-8859-1')
print (df.head())

import pickle
filename = "./models/model-20220413-061248.pickle"
loaded_model = pickle.load(open(filename, 'rb'))

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
interactions_train_imp['product_id'] = interactions_train['PRODUCT_NAME'].map(item_to_index)

# prepare the data for CSR creation
# data = sample_weight_train
data = interactions_train_imp ['count']
rows = interactions_train_imp['user_id']
cols = interactions_train_imp['product_id']

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

def test_trainid (traid):
    alpha = 1
    recommendations = recommend_users(
        als_model=loaded_model,
    #     plays_matrix=alpha * plays.tocsr(),
        plays_matrix=item_users_imp,
        traid=traid,
        n=10,
        tracks_mapping=traid_tracode,
        users_mapping=usercode_userids,
    )
    # print (recommendations)
    return (recommendations.user[0])

def test_score (traid):
    alpha = 1
    recommendations = recommend_users(
        als_model=loaded_model,
    #     plays_matrix=alpha * plays.tocsr(),
        plays_matrix=item_users_imp,
        traid=traid,
        n=10,
        tracks_mapping=traid_tracode,
        users_mapping=usercode_userids,
    )
    # print (recommendations)
    return (recommendations.score[0])

df_test = interactions_train_imp
df_test["train_id"] = df_test['product_id'].apply(lambda x: test_trainid (x))
df_test["score"] = df_test['product_id'].apply(lambda x: test_score (x))

df_test