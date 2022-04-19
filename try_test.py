import pandas as pd
df = pd.read_csv('data/data_gab.csv')
df.head()

import pickle
filename = "./models/model-20220419-084236.pickle"
loaded_model = pickle.load(open(filename, 'rb'))

interactions_train = pd.DataFrame(
    df[["id match", "product"]].dropna().groupby(["id match", "product"]).size(),
    columns=["count"],
).reset_index()
interactions_train.head()

#matrix using count
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# create zero-based index position <-> user/item ID mappings
index_to_user = pd.Series(np.sort(np.unique(interactions_train['id match'])))
index_to_item = pd.Series(np.sort(np.unique(interactions_train['product'])))

# create reverse mappings from user/item ID to index positions
user_to_index = pd.Series(data=index_to_user.index, index=index_to_user.values)
item_to_index = pd.Series(data=index_to_item.index, index=index_to_item.values)

# convert user/item identifiers to index positions
interactions_train_imp = interactions_train.copy()
interactions_train_imp['user_id'] = interactions_train['id match'].map(user_to_index)
interactions_train_imp['product_id'] = interactions_train['product'].map(item_to_index)

# prepare the data for CSR creation
# data = sample_weight_train
data = interactions_train_imp ['count']
rows = interactions_train_imp['user_id']
cols = interactions_train_imp['product_id']

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