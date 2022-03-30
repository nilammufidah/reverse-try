import random

import implicit
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import mean_average_precision_at_k, train_test_split
from implicit.nearest_neighbours import CosineRecommender
from implicit.utils import nonzeros
from IPython.display import set_matplotlib_formats
from optuna import samplers
from pandas_profiling import ProfileReport
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

from utils import recommend_users, save_als_model

%matplotlib inline
set_matplotlib_formats("svg")

RANDOM_SEED = 27
SAVE_DIR = "./models/"

tracks_df = pd.read_csv("./datapoc/jan_2021_match_id.csv", sep=';',encoding='iso-8859-1')

print(
    f'{100 * len(tracks_df[(tracks_df["PRODUCT_NAME"].isna()) & (~tracks_df["traname"].isna())]) / len(tracks_df):.2f}% of dataset will be lost to missing id'
)

plays_df = pd.DataFrame(
    tracks_df[["MATCHED_ID", "PRODUCT_NAME"]].dropna().groupby(["MATCHED_ID", "PRODUCT_NAME"]).size(),
    columns=["count"],
).reset_index()

print (len (plays_df))
plays_df[["MATCHED_ID", "PRODUCT_NAME"]].drop_duplicates().groupby("MATCHED_ID").size().describe()

songs_plays_df = plays_df.groupby(["PRODUCT_NAME"]).sum().reset_index()

N = 7
print(
    f'{100*len(songs_plays_df[songs_plays_df["plays"] < N]) / len(songs_plays_df):.2f}% of songs have been played less than {N} times'
)

plays_df["MATCHED_ID"] = plays_df["MATCHED_ID"].astype("category")
plays_df["PRODUCT_NAME"] = plays_df["PRODUCT_NAME"].astype("category")

# create a sparse matrix of all the tracks/user/play triples
# The Matrix is actually transposed, meaning that we pass tracks instead of users to our model
plays = coo_matrix(
    (
        plays_df["count"].astype(np.float64),
        (plays_df["MATCHED_ID"].cat.codes, plays_df["PRODUCT_NAME"].cat.codes),
    )
)

# random_state parameter implemented in https://github.com/benfred/implicit/pull/411
# but still not pushed to conda/pip
train_user_items, test_user_items = train_test_split(
    plays,
    train_percentage=0.8,
    # random_state=RANDOM_SEED,
)

matrix_size = (
    plays.shape[0] * plays.shape[1]
)  # Number of possible interactions in the matrix
num_purchases = len(plays.nonzero()[0])  # Number of items interacted with
sparsity = 100 * (1 - (num_purchases / matrix_size))
print(f"The matrix is {sparsity:.2f}% sparse")

# rough optimization of the NN model
N_TRIALS = 5
K = 5


def optimize_nn(trial):
    neighbors = trial.suggest_int("K", 20, 100, 10)
    nn = CosineRecommender(K=neighbors)
    nn.fit(train_user_items)
    map_at_k = mean_average_precision_at_k(
        nn, train_user_items.T, test_user_items.T, K=K
    )
    return map_at_k


sampler = samplers.TPESampler(seed=RANDOM_SEED)
nn_study = optuna.create_study(
    direction="maximize", study_name="nearest_neighbors_study"
)
nn_study.optimize(optimize_nn, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

print(f"MAP@5 for our best Cosine model: {nn_study.best_trial.value:.2f}")

# Optimization of the ALS MF model
K = 5
# As this is quite computationally expensive we'll just run it for a few iterations
N_TRIALS = 5


def optimize_als(trial):
    """Runs multiple "trials" with various parameters to
    determine the best hyper parameters to use

    Args:
        trial (Trial): a Trial object which does not need to be instanciated
    """
    # matrix__confidence = trial.suggest_int("matrix__confidence", 2, 100, 5)
    factors = trial.suggest_int("factors", 300, 50, 10)
    regularization = trial.suggest_loguniform(
        "regularization", 1e-4, 1
    )  # cannot start at 0
    iterations = trial.suggest_int("iterations", 10, 100, 10)
    # alpha is a scaling factor for the raw ratings matrix https://github.com/benfred/implicit/issues/199
    alpha = trial.suggest_int("alpha", 1, 15, 1)
    als = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=RANDOM_SEED,
    )
    als.fit(alpha * train_user_items)
    map_at_k = mean_average_precision_at_k(
        als, alpha * train_user_items.T, alpha * test_user_items.T, K=K, num_threads=0
    )
    # as a safety measure let's save the models
    save_als_model(als_model=als, trial=trial, mapk=map_at_k, save_dir=SAVE_DIR)
    return map_at_k


study = optuna.create_study(direction="maximize", study_name="als_study")
study.optimize(optimize_als, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

print(f"MAP@5 of the best ALS Model: {study.best_value:.2f}")
