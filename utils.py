import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle


def save_als_model(als_model, trial, mapk, save_dir):
    """Saves the als model on disk as a .pickle

    Args:
        als_model ([type]): Implicit ALS model
        trial (Trial): Optuna trial object
        mapk (float): Map@k for this model
        save_dir (str): Directory to save into
    """
    Path(save_dir).mkdir(exist_ok=True)
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"{trial.number}-{date_time}-{trial.number}-{mapk:.2f}:.pickle"
    file_path = Path(save_dir) / file_name
    with open(file_path, "wb") as fout:
        pickle.dump(als_model, fout)


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
    recommendations = als_model.recommend(
        userid=item_id, user_items=plays_matrix.T, N=n
    )
    recommendations = [
        (users_mapping[x[0]], x[1], als_model.user_factors[x[0]])
        for x in recommendations
    ]
    return pd.DataFrame(recommendations, columns=["userid", "score", "factor"])