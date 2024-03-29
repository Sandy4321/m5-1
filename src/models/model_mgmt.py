import os
from joblib import load, dump
import pandas as pd

def draw_id(file_path, create_new=False, size=10000):
    """ Draws a unique ID from a file of model IDs and then removes it from the
        model ID file. The ID file can be created if it does not exist yet.
    """
    if create_new:
        ids = random.sample(range(size), size)
    else:
        with open(file_path, 'rb') as f:
            ids = load(f)
    model_id = ids.pop(0)
    with open(file_path, 'wb') as f:
        dump(ids, f)
    return model_id

def save_metadata(model, model_id, model_score, importances, dir_path):
    """ Formats the model metadata as a pandas series and saves it. Takes model
        score and feature importances as separate arguments in case they have
        been averaged over multiple folds.
    """
    meta_path = os.path.join(dir_path, ''.join(['model_', str(model_id), '_metadata.csv']))
    model_meta = pd.Series({'model_id': model_id,
                        'feature_names': list(model.feature_name()),
                        'feature_importances': importances,
                        'score': model_score})
    model_meta.to_csv(meta_path, header=False)
    return model_meta
