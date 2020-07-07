import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def le_cat_features(feature_df, cat_features):
    """ Label encodes categorical features and saves the fitted label encoders
        to be used to transform test data.
    """
    les = []
    features = np.empty((feature_df.shape[0], len(cat_features)))
    for i, feature in enumerate(cat_features):
        le = LabelEncoder()
        features[:,i] = le.fit_transform(feature_df[feature].fillna('None'))
        les.append(le)
    return pd.DataFrame(features, columns=cat_features), les
