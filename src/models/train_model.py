import os
from joblib import dump
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from lightgbm import LGBMRegressor

sample_path = os.path.join('..', '..', 'data', 'raw', 'sample_submission.csv')
sample = pd.read_csv(sample_path)

feature_path = os.path.join('..', '..', 'data', 'interim', 'feature_set.csv')
df = pd.read_csv(feature_path)
y = df.value

quant_features = ['last_week', 'last_year', 'days', 'ma30']
get_quant_features = FunctionTransformer(lambda x: x[quant_features],
                                         validate=False)

cat_features = ['weekday', 'store_id']
cat_to_dict = FunctionTransformer(lambda x: x[cat_features].to_dict('records'),
                                  validate=False)

pipe = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('quant_features', Pipeline([
                    ('selector', get_quant_features)
                ])),
                ('cat_features', Pipeline([
                    ('to_dict', cat_to_dict),
                    ('dict_vectorizer', DictVectorizer())
                ]))
            ]
        )),
        ('reg', LGBMRegressor)
    ])

pipe.fit(df, y)

pipe_path = os.path.join('..', '..', 'models', 'pipe_1.joblib')
dump(pipe, pipe_path)
