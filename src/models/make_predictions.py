import os
import random
from joblib import dump
import numpy as np
import pandas as pd
import calendar as cal
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error as MSE
import lightgbm as lgb

from src.features.build_train_features import *
from src.features.build_test_features import *
from src.models.preprocess import *
from src.models.model_mgmt import *

calendar_path = os.path.join('data', 'calendar.csv')
sales_path = os.path.join('data', 'sales_train_evaluation.csv')
price_path = os.path.join('data', 'sell_prices.csv')
sample_path = os.path.join('data', 'sample_submission.csv')

calendar = pd.read_csv(calendar_path, parse_dates=['date'])
sales = pd.read_csv(sales_path)
prices = pd.read_csv(price_path)
sample = pd.read_csv(sample_path)

ar_lags = {'days':[7]}

cat_features = ['weekday', 'month', 'year',  'store_id',
                'cat_id', 'state_id', 'event_name_1', 'event_name_2']
quant_features = ['days', 'sell_price', 'mean']
str_features = ['store_id', 'cat_id', 'state_id',
                'event_name_1', 'event_name_2', 'weekday']

ma_periods = [30]
other_features = ['days', 'weekday', 'sell_price'] #, # 'store_id',
                 # 'event_name_1', 'event_name_2']
arma_prefixes = ['ma_', 'days_', 'months_', 'years_']

train_dates = calendar.date[:-28].values
test_dates = calendar.date[-28:].values

train_features = build_train_features(
        train_dates=train_dates, ar_lags=ar_lags, ma_periods=ma_periods,
        other_features=other_features, calendar=calendar,
        sales=sales, prices=prices)
arma_features = [col for col in train_features.columns
                 if any([ap in col for ap in arma_prefixes])]
feature_names = other_features + arma_features
cat_names = [fn for fn in feature_names if fn in cat_features]
quant_names = arma_features + [fn for fn in feature_names
                               if fn in quant_features]
str_names = [fn for fn in feature_names if fn in str_features]

cat_df, les = le_cat_features(train_features, str_names)

X_train = pd.concat([train_features[quant_names].reset_index(drop=True),
                        cat_df], axis=1)

    #X_train = train[quant_features + cat_features]
y_train = train_features.value

lgb_data = lgb.Dataset(X_train, label = y_train,
                       categorical_feature=cat_names, free_raw_data=False)

print('Finished building lgb_data')

params = {
    "objective" : "poisson",
    "metric" :"rmse",
    "learning_rate" : 0.075,
    "sub_row" : 0.75,
    "bagging_freq" : 1,
    "lambda_l2" : 0.1,
    "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : 1200,
    'num_leaves': 128,
    "min_data_in_leaf": 100,
}

reg = lgb.train(params, lgb_data)

print('Finished training lgb model')

model_id = draw_id(os.path.join('models', 'model_ids.joblib'))

model_path = os.path.join('models',
                          ''.join(['model_', str(model_id), '.joblib']))
with open(model_path, 'wb') as f:
    dump(reg, f)

test_features = build_test_features(
                    test_dates=test_dates, train_dates=train_dates,
                    ar_lags=ar_lags, ma_periods=ma_periods,
                    other_features=other_features, calendar=calendar,
                    sales=sales, prices=prices)

print('Finished building test features')

cat_test = [le.transform(test_features[str_names[i]])
            for i, le in enumerate(les)]
cat_test_df = pd.concat([pd.DataFrame(cp) for cp in cat_test], axis=1)
cat_test_df.columns = str_names

print('Finished label encoding')

X_test = pd.concat([test_features[quant_names], cat_test_df], axis=1)

print('Finished making X_test')

X_test['predictions'] = reg.predict(X_test)

print('Finished predicting')

with open(os.path.join('models', 'predictions.joblib'), 'wb') as f:
    dump(X_test, f)
