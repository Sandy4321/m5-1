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

le_path = 'models/le_5961.joblib'
with open(le_path, 'rb') as f:
    les = load(f)

cat_train = [le.transform(train_features[str_names[i]])
            for i, le in enumerate(les)]
cat_train_df = pd.concat([pd.DataFrame(cp) for cp in cat_train], axis=1)
cat_train_df.columns = str_names

print('Finished label encoding')

X_train = pd.concat([train_features[quant_names].reset_index(drop=True),
                        cat_train_df], axis=1)

model_path = 'models/model_5961.joblib'
with open(model_path, 'rb') as f:
    reg = load(f)

lgbm_preds = reg.predict(X_train)

lgbm_pred_path = 'models/model_5961_preds.joblib'
with open(lgbm_pred_path, 'wb') as f:
    dump(lgbm_preds, f)

y_train = train_features.value
