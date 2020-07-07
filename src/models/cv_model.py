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
sales_train_path = os.path.join('data', 'sales_train_validation.csv')
price_path = os.path.join('data', 'sell_prices.csv')
sample_path = os.path.join('data', 'sample_submission.csv')

calendar = pd.read_csv(calendar_path, parse_dates=['date'])
sales = pd.read_csv(sales_train_path)
prices = pd.read_csv(price_path)
sample = pd.read_csv(sample_path)

#the set of possible feature names
cat_features = ['weekday', 'month', 'year',  'store_id',
                'cat_id', 'state_id', 'event_name_1', 'event_name_2']
quant_features = ['days', 'sell_price', 'mean']
str_features = ['store_id', 'cat_id', 'state_id',
                'event_name_1', 'event_name_2', 'weekday']
arma_prefixes = ['ma_', 'days_', 'months_', 'years_']

#specify model features
ar_lags = {'days':[7]}
ma_periods = [30]
other_features = ['days', 'weekday', 'sell_price']

fold = 1
scores = []
importances = []

for train_idx, test_idx in (TimeSeriesSplit(3)
                            .split(calendar[calendar.d.isin(sales.columns)])):

    train_dates = calendar.date[train_idx].values
    test_dates = calendar.date[test_idx].values

    #build feature set for the training data
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
    y_train = train_features.value

    #create lgb Dataset and train model
    lgb_data = lgb.Dataset(X_train, label = y_train,
                         categorical_feature=cat_names, free_raw_data=False)
    print('Finished building lgb_data for fold ' + str(fold) + '...')
    params = {
        'objective' : 'poisson',
        'learning_rate' : 0.075,
        'sub_row' : 0.75,
        'bagging_freq' : 1,
        'lambda_l2' : 0.1,
        'metric': ['rmse'],
        'verbosity': 1,
        'num_iterations' : 1200,
        'num_leaves': 128,
        'min_data_in_leaf': 100,
    }
    reg = lgb.train(params, lgb_data)
    print('Finished training lgb model for fold ' + str(fold) + '...')

    #build feature set for the test data
    test_features = build_test_features(
                        test_dates=test_dates, train_dates=train_dates,
                        ar_lags=ar_lags, ma_periods=ma_periods,
                        other_features=other_features, calendar=calendar,
                        sales=sales, prices=prices)
    print('Finished building test features for fold ' + str(fold) + '...')
    cat_test = [le.transform(test_features[str_names[i]])
                for i, le in enumerate(les)]
    cat_test_df = pd.concat([pd.DataFrame(cp) for cp in cat_test], axis=1)
    cat_test_df.columns = str_names
    print('Finished label encoding for fold ' + str(fold) + '...')
    X_test = pd.concat([test_features[quant_names], cat_test_df], axis=1)
    print('Finished making X_test for fold ' + str(fold) + '...')
    y_test = (sales[calendar[calendar.date.isin(test_dates)].d]
                   .values.flatten('F'))

    #add score and importances for the fold
    scores.append(MSE(y_test, reg.predict(X_test)))
    importances.append(reg.feature_importance())
    print('Finished predicting for fold ' + str(fold) + '...')

    fold+=1

#average scores and importances over folds
model_score = np.mean(scores)
mn_importances = np.mean(importances, axis=0).tolist()

print('Finished cross validating. Model score: ' + str(model_score))

#save model metadata
model_id = draw_id(os.path.join('models', 'model_ids.joblib'))
meta_dir = os.path.join('models', 'metadata')
model_meta = save_metadata(model=reg, model_id=model_id,
                           model_score=model_score,
                           importances=mn_importances, dir_path=meta_dir)

#compare to best model
best_model_path = os.path.join('models', 'metadata', 'best_model.csv')
if os.path.exists(best_model_path):
    best_model = pd.read_csv(best_model_path, squeeze=True, index_col=0)
else:
    best_model = model_meta
if model_score <= float(best_model.loc['score']):
    print('(this is the best model so far)')
    model_meta.to_csv(best_model_path, header=False)
