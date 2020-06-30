import numpy as np
import pandas as pd
from itertools import chain

def make_ar_train(sales_df, lags):
    offsets = [[*map(lambda e: pd.DateOffset(**{k:e}), v)]
                for k,v in lags.items()]
    offsets = [*chain.from_iterable(offsets)]
    lag_dfs = [pd.melt(sales_df.shift(freq=offset).reset_index(),
               id_vars='date',
               value_name=offset.freqstr[13:-1].replace('=', '_'))
               for offset in offsets]
    lag_df = pd.concat(lag_dfs, axis=1)
    return lag_df.loc[:,~lag_df.columns.duplicated()]

def make_ma_train(sales_df, periods):
    ma_dfs = [pd.melt(sales_df.rolling(period).mean().reset_index(),
                      id_vars='date',
                      value_name=''.join(['ma_', str(period)]))
                      for period in periods]

    ma_df = pd.concat(ma_dfs, axis=1)
    return ma_df.loc[:,~ma_df.columns.duplicated()]

def build_train_features(train_dates, ar_lags, ma_periods, other_features,
                         calendar, sales, prices):

    sales_df = sales[calendar[calendar.date.isin(train_dates)].d].T
    sales_df['date'] = train_dates
    sales_df.set_index('date', inplace=True)

    sales_values = pd.melt(sales_df.reset_index(), id_vars='date').dropna()

    lag_df = make_ar_train(sales_df, ar_lags)
    ma_df = make_ma_train(sales_df, ma_periods)

    print('Finished with pre-merge steps')

    feature_df = (sales_values.merge(lag_df, how='left', on=['date', 'variable'])
                              .merge(ma_df, how='left', on=['date', 'variable']))

    cal_features = ['weekday', 'event_name_1', 'event_name_2']
    cal_features = [cf for cf in cal_features if cf in other_features]

    if 'sell_price' in other_features:
        cal_features = cal_features + ['wm_yr_wk']

    if len(cal_features) > 0:
        feature_df = feature_df.merge(calendar[cal_features + ['date']], how='left', on='date')

    if 'mean' in other_features:
        mean_df = sales_df.mean().reset_index()
        mean_df.columns = ['variable', 'mean']
        feature_df = feature_df.merge(mean_df, how='left', on='variable')

    feature_df = feature_df.merge(sales[['item_id', 'cat_id', 'store_id']], how='left',
                                  left_on='variable', right_index=True)

    if 'sell_price' in other_features:
        feature_df = feature_df.merge(prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])

    print('Finished merging')

    if 'month' in other_features:
        feature_df['month'] = feature_df.date.dt.month

    if 'year' in other_features:
        feature_df['year'] = feature_df.date.dt.year

    if 'days' in other_features:
        feature_df['days'] = [diff.days for diff in (feature_df.date - calendar.date.min())]

    print('Added days')

    return feature_df
