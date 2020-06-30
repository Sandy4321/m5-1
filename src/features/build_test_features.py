import numpy as np
import pandas as pd
from itertools import chain

def make_ar_test(sales_df, test_date, lags):
    test_date = pd.to_datetime(test_date)
    offsets = [[*map(lambda e: pd.DateOffset(**{k:e}), v)]
                for k,v in lags.items()]
    offsets = [*chain.from_iterable(offsets)]
    offsets_used = offsets.copy()
    for i, offset in enumerate(offsets):
        multiple = 1
        while test_date - offsets_used[i] not in sales_df.index:
            multiple+=1
            offsets_used[i] = offset * multiple
            if multiple > 200:
                print("Couldn't find ar feature for " + str(offset))
                break
    ar_dfs = [sales_df.loc[test_date - ou] for ou in offsets_used]
    ar_df = pd.concat(ar_dfs, axis=1)
    ar_df.columns = [offset.freqstr[13:-1].replace('=', '_')
                     for offset in offsets]
    return ar_df

def make_ma_test(sales_df, periods):
    ma_features = [sales_df[-period:].mean() for period in periods]
    feature_df =  pd.concat(ma_features, axis=1)
    feature_df.columns = [''.join(['ma_', period])
                          for period in map(str, periods)]
    return feature_df

def build_test_features(test_dates, train_dates, ar_lags, ma_periods,
                        other_features, calendar, sales, prices):

    n_obs = sales.shape[0]
    n_dates = len(test_dates)

    sales_df = sales[calendar[calendar.date.isin(train_dates)].d].T
    sales_df['date'] = train_dates
    sales_df.set_index('date', inplace=True)

    ma_df = pd.concat([make_ma_test(sales_df, ma_periods)] * n_dates)
    ar_df = pd.concat([make_ar_test(sales_df, td, ar_lags)
                       for td in test_dates])

    print("Added arma features...")

    test_features = pd.concat([ma_df, ar_df], axis=1).reset_index(drop=True)

    test_features['date'] = np.repeat(test_dates, n_obs)

    test_dates_dt = pd.to_datetime(test_dates)

    if 'mean' in other_features:
        test_features['mean'] = np.tile(sales_df.mean(), n_dates)
    if 'month' in other_features:
        test_features['month'] = np.repeat([td.month
                                            for td in test_dates_dt], n_obs)
    if 'year' in other_features:
        test_features['year'] = np.repeat([td.year
                                           for td in test_dates_dt], n_obs)
    if 'days' in other_features:
        test_features['days'] = np.repeat([(td - calendar.date.min()).days
                                          for td in test_dates_dt], n_obs)
    if 'weekday' in other_features:
        test_wd = [calendar.weekday[calendar.date == td].values[0]
                   for td in test_dates]
        test_features['weekday'] = np.repeat(test_wd, n_obs)
    if 'store_id' in other_features:
        test_features['store_id'] = np.tile(sales['store_id'], n_dates)
    if 'event_name_1' in other_features:
        test_ev1 = [calendar.event_name_1[calendar.date == td].values[0]
                    for td in test_dates]
        test_features['event_name_1'] = np.repeat(test_ev1, n_obs)
    if 'event_name_2' in other_features:
        test_ev2 = [calendar.event_name_2[calendar.date == td].values[0]
                    for td in test_dates]
        test_features['event_name_2'] = np.repeat(test_ev2, n_obs)
    if 'sell_price' in other_features:
        test_features['item_id'] = np.tile(sales['item_id'], n_dates)
        test_wyw = [calendar.wm_yr_wk[calendar.date == td].values[0]
                    for td in test_dates]
        test_features['wm_yr_wk'] = np.repeat(test_wyw, n_obs)
        test_features = test_features.merge(prices, how='left',
                                        on=['store_id', 'item_id', 'wm_yr_wk'])

    return test_features
