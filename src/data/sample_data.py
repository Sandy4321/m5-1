import os
import random
import pandas as pd

calendar_path = os.path.join('..', '..', 'data', 'raw', 'calendar.csv')
sales_train_path = os.path.join('..', '..', 'data', 'raw', 'sales_train_validation.csv')

calendar = pd.read_csv(calendar_path, parse_dates=['date'])
sales_train = pd.read_csv(sales_train_path)

mask = ['d_' in col for col in sales_train.columns]
df = sales_train.iloc[:, mask].T

# Set aside the last 28 days as test set
train_df = df.iloc[:-28,:]
test_df = df.iloc[-28:,:]

# Sample 1000 item ids
random.seed(22)
item_sample = random.sample(set(df.columns), 1000)
sample_ids = sales_train.id[item_sample]

train_sample = train_df[item_sample].T
train_sample.insert(loc=0, column='id', value=sample_ids)

test_sample = test_df[item_sample].T
test_sample.insert(loc=0, column='id', value=sample_ids)

future_dates = calendar.date.values[train_df.shape[0]:df.shape[0]]
future = pd.DataFrame({'ds': future_dates})

train_path = os.path.join('..', '..', 'data', 'train', 'train_sample.csv')
test_path = os.path.join('..', '..', 'data', 'train', 'test_sample.csv')
future_path = os.path.join('..', '..', 'data', 'train', 'future.csv')

train_sample.to_csv(train_path, index=False)
test_sample.to_csv(test_path, index=False)
future.to_csv(future_path, index=False)
