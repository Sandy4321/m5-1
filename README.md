m5
==============================

Submission for the Kaggle m5 competition, a forecasting competition to predict daily demand for over three thousand products sold at ten Walmart locations in three states. A LightGBM model was trained on only a small set of the available features and still scored in the top 13% of entries. The features used were:

- `weekday` - Day of the week.
- `sell_price` - Price listed for the product in the store.
- `days_7` (created) - Demand for the product in the store seven days ago. For predicting dates further out than seven days, the closest multiple was used instead (demand from 14 days ago, 21 days ago, etc.).
- `ma_30` (created) - Average sales of the product in the store for the last 30 days (or most recent 30 days available).
- `days` (created) - Number of days since the first date in the training set.

A set of functions are included that were written to ease creating arbitrary autoregressive and moving average features. Candidate models were cross validated using sklearn's TimeSeriesSplit.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Data files
    ├── models             <- Trained models
    │   ├── metadata       <- Model metadata
    ├── notebooks          <- Jupyter notebooks
    ├── reports            <- Rendered notebooks
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    ├── src                <- Source code
    │   ├── features       <- Code for feature engineering
    │   │   ├── build_test_features.py
    │   │   ├── build_train_features.py
    │   ├── models           <- Scripts and functions for training, testing and saving models
    │   │   └── cv_model.py
    │   │   └── make_predictions.py
    │   │   └── model_mgmt.py
    │   │   └── preprocess.py


--------

<p><small>Project structure loosely based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.
