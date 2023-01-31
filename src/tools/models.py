# Functions to build models

# Libraries
import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score as cvs
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# --------------------------------------------------------------------------------------------------------------------
# Lazypredict
def model_lazypredict(df):
    # define features and target
    X = df[["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]]
    y = df["price"]
    # split data train-test 80%-20%
    offset = int(X.shape[0] * 0.8)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    # model
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    return models

# --------------------------------------------------------------------------------------------------------------------
# h2o AutoML 
def model_automl(df):
    df = df.drop(['id'], axis=1) # dropping the 'id' column
    # split data train-test 80%-20%
    splits = df.split_frame(ratios = [0.8], seed = 1)
    train = splits[0]
    test = splits[1]
    # training automl models
    aml = H2OAutoML(max_runtime_secs = 60, seed = 1, project_name = "diamonds_automl", sort_metric = 'RMSE')
    aml.train(y = "price", training_frame = train, leaderboard_frame = test) # specity that price is the target
    return aml

# --------------------------------------------------------------------------------------------------------------------
# Manually training models with sklearn
def model_sklearn(df, condition, X_split, y_split):
    # models to test
    models = {
        "GBM": GradientBoostingRegressor(),
        "RF": RandomForestRegressor(),
        "XGB": xgb.XGBRegressor()
        }
    # define target variable -adapted for feature selection purposes
    X = df[X_split]
    y = df[y_split]
    # split data train-test 80%-20%
    X_train, X_test, y_train, y_test = tts (X, y, test_size = 0.2)
    # dictionary to store RMSE of each model
    all_rmse = {}
    for name, model in models.items():
        rmse = cvs(model, X_train, y_train, scoring = "neg_root_mean_squared_error", cv=5)
        all_rmse[name] = abs(np.mean(rmse))
    df_out = pd.DataFrame.from_dict(all_rmse, orient='index').reset_index().rename(columns={'index':'Model',0:'RMSE'})
    df_out['Condition'] = condition
    return df_out
 