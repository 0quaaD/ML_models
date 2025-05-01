import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def random_forest_xgboost(df, n_estimators, test_size):
  if(df.isnull().any().any()):
    raise ValueError('Data has missing values')
  else:
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=test_size, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)

    start_rf_train = time.time()
    rf.fit(train_X, train_y)
    end_rf_train = time.time()
    rf_train = end_rf_train - start_rf_train

    start_rf_test = time.time()
    y_pred_rf = rf.predict(test_X)
    end_rf_test = time.time()
    rf_test = end_rf_test - start_rf_test
    
    start_xgb_train = time.time()
    xgb.fit(train_X, train_y)
    end_xgb_train = time.time()
    xgb_train = end_xgb_train - start_xgb_train

    start_xgb_test = time.time()
    y_pred_xgb = xgb.predict(test_X)
    end_xgb_test = time.time()
    xgb_test = end_xgb_test - start_xgb_test

    print(f"Random Forest Train Time:\t{rf_train:.4f} seconds\t\tTest time: {rf_test:.4f} seconds")
    print(f"XGBoost train time:\t\t{xgb_train:.4f} seconds\t\tTest time: {xgb_test:.4f} seconds\n")
    
    mse_rf = mean_squared_error(test_y, y_pred_rf)
    mse_xgb = mean_squared_error(test_y, y_pred_xgb)

    r2_rf = r2_score(test_y, y_pred_rf)
    r2_xgb = r2_score(test_y, y_pred_xgb)
    print(f"Random Forest MSE: {mse_rf:.4f}")
    print(f"XGBoost MSE: {mse_xgb:.4f}\n")
    print(f"Random Forest R2 Score: {r2_rf:.4f}")
    print(f"XGBoost R2 Score: {r2_xgb:.4f}")

df = pd.read_csv('california_housing.csv')
random_forest_xgboost(df, 100, 0.2)
