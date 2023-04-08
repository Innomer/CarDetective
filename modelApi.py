import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import load


def predict_price_textual():
    lr_drop = load('./Models/linear_regression_drop.joblib')
    rfr_drop = load('./Models/random_forest_drop.joblib')
    lr_price = load('./Models/linear_regression_price.joblib')
    rfr_price = load('./Models/random_forest_price.joblib')