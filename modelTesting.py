import pandas as pd
import tensorflow as tf
import numpy as np
import keras
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Preprocessing


def preprocess(df):
    # nan_cols = df.isna().any()
    # print("Empty Cols = ")
    # print(nan_cols[nan_cols == True].index.tolist())
    df.dropna(inplace=True)
    df['Fuel_Type_Encoded'] = df['Fuel_Type'].astype('category').cat.codes
    df['Transmission_Encoded'] = df['Transmission'].astype(
        'category').cat.codes
    df['Owner_Type_Encoded'] = df['Owner_Type'].astype('category').cat.codes
    df['New_Price'] = df['New_Price'].str.replace('Lakh', '')
    df['New_Price'] = df['New_Price'].str.replace('Cr', '')
    df['Power'] = df['Power'].str.replace('bhp', '')
    df['Engine'] = df['Engine'].str.replace('CC', '')
    df['Mileage'] = df['Mileage'].str.replace('kmpl', '')
    df['Mileage'] = df['Mileage'].str.replace('km/kg', '')
    df['New_Price'] = df['New_Price'].str.strip()
    df['Power'] = df['Power'].str.strip()
    df['Engine'] = df['Engine'].str.strip()
    df['Mileage'] = df['Mileage'].str.strip()
    df['Mileage'] = df['Mileage'].astype(float)
    df['Engine'] = df['Engine'].astype(float)
    df['Power'] = df['Power'].astype(float)
    df['New_Price'] = df['New_Price'].astype(float)
    df = df.reset_index(drop=True)
    return df

# Feature Engineering


def calcAge(year):
    current_year = datetime.now().year
    return current_year-year


def norm(srs):
    scaler = MinMaxScaler()
    scaled_srs = scaler.fit_transform(srs.values.reshape(-1, 1))
    return scaled_srs


def feature_engineering(df):
    df['Age'] = df['Year'].apply(calcAge)
    df['Kilometers_Driven'] = norm(df['Kilometers_Driven'])
    df['Engine'] = norm(df["Engine"])
    df['Power'] = norm(df["Power"])
    df['Mileage'] = norm(df["Mileage"])
    df['Price_Drop']=df['New_Price']-df['Price']
    df['Price']=norm(df["Price"])
    df['Price_Drop']=norm(df["Price_Drop"])
    df['New_Price'] = norm(df["New_Price"])
    df['Fuel_Type'] = df['Fuel_Type_Encoded']
    df['Owner_Type'] = df['Owner_Type_Encoded']
    df['Transmission'] = df['Transmission_Encoded']
    df['Name'] = df['Name'].str.extract('(\w+)')
    df['Name'] = df['Name'].astype('category').cat.codes
    df.drop(['Transmission_Encoded','Owner_Type_Encoded','Fuel_Type_Encoded','Location','New_Price','Price','Year'],axis=1,inplace=True)
    # df.drop(['Transmission_Encoded', 'Owner_Type_Encoded',
    #         'Fuel_Type_Encoded', 'Location', 'Year'], axis=1, inplace=True)
    return df


def test_features(df):
    df['Age'] = df['Year'].apply(calcAge)
    df['Kilometers_Driven'] = norm(df['Kilometers_Driven'])
    df['Engine'] = norm(df["Engine"])
    df['Power'] = norm(df["Power"])
    df['Mileage'] = norm(df["Mileage"])
    # df['Price_Drop']=df['New_Price']-df['Price']
    # df['Price']=norm(df["Price"])
    # df['Price_Drop']=norm(df["Price_Drop"])
    df['New_Price'] = norm(df["New_Price"])
    df['Fuel_Type'] = df['Fuel_Type_Encoded']
    df['Owner_Type'] = df['Owner_Type_Encoded']
    df['Transmission'] = df['Transmission_Encoded']
    df['Name'] = df['Name'].str.extract('(\w+)')
    df['Name'] = df['Name'].astype('category').cat.codes
    df.drop(['Transmission_Encoded', 'Owner_Type_Encoded',
            'Fuel_Type_Encoded', 'Location', 'Year'], axis=1, inplace=True)
    return df


traindf = preprocess(pd.read_csv('./Datasets/train_data.csv'))
traindf = feature_engineering(traindf)
testdf = preprocess(pd.read_csv('./Datasets/test_data.csv'))
testdf = test_features(testdf)


# X = traindf.drop('Price', axis=1)
# y = traindf['Price']
X = traindf.drop('Price_Drop', axis=1)
y = traindf['Price_Drop']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_rmse = mean_squared_error(y_test, lr_pred, squared=False)
lr_r2 = r2_score(y_test, lr_pred)

# Random Forest Regressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)
rfr_rmse = mean_squared_error(y_test, rfr_pred, squared=False)
rfr_r2 = r2_score(y_test, rfr_pred)

# SVM
# svr = SVR()
# svr.fit(X_train, y_train)
# svr_pred = svr.predict(X_test)
# svr_rmse = mean_squared_error(y_test, svr_pred, squared=False)
# svr_r2 = r2_score(y_test, svr_pred)

print('Linear Regression RMSE:', lr_rmse)
print('Linear Regression R^2:', lr_r2)
print('Random Forest Regression RMSE:', rfr_rmse)
print('Random Forest Regression R^2:', rfr_r2)
# print('Support Vector Regression RMSE:', svr_rmse)
# print('Support Vector Regression R^2:', svr_r2)

# Price Drop Pred
# Linear Regression RMSE: 0.06854988080410929
# Linear Regression R^2: 0.0007094859253358177
# Random Forest Regression RMSE: 0.047372304163081906
# Random Forest Regression R^2: 0.5227703975083304
# Support Vector Regression RMSE: 0.07531974990854874
# Support Vector Regression R^2: -0.20641324201886424

# Final Price Pred
# Linear Regression RMSE: 6.182162793359798
# Linear Regression R^2: 0.8060657719191427
# Random Forest Regression RMSE: 3.2735414592396417
# Random Forest Regression R^2: 0.9456237463209227
# Support Vector Regression RMSE: 14.09856302734958
# Support Vector Regression R^2: -0.00861005204204579

# Therefore a Random Forest Regression is a better choice to calculate a price drop whereas a Linear Regression Model as well as Random Forest both are good at calculating new Prices but RF is slightly better.


from joblib import dump
# dump(lr, './Models/linear_regression_price.joblib')
# dump(rfr, './Models/random_forest_price.joblib')
dump(lr, './Models/linear_regression_drop.joblib')
dump(rfr, './Models/random_forest_drop.joblib')