# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:01:34 2020

@author: foryou
"""

import numpy as np
import pandas as pd
from sklearn import ensemble, preprocessing

# 載入資料
def loading_data():
    train_data = r"train.csv"
    titanic_train = pd.read_csv(train_data)
    
    test_data = r"test.csv"
    titanic_test = pd.read_csv(test_data)
    
    submit_data = r'gender_submission.csv'
    submit = pd.read_csv(submit_data)
    
    return titanic_train, titanic_test, submit

# 填補遺漏值
def fill_values(data):
    median = np.nanmedian(data)
    data.fillna(median, inplace=True)
    return data

# 轉換數值
def value_encoder(data):
    label_encoder = preprocessing.LabelEncoder()
    data = label_encoder.fit_transform(data)
    return data

# 量化區間
def normalize(data, data_key):
    for key in data_key:
        data[key] = preprocessing.scale(data[key])
    return data

# 資料預處理
def data_processing(titanic_train, titanic_test):
    titanic_train["Age"] = fill_values(titanic_train["Age"])
    titanic_test["Age"] = fill_values(titanic_test["Age"])
    
    titanic_train["Sex"] = value_encoder(titanic_train["Sex"])
    titanic_test["Sex"] = value_encoder(titanic_test["Sex"])
    
    titanic_train["Embarked"] = np.where(titanic_train["Embarked"].isnull(), "S", titanic_train["Embarked"])
    titanic_test["Embarked"] = np.where(titanic_test["Embarked"].isnull(), "S", titanic_test["Embarked"])
    
    titanic_train["Embarked"] = value_encoder(titanic_train["Embarked"])
    titanic_test["Embarked"] = value_encoder(titanic_test["Embarked"])
  
    titanic_test["Fare"] = np.where(titanic_test["Fare"].isnull(), 0, titanic_test["Fare"])
    
    # print(np.isnan(titanic_test["Fare"]).sum())
    # print(np.isnan(titanic_test["Fare"]).sum())
    titanic_train = normalize(titanic_train, ["Embarked", "Fare", "Pclass", "Age", "Parch", "SibSp"])
    titanic_test = normalize(titanic_test, ["Embarked", "Fare", "Pclass", "Age", "Parch", "SibSp"])
    
    return titanic_train, titanic_test

# 建立訓練與測試資料
def data_split(titanic_train, titanic_test):
    titanic_X = pd.DataFrame([titanic_train["Pclass"],
                             titanic_train["Sex"],
                             titanic_train["Age"],
                             titanic_train["Fare"],
                             titanic_train["Embarked"]
    ]).T   
                     
    titanic_test_X = pd.DataFrame([titanic_test["Pclass"],
                             titanic_test["Sex"],
                             titanic_test["Age"],
                             titanic_test["Fare"],
                             titanic_test["Embarked"]
    ]).T
    
    titanic_Y = titanic_train["Survived"]
    
    return titanic_X, titanic_Y, titanic_test_X

# 建立隨機生成樹模型
def RandomForestmodel(titanic_X, titanic_Y):
    forest = ensemble.RandomForestClassifier(n_estimators = 200,
                                  min_samples_split = 20,
                                  min_samples_leaf = 1,
                                  oob_score = True,
                                  n_jobs = -1 )
    forest_fit = forest.fit(titanic_X, titanic_Y)
    
    test_y_predicted = forest.predict(titanic_test_X)
    
    return forest_fit, test_y_predicted

#輸出預測資料
def output_file(test_y_predicted):
    submit_data = r'gender_submission.csv'
    
    submit = pd.read_csv(submit_data)
    submit['Survived'] = test_y_predicted.astype(int)
    submit.to_csv(submit_data, index=False )
    
    return submit

if __name__ == '__main__':
    titanic_train, titanic_test, submit = loading_data()
    titanic_train, titanic_test = data_processing(titanic_train, titanic_test)
    
    titanic_X, titanic_Y, titanic_test_X = data_split(titanic_train, titanic_test)
    forest_fit, test_y_predicted = RandomForestmodel(titanic_X, titanic_Y)
    
    importances = forest_fit.feature_importances_
    print(importances)
    print( f'預測結果：' )
    print(forest_fit.oob_score_)
    
    output_file(test_y_predicted)

    # accuracy = metrics.accuracy_score(titanic_test_y, test_y_predicted)
    # print(accuracy)