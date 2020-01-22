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

#取得遺失資料筆數
def get_miss_count(data):
    missing = data.isnull().sum()
    missing = missing[missing>0]
    print(missing)
    
# 填補中值
def fill_median(data):
    median = np.nanmedian(data)
    data.fillna(median, inplace=True)
    return data

# 轉換數值
def value_encoder(data):
    label_encoder = preprocessing.LabelEncoder()
    data = label_encoder.fit_transform(data)
    # data = pd.get_dummies(data)

    return data

# 量化區間
def normalize(data, data_key):
    for key in data_key:
        data[key] = preprocessing.scale(data[key])
    return data

def Ticket_process(data):
    new = list()
    for x in data:
        new.append(x.split(" ")[0])
    return pd.DataFrame(new)

def Name_process(data):
    new = list()
    for x in data:
        new.append(x.split(",")[1].split(".")[0])
    return pd.DataFrame(new)

# 資料預處理
def data_processing(titanic_data):
    # titanic_data["Age"] = fill_median(titanic_data["Age"])
    
    titanic_data["Sex"] = value_encoder(titanic_data["Sex"])
    
    titanic_data["Cabin"] = np.where(titanic_data["Cabin"].isnull(), "No", titanic_data["Cabin"])
    titanic_data["Cabin"] = value_encoder(titanic_data["Cabin"])
    
    titanic_data["Embarked"] = np.where(titanic_data["Embarked"].isnull(), "N", titanic_data["Embarked"])
    titanic_data["Embarked"] = value_encoder(titanic_data["Embarked"])
    
    titanic_data["Ticket"] = Ticket_process(titanic_data["Ticket"])
    titanic_data["Ticket"] = value_encoder(titanic_data["Ticket"])
    
    titanic_data["Family"] = titanic_data["Name"].str.split(",",expand=True)[0]
    titanic_data["Family"] = value_encoder(titanic_data["Family"])
    
    titanic_data["Name"] = Name_process(titanic_data["Name"])
    Age_Mean = titanic_data[['Name','Age']].groupby( by=['Name'] ).mean()
    Age_Median = titanic_data[['Name','Age']].groupby( by=['Name'] ).median()
    
    Age_Mean.columns = ['Age Mean']
    Age_Median.columns = ['Age Median']
    Age_Mean.reset_index( inplace=True )
    Age_Median.reset_index( inplace=True )
    
    Age_Mean['Name']=Age_Mean['Name'].astype('category').astype('str')
    print(Age_Mean['Name'])
    aaa
    print(Age_Mean[['Name']],Age_Mean)
    # print(titanic_data.loc[(titanic_data.Name=='Master'),'Age'])
    titanic_data["Embarked"] = np.where(titanic_data["Name"].str.contains("Miss"), "N", titanic_data["Embarked"])
    # titanic_data.loc[(titanic_data.Name=='Master'),'Age'] = Age_Mean.loc[Age_Mean.Name=='Master','Age Mean'][0]
    # titanic_data.loc[(titanic_data.Name=='Miss'),'Age'] = Age_Mean.loc[Age_Mean.Name=='Miss','Age Mean'][1]
    # titanic_data.loc[(titanic_data.Name=='Mr'),'Age'] = Age_Mean.loc[Age_Mean.Name=='Mr','Age Mean'][2]
    # titanic_data.loc[(titanic_data.Name=='Mrs'),'Age'] = Age_Mean.loc[Age_Mean.Name=='Mrs','Age Mean'][3]
    aa
    titanic_data["Embarked"] = np.where(titanic_data["Name"].str.contains("Miss"), "N", titanic_data["Embarked"])
    titanic_data["Name"] = value_encoder(titanic_data["Name"])
    
    titanic_data["Fare"] = fill_median(titanic_data["Fare"])
    titanic_data = normalize(titanic_data, titanic_data.columns.drop('Survived').drop('PassengerId'))
    
    return titanic_data

# 建立訓練與測試資料
def data_split(titanic_data):
    titanic_train = titanic_data[ pd.notnull(titanic_data.Survived) ]
    titanic_test = titanic_data[ pd.isnull(titanic_data.Survived) ]
    
    titanic_X = titanic_train.drop(['PassengerId','Survived'], axis=1)
    titanic_test_X = titanic_test.drop(['PassengerId','Survived'], axis=1)
    # titanic_X=titanic_train[]
    titanic_Y = titanic_train["Survived"]
    
    return titanic_X, titanic_Y, titanic_test_X

# 建立隨機生成樹模型
def RandomForestmodel(titanic_X, titanic_Y, titanic_test_X):
    forest = ensemble.RandomForestClassifier(n_estimators = 200,
                                  min_samples_split = 20,
                                  min_samples_leaf = 1,
                                  oob_score = True,
                                  n_jobs = -1 )
    forest_fit = forest.fit(titanic_X, titanic_Y)
    
    test_y_predicted = forest.predict(titanic_test_X)
    
    return forest_fit, test_y_predicted

# 輸出預測資料
def output_file(test_y_predicted):
    submit_data = r'gender_submission.csv'
    
    submit = pd.read_csv(submit_data)
    submit['Survived'] = test_y_predicted.astype(int)
    submit.to_csv(submit_data, index=False )
    
    return submit

# 重點特徵數
def features_reduced(forest_fit, titanic_X, titanic_test_X):
    from sklearn.feature_selection import SelectFromModel
    model = SelectFromModel(forest_fit, prefit=True)
    n_features = model.transform(titanic_X).shape[1]
    titanic_X_reduced = model.transform(titanic_X)
    titanic_test_reduced = model.transform(titanic_test_X)
    print(titanic_X_reduced.shape)
    print(titanic_test_reduced.shape)
    
if __name__ == '__main__':
    titanic_train, titanic_test, submit = loading_data()
    titanic_data = titanic_train.append(titanic_test)
    get_miss_count(titanic_data)
    
    titanic_data = data_processing(titanic_data)
    titanic_X, titanic_Y, titanic_test_X = data_split(titanic_data)
    forest_fit, test_y_predicted = RandomForestmodel(titanic_X, titanic_Y, titanic_test_X)
    importances = forest_fit.feature_importances_

    Feature_Rank = pd.DataFrame( { 'Feature_Name':titanic_X.columns, 'Importance':importances } )
        
    print(Feature_Rank)
    print( f'預測結果：' )
    print(forest_fit.oob_score_)
    features_reduced(forest_fit, titanic_X, titanic_test_X)
    output_file(test_y_predicted)

    # accuracy = metrics.accuracy_score(titanic_test_y, test_y_predicted)
    # print(accuracy)