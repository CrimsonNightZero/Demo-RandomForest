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

def Cabin_process_by_Family(data):
    titanic_data["Cabin"] = np.where(titanic_data["Cabin"], titanic_data["Cabin"].str[0], titanic_data["Cabin"])

    for family in titanic_data['Family'].unique(): 
        exist_Cabin = ( titanic_data['Family']==family )&( titanic_data["Cabin"] )
        if len(titanic_data.loc[exist_Cabin,'Cabin']) > 0:
            Na_Cabin = ( titanic_data['Family']==family )&( titanic_data["Cabin"].isnull() )
            titanic_data.loc[Na_Cabin,'Cabin'] = titanic_data.loc[exist_Cabin,'Cabin'].iloc[0]
    return titanic_data['Cabin']

def Cabin_process_related(data, relate_col):
    for pclass in titanic_data['Pclass'].unique():
        for relate_value in titanic_data[relate_col].unique(): 
            exist_Cabin = ( titanic_data[relate_col]==relate_value )&( titanic_data['Pclass']==pclass )&( titanic_data["Cabin"] )
            if len( titanic_data.loc[exist_Cabin,'Cabin']) >0:
                Ticket_Pclass = pd.DataFrame(titanic_data.loc[exist_Cabin,'Cabin'])
                Ticket_Pclass.columns = ['Cabin']
                Ticket_Pclass =Ticket_Pclass['Cabin'].value_counts().reset_index( )  
                Ticket_Pclass.columns = ['Cabin','Count']
                Ticket_Pclass.sort_values(by='Count', ascending=False, inplace=True )

                Na_Cabin = ( titanic_data[relate_col]==relate_value )&( titanic_data['Pclass']==pclass )&( titanic_data["Cabin"].isnull() )
                titanic_data.loc[Na_Cabin,'Cabin'] = Ticket_Pclass['Cabin'][0]
    return titanic_data['Cabin']

def Age_process(titanic_data):
    Age_Mean = titanic_data[['Name','Age']].groupby( by=['Name'] ).mean()
    Age_Median = titanic_data[['Name','Age']].groupby( by=['Name'] ).median()
    
    Age_Mean.columns = ['Age Mean']
    Age_Median.columns = ['Age Median']
    Age_Mean.reset_index( inplace=True )
    Age_Median.reset_index( inplace=True )
    
    for index, name in enumerate(Age_Mean['Name']):
        titanic_data.loc[(( titanic_data['Name']==name )&( titanic_data["Age"].isnull() )),'Age'] = Age_Mean.loc[ Age_Mean['Name']==name,'Age Mean' ][index]
    
    return titanic_data['Age']

# 資料預處理
def data_processing(titanic_data):
    titanic_data["Embarked"] = np.where(titanic_data["Embarked"].isnull(), "N", titanic_data["Embarked"])

    titanic_data["Ticket"] = titanic_data["Ticket"].str.split(" ", expand=True)[0]

    titanic_data["Family"] = titanic_data["Name"].str.split(",", expand=True)[0]

    titanic_data["Name"] = titanic_data["Name"].str.split(",", expand=True)[1].str.split(".", expand=True)[0]
    
    get_miss_count(titanic_data['Cabin'])
    
    titanic_data['Family'] = titanic_data['Family'].str.strip()
    titanic_data['Cabin'] = titanic_data['Cabin'].str.strip()
    
    titanic_data['Cabin'] = Cabin_process_by_Family(titanic_data)
    
    titanic_data['Cabin'] = Cabin_process_related(titanic_data, 'Ticket')

    titanic_data['Cabin'] = Cabin_process_related(titanic_data, 'Fare')
    
    get_miss_count(titanic_data['Cabin'])  
    
    titanic_data["Cabin"] = np.where(titanic_data["Cabin"].isnull(), "No", titanic_data["Cabin"].str[0])

    titanic_data['Name'] = titanic_data['Name'].str.strip()
    
    titanic_data['Age'] = Age_process(titanic_data)

    titanic_data["Fare"] = np.where(titanic_data["Fare"] > 0 | titanic_data["Fare"].isnull(), np.log10(titanic_data["Fare"]), titanic_data["Fare"])
    titanic_data["Fare"] = fill_median(titanic_data["Fare"])
    titanic_data["Single"] = np.where((titanic_data["Parch"] == 0) & (titanic_data["SibSp"] == 0), 0, 1)
    get_miss_count(titanic_data)
    
    for col in ["Sex", "Embarked", "Ticket", "Family", "Cabin", "Name"]:
      titanic_data[col] = value_encoder(titanic_data[col])
      
    titanic_data = normalize(titanic_data, titanic_data.columns.drop('Survived').drop('PassengerId'))
    
    return titanic_data

# 建立訓練與測試資料
def data_split(titanic_data):
    titanic_train = titanic_data[ pd.notnull(titanic_data.Survived) ]
    titanic_test = titanic_data[ pd.isnull(titanic_data.Survived) ]
    
    titanic_X = titanic_train.drop(['PassengerId','Survived','Family'], axis=1)
    titanic_test_X = titanic_test.drop(['PassengerId','Survived','Family'], axis=1)

    titanic_Y = titanic_train["Survived"]
    
    return titanic_X, titanic_Y, titanic_test_X

# 建立隨機生成樹模型
def RandomForestmodel(titanic_X, titanic_Y, titanic_test_X):
    forest = ensemble.RandomForestClassifier(n_estimators = 1000,
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
    # n_features = model.transform(titanic_X).shape[1]
    titanic_X_reduced = model.transform(titanic_X)
    titanic_test_reduced = model.transform(titanic_test_X)
 
    print(titanic_X_reduced.shape)
    print(titanic_test_reduced.shape)
    return  titanic_X_reduced

def test():
    titanic_train["Fare"]=np.where(titanic_train["Fare"] > 0, np.log10(titanic_train["Fare"]), 0)
    for x in range(10):
        groups = titanic_train.groupby(['Survived', pd.cut(titanic_train.Fare, x+1)])
        print(groups.size().unstack())

    print(titanic_train[['Sex','Survived']].groupby(['Sex','Survived']).size().reset_index(name="Freq"))
    
if __name__ == '__main__':
    titanic_train, titanic_test, submit = loading_data()
    titanic_data = titanic_train.append(titanic_test)
 
    # print(titanic_train.describe())
    # print(titanic_test.describe())

    get_miss_count(titanic_train)
    get_miss_count(titanic_test)
    
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