# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn import tree
# save model
from sklearn.externals import joblib


train_df = pd.read_csv('data/train.csv')
#test_df = pd.read_csv('data/test.csv')

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
#test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
#combine = [train_df, test_df]


train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',\
'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
train_df['Title'] = train_df['Title'].map(title_mapping)
train_df['Title'] = train_df['Title'].fillna(0)
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

guess_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = train_df[(train_df['Sex'] == i) & (train_df['Pclass'] == j + 1)]['Age'].dropna()
age_guess = guess_df.median()
guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.
for i in range(0, 2):
    for j in range(0, 3):
        train_df.loc[(train_df.Age.isnull()) & (train_df.Sex == i) & (train_df.Pclass == j + 1), \
                     'Age'] = guess_ages[i, j]

train_df['Age'] = train_df['Age'].astype(int)

train_df.loc[train_df['Age'] <= 16, 'Age'] = 0
train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age'] = 1
train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age'] = 2
train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age'] = 3
train_df.loc[train_df['Age'] > 64, 'Age'] = 4


train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = 0
train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize', 'Embarked', 'Fare', 'Title'], axis=1)


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]

TreeClf = tree.DecisionTreeClassifier(max_depth= 6, min_samples_leaf=59)
TreeClf = TreeClf.fit(X_train,Y_train)
print(round(TreeClf.score(X_train, Y_train) * 100, 2))
print(TreeClf.feature_importances_)

tree.export_graphviz(TreeClf,
                    feature_names=list(X_train),
                    out_file = 'TreeGraph.txt',
                    class_names=['Survived', 'Did not survived'],
                    label = 'none',filled =True,leaves_parallel=True,impurity=False)

feat_importances = pd.Series(TreeClf.feature_importances_, index=X_train.columns)


print('A')
print('id=2 ' + str(TreeClf.predict_proba([X_train.loc[1]])))
print('id=341 ' + str(TreeClf.predict_proba([X_train.loc[340]])))
print('id=584 ' + str(TreeClf.predict_proba([X_train.loc[583]])))
print('id=413 ' + str(TreeClf.predict_proba([X_train.loc[412]])))
print('id=600 ' + str(TreeClf.predict_proba([X_train.loc[599]])))
print('id=178 ' + str(TreeClf.predict_proba([X_train.loc[177]])))
print('id=297 ' + str(TreeClf.predict_proba([X_train.loc[296]])))
print('id=253 ' + str(TreeClf.predict_proba([X_train.loc[252]])))
print('id=371 ' + str(TreeClf.predict_proba([X_train.loc[370]])))
print('id=358 ' + str(TreeClf.predict_proba([X_train.loc[257]])))

print('B')
print('id=139 ' + str(TreeClf.predict_proba([X_train.loc[138]])))
print('id=133 ' + str(TreeClf.predict_proba([X_train.loc[132]])))
print('id=115 ' + str(TreeClf.predict_proba([X_train.loc[114]])))
print('id=773 ' + str(TreeClf.predict_proba([X_train.loc[772]])))
print('id=506 ' + str(TreeClf.predict_proba([X_train.loc[505]])))
print('id=886 ' + str(TreeClf.predict_proba([X_train.loc[885]])))
print('id=231 ' + str(TreeClf.predict_proba([X_train.loc[230]])))
print('id=586 ' + str(TreeClf.predict_proba([X_train.loc[585]])))
print('id=10 ' + str(TreeClf.predict_proba([X_train.loc[9]])))
print('id=75 ' + str(TreeClf.predict_proba([X_train.loc[74]])))

pass
#logreg = LogisticRegression()
#logreg.fit(X_train, Y_train)

#acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
#print(acc_log)
#78.56
#joblib.dump(logreg, 'logreg.pkl')
#pass

#logreg = joblib.load('logreg.pkl')
