
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from flask import Flask, request, render_template
import pickle


url = "https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv"
df = pd.read_csv(url)

#No tienen valor predictivo la elimino
df =df.drop(['Cabin','PassengerId','Ticket','Name','SibSp'],axis=1)

#
df["Sex"]=df["Sex"].map({"male":1,"female":0})
df["Embarked"]=df["Embarked"].map({"S":2,"C":1,"Q":0})

df['Age_clean']=df['Age'].fillna(29)
df=df.drop(['Age'],axis=1)

X=df.drop(columns=['Survived'])
y = df[['Survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=520, test_size=0.2)

#Uso RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5)
modelo.fit(X_train, y_train)

y_train_pred=modelo.predict(X_train)
y_test_pred=modelo.predict(X_test)

#target_names = ['Muere', 'Vive']
#print(classification_report(y_train, y_train_pred, target_names=target_names))
#print(classification_report(y_test, y_test_pred, target_names=target_names))

filename = '../models/titanic_model.pickle'
pickle.dump(modelo, open(filename, 'wb'))

#filename = 'titanic_finalized_model.sav'
#pickle.dump(model, open(filename, 'wb'))



