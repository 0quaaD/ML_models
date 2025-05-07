import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, Binarizer
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

os.chdir('../dataset')
#print(os.getcwd())

df = pd.read_csv('winequality-red.csv',on_bad_lines = 'warn')
#print(df.head(10))
df = df.dropna()
corr_ = df.corr()
#print(corr_['quality'].sort_values(ascending=False))

X = df.drop(columns='quality',axis=1)
y = df['quality'].values.reshape(-1,1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

binary = Binarizer(threshold=6.5)
y = binary.fit_transform(y).ravel()


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=2500,min_samples_split=2,min_samples_leaf=1,max_depth=55, n_jobs=12, class_weight='balanced', random_state=42) 
model.fit(train_X, train_y)

y_pred = model.predict(test_X)


y_proba = model.predict_proba(test_X)[:, 1]
y_pred_custom = (y_proba > 0.3).astype(int)  # try 0.3 instead of 0.5
acc = accuracy_score(test_y,y_pred_custom)
print(f'Accuracy : {100 * acc:.1f}%')
#print(f'Confusion matrix :\n{confusion_matrix(test_y, y_pred_custom)}\n')
print(f'Classification report:\n{classification_report(test_y, y_pred_custom)}')

with open('../model/wine_model.pkl','wb') as file:
    pickle.dump(model,file)
print("Model Successfully saved!")

with open('../model/scaler.pkl','wb') as file:
    pickle.dump(scaler,file)
print('Scaler Successfully saved!')

