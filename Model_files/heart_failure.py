import numpy as np
import pandas as pd
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')
os.chdir('/home/elnokin/Desktop/files/Data Science/Pytorch-ML-AI/Datasets')
print(os.getcwd())

df = pd.read_csv('heart.csv', on_bad_lines = 'warn')
df = df.dropna()

one_hot = OneHotEncoder()
label = LabelEncoder()

X = df.drop(columns='HeartDisease',axis=1)
y = df['HeartDisease']

X_enc = one_hot.fit_transform(X)
y_enc = label.fit_transform(y.squeeze())

train_X, test_X, train_y, test_y = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(train_X, train_y)

y_pred = model.predict(test_X)

acc = accuracy_score(test_y, y_pred)
print(f'Accuracy: {np.round(100*acc,2)}%')
print(classification_report(test_y, y_pred))
