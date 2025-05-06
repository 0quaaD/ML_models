import pandas as pd
import numpy as np
import warnings
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
os.chdir('/home/elnokin/Desktop/files/Data Science/Pytorch-ML-AI/Datasets')
print(os.getcwd())

df = pd.read_csv('mushrooms.csv', on_bad_lines = 'warn')
df = df.dropna()

X = df.drop(columns='class',axis=1)
y = df['class']

lb_y = LabelEncoder()
one_hot = OneHotEncoder()

X_encoded = one_hot.fit_transform(X)
y_encoded = lb_y.fit_transform(y.squeeze())

train_X, test_X, train_y, test_y = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(train_X, train_y)

y_pred = model.predict(test_X)
acc = accuracy_score(test_y, y_pred)
print(f'Accuracy: {np.round(100*acc,2)}')
