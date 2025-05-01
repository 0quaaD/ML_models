import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklean.model_selection import train_test_split
from sklearn.KNeighborsClassifier import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def knnClassification(df, test_size):
  if(df.isnull().any().any()):
    raise ValueError('Dataframe contains null values')
  else:
    corr_mat = df.corr()
    corr_values = df.corr()['custcat'].drop('custcat').sort_values(ascending=False)
    X = df.drop('custcat',axis=1)
    y = df.custcat
    X_norm = StandardScaler().fit_transform(X)
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=test_size, random_state=4)
    
    # How to find the best K value for the dataset and Model with Plot?
    Ks = 120
    acc = np.zeros((Ks))
    std_acc = np.zeros((Ks))
    for n in range(1,Ks+1):
      knn_model = KNeighborsClassifier(n_neighbors=n).fit(train_X,train_y)
      y_hat = knn_model.predict(test_X)
      acc[n-1] = accuracy_score(test_y, y_hat)
      std_acc[n-1] = np.std(y_hat == test_y)/np.sqrt(y_hat.shape[0])

    print(f'Best Accuracy : {np.round(100 * acc.max(),2)}%\nBest K value : {acc.argmax()+1}')
    plt.plot(range(1,Ks+1),acc,'g')
    plt.fill_between(range(1,Ks+1), acc-1 * std_acc, acc+1 * std_acc, alpha=0.10)
    plt.tight_layout()
    plt.show()

df = pd.read_csv('teleCust1000t.csv')
knnClassification(df, 0.2)
