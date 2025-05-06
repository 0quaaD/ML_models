## import numpy as np
import pandas as pd
import pickle
import os 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Binarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings('ignore')
os.chdir('/home/elnokin/Desktop/files/Data Science/Pytorch-ML-AI/Datasets')
#print(os.getcwd())

df = pd.read_csv('winequality-red.csv', on_bad_lines = 'warn')
df = df.dropna()

# there is no null values
def plot_ph_sulphates_quality_relation(df):
    sns.scatterplot(data = df, x='pH', y='quality')
    plt.title('Relation of ph and Quality at Wine')
    plt.show()
    sns.scatterplot(data = df, y = 'sulphates', x = 'quality')
    plt.title('Relation of sulphates and quality')
    plt.show()

#plot_ph_sulphates_quality_relation(df)

corr_ = df.corr()
#print(corr_[['quality']].sort_values(by='quality',ascending=False))
X = df.drop(columns=['quality', 'free sulfur dioxide'],axis=1);

# Binary classification: 6+ is good wine
df['good'] = df['quality'] >= 6
y1 = df['good'].astype(int)

y = df['quality'].values
one_hot = OneHotEncoder()
label= LabelEncoder()
binary = Binarizer(threshold=6.5)

X = one_hot.fit_transform(X)
y = binary.fit_transform(y.reshape(-1,1))

train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.25, random_state=42)

choice = input('Decision Tree(tree) or Random Forest(rand)? -->')
if(choice == 'rand'):
    model = RandomForestClassifier(n_estimators=2000,max_depth=10000000, n_jobs=12, criterion='entropy')
    model.fit(train_X, train_y)
    imp = model.feature_importances_
    feature_name = one_hot.get_feature_names_out()
    imp_df = pd.DataFrame({'feature':feature_name, 'importance':imp})
    sns.barplot(data = imp_df.head(10), x='importance',y='feature')
    plt.show()

    y_pred = model.predict(test_X)

    acc = np.round(100 * accuracy_score(test_y, y_pred),2)
    print(f'Accuracy Random Forest: {acc}%')

elif(choice == 'tree'):
    model = DecisionTreeClassifier(max_depth=10,criterion='entropy')
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    acc = np.round(100 * accuracy_score(test_y, y_pred),2)
    print(f'Accuracy Decision Tree: {acc}%')
with open('wine_model.pkl','wb') as file:
    pickle.dump(model,file)
