import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay 
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore',category=FutureWarning)
print(os.getcwd())

df = pd.read_csv("weatherAUS.csv",on_bad_lines='warn')

df = df.dropna()
df = df.rename(columns={'RainToday': 'RainYest', 'RainTomorrow':'RainToday'})
df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia'])]

def date_to_season(date):
    month = date.month
    if(month == 12 or month == 1 or month == 2):
        return 'Summer'
    elif(month == 3 or month == 4 or month == 5):
        return 'Autumn'
    elif(month == 6 or month == 7 or month == 8):
        return 'Winter'
    elif(month == 9 or month == 10 or month == 11):
        return 'Spring'

df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)
df = df.drop(columns='Date',axis=1)

X = df.drop(columns='RainToday',axis=1)
y = df['RainToday']

train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print(train_X.info())
numeric_features = train_X.select_dtypes(include=['float64']).columns.tolist()
categorical_features = train_X.select_dtypes(include=['object','category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier', RandomForestClassifier()),
])

# true positive rate = 0.72 --> Random Forest
# true positive rate = 0.71 --> Logistic Regression
# Humidity3pm -- most important feature

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)  
grid_search.fit(train_X, train_y)
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

test_score = grid_search.score(test_X, test_y)  
print("Test set score: {:.2f}".format(test_score))

y_pred = grid_search.predict(test_X)
print("\nClassification Report:")

print(classification_report(test_y, y_pred))

conf_matrix = confusion_matrix(test_y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

N = 20  # Change this number to display more or fewer features
top_features = importance_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])



# update the model's estimator to use the new pipeline
grid_search.estimator = pipeline

# Define a new grid with Logistic Regression parameters
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}

grid_search.param_grid = param_grid

# Fit the updated pipeline with LogisticRegression
grid_search.fit(train_X, train_y)

# Make predictions
y_pred = grid_search.predict(test_X)

print(classification_report(test_y, y_pred))

# Generate the confusion matrix 
conf_matrix = confusion_matrix(test_y, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()
