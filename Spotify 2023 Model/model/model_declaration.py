import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
import pickle
import matplotlib.ticker as ticker

from sklearn.metrics import r2_score, mean_squared_error, make_scorer, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

warnings.filterwarnings('ignore')

if(os.getcwd() != '../dataset/'):
    os.chdir('../dataset')
print(os.getcwd())

df = pd.read_csv('spotify-2023.csv',on_bad_lines = 'warn', encoding='latin1')

df = df.dropna()

df = df.drop(columns=['track_name','artist(s)_name','key','mode'],axis=1)

df = df[['in_deezer_charts','streams','in_spotify_charts',
         'in_apple_charts','in_deezer_playlists',
         'in_apple_playlists','in_spotify_playlists']]

#print(df.corr()['streams'].sort_values(ascending=False))


feature_names = df.drop(columns=['streams']).columns.tolist()
X = df.drop(columns=['streams'],axis=1).to_numpy()
y = df['streams']


scaler = StandardScaler()
X = scaler.fit_transform(X)

train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.2,random_state=42)

with open('../model/test_X.pkl','wb') as file:
    pickle.dump(test_X, file)
with open('../model/test_y.pkl','wb') as file:
    pickle.dump(test_y,file)

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Gradient Boosting Hyperparameter Grid
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.05],
    'max_depth': [3, 5],
    'subsample': [1.0, 0.8]
}
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
# Random Forest Regressor Model --> R^2 = ~0.85-0.86
model = RandomForestRegressor(n_estimators = 2000, max_depth = 100, n_jobs = 12)
model_grid = GridSearchCV(model, rf_params, scoring = rmse_scorer, cv=5, n_jobs=-1, verbose=1)
model_grid.fit(train_X, train_y)
y_pred = model_grid.predict(test_X)
r2 = r2_score(test_y, y_pred)
print(f'\nR^2 Score Random Forest: {r2:.2f}')
print(f'RMSE Error Random Forest: {np.sqrt(mean_squared_error(test_y, y_pred)):.2f}')

# ---------------------
model_grad = GradientBoostingRegressor(learning_rate = 0.001, n_estimators = 10000, random_state=42)
model_grad_grid = GridSearchCV(model_grad, gb_params, scoring = rmse_scorer, cv=5, n_jobs=-1, verbose=1)
model_grad_grid.fit(train_X, train_y)

y_pred_grad = model_grad_grid.predict(test_X)

r2 = r2_score(test_y, y_pred_grad)
print(f'\nR^2 Score Gradient: {r2:.2f}')
print(f'RMSE Error Gradient: {np.sqrt(mean_squared_error(test_y, y_pred_grad)):.2f}')

# Plot for Random Regressor
sns.scatterplot(x = test_y, y = y_pred)
plt.plot([test_y.min(), test_y.max()], [y_pred.min(), y_pred.max()], 'r--')
plt.xlabel('Actual data')
plt.ylabel('Predicted data')
plt.title('Plot for Random Regressor')
plt.show()

# Plot for Gradient Boosting
sns.scatterplot(x = test_y, y = y_pred_grad)
plt.plot([test_y.min(), test_y.max()], [y_pred_grad.min(), y_pred_grad.max()], 'r--')
plt.xlabel('Actual data')
plt.ylabel('Predicted data')
plt.title('Plot for Gradient Boosting')
plt.show()

# Plot Feature Importance of the Random Forest Model
best_estimator_forest = model_grid.best_estimator_

imp = best_estimator_forest.feature_importances_
imp_df_forest = pd.DataFrame({'Feature':feature_names, 'Importance':imp}).sort_values(by='Importance',ascending=False)

sns.barplot(data=imp_df_forest,x='Importance',y='Feature')
plt.title('Feature Importance of the Random Forest Model')
plt.show()

# plot Feature Importance of the Gradient Boosting model
best_estimators_grad = model_grad_grid.best_estimator_
imp = best_estimators_grad.feature_importances_
imp_df_grad = pd.DataFrame({'Feature':feature_names, 'Importance': imp}).sort_values(by='Importance',ascending=False)

sns.barplot(data = imp_df_grad, x='Importance',y='Feature')
plt.title('Feature Importance of the Gradient Boosting Model')
plt.show()

with open('../model/spotify_2023_random_forest.pkl','wb') as file:
    pickle.dump(model_grid,file)


with open('../model/spotify_2023_gradient_boosting.pkl','wb') as file:
    pickle.dump(model_grad_grid,file)

