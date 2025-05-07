# 🍷Wine Quality Classifier using Machine Learning  
## Project Overview
This project aims to classify red wine quality as Good (1) or Bad (0) based on various chemical properties of the wine. 
We use a Random Forest Classifier to predict the quality based on 11 features extracted from the red wine dataset. 
The classification model is deployed on a web interface using Streamlit, allowing users to input wine characteristics and receive a prediction about the quality.

## Project Objectives

- **📊 Data preprocessing** : Clean the data by handling missing values and scaling the features.  
- **✴️ Model Training** : Train a machine learning model (Random Forest Classifier) to predict wine quality as good or bad.
- **📈 Model Evaluation** : Evaluate the model using metrics like accuracy, confusion matrix, precision, recall, and F1-score.
- **🌐 Web Application** : Create a user-friendly web interface to input wine features and predict the quality using the trained model.

## 👨‍💻 Used softwares and languages  
- **Python** : The primary programming language for model training, evaluation, and application development.
- **Scikit-Learn** : Machine learning library used for model training and evaluation.
- **Streamlit** : Python library used to build the interactive web application.
- **Pandas and NumPy** : Libraries for data manipulation and numerical computations.
- **Pickle** : Library used to serialize the trained model and scaler for later use in the Streamlit application.

## 📊 The Dataset  
The [dataset](dataset/winequality-red.csv) used in this project is the **Wine Quality Dataset** from the UCI Machine Learning Repository. 
It contains information about various chemical properties of red wines, and the goal is to predict the quality of the wine.

### 📋 Dataset Features
1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

The target variable is **'quality'**, which is a numerical rating between 0 and 10. 
The model predicts if the wine is "Good (1)" or "Bad (0)" based on these features.

## Project Steps
1. **📊 Data Preprocessing**
- The dataset is read and missing values are removed using `dropna()`.
- The features are scaled using `StandardScaler` to normalize the data and ensure that all features contribute equally to the model.
-  The target variable (`quality`) is binarized into `Good` or `Bad` wine using a threshold of `6.5`, where wines with a quality rating above 6.5 are considered `Good (1)`, and below or equal to 6.5 are considered `Bad (0)`
2. **✴️ Model Training**
- The dataset is split into training and testing sets using `train_test_split`.
- A `Random Forest Classifier` is trained on the data. Random forests are an ensemble learning method that creates multiple decision trees and combines their results for more accurate predictions.
- Hyperparameters such as `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` are tuned to improve model performance.
3. **📈 Model Evaluation**
- The trained model is evaluated using various metrics:
  - **Accuracy** : `~89%`
  - **Precision(Good)** : `~56%`
  - **Recall(Good)** : `~81%`
  - **F1-Score(Good)** : `~66%`
  - **Class Imbalance** : 347 Bad vs. 53 Good
 
    
  **Confusion Matrix** :
  
    ``[[313 34]
    [10 43]]``

# Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------:|
| 0     |   0.97    |  0.90  |   0.93   |     347 |
| 1     |   0.56    |  0.81  |   0.66   |      53 |
| **Accuracy** |           |        |   **0.89**   |     **400** |
| **Macro Avg** |   0.76    |  0.86  |   0.80   |     400 |
| **Weighted Avg** |   0.91    |  0.89  |   0.90   |     400 |



## 🏁 Conclusion  

This project demonstrates the use of machine learning to predict wine quality based on its chemical properties. The model is deployed in a user-friendly web interface, making it easy for users to input their wine data and get predictions. The Random Forest Classifier achieved a solid accuracy of ~89%, making it an effective tool for wine quality prediction.
