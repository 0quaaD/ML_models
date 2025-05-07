import pickle
import os
print(os.getcwd())
with open('classification_report.pkl','rb') as file:
    a = pickle.load(file)

print(f"Classification Report: \n{a}")
