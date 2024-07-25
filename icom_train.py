from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import joblib 


#create a model instance
#C is a positive floating-point number (1.0 by default) that defines the relative strength of regularization. Smaller values indicate stronger regularization.
model = LogisticRegression(solver='liblinear', random_state=0, C = 1.0)

#training data
#Al
data_al = pd.read_csv()
#splitting of data  
#x data need to reshape, so we apply .reshape() with the arguments -1 to get as many rows as needed and 1 to get one column.
X1 = np.array().shape(-1,1)
Y1 = np.array()

X1_train, X1_test, Y1_train, Y1_test = \
    train_test_split(X1, Y1, test_size=0.3,
                     random_state=2018)

model_al = model.fit(X1_train, Y1_train)
joblib.dump(model_al, '')

#model_al.coef_
#model_al.intercept_





#Mg
data_mg = pd.read_csv("/Users/kjx/Desktop/Mg.csv")
#splitting of data
X2 = np.array(data_mg["L"]).shape(-1,1)
Y2 = np.array(data_mg["Hazardous"])
#X2_train, X2_test, Y2_train, Y2_test = \
#    train_test_split(X2, Y2, test_size=0.3,
#                     random_state=2018)

model_mg = model.fit(X2, Y2)
joblib.dump(model_mg, '/Users/kjx/Desktop/Mg.ckp')


#Co
data_co = pd.read_csv()
#splitting of data
X3 =
Y3 =
X3_train, X3_test, Y3_train, Y3_test = \
    train_test_split(X3, Y3, test_size=0.3,
                     random_state=2018)

model_co = model.fit(X1_train, Y1_train)
joblib.dump(model_co, '')


#Fe
data_fe = pd.read_csv()
#splitting of data
X4 =
Y4 =
X4_train, X4_test, Y4_train, Y4_test = \
    train_test_split(X4, Y4, test_size=0.3,
                     random_state=2018)

model_fe = model.fit(X1_train, Y1_train)
joblib.dump(model_fe, '')

#Cu
data_cu = pd.read_csv()
#splitting of data
X5 =
Y5 =
X5_train, X5_test, Y5_train, Y5_test = \
    train_test_split(X5, Y5, test_size=0.3,
                     random_state=2018)

model_cu = model.fit(X1_train, Y1_train)
joblib.dump(model_cu, '')
