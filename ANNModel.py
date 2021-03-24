# Predict if a customer will leave the bank or not
import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values  #the first 3 columns are irrelevant
y = dataset.iloc[:, -1].values

# LabelEncoding -> Gender column -> 0=Female 1=Male
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One Hot Encoding-> Geography column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split dataset 80% training - 20% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Artificial Neural Network - fully connected - 10 input neurons
from tensorflow.keras.layers import Dense
ANNmodel = tf.keras.models.Sequential()
# Input and first hidden layer
ANNmodel.add(Dense(units=5, activation='relu'))
# Second hidden layer
ANNmodel.add(Dense(units=6, activation='relu'))
# Output layer -> sigmoid because we need binary output : stays or leave
ANNmodel.add(Dense(units=1, activation='sigmoid'))

ANNmodel.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Training
ANNmodel.fit(X_train, y_train, batch_size = 32, epochs = 50)
##
"""
Does the customer with the below informations leave or stay?
Geography: France
Credit Score: 200
Gender: Female
Age: 42 years old
Tenure: 3 years
Balance: $ 50000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
"""
print(ANNmodel.predict(sc.transform([[1, 0, 0, 200, 0, 42, 3, 50000, 3, 1, 1, 45000]])) > 0.8)
