import json
import pickle
import numpy as np
import os
import random
import re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
import time

# https://www.basilica.ai/tutorials/how-to-train-an-image-model/

EMB_DIR = './embeddings/'

files = [f for f in os.listdir(EMB_DIR)]
# shuffle files in list
random.seed(42)
random.shuffle(files)

# setting train size to 80% of data
train_size = int(len(files)*0.8)

# initializing train and target to 0 vectors/matrices
x_train = np.zeros((train_size, 768))
x_test = np.zeros((len(files)-train_size, 768))
y_train = np.zeros(train_size, dtype=int)
y_test = np.zeros(len(files)-train_size, dtype=int)

# loop through train indices and store in x_train and y_train
for i in range(train_size):
    filename = files[i]
    with open(EMB_DIR + filename, 'r') as f:
        x_train[i] = json.load(f)
        y_train[i] = (0 if re.match('.*neg.*', filename) else 1)

# loop trhough test indices and store in x_test and y_test
for i in range(len(files) - train_size):
    filename = files[train_size+i]
    with open(EMB_DIR + filename, 'r') as f:
        x_test[i] = json.load(f)
        y_test[i] = (0 if re.match('.*neg.*', filename) else 1)

# make sure pos/neg split is correct
print("Negative/Positive Breakdown")
print(y_train.mean(), y_test.mean())

# normalize data
x_train = normalize(x_train)
x_test = normalize(x_test)

lib_path = "./app/lib/"

# MODEL 1 - Logistic Regression
# instantiate model
model = LogisticRegression()
# fit model to training data
model.fit(x_train, y_train)
# see resulting model's score on test data
print("Logistic Regression Test Score:")
print(model.score(x_test, y_test))
# last score: 95%
# save model
filename= lib_path+"lr.pkl"  
with open(filename, 'wb') as file:  
    pickle.dump(model, file)

# MODEL 2 - Random Forest
# instantiate model
# model = RandomForestClassifier()
# fit model to training data
# model.fit(x_train, y_train)
# see resuling model's score
# print("Random Forest Test Score:")
# print(model.score(x_test, y_test))
# last score: 93.48%


# MODEL 3 - LinearSVC
# instantiate model
model = LinearSVC()
# fit model on training data
model.fit(x_train, y_train)
# see resulting model's score
print("Linear SVC:")
print(model.score(x_test, y_test))
# last score:
# save model
filename= lib_path+"svc.pkl"  
with open(filename, 'wb') as file:  
    pickle.dump(model, file)