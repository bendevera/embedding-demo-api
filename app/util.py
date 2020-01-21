import basilica
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
import os
import numpy as np

model = LogisticRegression()
filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib/lr.pkl')

with open(filename, 'rb') as file:  
    model = pickle.load(file)

img_model = RandomForestClassifier()
filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib/rf.pkl')

with open(filename, 'rb') as file:  
    img_model = pickle.load(file)

filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib/class_map.json')
class_map = {}

with open(filename, 'r') as file:
    class_map = json.load(file)

new_class_map = {}

for key, value in class_map.items():
    new_class_map[value] = key

def get_embedding(review):
    with basilica.Connection('8b7c2d45-55d5-66ea-6128-9bbd53df7d7e') as c:
        return normalize([c.embed_sentence(review, model='product-reviews')])

def get_image_embedding(image):
    with basilica.Connection('8b7c2d45-55d5-66ea-6128-9bbd53df7d7e') as c:
        return normalize([c.embed_image(image.read())])

def get_prediction(review):
    x = np.zeros((1, 768))
    x[0] = get_embedding(review)
    prediction = model.predict_proba(x)[0]
    if prediction[0] > prediction[1]:
        return {"class": "Negative", "confidence": prediction[0], "review": review}
    else:
        return {"class": "Positive", "confidence": prediction[1], "review": review}

def get_image_prediction(image):
    x = np.zeros((1, 2048))
    x[0] = get_image_embedding(image)
    prediction = img_model.predict_proba(x)[0]
    prediction = list(prediction)
    best_prob = max(prediction)
    best_class = new_class_map[img_model.classes_[prediction.index(best_prob)]]
    return {"class": best_class, "confidence": best_prob, "review": "thank you for your feedback!"}
