import basilica
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import os
import numpy as np

model = LogisticRegression()
filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib/lr.pkl')

with open(filename, 'rb') as file:  
    model = pickle.load(file)

def get_embedding(review):
    with basilica.Connection('8b7c2d45-55d5-66ea-6128-9bbd53df7d7e') as c:
        return normalize([c.embed_sentence(review, model='product-reviews')])

def get_prediction(review):
    x = np.zeros((1, 768))
    x[0] = get_embedding(review)
    print(x)
    prediction = model.predict_proba(x)[0]
    if prediction[0] > prediction[1]:
        return {"sentiment": "Negative", "confidence": prediction[0]}
    else:
        return {"sentiment": "Positive", "confidence": prediction[1]}
