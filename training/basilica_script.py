import pandas as pd
import basilica 
import json
import random
from config import Config

DATA_PATH = './data/'

positive = pd.read_csv(DATA_PATH+"pos_subset.csv", names=['sentiment', 'title', 'text'])
positive = positive.reset_index()
negative = pd.read_csv(DATA_PATH+"neg_subset.csv", names=['sentiment', 'title', 'text'])
negative = negative.reset_index()

EMB_DIR = './embeddings/'

start_point = 91503

with basilica.Connection(Config.BASILICA_KEY) as c:

    def embed_reviews(cell, text_class):
        global start_point
        if cell['index'] >= start_point:
            embedding = c.embed_sentence(cell['text'], model='product-reviews')
            filename = EMB_DIR+text_class+'-'+str(cell['index'])+'.emb'
            print(f"Saving {filename} | {text_class} | {cell['index']}")
            with open(filename, 'w') as f:
                f.write(json.dumps(embedding))

    print("Embedding Positive Lines")
    # positive.apply(embed_reviews, args=("pos",), axis=1)
    print("Embedding Negative Lines")
    negative.apply(embed_reviews, args=("neg",), axis=1)
    