import pandas as pd
import basilica 
import json
import random

DATA_PATH = './data/'

# positive = pd.read_csv(DATA_PATH+"pos_subset.csv", names=['sentiment', 'title', 'text'])
# positive = positive.reset_index()
negative = pd.read_csv(DATA_PATH+"neg_subset.csv", names=['sentiment', 'title', 'text'])
negative = negative.reset_index()

EMB_DIR = './embeddings/'

with basilica.Connection('8b7c2d45-55d5-66ea-6128-9bbd53df7d7e') as c:

    def embed_reviews(cell, text_class):
        embedding = c.embed_sentence(cell['text'], model='product-reviews')
        filename = EMB_DIR+text_class+'-'+str(cell['index'])+'.emb'
        print(f"Saving {filename} | {text_class} | {cell['index']}")
        with open(filename, 'w') as f:
            f.write(json.dumps(embedding))

    # print("Embedding Positive Lines")
    # positive.apply(embed_reviews, args=("pos",), axis=1)
    print("Embedding Negative Lines")
    negative.apply(embed_reviews, args=("neg",), axis=1)
    