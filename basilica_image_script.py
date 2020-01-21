import basilica 
import json
import random
import os
from config import Config

'''
Natural Image Dataset: https://www.kaggle.com/prasunroy/natural-images
'''

IMAGE_DATA_PATH = './data/images/'

EMB_DIR = './image-embeddings/'

issue_count = 0

with basilica.Connection(Config['basilica_key']) as c:

    def embed_image(image_file, image_class, count):
        global issue_count
        try:
            embedding = c.embed_image_file(image_file)
            filename = EMB_DIR+image_class+'-'+count+'.emb'
            print(f"Saving {filename} | {image_class} | {count} | {len(embedding)} ")
            with open(filename, 'w') as f:
                f.write(json.dumps(embedding))
        except Exception as e:
            issue_count += 1
            print(f"count {count} didn't go through")

    class_list = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
    completed = ['cat', 'car', 'fruit']
    for directory in os.listdir(IMAGE_DATA_PATH):
        if directory in class_list and directory not in completed:
            for image in os.listdir(os.path.join(IMAGE_DATA_PATH, directory)):
                count = image.split("_")[1][:4]
                embed_image(os.path.join(IMAGE_DATA_PATH, directory, image), directory, count)
        else:
            print(f"found a non class item or completed item: {directory}")
    print(f"Num issues: {issue_count}")
