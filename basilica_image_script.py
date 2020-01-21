import basilica 
import json
import random
import os

IMAGE_DATA_PATH = './data/images/'

EMB_DIR = './image-embeddings/'

with basilica.Connection('8b7c2d45-55d5-66ea-6128-9bbd53df7d7e') as c:

    def embed_image(image_file, person, count):
        try:
            embedding = c.embed_image_file(image_file)
            filename = EMB_DIR+person+'-'+count+'.emb'
            print(f"Saving {filename} | {person} | {count} | {len(embedding)} ")
            with open(filename, 'w') as f:
                f.write(json.dumps(embedding))
        except Exception as e:
            print(f"count {count} didn't go through")

    log = {}
    for directory in os.listdir(IMAGE_DATA_PATH):
        person = directory.lower().replace(" ", "-")
        images = os.listdir(os.path.join(IMAGE_DATA_PATH, directory))
        log[person] = len(images)
    top10 = sorted(log.values(), reverse=True)[:10]
    people = []
    for key, value in log.items():
        for val in top10:
            if val == value:
                people.append(key)
    print(people)

    # for directory in os.listdir(IMAGE_DATA_PATH):
    #     person = directory.lower().replace(" ", "-")
    #     if person in people:
    #         for image in os.listdir(os.path.join(IMAGE_DATA_PATH, directory)):
    #             count = image.split(".")[0]
    #             embed_image(os.path.join(IMAGE_DATA_PATH, directory, image), person, count)