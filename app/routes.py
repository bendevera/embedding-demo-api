from app import app as api 
from flask import jsonify, request 
from app.util import get_sentiment_prediction, get_image_prediction
from app.models import ConfusionMatrix


@api.route('/')
def index():
    return jsonify({"message": "Hello, from product-review-api!"})

@api.route('/accuracy/sentiment', methods=["POST"])
def accuracy():
    params = request.json 
    matrix = ConfusionMatrix.query.all()[0]
    matrix.add_prediction(params['answer'], params['predicted'])
    response = matrix.get_matrix()
    print(response)
    return jsonify(response)

@api.route('/predict/sentiment', methods=["POST"])
def predict():
    params = request.json 
    prediction = get_sentiment_prediction(params['review'])
    return jsonify(prediction)

@api.route('/predict/image', methods=["POST"])
def predict_image():
    img = request.files['file']
    if img is None:
        return jsonify({"message": "No file attached."})
    prediction = get_image_prediction(img)
    return jsonify(prediction)
