from app import app as api 
from flask import jsonify, request 
from app.util import get_sentiment_prediction, get_image_prediction


@api.route('/')
def index():
    return jsonify({"message": "Hello, from product-review-api!"})

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
