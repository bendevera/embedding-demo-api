from app import app as api 
from flask import jsonify, request 
from app.util import get_prediction


@api.route('/')
def index():
    return jsonify({"message": "Hello, from product-review-api!"})

@api.route('/predict', methods=["POST"])
def predict():
    params = request.json 
    print(params)
    prediction = get_prediction(params['review'])
    return jsonify(prediction)