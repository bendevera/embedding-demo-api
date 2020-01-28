# embedding-demo-api

REST API for review sentiment and natural image classification web app.

## setup 
- install dependencies running `pip install -r requirements.txt`
- get basilica API key and input into basilica scripts
- run both basilica scripts
  - `python basilica_scripty.py`
  - `python basilica_image_script.py`
- run build model script 
  - `python build_model.py both`
- run `export FLASK_APP=review_api.py`
- setup db
  - `flask db init`
  - `flask db migrate -m "setting up db"`
  - `flask db upgrade`
- run app:
  - `flask run`
  
## routes
- "/" - [GET] - simple dummy route to ping when needing to know if api is running

- "/predict/sentiment" - [POST] - get product review sentiment prediction
ex request json:
{
  "review": "Text of the review"
}
ex response json:
{
  "class": "positive"/"negative",
  "confidence": 1-0,
  "review": "text of the review"
}

- "/predict/image" - [POST] - get natural image class prediction
request body has image named as "file".
ex response json:
{
  "class": "airplane"/"person"/"car"/"cat"/"dog"/"fruit"/"flower"/"motorbike",
  "confidence": 1-0
}

- "/accuracy/sentiment" - [POST] - route to track whether prediction was correct or incorrect
ex request json:
{
  "answer": "correct"/"incorrect",
  "prediction": "positive"/"negative"
}
ex response json:
{
  "total": int,
  "true_positives": int,
  "true_negatives": int,
  "false_positives": int,
  "false_negatives": int
}
