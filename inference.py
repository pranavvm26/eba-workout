import json
import logging
import sys
import os
import xgboost
import joblib

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# defining model and loading weights to it.
def model_fn(model_dir): 
    print("model dir ----->", model_dir)
    print("model list dir ----->",  os.listdir(model_dir))
    model = joblib.load(os.path.join(model_dir, "eba_xgboost_model.pkl"))
    return model


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    data = json.loads(request_body)["inputs"]["data"]
    print("rx data", data)
    return data


# inference
def predict_fn(input_object, model):
    prediction = model.predict(input_object)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    return json.dumps({"predictions":predictions.tolist()})

# def run():
#     model = model_fn("./")
#     input_data = input_fn(request_body='{"inputs": {"data": [[6.1, 2.8, 4.7, 1.2], [5.7, 3.8, 1.7, 0.3], [7.7, 2.6, 6.9, 2.3], [6.0, 2.9, 4.5, 1.5], [6.8, 2.8, 4.8, 1.4], [5.4, 3.4, 1.5, 0.4], [5.6, 2.9, 3.6, 1.3], [6.9, 3.1, 5.1, 2.3], [6.2, 2.2, 4.5, 1.5], [5.8, 2.7, 3.9, 1.2], [6.5, 3.2, 5.1, 2.0], [4.8, 3.0, 1.4, 0.1], [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [5.1, 3.8, 1.5, 0.3], [6.3, 3.3, 4.7, 1.6], [6.5, 3.0, 5.8, 2.2], [5.6, 2.5, 3.9, 1.1], [5.7, 2.8, 4.5, 1.3], [6.4, 2.8, 5.6, 2.2], [4.7, 3.2, 1.6, 0.2], [6.1, 3.0, 4.9, 1.8], [5.0, 3.4, 1.6, 0.4], [6.4, 2.8, 5.6, 2.1], [7.9, 3.8, 6.4, 2.0], [6.7, 3.0, 5.2, 2.3], [6.7, 2.5, 5.8, 1.8], [6.8, 3.2, 5.9, 2.3], [4.8, 3.0, 1.4, 0.3], [4.8, 3.1, 1.6, 0.2]]}}', request_content_type="application/json")
#     predictions = predict_fn(input_data, model)
#     return output_fn(predictions, content_type ="application/json")

# if __name__ == "__main__":
#     print(run())