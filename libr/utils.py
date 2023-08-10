# Code owner : Shashank Sahoo
import pandas as pd
from libr import model, map_dict


def predict(X, model):
    prediction = model.predict(X)[0]
    return prediction


def get_model_response(input, gender_test,stream_test,subject_test):
    
    X = pd.json_normalize(input.__dict__)
    X["gender_code"] = gender_test
    X["stream_code"] = stream_test
    X["subject_code"] = subject_test
    prediction = predict(X, model)
    course_predicted = [v for k,v in map_dict["course"].items() if int(k) == prediction]
    label = course_predicted[0]
    assert isinstance(label, str), "prediction type is incorrect"
    return {
        'label': label,
        'prediction': int(prediction)
    }

