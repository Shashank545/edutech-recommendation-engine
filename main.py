# Code owner : Shashank Sahoo
# Local imports
import datetime

# Third party imports
from pydantic import BaseModel, Field

from libr import app, map_dict
from libr.utils import get_model_response


model_name = "Course Recommendation System"
version = "v1.0.0"


# Input for data validation
class Input(BaseModel):
    gender_code: str = Field()
    stream_code: str = Field()
    subject_code: str = Field()
    marks: int = Field(..., gt=0)


# Ouput for data validation
class Output(BaseModel):
    label: str
    prediction: int


@app.get('/modelinfo')
async def model_info():
    """Show information about the model system"""
    return {
        "name": model_name,
        "version": version
    }


@app.get('/apphealth')
async def service_health():
    """Return service health"""
    return {
        "ok"
    }


@app.post('/predict', response_model=Output)
async def model_predict(input: Input):
    """Predict with given test      input"""

    gender_test = [k for k,v in map_dict["gender"].items() if v == input.gender_code]
    stream_test = [k for k,v in map_dict["stream"].items() if v == input.stream_code]
    subject_test = [k for k,v in map_dict["subject"].items() if v == input.subject_code]
    
    assert isinstance(gender_test, list), "Gender datatype is incorrect"
    assert isinstance(stream_test, list), "Stream datatype is incorrect"
    assert isinstance(subject_test, list), "Subject datatype is incorrect"

    response = get_model_response(input, gender_test,stream_test,subject_test)
    return response 