# Code owner : Shashank Sahoo
# Imports
from fastapi import FastAPI
import joblib, pickle

# Initialize FastAPI app
app = FastAPI()

# Load model
model = joblib.load("model/model_binary.dat.gz")

# Load Model target mapping
with open("model/model_target_map.pkl", 'rb') as f:
    map_dict = pickle.load(f)