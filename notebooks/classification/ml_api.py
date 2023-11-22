# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model
import pydantic
import pycaret
import fastapi

print("Version PyCaret: ", pycaret.__version__)
print("Version Pydantic: ", pydantic.__version__)
print("Version FastAPI: ", fastapi.__version__)

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("ml_api")

# Create input/output pydantic models
input_model = create_model("ml_api_input", **{'id': 8810987.0, 'radius_mean': 13.859999656677246, 'texture_mean': 16.93000030517578, 'perimeter_mean': 90.95999908447266, 'area_mean': 578.9000244140625, 'smoothness_mean': 0.10260000079870224, 'compactness_mean': 0.1517000049352646, 'concavity_mean': 0.09900999814271927, 'concave points_mean': 0.05601999908685684, 'symmetry_mean': 0.21060000360012054, 'fractal_dimension_mean': 0.06915999948978424, 'radius_se': 0.2563000023365021, 'texture_se': 1.194000005722046, 'perimeter_se': 1.9329999685287476, 'area_se': 22.690000534057617, 'smoothness_se': 0.005960000213235617, 'compactness_se': 0.034380000084638596, 'concavity_se': 0.039090000092983246, 'concave points_se': 0.014349999837577343, 'symmetry_se': 0.01939000003039837, 'fractal_dimension_se': 0.004559999797493219, 'radius_worst': 15.75, 'texture_worst': 26.93000030517578, 'perimeter_worst': 104.4000015258789, 'area_worst': 750.0999755859375, 'smoothness_worst': 0.1459999978542328, 'compactness_worst': 0.43700000643730164, 'concavity_worst': 0.4636000096797943, 'concave points_worst': 0.16539999842643738, 'symmetry_worst': 0.3630000054836273, 'fractal_dimension_worst': 0.10589999705553055})
output_model = create_model("ml_api_output", prediction='M')


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
