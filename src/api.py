from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
import util as utils
import feature_engineering as feature_engineering

config = utils.load_config()
production_model = utils.load_pickle(config["production_model"])
ohe_train = utils.load_pickle(config['ohe_train_path'])
standard_scaler_train = utils.load_pickle(config['standard_scaler_path'])

class api_data(BaseModel):
    GR : float
    ILD_log10 : float
    DeltaPHI : float
    PHIND : float
    PE : float
    NM_M : int

app = FastAPI()     # create object FastAPI

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")     # this 
def predict(data_api: api_data):    
    # Convert data api to dataframe
    data_api = pd.DataFrame(data_api).set_index(0).T.reset_index(drop = True)  # type: ignore
    data_api.columns = config["api_columns"]

    data_api = pd.concat(
        [
            data_api[config["api_columns"][:5]].astype(np.float64),  # type: ignore
            data_api[config["api_columns"][5:]].astype(np.int64)  # type: ignore
        ],
        axis = 1
    )
    
    X_api_numerical, X_api_categorical = feature_engineering.split_numerical_categorical(data=data_api)
    X_api_categorical = feature_engineering.categorical_handling_test_data(data=X_api_categorical, ohe=ohe_train)
    X_api_test = feature_engineering.normalize_test_data(numerical_data=X_api_numerical,
                                                         categorical_data=X_api_categorical,
                                                         scaler=standard_scaler_train)

    # Predict data
    y_pred = production_model.predict(X_api_test.values)
    facies_label_decoder = ['Sandstone','Coarse Siltstone','Fine Siltstone','Marine Siltstone and Shale',
                            'Mudstone','Wackestone','Dolomite','Packstone-Grainstone','Phylloid-Algal Bafflestone']
    return {"res" : facies_label_decoder[y_pred[0]], "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
