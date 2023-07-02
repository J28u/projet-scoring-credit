import __main__
from warnings import simplefilter, filterwarnings
import dill
import pandas as pd
import os
import json
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.params import Query

filterwarnings("ignore", message=".*The 'nopython' keyword.*")
simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import shap

# import logging

__main__.pd = pd

app = FastAPI(title='DefaultRiskApp',
              description="""API qui alimente le tableau de bord client, construit dans le cadre du 
              projet 7 du parcours Datascientist proposé par OpenClassrooms""",
              version="0.1")

# my_logger = logging.getLogger()
# my_logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG, filename='logs.log')

print(os.listdir())
DATA_PATH = os.environ.get('DATA_PATH')
MODEL_PATH = os.environ.get('MODEL_PATH')

classifier = None
nearest_neighbors = None

client_database = pd.DataFrame()
best_params = pd.DataFrame()


@app.on_event("startup")
def load_models():
    global classifier
    global nearest_neighbors
    # my_logger.info('start loading models')
    print('start loading models')
    try:
        classifier = dill.load(open(MODEL_PATH + "/classifier.pkl", "rb"))
        nearest_neighbors = dill.load(open(MODEL_PATH + "/nearest_neighbors.pkl", "rb"))
        print('Models loaded')
        # my_logger.info('Models loaded')
    except BaseException as e:
        print("Error while trying to load models : " + str(e))
        # my_logger.error("Error while trying to load models : " + str(e))
        

@app.on_event("startup")
def load_data():
    global client_database
    global best_params
    # my_logger.info('start loading data')
    print('start loading data')
    try:
        client_database = pd.read_pickle(DATA_PATH + '/X_sample.pkl')
        best_params = pd.read_pickle(DATA_PATH + '/BestParams.pkl')
        # my_logger.info('Data loaded')
        print('Data loaded')
    except BaseException as e:
        print("Error while trying to load models : " + str(e))
        # my_logger.error("Error while trying to load models : " + str(e))


@app.get('/in_database')
def check_client_in_database(client_id: int):
    try:
        check = client_id in client_database.index
        return {'check': check}
    except BaseException as e:
        print("Error while checking whether client is in file or not :" + str(e))
        # my_logger.error("Error while checking whether client is in file or not :" + str(e))
        return {'check': False}


@app.get('/threshold')
def get_default_threshold():
    try:
        thresh = best_params.loc[best_params['Param'] == 'thresh', 'Best Param'].values[0]
        # print('thresh : ' + str(thresh))
        return {"threshold": thresh}
    except BaseException as e:
        print("Error while reading threshold in Best_Params.pkl file :" + str(e))
        # my_logger.error("Error while reading threshold in Best_Params.pkl file :" + str(e))
        return {"threshold": 0}


@app.get('/client_info')
def get_client_info(client_id: int):
    try:
        client_info = client_database.loc[[client_id]]
        return Response(client_info.to_json(orient='records'), media_type="application/json")
    except BaseException as e:
        print('Error while retrieving client info: ' + str(e))
        # my_logger.error('Error while retrieving client info: ' + str(e))


@app.get('/download_database')
def download_client_database():
    print('start download')
    file_path = DATA_PATH + '/X_sample.pkl'
    if os.path.exists(file_path):
        print('file exists')
        return FileResponse(path=file_path, filename=file_path, media_type='application/pickle')
    # my_logger.error('Client database file not found')
    print('Client database file not found: ' + file_path)
    return {"message": file_path + " file not found"}


@app.get('/predict_default')
def predict_default(client_id: int):
    try:
        client_info = client_database.loc[[client_id]]
        thresh = best_params.loc[best_params['Param'] == 'thresh', 'Best Param'].values[0]

        client_proba = classifier.predict_proba(client_info)
        client_default_proba = client_proba[:, 1][0]

        if client_default_proba > thresh:
            prediction = "Crédit refusé"
        else:
            prediction = "Crédit accordé"

        return {'prediction': prediction, 'proba_default': client_default_proba}
    except BaseException as e:
        print('Error while predicting client default proba: ' + str(e))
        # my_logger.error('Error while predicting client default proba: ' + str(e))
        return {'prediction': "error", 'proba_default': 0}


@app.get('/predict_default_all_clients')
def predict_default_all(client_ids: list[int] = Query(...)):
    print('start predicting several client defaults')
    try:
        client_info = client_database.loc[client_ids]
        client_proba = classifier.predict_proba(client_info)
        client_default_proba = client_proba[:, 1]

        return {'proba_default': client_default_proba.tolist()}
    except BaseException as e:
        print('Error while trying to predict dafault proba for all clients: ' + str(e))
        # my_logger.error('Error while trying to predict dafault proba for all clients: ' + str(e))
        return {'proba_default': []}


@app.get('/numeric_features_list')
def get_numeric_features():
    try:
        numeric_features = classifier[0].named_transformers_['pipeline-3'][0].feature_names_in_
        # print('numeric features : ' + str(len(numeric_features.tolist())))
        return {'numeric_features': numeric_features.tolist()}
    except BaseException as e:
        # my_logger.error('Error while trying to retrieve numeric features as a list: ' + str(e))
        print('Error while trying to retrieve numeric features as a list: ' + str(e))
        return {'numeric_features': []}


@app.get('/shap_values_default')
def get_shap_values(client_id: int):
    try:
        client_info = client_database.loc[[client_id]]
        explainer = shap.TreeExplainer(classifier[-1], model_outpout='predict_proba')

        binary_features = classifier[0].named_transformers_['pipeline-1'][0].feature_names_in_
        dummies_features = classifier[0].named_transformers_['pipeline-2'][0].get_feature_names()
        numeric_features = classifier[0].named_transformers_['pipeline-3'][0].feature_names_in_

        features = [*binary_features, *dummies_features, *numeric_features]

        client_info_transformed = classifier[0].transform(client_info)
        shap_vals = explainer.shap_values(client_info_transformed)
        expected_values = explainer.expected_value

        return {"shap_values": json.dumps(shap_vals[1].tolist()),
                "expected_values": expected_values[1],
                "features": features}
    except BaseException as e:
        print('Error while trying to compute shap values: ' + str(e))
        # my_logger.error('Error while trying to compute shap values: ' + str(e))
        return {"shap_values": [],
                "expected_values": 0,
                "features": []}


@app.get('/client_ids')
def get_client_ids():
    try: 
        ids_list = client_database.index.to_list()
        # print('client ids : ' + str(len(ids_list)))
        return {"ids": ids_list}
    except BaseException as e:
        print('Error while trying to retrieve client ids list ' + str(e))
        return {"ids": []}


@app.get('/nearest_neighbors_ids')
def get_nearest_neighbors_ids(client_id: int, n_neighbors: int):
    try:
        transformed_data = nearest_neighbors[0].transform(client_database.loc[[client_id]])
        indices = nearest_neighbors[1].kneighbors(transformed_data, n_neighbors=n_neighbors, return_distance=False)
        neighbors_ids = client_database.iloc[indices[0].tolist()].index.tolist()

        return {"nearest_neighbors_ids": neighbors_ids}
    except BaseException as e:
        print('Error while trying to find nearest_neighbors: ' + str(e))
        # my_logger.error('Error while trying to find nearest_neighbors: ' + str(e))
        return {"nearest_neighbors_ids": []}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
# uvicorn main:app --reload
