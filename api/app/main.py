import __main__
from warnings import simplefilter
import dill
import pandas as pd
import shap
import uvicorn
import json
import os
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.params import Query
import logging

__main__.pd = pd

app = FastAPI(title='DefaultRiskApp',
              description="""API qui alimente le tableau de bord client, construit dans le cadre du 
              projet 7 du parcours Datascientist proposé par OpenClassrooms""",
              version="0.1")

my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, filename='logs.log')

DATA_PATH = os.environ.get('DATA_PATH')
MODEL_PATH = os.environ.get('MODEL_PATH')

simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

classifier = None
nearest_neighbors = None

client_database = pd.DataFrame()
best_params = pd.DataFrame()


@app.on_event("startup")
def load_models():
    global classifier
    global nearest_neighbors
    my_logger.info('start loading models')
    try:
        classifier = dill.load(open(MODEL_PATH + "/classifier.pkl", "rb"))
        nearest_neighbors = dill.load(open(MODEL_PATH + "/nearest_neighbors.pkl", "rb"))
        my_logger.info('Models loaded')
    except BaseException as e:
        my_logger.error("Error while trying to load models : " + str(e))
        

@app.on_event("startup")
def load_data():
    global client_database
    global best_params
    my_logger.info('start loading data')
    try:
        client_database = pd.read_pickle(DATA_PATH + '/X_sample.pkl')
        best_params = pd.read_pickle(DATA_PATH + '/BestParams.pkl')
        my_logger.info('Data loaded')
    except BaseException as e:
        my_logger.error("Error while trying to load models : " + str(e))


@app.get('/in_database')
def check_client_in_database(client_id: int):
    try:
        check = client_id in client_database.index
        return {'check': check}
    except BaseException as e:
        my_logger.error("Error while checking whether client is in file or not :" + str(e))


@app.get('/threshold')
def get_default_threshold():
    try:
        thresh = best_params.loc[best_params['Param'] == 'thresh', 'Best Param'].values[0]
        return {"threshold": thresh}
    except BaseException as e:
        my_logger.error("Error while reading threshold in Best_Params.pkl file :" + str(e))


@app.get('/client_info')
def get_client_info(client_id: int):
    try:
        client_info = client_database.loc[[client_id]]

        return Response(client_info.to_json(orient='records'), media_type="application/json")
    except BaseException as e:
        my_logger.error('Error while retrieving client info: ' + str(e))


@app.get('/download_database')
def download_client_database():
    file_path = DATA_PATH + '/X_sample.pkl'
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=file_path, media_type='application/pickle')
    my_logger.error('Client database file not found')
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
        my_logger.error('Error while predicting client default proba: ' + str(e))


@app.get('/predict_default_all_clients')
def predict_default_all(client_ids: list[int] = Query(...)):
    try:
        client_info = client_database.loc[client_ids]
        client_proba = classifier.predict_proba(client_info)
        client_default_proba = client_proba[:, 1]

        return {'proba_default': client_default_proba.tolist()}
    except BaseException as e:
        my_logger.error('Error while trying to predict dafault proba for all clients: ' + str(e))


@app.get('/numeric_features_list')
def get_numeric_features():
    try:
        numeric_features = classifier[0].named_transformers_['pipeline-3'][0].feature_names_in_
        return {'numeric_features': numeric_features.tolist()}
    except BaseException as e:
        my_logger.error('Error while trying to retrieve numeric features as a list: ' + str(e))


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
        my_logger.error('Error while trying to compute shap values: ' + str(e))


@app.get('/client_ids')
def get_client_ids():
    return {"ids": client_database.index.to_list()}


@app.get('/nearest_neighbors_ids')
def get_nearest_neighbors_ids(client_id: int, n_neighbors: int):
    try:
        transformed_data = nearest_neighbors[0].transform(client_database.loc[[client_id]])
        indices = nearest_neighbors[1].kneighbors(transformed_data, n_neighbors=n_neighbors, return_distance=False)
        neighbors_ids = client_database.iloc[indices[0].tolist()].index.tolist()

        return {"nearest_neighbors_ids": neighbors_ids}
    except BaseException as e:
        my_logger.error('Error while trying to find nearest_neighbors: ' + str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn main:app --reload
