import __main__
from warnings import simplefilter, filterwarnings
import dill
import pandas as pd
import os
import json
import uvicorn
import logging
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.params import Query
import time

filterwarnings("ignore", message=".*The 'nopython' keyword.*")
simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import shap

__main__.pd = pd

my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, filename='logs.log')

DATA_PATH = os.environ.get('DATA_PATH')
MODEL_PATH = os.environ.get('MODEL_PATH')


class App:
    """
    Classe qui représente une API construite avec FastAPI. 
    Cette API permet d'alimenter un dashboard interactif à destination des gestionnaires 
    de la relation client de la société "Prêt à dépenser". 
    Le but du dashboard étant d'interpréter plus facilement les prédictions 
    d'un modèle de scoring qui calcule la probabilité de faillite d'un client.
    
    Attributes
    -------------------------------------
    title : str : titre de l'api
    description : str : description de l'api
    version : str : numéro de version de l'api
    classifier : : modèle de classification
    nearest_neighbors : : modèle qui renvoie les n plus proches voisins d'un client
    client_database : pd.DataFrame : informations descriptives relatives aux clients
    best_params : pd.DataFrame : hyperparamètres optimisés du modèle de classification
    
    Methods
    -------------------------------------
    create_app(): renvoie une api
    """
    def __init__(self, title: str, description: str, version: str):
        """
        Construit tous les attributs nécessaires pour une instance de la classe App 
        
        Positional Arguments
        -------------------------------------
        title : str : titre de l'api
        description : str : description de l'api
        version : str : numéro de version de l'api
        """
        self.title = title
        self.description = description
        self.version = version
        self.classifier = None
        self.nearest_neighbors = None
        self.client_database = pd.DataFrame()
        self.best_params = pd.DataFrame()

    def create_app(self):
        """
        Renvoie une application FastAPI
        
        Returns : 
        -------------------------------------
        app : fastapi.FastAPI : une application FastAPI
        """
        app = FastAPI(title=self.title,
                      description=self.description,
                      version=self.version)

        @app.on_event("startup")
        def load_models():
            """
            Charge les modèles (classifieur et plus proches voisins)
            
            Returns : 
            -------------------------------------
            None
            """
            my_logger.info('start loading models')
            print('start loading models')
            start = time.perf_counter()
            try:
                self.classifier = dill.load(open(MODEL_PATH + "/classifier.pkl", "rb"))
                self.nearest_neighbors = dill.load(open(MODEL_PATH + "/nearest_neighbors.pkl", "rb"))
                my_logger.info('Models loaded')
                print("Models loaded in {:0.4f} seconds".format(time.perf_counter() - start))
            except BaseException as e:
                print("Error while trying to load models : " + str(e))
                my_logger.error("Error while trying to load models : " + str(e))

        @app.on_event("startup")
        def load_data():
            """
            Charge, à partir de fichiers pickle, les informations clients et le seuil nécessaires
            pour la classification.
            
            Returns : 
            -------------------------------------
            None
            """
            my_logger.info('start loading data')
            print('start loading data')
            start = time.perf_counter()
            try:
                self.client_database = pd.read_pickle(DATA_PATH + '/X_sample.pkl')
                self.best_params = pd.read_pickle(DATA_PATH + '/BestParams.pkl')
                my_logger.info('Data loaded')
                print("Data loaded in {:0.4f} seconds".format(time.perf_counter() - start))
            except BaseException as e:
                print("Error while trying to load models : " + str(e))
                my_logger.error("Error while trying to load models : " + str(e))

        @app.get('/in_database')
        async def check_client_in_database(client_id: int):
            """
            Renvoie True si l'identifiant se trouve dans le dataframe 
            de données chargées, False sinon.

            Positional arguments : 
            -------------------------------------
            client_id : int : identifiant client (SK_ID_CURR)
            """
            start = time.perf_counter()
            try:
                check = client_id in self.client_database.index
                print("check_client_in_database in {:0.4f} seconds".format(time.perf_counter() - start))
                return {'check': check}
            except BaseException as e:
                print("Error while checking whether client is in file or not :" + str(e))
                my_logger.error("Error while checking whether client is in file or not :" + str(e))
                print("check_client_in_database in {:0.4f} seconds".format(time.perf_counter() - start))
                return {'check': False}

        @app.get('/threshold')
        async def get_default_threshold():
            """
            Renvoie le seuil optimisé utilisé pour classifier les clients.  
            Si la probabilité de défaut du client est strictement au dessus de ce seuil 
            le crédit n'est pas accordé.
            """
            start = time.perf_counter()
            try:
                thresh = self.best_params.loc[self.best_params['Param'] == 'thresh', 'Best Param'].values[0]
                print("get_default_threshold in {:0.4f} seconds".format(time.perf_counter() - start))
                return {"threshold": thresh}
            except BaseException as e:
                print("Error while reading threshold in Best_Params.pkl file :" + str(e))
                my_logger.error("Error while reading threshold in Best_Params.pkl file :" + str(e))
                return {"threshold": 0}

        @app.get('/client_info')
        async def get_client_info(client_id: int):
            """
            Renvoie les informations descriptives relatives à un client.
            
            Positional arguments : 
            -------------------------------------
            client_id : int : identifiant client (SK_ID_CURR)
            """
            start = time.perf_counter()
            try:
                client_info = self.client_database.loc[[client_id]]
                print("get_client_info in {:0.4f} seconds".format(time.perf_counter() - start))
                return Response(client_info.to_json(orient='records'), media_type="application/json")
            except BaseException as e:
                print('Error while retrieving client info: ' + str(e))
                my_logger.error('Error while retrieving client info: ' + str(e))

        @app.get('/download_database')
        async def download_client_database(file_name='/X_sample.pkl'):
            """
            Télecharge le fichier pickle contenant toutes les informations clients.

            Positional arguments : 
            -------------------------------------
            file_name : str : "/" + nom du fichier pickle à télécharger
            """
            print('start download')
            file_path = DATA_PATH + file_name
            if os.path.exists(file_path):
                print('file exists')
                return FileResponse(path=file_path, filename=file_path, media_type='application/pickle')
            my_logger.error('Client database file not found')
            print('Client database file not found: ' + file_path)
            return {"message": file_path + " file not found"}

        @app.get('/predict_default')
        async def predict_default(client_id: int):
            """
            Renvoie la probabilité de défaut de remboursement d'un client ainsi qu'un 
            message indiquant si la demande de crédit est acceptée ou refusée.

            Positional arguments : 
            -------------------------------------
            client_id : int : identifiant client (SK_ID_CURR)
            """
            start = time.perf_counter()
            try:
                client_info = self.client_database.loc[[client_id]]
                thresh = self.best_params.loc[self.best_params['Param'] == 'thresh', 'Best Param'].values[0]

                client_proba = self.classifier.predict_proba(client_info)
                client_default_proba = client_proba[:, 1][0]

                if client_default_proba > thresh:
                    prediction = "Crédit refusé"
                else:
                    prediction = "Crédit accordé"
                    
                print("predict_default in {:0.4f} seconds".format(time.perf_counter() - start))
                return {'prediction': prediction, 'proba_default': client_default_proba}
            except BaseException as e:
                print('Error while predicting client default proba: ' + str(e))
                my_logger.error('Error while predicting client default proba: ' + str(e))
                return {'prediction': "error", 'proba_default': 0}

        @app.get('/predict_default_all_clients')
        async def predict_default_all(client_ids: list[int] = Query(...)):
            """
            Renvoie, pour une liste de clients, une liste contenant la probabilité 
            de défaut de remboursement de chacun.

            Positional arguments : 
            -------------------------------------
            client_ids : list of integers : liste d'identifiants clients (SK_ID_CURR)
            """
            start = time.perf_counter()
            try:
                client_info = self.client_database.loc[client_ids]
                client_proba = self.classifier.predict_proba(client_info)
                client_default_proba = client_proba[:, 1]
                
                print("predict_default_all in {:0.4f} seconds".format(time.perf_counter() - start))
                return {'proba_default': client_default_proba.tolist()}
            except BaseException as e:
                print('Error while trying to predict dafault proba for all clients: ' + str(e))
                my_logger.error('Error while trying to predict dafault proba for all clients: ' + str(e))
                return {'proba_default': []}

        @app.get('/numeric_features_list')
        async def get_numeric_features():
            """
            Renvoie la liste des variables numériques utilisées pour la classification.            
            """
            start = time.perf_counter()
            try:
                numeric_features = self.classifier[0].named_transformers_['pipeline-3'][0].feature_names_in_
                print("get_numeric_features in {:0.4f} seconds".format(time.perf_counter() - start))
                return {'numeric_features': numeric_features.tolist()}
            except BaseException as e:
                my_logger.error('Error while trying to retrieve numeric features as a list: ' + str(e))
                print('Error while trying to retrieve numeric features as a list: ' + str(e))
                return {'numeric_features': []}

        @app.get('/shap_values_default')
        async def get_shap_values(client_id: int):
            """
            Renvoie, pour un client donnée, la liste des shap_values, 
            la valeur attendue ("expected_value") ainsi que 
            la liste des variables du modèle.
            
            Positional arguments : 
            -------------------------------------
            client_id : int : identifiant client (SK_ID_CURR)
            """
            start = time.perf_counter()
            try:
                client_info = self.client_database.loc[[client_id]]
                explainer = shap.TreeExplainer(self.classifier[-1], model_outpout='predict_proba')

                binary_features = self.classifier[0].named_transformers_['pipeline-1'][0].feature_names_in_
                dummies_features = self.classifier[0].named_transformers_['pipeline-2'][0].get_feature_names()
                numeric_features = self.classifier[0].named_transformers_['pipeline-3'][0].feature_names_in_

                features = [*binary_features, *dummies_features, *numeric_features]

                client_info_transformed = self.classifier[0].transform(client_info)
                shap_vals = explainer.shap_values(client_info_transformed)
                expected_values = explainer.expected_value

                print("get_shap_values in {:0.4f} seconds".format(time.perf_counter() - start))
                return {"shap_values": json.dumps(shap_vals[1].tolist()),
                        "expected_values": expected_values[1],
                        "features": features}
                        
            except BaseException as e:
                print('Error while trying to compute shap values: ' + str(e))
                my_logger.error('Error while trying to compute shap values: ' + str(e))
                return {"shap_values": "[[]]",
                        "expected_values": 0,
                        "features": []}

        @app.get('/client_ids')
        async def get_client_ids():
            """
            Renvoie la liste de tous les identifiants clients présents 
            dans le jeu de données chargé.
            """
            start = time.perf_counter()
            try:
                ids_list = self.client_database.index.to_list()
                print("get_client_ids in {:0.4f} seconds".format(time.perf_counter() - start))
                return {"ids": ids_list}
            except BaseException as e:
                print('Error while trying to retrieve client ids list ' + str(e))
                return {"ids": []}

        @app.get('/nearest_neighbors_ids')
        async def get_nearest_neighbors_ids(client_id: int, n_neighbors: int):
            """
            Renvoie, pour un client donnée, la liste des identifiants 
            des n clients les plus proches.

            Positional arguments : 
            -------------------------------------
            client_id : int : identifiant du client de référence (SK_ID_CURR)
            n_neighbors : int : nombre de plus proches voisins à retourner
            """
            start = time.perf_counter()
            try:
                transformed_data = self.nearest_neighbors[0].transform(self.client_database.loc[[client_id]])
                indices = self.nearest_neighbors[1].kneighbors(transformed_data, n_neighbors=n_neighbors,
                                                               return_distance=False)
                neighbors_ids = self.client_database.iloc[indices[0].tolist()].index.tolist()
                
                print("get_nearest_neighbors_ids in {:0.4f} seconds".format(time.perf_counter() - start))
                return {"nearest_neighbors_ids": neighbors_ids}

            except BaseException as e:
                print('Error while trying to find nearest_neighbors: ' + str(e))
                my_logger.error('Error while trying to find nearest_neighbors: ' + str(e))
                return {"nearest_neighbors_ids": []}

        return app


app_fastapi = App(title='DefaultRiskApp',
                  description="""API qui alimente le tableau de bord client, construit dans le cadre du 
                  projet 7 du parcours Datascientist proposé par OpenClassrooms""",
                  version="0.1").create_app()


if __name__ == '__main__':
    uvicorn.run(app_fastapi, host='0.0.0.0', port=80)
# uvicorn main:app_fastapi --reload
