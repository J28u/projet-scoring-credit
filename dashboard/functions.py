import os
from datetime import datetime
from warnings import filterwarnings
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import more_itertools as mit
import time

filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap

URL_API = os.environ.get('URL_API')
DATABASE_PATH = os.environ.get('DATABASE_PATH')


def store_request(response, response_time: datetime, params: str, endpoint: str, result=np.nan):
    """
    Enregistre des informations sur les requêtes faites à l'API 'DefaultRiskApp' 
    (heure de requête, paramètres, nom du endpoint, status code, résultat)
            
    Positional arguments : 
    -------------------------------------
    response : : réponse renvoyée par la requête
    response_time : dt.datetime : heure à laquelle la requête a renvoyée une réponse
    params : str : paramètres envoyés dans la requête
    endpoint : str : nom du endpoint requêté
    
    Optional arguments : 
    -------------------------------------
    results : float : résultat de la requête
    
    Returns : 
    -------------------------------------
    None
    """
    request_log = pd.DataFrame([{'time': response_time, 'params': params, 'endpoint': endpoint,
                                'status': response.status_code, 'result': result}])
    st.session_state.requests_history = pd.concat([st.session_state.requests_history, request_log])


@st.cache_data
def get_all_client_ids():
    """
    Renvoie la liste de tous les identifiants clients répertoriés
    (à partir de la réponse renvoyée par l'API 'DefaultRiskApp').
            
    Returns : 
    -------------------------------------
    list of integers
    """
    start = time.perf_counter()
    response = requests.get(URL_API + "client_ids")
    store_request(response, datetime.now(), "no params", "GET_client_ids")
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    print("get_all_client_ids in {:0.4f} seconds".format(time.perf_counter() - start))
    return response.json()['ids']


@st.cache_data
def get_numeric_features():
    """
    Renvoie la liste des variables numériques utilisées par le modèle de classification
    (à partir de la réponse envoyée par l'API 'DefaultRiskApp').
            
    Returns : 
    -------------------------------------
    list of str
    """
    start = time.perf_counter()
    response = requests.get(URL_API + 'numeric_features_list')
    store_request(response, datetime.now(), "no params", "GET_numeric_features_list")
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    print("get_numeric_features in {:0.4f} seconds".format(time.perf_counter() - start))
    return response.json()['numeric_features']


@st.cache_data
def get_all_client_info():
    """
    Renvoie les informations descriptives relatives à tous les clients répertoriés
    (à partir de la réponse envoyée par l'API 'DefaultRiskApp').
            
    Returns : 
    -------------------------------------
    client_dataset : pd.DataFrame : informations descriptives relatives aux clients
    """
    start = time.perf_counter()
    response = requests.get(URL_API + "download_database")
    
    if response.status_code != 200:
        store_request(response, datetime.now(), "no params", "GET_download_database")
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
            
    if response.headers['content-type'] == 'application/json':
        store_request(response, datetime.now(), "no params", "GET_download_database", response.json()['message'])
        client_dataset = pd.DataFrame()
        
    else:
        open('database.pkl', "wb").write(response.content)
        client_dataset = pd.read_pickle('database.pkl')

    print("get_all_client_info in {:0.4f} seconds".format(time.perf_counter() - start))
    return client_dataset


@st.cache_data
def get_default_threshold():
    """
    Renvoie la liste de tous les identifiants clients répertoriés
    (à partir de la réponse envoyée par l'API 'DefaultRiskApp').
            
    Returns : 
    -------------------------------------
    list of integers
    """
    start = time.perf_counter()
    response = requests.get(URL_API + "threshold")
    if response.status_code != 200:
        store_request(response, datetime.now(), "no params", "GET_threshold")
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    thresh = response.json()['threshold']
    store_request(response, datetime.now(), "no params", "GET_threshold", thresh)
    
    print("get_default_threshold in {:0.4f} seconds".format(time.perf_counter() - start))
    return thresh


@st.cache_data
def get_nearest_neighbors_ids(client_id: int, n_neighbors: int):
    """
    Renvoie, pour un client donné, la liste des identifiants de ses n 
    plus proches voisins(à partir de la réponse renvoyée par l'API 'DefaultRiskApp').
    
    Positional arguments : 
    -------------------------------------
    client_id : int : identifiant (SK_ID_CURR) du client de référence
    n_neighbors : int : nombre de plus proches voisins
            
    Returns : 
    -------------------------------------
    list of integers
    """
    start = time.perf_counter()
    params = {'client_id': client_id, 'n_neighbors': n_neighbors}
    response = requests.get(URL_API + 'nearest_neighbors_ids', params=params)
    store_request(response, datetime.now(), str(params), "GET_nearest_neighbors_ids")

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    print("get_nearest_neighbors_ids in {:0.4f} seconds".format(time.perf_counter() - start))
    return response.json()["nearest_neighbors_ids"]


@st.cache_data
def get_client_default_proba_one(client_id: int):
    """
    Renvoie la probabilité de défaut de remboursement d'un client ainsi qu'un message
    indiquant si la demande de crédit a été acceptée ou non
    (à partir de la réponse renvoyée par l'API 'DefaultRiskApp').
    
    Positional arguments : 
    -------------------------------------
    client_id : int : identifiant du client(SK_ID_CURR) 
            
    Returns : 
    -------------------------------------
    dict
    """
    start = time.perf_counter()
    params = {'client_id': client_id}
    response = requests.get(URL_API + "predict_default_one", params=params)

    if response.status_code != 200:
        store_request(response, datetime.now(), str(params), "GET_predict_default_one")
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    store_request(response, datetime.now(), str(params), "GET_predict_default_one", response.json()['proba_default'])
    
    print("get_client_default_proba in {:0.4f} seconds".format(time.perf_counter() - start))
    return response.json()


@st.cache_data
def get_clients_default_proba_many(client_ids: list[int]):
    """
    Renvoie, pour une liste de clients, une liste contenant la probabilité de défaut 
    de remboursement de chacun(à partir de la réponse renvoyée par l'API 'DefaultRiskApp').
    
    Positional arguments : 
    -------------------------------------
    client_ids : list of integers : liste d'identifiants client(SK_ID_CURR) 
            
    Returns : 
    -------------------------------------
    results : list of floats : liste de probabilités de défault
    """
    start = time.perf_counter()
    results = []
    chunks = list(mit.chunked(client_ids, 1_000))
    for i, chunk in enumerate(chunks):
        params = {'client_ids': chunk}
        response = requests.get(URL_API + 'predict_default_many', params=params)
        store_request(response, datetime.now(), str({"client_id": "chunk " + str(i+1) + "/" + str(len(chunks))}),
                      "GET_predict_default_many")

        if response.status_code != 200:
            raise Exception(
                "Request failed with status {}, {}".format(response.status_code, response.text))
        results.extend(response.json()['proba_default'])
    
    print("get_clients_default_proba_many in {:0.4f} seconds".format(time.perf_counter() - start))
    return results
    

@st.cache_data
def get_client_info(client_id: int):
    """
    Renvoie les informations descriptives relatives à un client donné 
    (à partir de la réponse renvoyée par l'API 'DefaultRiskApp').
    
    Positional arguments : 
    -------------------------------------
    client_id : int : identifiant client(SK_ID_CURR) 
            
    Returns : 
    -------------------------------------
    dict
    """
    start = time.perf_counter()
    params = {"client_id": client_id}
    response = requests.get(URL_API + "client_info", params=params)
    store_request(response, datetime.now(), str(params), "GET_client_info")
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    print("get_client_info in {:0.4f} seconds".format(time.perf_counter() - start))
    return response.json()[0]


@st.cache_data
def check_client_in_database(client_id):
    """
    Renvoie True si un client est répertorié, False sinon
    (à partir de la réponse renvoyée par l'API 'DefaultRiskApp').
    
    Positional arguments : 
    -------------------------------------
    client_id : int : identifiant client(SK_ID_CURR) 
            
    Returns : 
    -------------------------------------
    bool
    """
    start = time.perf_counter()
    params = {"client_id": client_id}
    response = requests.get(URL_API + "in_database", params=params)
    if response.status_code != 200:
        store_request(response, datetime.now(), str(params), "GET_in_database")
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    store_request(response, datetime.now(), str(params), "GET_in_database", response.json()['check'])
    
    print("check_client_in_database in {:0.4f} seconds".format(time.perf_counter() - start))
    return response.json()['check']


@st.cache_data
def build_waterfall_plot(client_id: int):
    """
    Renvoie un graphique "waterfall" construit avec la librairie shap 
    (à partir de la réponse renvoyée par l'API 'DefaultRiskApp').
    
    Positional arguments : 
    -------------------------------------
    client_id : int : identifiant client(SK_ID_CURR) 
            
    Returns : 
    -------------------------------------
    fig
    """
    start = time.perf_counter()
    params = {"client_id": client_id}
    response = requests.get(URL_API + "shap_values_default", params=params)
    store_request(response, datetime.now(), str(params), "GET_shap_values_default")

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    sns.set_theme(style='white')
    shap_values = np.asarray(json.loads(response.json()['shap_values']))
    shap_vals_explanation = shap.Explanation(shap_values,
                                             base_values=response.json()['expected_values'],
                                             feature_names=response.json()['features'])

    fig, ax = plt.subplots()
    ax = shap.plots.waterfall(shap_vals_explanation[0])
    plt.grid(False, axis='x')
    
    print("build_waterfall_plot in {:0.4f} seconds".format(time.perf_counter() - start))
    return fig


def build_donut(dataset: pd.DataFrame, categ_var: str, text_color='#595959',
                colors='colorblind', labeldistance=1.1):
    """
    Renvoie un graphique en forme de donut représentant la répartition d'une variable qualitative.
    
    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    categ_var : str : nom de la colonne contenant les valeurs de la variable qualitative
    
    Optionnal arguments : 
    -------------------------------------
    text_color : str : couleur du texte
    colors : str : palette de couleurs seaborn utilisée pour colorer le donut
    labeldistance : float : distance à laquelle placer les labels
            
    Returns : 
    -------------------------------------
    fig
    """
    start = time.perf_counter()            
    with plt.style.context('seaborn-white'):
        sns.set_theme(style='white')
        fig, ax = plt.subplots()
        plt.rcParams.update(
            {'axes.labelcolor': text_color, 'axes.titlecolor': text_color, 'legend.labelcolor': text_color,
             'axes.titlesize': 16, 'axes.labelpad': 10})

    pie_series = dataset[categ_var].value_counts(sort=False, normalize=True)
    patches, texts, autotexts = ax.pie(pie_series, labels=pie_series.index, autopct='%.0f%%', pctdistance=0.85,
                                       colors=sns.color_palette(colors), labeldistance=labeldistance,
                                       textprops={'fontsize': 20, 'color': '#595959', 'fontname': 'Open Sans'},
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)

    centre_circle = plt.Circle((0, 0), 0.7, fc='white')

    ax.axis('equal')
    ax.add_artist(centre_circle)
    plt.tight_layout()

    print("build_donut in {:0.4f} seconds".format(time.perf_counter() - start))
    return fig


def build_gauge(color: str):
    """
    Renvoie un graphique en forme de gauge représentant la probabilité 
    de défaut de remboursement d'un client.
    
    Positional arguments : 
    -------------------------------------
    color : str : jeu de données
         
    Returns : 
    -------------------------------------
    fig
    """
    start = time.perf_counter() 
    fig = go.Figure(go.Indicator(mode="gauge+number",
                                 value=st.session_state.proba_default_int,
                                 domain={'x': [0, 1], 'y': [0, 1]},
                                 title={'text': 'Probabilité de défaut (%)', 'font': {'size': 14, 'color': "#262730"},
                                        'align': "left"},
                                 gauge={'axis': {'range': [None, 100],
                                                 'tickwidth': 2,
                                                 'tickcolor': "#262730",
                                                 'tickvals': [0, 25, 50, 75, 100]},
                                        'bar': {"color": color},
                                        'bgcolor': 'white',
                                        'borderwidth': 2,
                                        'bordercolor': "#262730",
                                        'threshold': {'value': st.session_state.thresh_int,
                                                      'thickness': 0.75,
                                                      'line': {'width': 2, 'color': "#262730"}},

                                        }
                                 )
                    )
    fig.update_layout(paper_bgcolor='rgba(28, 131, 225, 0.1)', height=175, margin={'b': 50, 't': 60, 'r': 20, 'l': 20},
                      font={"color": "#262730"})
    fig.add_annotation(x=0, y=-0.6,
                       text='seuil: ' + st.session_state.thresh,
                       showarrow=False,
                       font={'size': 16, 'color': color})
    print("build_gauge in {:0.4f} seconds".format(time.perf_counter() - start))
    return fig


def build_scatter_plot(dataset: pd.DataFrame, x_var: str, y_var: str, colors='temps', with_hue=False):
    """
    Renvoie une figure graphique nuage de point pour comparer un client 
    à un groupe de clients.
    
    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    x_var : str : nom de la variable quantitative à mettre en abscisse
    y_var : str : nom de la variable quantitative à mettre en ordonnée
    
    Optionnal arguments : 
    -------------------------------------
    colors : str : palette de couleurs seaborn à utiliser pour colorer 
    les points selon la probabilité de défaut de chaque client
    with_hue : bool : colorer ou non les points en fonction de la proba de défaut
            
    Returns : 
    -------------------------------------
    fig
    """
    start = time.perf_counter() 
    dataset['CLIENT_ID'] = dataset.index.tolist()
    dataset['CLIENT_TAG'] = 'other clients (n=' + str(dataset.shape[0]-1) + ')'
    dataset.loc[st.session_state.selected_client, 'CLIENT_TAG'] = 'selected client'

    if with_hue:
        fig = px.scatter(dataset, x=x_var, y=y_var, color='DEFAULT_PROBA', symbol='CLIENT_TAG', opacity=.9,
                         color_continuous_scale=colors,
                         hover_data=['CLIENT_TAG', 'CLIENT_ID', 'DEFAULT_PROBA', x_var, y_var])
    else:
        fig = px.scatter(dataset, x=x_var, y=y_var, symbol='CLIENT_TAG', opacity=.9,
                         hover_data=['CLIENT_TAG', 'CLIENT_ID', x_var, y_var])

    fig.update_traces(marker=dict(size=20, line=dict(color='DarkSlateGrey', width=2), opacity=1),
                      selector=({"name": 'selected client'}))
    fig.data = (fig.data[1], fig.data[0])
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", xanchor="center", y=1, x=.5), plot_bgcolor='white')
    fig.update_xaxes(gridcolor='lightgrey', linecolor='lightgrey', linewidth=2, showline=True, showgrid=False)
    fig.update_yaxes(gridcolor='lightgrey', linecolor='lightgrey', linewidth=2, showline=True, showgrid=False)
    
    print("build_scatter_plot in {:0.4f} seconds".format(time.perf_counter() - start))
    return fig


def build_hist(dataset: pd.DataFrame, x_var: str, labels: dict, client_id: int, hue_var=None, palette='colorblind'):
    """
    Renvoie un histogramme
    
    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    x_var : str : nom de la variable quantitative à mettre en abscisse
    labels : dict : dictionnaire contenant les titres des axis {'x' : blabla, 'y': bloblo}
    client_id : int : identifiant du client sélectionné
    
    Optionnal arguments : 
    -------------------------------------
    hue_var : str : nom de la variable utilisée pour colorer le graphique
    palette : str : palette de couleurs seaborn à utiliser
            
    Returns : 
    -------------------------------------
    fig
    """
    start = time.perf_counter()
    rgb_text = sns.color_palette('Greys', 15).as_hex()[12]
    fig = px.histogram(dataset, x=x_var, color=hue_var, nbins=20, labels=labels, opacity=.8, 
                       range_x=[20, 70], histnorm='percent', 
                       color_discrete_sequence=sns.color_palette(palette).as_hex()[:2],
                       template="plotly_white")
    
    fig.update_layout(bargap=0.05, font_family="Corbel", font_color=rgb_text, yaxis_title=labels['y'])
    fig.update_yaxes(showgrid=True)
    fig.add_vline(x=dataset.loc[[client_id], x_var].values[0], type='line',
                  line_dash = 'dash', line_color = 'darkmagenta', line_width=3)
    fig.add_annotation(text="selected client", x=dataset.loc[[client_id], x_var].values[0] - 3, y=32, showarrow=False)
    
    print("build_hist in {:0.4f} seconds".format(time.perf_counter() - start))
    return fig


def format_amount(amount: float):
    """
    Renvoie un montant arrondi, converti en chaine de caractère, avec une unité de mesure. 
    
    Positional arguments : 
    -------------------------------------
    amount : float : montant à convertir
           
    Returns : 
    -------------------------------------
    formatted_amount : str : montant arrondi et converti en chaîne de caractères
    """
    if amount >= 1_000_000:
        formatted_amount = str(round(int(amount) / 1_000_000, 1)) + "M"
    elif amount >= 1_000:
        formatted_amount = str(round(int(amount) / 1_000, 1)) + "k"
    else:
        formatted_amount = str(int(round(amount, 0)))

    return formatted_amount


def load_graphs():
    """
    Enregistre les figures graphiques (histogramme du nombre de client par tranche d'âge et 
    donut de la répartition des genres)

    Returns : 
    -------------------------------------
    None 
    """
    start = time.perf_counter()
    dataset = st.session_state.dataset_original
    dataset['AGE'] = dataset['DAYS_BIRTH'].abs() // 365.25
    st.session_state.age_mean = str(np.abs(dataset['AGE'].mean()).astype(int)) + " ans"
    st.session_state.income_mean = format_amount(dataset['AMT_INCOME_TOTAL'].mean())
    st.session_state.donut_gender = build_donut(dataset, 'CODE_GENDER')

    labels_age = {"AGE": "Tranches d'âge", "y": "Clients par tranche d'âge (%)"}
    st.session_state.hist_age = build_hist(dataset, 'AGE', labels_age, 
                                           st.session_state.selected_client, 'CODE_GENDER')
    print("load_graphs in {:0.4f} seconds".format(time.perf_counter() - start))


def filter_dataset():
    """
    Applique les filtres au jeu de données client
           
    Returns : 
    -------------------------------------
    None 
    """
    start = time.perf_counter()
    dataset = st.session_state.dataset_original
    if not dataset.empty:
        dataset['AGE'] = dataset['DAYS_BIRTH'].abs() // 365.25

        nearest_neighbors_ids = get_nearest_neighbors_ids(st.session_state.selected_client,
                                                          st.session_state.n_neighbors + 1)
        filtered_dataset = dataset.loc[nearest_neighbors_ids]

        if st.session_state.m and st.session_state.f:
            st.session_state.filter_gender = ['F', 'M']
        elif st.session_state.m:
            st.session_state.filter_gender = ['M']
        elif st.session_state.f:
            st.session_state.filter_gender = ['F']
        else:
            st.session_state.filter_gender = []

        mask_gender = (filtered_dataset['CODE_GENDER'].isin(st.session_state.filter_gender))
        mask_age_min = (filtered_dataset['AGE'] >= st.session_state.age_slider[0])
        mask_age_max = (filtered_dataset['AGE'] <= st.session_state.age_slider[1])

        filtered_dataset = filtered_dataset.loc[mask_gender & mask_age_min & mask_age_max]
        st.session_state.dataset = filtered_dataset
        reload_scatter_plot()
    print("filter_dataset in {:0.4f} seconds".format(time.perf_counter() - start))


def reset_filter():
    """
    Enlève les filtres appliqués au jeu de données client
    
    Returns : 
    -------------------------------------
    None 
    """
    st.session_state.dataset = st.session_state.dataset_original
    st.session_state['f'] = True
    st.session_state['m'] = True
    st.session_state['age_slider'] = (10, 100)
    st.session_state['neighbors_input'] = st.session_state.number_of_clients - 1

    if not st.session_state.dataset.empty:
        reload_scatter_plot()


def reload_scatter_plot():
    """
    Construit et enregistre le graphique type nuage de points utilisé
    pour comparer les informations d'un client à celles d'un groupe de clients.
    
    Returns : 
    -------------------------------------
    None
    """
    with_hue = False
    if st.session_state.dataset.shape[0] <= 5_000:
        ids = st.session_state.dataset.index.to_list()
        st.session_state.dataset['DEFAULT_PROBA'] = get_clients_default_proba_many(ids)
        with_hue = True
    st.session_state.scatter = build_scatter_plot(st.session_state.dataset,
                                                  st.session_state.x_var,
                                                  st.session_state.y_var,
                                                  with_hue=with_hue)


def initialize_dashboard():
    """
    Initialize le dashboard, i.e. récupère et enregistre tous les ids clients répertoriés, 
    la liste des variables numériques du modèle, 
    le seuil optimisé utilisé pour classifier les clients,
    et les informations descriptives de tous les clients.
  
    Returns : 
    -------------------------------------
    None
    """
    start = time.perf_counter()
    st.session_state.requests_history = pd.DataFrame(columns=['time', 'params', 'endpoint', 'status', 'result'])
    st.session_state.ids = get_all_client_ids()
    st.session_state.numeric_features = get_numeric_features()
    st.session_state.thresh_int = np.round(get_default_threshold() * 100).astype(int)
    st.session_state.thresh = str(st.session_state.thresh_int) + "%"
    st.session_state.dataset_original = pd.read_pickle(DATABASE_PATH + 'database.pkl')
    st.session_state.dataset = st.session_state.dataset_original
    st.session_state.number_of_clients = len(st.session_state.ids)
    print("initialize_dashboard in {:0.4f} seconds".format(time.perf_counter() - start))


def load_client_info(client_id: int):
    """
    Récupère et enregistre la probabilité prédite qu'un client donné ne rembourse 
    pas son crédit ainsi que ses informations descriptives.
    
    Positional arguments : 
    -------------------------------------
    client_id : int : identifiant client (SK_ID_CURR)
  
    Returns : 
    -------------------------------------
    None
    """
    start = time.perf_counter()
    if check_client_in_database(client_id):
        data = get_client_default_proba_one(client_id)
        info = get_client_info(client_id)

        st.session_state.selected_client = client_id
        st.session_state.prediction = data['prediction']
        st.session_state.proba_default = str(np.round(data['proba_default'] * 100).astype(int)) + "%"
        st.session_state.proba_default_int = np.round(data['proba_default'] * 100).astype(int)
        st.session_state.waterfall_plot = build_waterfall_plot(client_id)
        st.session_state.amt_credit = format_amount(info['AMT_CREDIT'])
        st.session_state.amt_annuity = format_amount(info['AMT_ANNUITY'])
        st.session_state.gender = info['CODE_GENDER']
        st.session_state.contract_type = info['NAME_CONTRACT_TYPE']
        st.session_state.income = format_amount(info['AMT_INCOME_TOTAL'])
        st.session_state.age = str(np.abs(info['DAYS_BIRTH'] // 365.25).astype(int)) + " ans"
        st.session_state.kids = str(info['CNT_CHILDREN'])
        st.session_state.income_type = info['NAME_INCOME_TYPE']
        st.session_state.occupation = info['OCCUPATION_TYPE']
        st.session_state.housing = info['NAME_HOUSING_TYPE']
        st.session_state.family = info['NAME_FAMILY_STATUS']
        st.session_state.education = info['NAME_EDUCATION_TYPE']

        if not st.session_state.dataset_original.empty:
            load_graphs()
        st.session_state.missing_id = False

    else:
        st.session_state.missing_id = True
        
    print("load_client_info in {:0.4f} seconds".format(time.perf_counter() - start))
    
