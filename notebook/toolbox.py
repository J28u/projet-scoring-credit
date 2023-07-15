import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import ssl
import time
import gc
from skimage import io
from math import ceil
from functools import partial
from hyperopt import fmin, hp, tpe, Trials
from enum import Enum
from contextlib import contextmanager

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.metrics import roc_auc_score, make_scorer, confusion_matrix, accuracy_score, recall_score, precision_score, balanced_accuracy_score
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import TransformerMixin

from lightgbm import LGBMClassifier

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging


class DataFile(Enum):
    """
    Classe qui représente le type de pré-traitement à appliquer à un jeu de données selon son contenu.
    """
    BUREAU = 0
    PREV = 1
    POS = 2
    INS = 3
    CC = 4
    
    
class GetDummiesTransformer(TransformerMixin):
    """
    Classe qui représente un transformer utilisant la methode get_dummies de pandas 
    pour encoder les variables catégorielles. 
    
    Attributes
    -------------------------------------
    input_columns : list of str : liste des variables catégorielles à encoder
    nan_as_category : bool : considérer ou non les nans comme une catégorie
    final_columns : list of str : liste des variables après transformation

    Methods
    -------------------------------------
    fit : entraîne le transformer sur le jeu d'entrainement
    transform : applique la transformation
    get_feature_names : renvoie les noms des variables après transformation
    """
    def __init__(self, input_columns=None, nan_as_category=False):
        """
        Construit tous les attributs nécessaires pour une instance de la classe GetDummiesTransformer
        
        Optional Arguments
        -------------------------------------
        input_columns : list of str : liste des variables catégorielles à encoder
        nan_as_category : bool : considérer ou non les nans comme une catégorie
        
        Returns
        -------------------------------------
        None
        """
        self.input_columns = input_columns
        self.nan_as_category = nan_as_category
        self.final_columns = None
        
    def fit(self, X, y=None, **kwargs):
        """
        Entraîne une instance de la classe GetDummiesTransformer sur un jeu d'entrainement
        
        Positional Arguments
        -------------------------------------
        X : np.array : jeu d'entrainement
        
        Optional Arguments
        -------------------------------------
        y : np.array : cibles à prédire
        
        Returns
        -------------------------------------
        self : GetDummiesTransformer : instance entrainée 
        """
        X = pd.get_dummies(X, columns=self.input_columns, dummy_na= self.nan_as_category)
        self.final_columns = X.columns
        return self
    
    def transform(self, X, y=None, **kwargs):
        """
        Transforme les variables catégorielles d'un jeu de données en variables numériques. 
        En créant une colonne par catégorie contenant des 0 et des 1 : 1 si l'individu appartient à la catégorie, 0 sinon.
        
        Positional Arguments
        -------------------------------------
        X : np.array : jeu de données
        
        Optional Arguments
        -------------------------------------
        y : np.array : cibles à prédire
        
        Returns
        -------------------------------------
        X : pd.DataFrame : jeu de données transformé
        """
        X = pd.get_dummies(X, columns=self.input_columns, dummy_na= self.nan_as_category)
        X_columns = X.columns
        missing = set(self.final_columns) - set (X_columns)
        for c in missing:
            X[c]=0
            
        return X[self.final_columns]
    
    def get_feature_names(self):
        """
        Renvoie les noms des variables après transformation.
        
        Returns
        -------------------------------------
        tuple
        """
        return tuple(self.final_columns)

    
def display_image_from_url(url: str, title: str, fig_size: tuple):
    """
    Affiche une image à partir de son url

    Positional arguments : 
    -------------------------------------
    url : str : url de l'image à afficher 
    title : str : titre à afficher au dessus de l'image
    figsize : tuple : taille de la zone d'affichage de l'image (largeur, hauteur)
    
    Returns
    -------------------------------------
    None
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    img = io.imread(url)
    plt.figure(figsize=fig_size)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=20, fontname='Corbel', pad=20)
    plt.imshow(img)

    plt.show()
    

def plot_donut(dataset: pd.DataFrame, categ_var: str, title: str, figsize: tuple, text_color='#595959',
               colors={'outside': sns.color_palette('Set2')}, nested=False, sub_categ_var=None, labeldistance=1.1,
               textprops={'fontsize': 20, 'color': '#595959', 'fontname': 'Open Sans'}):
    """
    Affiche un donut de la répartition d'une variable qualitative

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à afficher
    categ_var : str : nom de la colonne contenant les valeurs de la variable qualitative
    
    palette : strings : nom de la palette seaborn à utiliser
    title : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optionnal arguments : 
    -------------------------------------
    text_color : str : couleur du texte
    colors : dict : couleurs du donut extérieur et couleurs du donut intérieur
    nested : bool : créer un double donut ou non
    sub_categ_var : str : nom de la colonne contenant les catégories à afficher dans le donut intérieur
    labeldistance : float : distance à laquelle placer les labels du donut extérieur
    textprops : dict : personnaliser les labels du donut extérieur (position, couleur ...)
    
    Returns
    -------------------------------------
    None
    """
    with plt.style.context('seaborn-white'):
        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, fontname='Corbel', fontsize=30)
        plt.rcParams.update(
            {'axes.labelcolor': text_color, 'axes.titlecolor': text_color, 'legend.labelcolor': text_color,
             'axes.titlesize': 16, 'axes.labelpad': 10})

    pie_series = dataset[categ_var].value_counts(sort=False, normalize=True)
    patches, texts, autotexts = ax.pie(pie_series, labels=pie_series.index, autopct='%.0f%%', pctdistance=0.85,
                                       colors=colors['outside'], labeldistance=labeldistance,
                                       textprops=textprops,
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)

    centre_circle = plt.Circle((0, 0), 0.7, fc='white')

    if nested:
        inside_pie_series = dataset[sub_categ_var].value_counts(sort=False, normalize=True)
        patches_sub, texts_sub, autotexts_sub = ax.pie(inside_pie_series, autopct='%.0f%%', pctdistance=0.75,
                                                       colors=colors['inside'], radius=0.7,
                                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})

        for autotext in autotexts_sub:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)

        plt.legend(patches_sub, inside_pie_series.index, title=sub_categ_var, fontsize=14, title_fontsize=16, loc=0)
        centre_circle = plt.Circle((0, 0), 0.4, fc='white')

    ax.axis('equal')
    ax.add_artist(centre_circle)

    plt.tight_layout()
    plt.show()


def compare_donuts(dataset_before: pd.DataFrame, dataset_after: pd.DataFrame, categ_var: str, title: str, 
                  figsize: tuple, text_color='#595959', colors=sns.color_palette('Set2'), labeldistance=1.1,
                  textprops={'fontsize': 20, 'color': '#595959', 'fontname': 'Open Sans'}, top=0.9, wspace=0.1, hspace=0.7
                  ):
    """
    Affiche un donut de la répartition d'une variable qualitative avant et après transformation

    Positional arguments : 
    -------------------------------------
    dataset_before : pd.DataFrame : jeu de données original
    dataset_after : pd.DataFrame : jeu de données transformé
    categ_var : str : nom de la colonne contenant les valeurs de la variable qualitative
    title : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optionnal arguments : 
    -------------------------------------
    text_color : str : couleur du texte
    colors : str : couleurs des donuts
    labeldistance : float : distance à laquelle placer les labels
    textprops : dict : personnaliser les labels (position, couleur ...)
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    
    Returns
    -------------------------------------
    None
    """
    with plt.style.context('seaborn-white'):
        sns.set_theme(style='whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.tight_layout()
        fig.suptitle(title, fontname='Corbel', fontsize=30)
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=top, wspace=wspace, hspace=hspace)
    
    data = [{'title': 'avant ({:_} individus)', 'ax': 0, 'dataset': dataset_before}, 
            {'title': 'après ({:_} individus)', 'ax': 1, 'dataset': dataset_after}]
    
    for d in data :
        
        pie_series = d['dataset'][categ_var].value_counts(sort=False, normalize=True)
        patches, texts, autotexts = axes[d['ax']].pie(pie_series, labels=pie_series.index, autopct='%.0f%%', 
                                                      pctdistance=0.85, startangle=45,
                                                      colors=colors, labeldistance=labeldistance,
                                                      textprops=textprops,
                                                      wedgeprops={'edgecolor': 'white', 'linewidth': 2})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(16)

        centre_circle = plt.Circle((0, 0), 0.7, fc='white')
        
        axes[d['ax']].set_title(d['title'].format(d['dataset'].shape[0]), fontname='Corbel', fontsize=20, pad=10)
        axes[d['ax']].axis('equal')
        axes[d['ax']].add_artist(centre_circle)

    plt.show()

    
def display_roc_curve(fpr: np.array, tpr: np.array, figsize=(10, 6), palette='muted'):
    """
    Affiche la courbe ROC

    Positional arguments : 
    -------------------------------------
    fpr : np.array : taux de faux positifs
    tpr : np.array : taux de vrais positifs
    
    Optionnal arguments : 
    -------------------------------------
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    palette : str : couleur de la courbe

    Returns
    -------------------------------------
    None
    """
    
    with plt.style.context('seaborn-white'):
        sns.set_theme(style='whitegrid', palette=palette)
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title('Courbe ROC', fontsize=20, fontname='Corbel')
        plt.xlabel('1-Specificité', fontsize=14, fontname='Corbel')
        plt.ylabel('Sensibilité', fontsize=14, fontname='Corbel')
        plt.show()

    
def missing_values_by_column(dataset: pd.DataFrame):
    """
    Retourne un dataframe avec le nombre et le pourcentage de valeurs manquantes par colonnes

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les colonnes dont on veut connaitre le pourcentage de vide
    
    Returns
    -------------------------------------
    missing_values_df : pd.DataFrame : nombre et pourcentage de valeurs manquantes par colonnes
    """ 
    missing_values_series = dataset.isnull().sum()
    missing_values_df = missing_values_series.to_frame(name='Number of Missing Values')
    missing_values_df = missing_values_df.reset_index().rename(columns={'index': 'VARIABLES'})

    missing_values_df['Missing Values (%)'] = round(
        missing_values_df['Number of Missing Values'] / (dataset.shape[0]) * 100, 2)

    missing_values_df = missing_values_df.sort_values('Number of Missing Values')

    return missing_values_df


def mode(row: pd.Series):
    """
    Renvoie le mode de la colonne

    Positional arguments :
    -------------------------------------
    row : pd.Series : colonne dont on souhaite connaitre le mode
    
    Returns
    -------------------------------------
    mode : float ou int ou str : mode de la colonne
    """
    mode = pd.Series.mode(row)
    if len(mode) == 0:
        mode = np.nan
    elif mode.dtype == np.ndarray:
        mode = mode[0]
        
    return mode


def display_barplot(dataset: pd.DataFrame, x_column: str, y_column: str, titles: dict, figsize: tuple, hue=None, legend=False, palette=None):
    """
    Affiche un barplot

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    x_column : str : nom de la variable à mettre sur l'axe des abscisses
    y_column : str : nombre de la variable à mettre sur l'axe des ordonnées
    titles : dict : dictionnaire contenant les titres à afficher sur le graphique {"title": 'titre du graph', 'xlabel' : 'titre de l'axe des abscisses'}
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optional arguments : 
    -------------------------------------
    hue : str : nom de la colonne à utiliser pour colorer les barres
    legend : bool : si True,  affiche la légende
    palette : str : couleurs à utiliser
    
    Returns
    -------------------------------------
    None
    """ 
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=x_column, y=y_column, hue=hue, data=dataset, palette=palette)

    plt.title(titles['title'], size=25, fontname='Corbel', pad=40) 
    plt.ylabel(y_column, fontsize=20, fontname='Corbel')
    plt.xlabel(titles['xlabel'], fontsize=20, fontname='Corbel')
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    if legend:
        plt.legend(title=hue, fontsize=20, title_fontsize=20)
    
    plt.show()

    
def plot_feature_importance_tree_model(tree_models: [dict], features : [str], figsize: tuple, top_n=10, palette="Set2", 
                                       top=0.8, wspace=0.8, hspace=0.5):
    """
    Affiche les variables ayant la plus grande importance dans un ou plusieurs modèles ensemblistes utilisant des arbres de décision

    Positional arguments : 
    -------------------------------------
    tree_models : list of dictionnaries : liste des modèles à analyser
    features : list of strings : liste des variables
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optional arguments : 
    -------------------------------------
    top_n : int : nombre de variables à afficher
    palette : str : nom de la palette de couleur seaborn à utiliser
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    
    Returns
    -------------------------------------
    None
    """ 
    sns.set_theme(style='white')
    rgb_text = sns.color_palette('Greys', 15)[12]
    plt.figure(figsize=figsize)
    color_list =  sns.color_palette(palette, len(features))
    
    fig, axs = plt.subplots(1, len(tree_models), figsize=figsize, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = hspace, wspace=wspace, top=top)
    axs = axs.ravel()
    for i in range(len(tree_models)):
        feature_importance = tree_models[i]["model"][-1].feature_importances_
        indices = np.argsort(feature_importance)
        indices = indices[-top_n:]

        bars = axs[i].barh(range(len(indices)), feature_importance[indices], color='b', align='center') 
        axs[i].set_title(tree_models[i]["name"], fontsize=30, fontname='Corbel', color=rgb_text)
        axs[i].set_xlabel(tree_models[i]['x_label'], fontsize=20, fontname='Corbel', labelpad=20, color=rgb_text)

        plt.sca(axs[i])
        plt.yticks(range(len(indices)), [features[j] for j in indices], fontsize=14) 
        plt.tick_params(axis='both', which='major', labelsize=14)

        for i, ticklabel in enumerate(plt.gca().get_yticklabels()):
            ticklabel.set_color(color_list[indices[i]])  

        for i,bar in enumerate(bars):
            bar.set_color(color_list[indices[i]])
        plt.box(False)

    plt.suptitle("Top des {} variables les plus 'importantes'".format(str(top_n)), fontsize=40, fontname='Corbel', color=rgb_text)
    
    plt.show()
    
    
def build_trial_df(trials: Trials, loss: str):
    """
    Retourne un dataframe contenant des informations sur les itérations de l'optimisation réalisée avec hyperopt 
    (score, paramètres testés)

    Positional arguments : 
    -------------------------------------
    trials : hyperopt.Trials : objet Trials contenant les informations sur chaque itération 
    loss : str : score à minimiser lors de l'optimisation
    
    Returns : 
    -------------------------------------
    trials_df : pd.DataFrame : informations sur les itérations de l'optimisation
    """ 
    trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(lambda row: row[0]) for t in trials])
    trials_df[loss] = [t["result"]["loss"] for t in trials]
    trials_df["trial_number"] = trials_df.index
    
    return trials_df


def display_lineplot(dataset: pd.DataFrame, x_column: str, y_column: str, figsize: tuple, titles:dict, grid_x=True, palette='husl'):
    """
    Affiche un graphique représentant l'évolution du score à chaque itération

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : tableau contenant les scores et les itérations 
    x_column : str : nom de la colonne contenant les itérations
    y_column : str : nom de la colonne contenant les scores
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    titles : dict : titres du graphique
    
    Optional arguments : 
    -------------------------------------
    grid_x : bool : si True affiche la grille de l'axe des abscisses
    palette : str : nom de la palette seaborn à utiliser
    
    Returns : 
    -------------------------------------
    None
    """ 
    sns.set_theme(style='whitegrid', palette=palette)
    plt.figure(figsize=figsize)
    sns.lineplot(dataset, x=x_column, y=y_column)
    plt.title(titles['title'], fontname='Corbel', fontsize=30, pad=30)
    plt.ylabel(titles['ylabel'], fontname='Corbel', fontsize=20)
    plt.xlabel(titles['xlabel'], fontname='Corbel', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    if not grid_x :
        plt.grid(False, axis='x')
        
    plt.show()

    
@contextmanager
def timer(title: str):
    """
    Affiche le temps d'execution d'une tâche

    Positional arguments : 
    -------------------------------------
    title : str : tâche chronométrée
    
    Returns : 
    -------------------------------------
    None
    """
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

    
def preprocess_application_data(file_path: str, num_rows=None):
    """
    Nettoie un jeu de données contenant les demandes de crédits auprès de la société 'Prêt à dépenser' 
    et crée de nouvelles variables.

    Positional arguments : 
    -------------------------------------
    file_path : str : emplacement du fichier contenant le jeu de données

    Optional arguments : 
    -------------------------------------
    num_rows : int : taille des échantillons train et test (si None, garde les jeux de données entiers)
    
    Returns : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données nettoyé
    """

    dataset = pd.read_csv(file_path, nrows=num_rows)

    # Nettoyage :
    # Supprime les lignes pour lesquelles la variables CODE_GENDER n'est ni F ni M
    # Remplace variables DAYS_EMPLOYED = 365243 par nan

    dataset = dataset[dataset['CODE_GENDER'].isin(['F', 'M'])]
    dataset['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Feature engineering : ajout de pourcentages
    dataset['DAYS_EMPLOYED_PERC'] = dataset['DAYS_EMPLOYED'] / dataset['DAYS_BIRTH']
    dataset['INCOME_CREDIT_PERC'] = dataset['AMT_INCOME_TOTAL'] / dataset['AMT_CREDIT']
    dataset['INCOME_PER_PERSON'] = dataset['AMT_INCOME_TOTAL'] / dataset['CNT_FAM_MEMBERS']
    dataset['ANNUITY_INCOME_PERC'] = dataset['AMT_ANNUITY'] / dataset['AMT_INCOME_TOTAL']
    dataset['PAYMENT_RATE'] = dataset['AMT_ANNUITY'] / dataset['AMT_CREDIT']

    col_percentage = ['DAYS_EMPLOYED_PERC', 'INCOME_CREDIT_PERC',
                      'INCOME_PER_PERSON', 'ANNUITY_INCOME_PERC', 'PAYMENT_RATE']

    for col in col_percentage:
        dataset[np.isinf(dataset[col])] = 0

    return dataset


def preprocess_bureau_and_balance(bureau_file_path: str, balance_file_path: str, save=True, data_dir='data/'):
    """
    Agrège les lignes de bureau_balance.csv (données mensuelles) par identifiant de demande de prêt 'SK_ID_BUREAU', 
    les fusionne avec bureau.csv sur 'SK_ID_BUREAU',
    puis agrège les demandes de prêt par identifiant actuel du client ('SK_ID_CURR'), 
    ajoute les mêmes colonnes numériques filtrées sur les prêts actifs, 
    ajoute les mêmes colonnes numériques filtrées sur les prêts fermés.

    Positional arguments : 
    -------------------------------------
    bureau_file_path : str : emplacement du fichier contenant les demandes recensées par le bureau de crédit
    balance_file_path : str : emplacement du fichier contenant le détail des mensualités recensées par le bureau de crédit

    Optional arguments : 
    -------------------------------------
    save : bool : sauvegarder les données nettoyées sous format pickle ou non
    data_dir : str : nom du dossier dans lequel enregistrer le fichier pickle
    
    Returns : 
    -------------------------------------
    bureau_agg : pd.DataFrame : jeu de données nettoyé
    """
    bureau = pd.read_csv(bureau_file_path)
    bb = pd.read_csv(balance_file_path)

    bureau_cat = [col for col in bureau.columns
                  if bureau[col].dtype == 'object']
    bb_cat = [col for col in bb.columns if bb[col].dtype == 'object']

    # Agrège les mensualités par prêt et renvoie le min, le max et le nombre de mensualités + le mode du status
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = [mode]

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper()
                              for e in bb_agg.columns.tolist()])

    # Fusionne les deux jeux de données :
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Fonctions d'agrégation utilisées pour les variables numériques
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

    # Fonctions d'agrégation utilisées pour les variables catégorielles : le mode
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = [mode]
    for cat in bb_cat:
        cat_aggregations[cat + "_MODE"] = [mode]

    # Agrège les données par demande déposée auprès de la société 'Prêt à dépenser'
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        {**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper()
                                   for e in bureau_agg.columns.tolist()])

    # Feature Engineering : ajoute les mêmes colonnes numériques filtrées sur les prêts actifs
    active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper()
                                   for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Feature Engineering : ajoute les mêmes colonnes numériques filtrées sur les prêts clôturés
    closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper()
                                   for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()

    print("Bureau df shape:", bureau_agg.shape)
    if save:
        bureau_agg.to_pickle(data_dir + 'bureau_data_clean.pkl')

    return bureau_agg


def preprocess_previous_applications(file_path: str, save=True, data_dir='data/'):
    """
    Ouvre et nettoie le jeu de données, crée une nouvelle variable, 
    agrège les demandes de prêts par identifiant client actuel ('SK_ID_CURR'),
    ajoute les mêmes colonnes numériques filtrées sur les demandes de prêt acceptées, 
    ajoute les mêmes colonnes numériques filtrées sur les demandes de prêt refusées.

    Positional arguments : 
    -------------------------------------
    file_path : str : emplacement du fichier

    Optional arguments : 
    -------------------------------------
    save : bool : sauvegarder le jeu de données nettoyé sous format pickle ou non
    data_dir : str : nom du dossier dans lequel enregistrer le fichier pickle
    
    Returns : 
    -------------------------------------
    prev_agg : pd.DataFrame : jeu de données nettoyé
    """
    prev = pd.read_csv(file_path)
    categ_var = [col for col in prev.columns if prev[col].dtype == 'object']

    # Nettoyage : remplace les 365243 par nan :
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Feature engineering : ratio montant demandé / montant reçu
    # (ex : 1 -> montant reçu égal au montant demandé, 2 -> montant reçu est deux fois plus petit que le montant demandé)
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev[np.isinf(prev['APP_CREDIT_PERC'])] = 0

    # Fonctions d'agrégation utilisées pour les variables numériques
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Fonctions d'agrégations utilisées pour les variables catégorielles
    cat_aggregations = {}
    for cat in categ_var:
        cat_aggregations[cat] = [mode]

    # Agrège les anciennes demandes de prêts par identifiant client actuel
    prev_agg = prev.groupby('SK_ID_CURR').agg(
        {**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper()
                                 for e in prev_agg.columns.tolist()])

    # Feature Engineering : ajoute les mêmes variables numériques filtrées sur les anciennes demandes acceptées
    approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved']
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper()
                                     for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Feature Engineering : ajoute les mêmes variables numériques filtrées sur les anciennes demandes refusées
    refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper()
                                    for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()

    print("Previous applications df shape:", prev_agg.shape)
    if save:
        prev_agg.to_pickle(data_dir + 'previous_applications_data_clean.pkl')

    return prev_agg


def preprocess_pos_cash(file_path: str, save=True, data_dir='data/'):
    """
    Agrège les données mensuelles par identifiant client actuel ('SK_ID_CURR'),
    ajoute une nouvelle variable contenant le nombre de POS_CASH par client.

    Positional arguments : 
    -------------------------------------
    file_path : str : emplacement du jeu de données

    Optional arguments : 
    -------------------------------------
    save : bool : sauvegarder le jeu de données nettoyé sous format pickle ou non
    data_dir : str : nom du dossier dans lequel enregistrer le fichier pickle
    
    Returns : 
    -------------------------------------
    pos_agg : pd.DataFrame : jeu de données nettoyé
    """
    pos = pd.read_csv(file_path)
    categ_var = [col for col in pos.columns if pos[col].dtype == 'object']

    # Fonctions d'agrégation utilisées pour les variables numériques
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    # Fonctions d'agrégations utilisées pour les variables catégorielles
    for cat in categ_var:
        aggregations[cat] = [mode]

    # Agrège les données mensuelles par demande de prêt
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper()
                                for e in pos_agg.columns.tolist()])

    # Feature engineering : nombre de mensualités par client
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()

    print("Pos-cash balance df shape:", pos_agg.shape)
    if save:
        pos_agg.to_pickle(data_dir + 'pos_cash_balance_data_clean.pkl')

    return pos_agg


def preprocess_installments_payments(file_path: str, save=True, data_dir='data/'):
    """
    Ouvre le jeu de données et crée de nouvelles variables,
    agrège les paiements par identifiant client actuel ('SK_ID_CURR'),
    ajoute une nouvelle variable contenant le nombre de paiements par client.

    Positional arguments : 
    -------------------------------------
    file_path : str : emplacement du fichier

    Optional arguments : 
    -------------------------------------
    save : bool : sauvegarder le jeu de données nettoyé sous format pickle ou non
    data_dir : str : nom du dossier dans lequel enregistrer le fichier pickle
    
    Returns : 
    -------------------------------------
    ins_agg : pd.DataFrame : jeu de données nettoyé
    """
    ins = pd.read_csv(file_path)
    categ_var = [col for col in ins.columns if ins[col].dtype == 'object']

    # Feature engineering : ratio montant versé / montant dû, et différence montant dû - montant payé
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins[np.isinf(ins['PAYMENT_PERC'])] = 0
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # Feature engineering : Days past due and days before due (no negative values)
    # nombre de jours entre date due et la date de versement effectif (si > 0 - payé en retard)
    # nombre de jours entre la date de versement et la date due (si > 0 - payé en avance)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # Fonctions d'agrégation utilisées sur les variables numériques
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    # Fonctions d'agrégation utilisées sur les variables catégorielles
    for cat in categ_var:
        aggregations[cat] = [mode]

    # Agrège les paiements par identifiant client actuel
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper()
                                for e in ins_agg.columns.tolist()])

    # Feature engineering : ajoute le nombre de versements par client
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()

    print("Installments payments df shape:", ins_agg.shape)
    if save:
        ins_agg.to_pickle(data_dir + 'installments_data_clean.pkl')

    return ins_agg


def preprocess_credit_card_balance(file_path: str, save=True, data_dir='data/'):
    """
    Agrège les cartes de crédit par identifiant client actuel ('SK_ID_CURR'),
    ajoute une nouvelle variable contenant le nombre de cartes de crédit par client.

    Positional arguments : 
    -------------------------------------
    file_path : str : emplacement du jeu de données

    Optional arguments : 
    -------------------------------------
    save : bool : sauvegarder le jeu de données nettoyé sous format pickle ou non
    data_dir : str : nom du dossier dans lequel enregistrer le fichier pickle
    
    Returns : 
    -------------------------------------
    cc_agg : pd.DataFrame : jeu de données nettoyé
    """
    cc = pd.read_csv(file_path)
    cc.drop(columns=['SK_ID_PREV'], axis=1, inplace=True)
    numeric_var = cc._get_numeric_data().columns

    # Fonctions d'agrégation utilisées
    aggregations = {'NAME_CONTRACT_STATUS': [mode]}

    for num in numeric_var:
        aggregations[num] = ['min', 'max', 'mean', 'sum', 'var']

    # Agrège les cartes de crédit par identifiant client actuel
    cc_agg = cc.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper()
                              for e in cc_agg.columns.tolist()])

    # Feature engineering : ajoute le nombre de cartes de crédit par client
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()

    print("Credit card balance df shape:", cc_agg.shape)
    if save:
        cc_agg.to_pickle(data_dir + 'credit_card_data_clean.pkl')

    return cc_agg


def join_dataframes(dataset: pd.DataFrame, df_to_join: [pd.DataFrame], key: str):
    """
    Joint plusieurs jeux de données

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : premier jeu de données, le plus à gauche, sur lequel coller les autres jeux de données
    df_to_join : list of pd.DataFrame : liste de jeux de données à joindre
    key : str : nom de la colonne commune à tous les jeux de données
    
    Returns : 
    -------------------------------------
    new_dataset : pd.DataFrame : nouveau jeu de données obtenu après fusion
    """
    new_dataset = dataset.copy()
    for dataframe in df_to_join:
        new_dataset = new_dataset.join(dataframe, how='left', on=key)

    return new_dataset


def preprocess_data(data_name: DataFile, data_path: [str]):
    """
    Applique un prétraitement spécifique (i.e. nettoyage + agrégation + feature engineering) 
    à un ou plusieurs jeux de données

    Positional arguments : 
    -------------------------------------
    data_name : DataFile : Enum de la classe DataFile indiquant le prétraitement à appliquer 
    data_path : list of strings : emplacements du ou des jeux de données à prétraiter
    
    Returns : 
    -------------------------------------
    clean_data : pd.DataFrame : jeu de données nettoyé
    """
    clean_data = pd.DataFrame()
    with timer("Process " + data_name.name):
        if data_name.name == 'BUREAU':
            clean_data = preprocess_bureau_and_balance(*data_path)
        elif data_name.name == 'PREV':
            clean_data = preprocess_previous_applications(*data_path)
        elif data_name.name == 'POS':
            clean_data = preprocess_pos_cash(*data_path)
        elif data_name.name == 'INS':
            clean_data = preprocess_installments_payments(*data_path)
        elif data_name.name == 'CC':
            clean_data = preprocess_credit_card_balance(*data_path)

    return clean_data


def data_treatment(application_file_path: str, raw_file_path: dict, train_set=False, clean_file_path={}, data_dir='data/'):
    """
    Prétraite tous les jeux de données et les fusionne sur la clé 'SK_ID_CURR', de sorte à obtenir 
    un unique jeu de données avec une ligne par demande de prêt faite auprès de la société 'Prêt à dépenser'.

    Positional arguments : 
    -------------------------------------
    application_file_path : str : emplacement du jeu de données contenant les demandes de prêts à classifier
    raw_file_path : dict : dictionnaire contenant les emplacements des jeux de données

    Optional arguments : 
    -------------------------------------
    train_set : bool : séparer la colonne cible du reste ou non
    clean_file_path : dict : dictionnaire contenant les emplacements des données déjà nettoyées
    data_dir : str : dossier dans lequel se trouvent les données nettoyées
    
    Returns : 
    -------------------------------------
    clean_datasets : dict : dictionnaire contenant le jeu de données nettoyé et les cibles associées
    """
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    dataset = preprocess_application_data(application_file_path)
    target = pd.DataFrame()
    print('Dataset shape: ', dataset.shape)

    if train_set:
        target = dataset[['TARGET']]
        dataset = dataset.drop(columns=['TARGET'])

    df_to_join = []
    for data_name, data_path in raw_file_path.items():
        if data_name in clean_file_path:
            clean_data_df = pd.read_pickle(data_dir + clean_file_path[data_name])
        else:
            clean_data_df = preprocess_data(data_name, data_path)

        df_to_join.append(clean_data_df)

    dataset = join_dataframes(dataset, df_to_join, 'SK_ID_CURR')

    del clean_data_df, df_to_join
    gc.collect()

    clean_datasets = {'y': target, 'X': dataset}

    return clean_datasets


def build_score_dict_from_grid(grid, set_name: str, scores: list):
    """
    Renvoie un dictionnaire des meilleurs résultats obtenus, pour un modèle sur un jeu de données particulier,
    construit à partir de la grille renvoyées par 
    la méthode GridSearchCV de scikit-learn

    Positional arguments : 
    -------------------------------------
    grid :  : grille renvoyées par la méthode GridSearchCV de scikit-learn
    set_name : str : nom du jeu de données ("test" ou "train")
    scores : list : noms des scores 
    
    Returns : 
    -------------------------------------
    data : dict : meilleurs résultats sur un jeu de données
    """
    best_model_index = grid.best_index_
    data = {'set': set_name}
    data['params'] = grid.best_params_
    data['mean_time_fit'] = grid.cv_results_['mean_fit_time'][best_model_index]
    data['std_time_fit'] = grid.cv_results_['std_fit_time'][best_model_index]

    for score_name in scores:
        data['mean_{}'.format(score_name)] = grid.cv_results_[
            'mean_{}_{}'.format(set_name, score_name)][best_model_index]
        data['std_{}'.format(score_name)] = grid.cv_results_[
            'std_{}_{}'.format(set_name, score_name)][best_model_index]

    return data


def build_score_df_from_grid(grid, model_name: str, set_names: [str], scores: list):
    """
    Renvoie un dataframe des meilleurs résultats obtenus par validation croisée, pour un modèle, 
    construit à partir de la grille renvoyées par la méthode GridSearchCV de scikit-learn

    Positional arguments : 
    -------------------------------------
    grid :  : grille renvoyées par la méthode GridSearchCV de scikit-learn
    model_name : str : nom du modèle testé
    set_name : list of str : liste des noms des jeux de données testés. ex: ["test", "train"]
    scores : list : noms des scores 
    
    Returns : 
    -------------------------------------
    scores_df : pd.DataFrame : meilleurs résultats sur un ou plusieurs jeux de données
    """
    scores_dict = []
    for set_name in set_names:
        scores_dict.append(build_score_dict_from_grid(grid, set_name, scores))

    scores_df = pd.DataFrame(scores_dict)
    scores_df.insert(0, 'model', model_name)

    return scores_df


def make_preprocessor(transformers: [dict]):
    """
    Retourne un objet preprocessor contenant les transformations à appliquer aux données avant l'entrainement du modèle

    Positional arguments : 
    -------------------------------------
    transformers : list of dict : liste des modifications à appliquer 
    
    Returns : 
    -------------------------------------
    preprocessor : sklearn.compose.ColumnTransformer : objet contenant les transformations à appliquer
    """
    steps = []
    for transformer in transformers:
        pipeline = make_pipeline(*transformer['estimator'])
        steps.append((pipeline, transformer['feature']))

    preprocessor = make_column_transformer(*steps, remainder='passthrough')

    return preprocessor


def build_scorers():
    """
    Retourne dictionnaire de scorers à utiliser lors de la validation croisée
    
    Returns : 
    -------------------------------------
    scorers : dict : scorers à utiliser lors de la validation croisée
    """
    scorers = {'custom_score': make_scorer(custom_cost_score, needs_proba=True, greater_is_better=False),
               'accuracy': make_scorer(accuracy_score),
               'balanced_accuracy': make_scorer(balanced_accuracy_score),
               'recall': make_scorer(recall_score),
               'precision': make_scorer(precision_score, zero_division=0),
               'roc_auc': make_scorer(roc_auc_score)
               }

    return scorers


def score_test(model, model_name: str, y_pred: np.array, y: pd.DataFrame, thresh=0.5):
    """
    Renvoie un dictionnaire de scores calculés sur le jeu de test

    Positional arguments : 
    -------------------------------------
    model :  : modèle de classification
    model_name : str : nom du modèle de classification
    y_pred : np.array : array contenant pour chaque individu les probabilités d'appartenir à chaque classe
    y : pd.DataFrame : cibles

    Optional arguments : 
    -------------------------------------
    thresh : float : seuil au dessus duquel l'individu est classé dans la classe 1
    
    Returns : 
    -------------------------------------
    scores : dict : scores calculés sur le jeu de test
    """
    scores = {'model': model_name,
              'threshold': thresh,
              'custom_score': cost(y, y_pred),
              'accuracy_score': accuracy_score(y, y_pred),
              'balanced_accuracy_score': balanced_accuracy_score(y, y_pred),
              'roc_auc_score': roc_auc_score(y, y_pred)
              }

    return scores


def build_model(classifier, transformers: [dict], transformers_dataset=[]):
    """
    Renvoie un modèle construit avec un pipeline de transformations

    Positional arguments : 
    -------------------------------------
    classifier :  : modèle de classification
    transformers : list of dict : liste des transformations à appliquer sur des variables spécifiques avant l'entrainement
    transformeds_dataset : list : liste des transformations à appliquer sur le jeu d'entrainement entier
    
    Returns : 
    -------------------------------------
    model : imblearn.pipeline.Pipeline : objet Pipeline contenant des transformations et un modèle de classification
    """
    preprocessor = make_preprocessor(transformers)
    model = imb_make_pipeline(preprocessor, *transformers_dataset, classifier)

    return model


def cost(y_true: pd.DataFrame, y_pred: np.array):
    """
    Renvoie le coût métier, construit tq Coût FN = 10*Coût FP

    Positional arguments : 
    -------------------------------------
    y_true : pd.DataFrame : cibles à prédire
    y_pred : np.array : pour chaque individu [probabilité prédite d'appartenir à la classe 0, 
                                              probabilité prédite d'appartenir à la classe 1]
                                              
    Returns : 
    -------------------------------------
    total_cost : float : coût métier
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                      labels=[0, 1],
                                      normalize='all').ravel()
    total_cost = (10*fn + fp)/10

    return total_cost


def custom_cost_score(y_true: pd.DataFrame, y_pred_positive_proba: np.array, return_thresh=False):
    """
    Renvoie le score métier et le seuil optimisé qui détermine la classe 0 ou 1

    Positional arguments : 
    -------------------------------------
    y_true : pd.DataFrame : cibles à prédire
    y_pred_positive_proba : np.array : probabilité prédite d'appartenir à la classe 1, pour chaque individu.

    Optional arguments : 
    -------------------------------------
    return_thresh : bool : renvoyer, ou pas, le seuil au dessus duquel un individu est classé dans la classe 1.
    
    Returns : 
    -------------------------------------
    : float ou float, float : score métier seul ou score métier et seuil optimisé
    """
    scores = []
    thresholds = np.arange(0, 1, 0.001)
    for thresh in thresholds:
        y_pred_labels = (y_pred_positive_proba >= thresh).astype('int')
        scores.append(cost(y_true, y_pred_labels))

    best_thresh_index = np.argmin(scores)

    if return_thresh:
        return scores[best_thresh_index], thresholds[best_thresh_index]

    return scores[best_thresh_index]


def grid_search_cv_and_score(classifiers: [dict], X: pd.DataFrame, y: pd.DataFrame,
                             X_test: pd.DataFrame, y_test: pd.DataFrame,
                             transformers_features: [dict],
                             transformers_dataset=[], n_splits=5):
    """
    Renvoie les meilleurs scores et meilleurs hyperparamètres, obtenus par validation croisée, 
    de modèles de classification. 
    Ces scores sont également enregistrés via MLFlow Tracking.

    Positional arguments : 
    -------------------------------------
    classifiers : list of dict : liste de dictionnaires contenant les informations nécessaires pour optimiser 
    chaque modèle de classification que l'on souhaite tester (nom du modèle, modèle, paramètres à optimiser)
    ex: {'name': 'dummy', 'classifier': DummyClassifier(), 'params': {'dummyclassifier__strategy':(['most_frequent'])}}
    X : pd.DataFrame : jeu de données à diviser en entrainement et validation
    y : pd.DataFrame : cibles à prédire
    X_test : pd.DataFrame : jeu de test
    y_test : pd.DataFrame : cibles du jeu de test
    transformers_features : list of dict : liste des transformations à appliquer sur des variables spécifiques 

    Optional arguments : 
    -------------------------------------
    transformers_dataset : list : liste des transformations à appliquer sur le jeu d'entrainement entier
    n_splits : int : nombre de folds dans la validation croisée
    
    Returns : 
    -------------------------------------
    scores_cv_all, scores_test_all : pd.DataFrame, pd.DataFrame : scores moyens sur validation croisée, scores sur jeu de test
    """
    scores_cv_all = pd.DataFrame()
    scores_test = []
    preprocessor = make_preprocessor(transformers_features)

    for model in classifiers:
        classifier = imb_make_pipeline(preprocessor,
                                       *transformers_dataset,
                                       model['classifier'])  # verbose=True

        scorers = build_scorers()

        grid = GridSearchCV(estimator=classifier,
                            param_grid=model['params'],
                            scoring=scorers,
                            cv=StratifiedKFold(n_splits),
                            return_train_score=True,
                            refit='custom_score',
                            verbose=1)

        # Log parameter, metrics, and model to MLflow
        with mlflow.start_run():
            grid.fit(X, y)

            best_model_index = grid.best_index_
            mlflow.log_metric("mean_time_fit",
                              grid.cv_results_['mean_fit_time'][best_model_index])
            mlflow.log_metric("std_time_fit",
                              grid.cv_results_['std_fit_time'][best_model_index])

            for param, best_value in grid.best_params_.items():
                mlflow.log_param(param, best_value)

            for set_name in ['test', 'train']:
                for score in scorers.keys():
                    mlflow.log_metric("mean_{}_cv_{}".format(score, set_name),
                                      grid.cv_results_['mean_{}_{}'.format(set_name, score)][best_model_index])
                    mlflow.log_metric("std_{}_cv_{}".format(score, set_name),
                                      grid.cv_results_['std_{}_{}'.format(set_name, score)][best_model_index])

            y_pred = grid.predict_proba(X)
            score, thresh = custom_cost_score(y,
                                              y_pred[:, 1],
                                              return_thresh=True)
            y_pred_test = grid.predict_proba(X_test)
            y_pred_test = (y_pred_test[:, 1] >= thresh).astype('int')

            mlflow.log_param("threshold", thresh)
            mlflow.log_metric("custom_score_test", cost(y_test, y_pred_test))
            mlflow.log_metric("accuracy_test",
                              accuracy_score(y_test, y_pred_test))
            mlflow.log_metric("balanced_accuracy_test",
                              balanced_accuracy_score(y_test, y_pred_test))
            mlflow.log_metric("roc_auc_test",
                              roc_auc_score(y_test, y_pred_test))
            mlflow.sklearn.log_model(grid, "model")

        scores_cv = build_score_df_from_grid(grid,
                                             model['name'],
                                             ['test', 'train'],
                                             scorers.keys())

        scores_test.append(score_test(grid, y_pred_test,
                           y_test, model['name'], thresh))

        scores_cv_all = pd.concat([scores_cv_all, scores_cv],
                                  ignore_index=True)

    scores_test_all = pd.DataFrame(scores_test)
    mlflow.end_run()

    return scores_cv_all, scores_test_all


def gb_cv(params, X: np.array, y: np.array, transformers: [dict], sampler: list, random_state=8, cv=3):
    """
    Renvoie - le score métier moyen, après validation croisée sur le modèle GradientBoost
    fonction à minimiser lors de l'optimisation des hyperparamètres du modèle GradientBoost

    Positional arguments : 
    -------------------------------------
    params :  : combinaison d'hyperparmètres à tester 
    X : np.array : jeu d'entrainement
    y : np.array : cibles à prédire
    transformers : list of dict : liste des transformations à appliquer sur des variables spécifiques
    sampler : list : liste des transformations à appliquer sur le jeu d'entrainement entier

    Optional arguments : 
    -------------------------------------
    random_state : int : si renseigné résultats reproductibles à chaque appel
    cv : int : nombre de folds dans la validation croisée
    
    Returns : 
    -------------------------------------
    score : float : -score métier moyen
    """
    clf = GradientBoostingClassifier(random_state=random_state, **params)
    model = build_model(clf, transformers, sampler)

    custom_score = make_scorer(custom_cost_score,
                               needs_proba=True,
                               greater_is_better=False)

    score = -cross_val_score(model, X, y, cv=cv,
                             scoring=custom_score, n_jobs=-1).mean()

    return score


def score_best_model(classifier, transformers: [dict], best_params: dict, datasets: dict, sampler=[],
                     cv=5, with_eval_set=False):
    """
    Affiche score métier et AUC sur jeu de test/train + temps d'entrainement du modèle optimisé 
    et retourne le modèle optimisé entrainé ainsi qu'un tableau contenant les hyperparamètres du meilleur modèle.

    Positional arguments : 
    -------------------------------------
    classifier :  : modèle de classification optimisé
    transformers : list of dict : liste des transformations à appliquer sur des variables spécifiques
    best_params : dict : dictionnaire des paramètres avec lesquels configurer le modèle optimisé
    datasets : dict : dictionnaire contenant les jeux de données de test et d'entrainement

    Optional arguments : 
    -------------------------------------
    sampler : list : liste des transformations à appliquer sur le jeu d'entrainement entier
    cv : int : nombre de folds dans la validation croisée
    with_eval_set : bool : utiliser ou non un jeu de validation
    
    Returns : 
    -------------------------------------
    model_opt, best_param_df : classifier, pd.DataFrame : modèle optimisé, tableau contenant les hyperparamètres après optimisation
    """
    model_opt = build_model(classifier, transformers, sampler)
    custom_score = make_scorer(custom_cost_score,
                               needs_proba=True,
                               greater_is_better=False)

    t1_opt_start = perf_counter()

    if with_eval_set:
        x_train, x_valid, y_train, y_valid = train_test_split(datasets['X_train'], datasets['y_train'], test_size=0.3,
                                                              random_state=8, stratify=datasets['y_train'])
        model_opt.fit(x_train, y_train,
                      eval_set=[(x_train, y_train), (x_valid, y_valid)],
                      eval_metric=custom_score)
    else:
        model_opt.fit(datasets['X_train'], datasets['y_train'])

    t1_opt_stop = perf_counter()

    y_pred = model_opt.predict_proba(datasets['X_train'])
    score, thresh = custom_cost_score(datasets['y_train'],
                                      y_pred[:, 1],
                                      return_thresh=True)
    y_pred_test = model_opt.predict_proba(datasets['X_test'])
    y_pred_test = (y_pred_test[:, 1] >= thresh).astype('int')

    best_param_df = pd.DataFrame(best_params.items(), columns=[
                                 'Param', 'Best Param'])
    best_param_df = pd.concat([best_param_df,
                               pd.DataFrame([{'Param': 'thresh', 'Best Param': thresh}])])
    display(best_param_df)

    print("Modèle optimisé : ")
    print("Test Score métier : {:.4f}".format(cost(datasets['y_test'],
                                                   y_pred_test)))
    print("Test AUC: {:.4f}".format(roc_auc_score(datasets['y_test'], y_pred_test)))
    print("Temps d'entrainement : {:.3f}".format(t1_opt_stop - t1_opt_start))
    print("Train Score métier : {:.4f}".format(score))

    return model_opt, best_param_df


def lgbm_cv(params, X: np.array, y: np.array, transformers: [dict], sampler: list, random_state=8, cv=3):
    """
    Renvoie - le score métier moyen, après validation croisée sur le modèle LightGBM
    fonction à minimiser lors de l'optimisation des hyperparamètres du modèle LightGBM

    Positional arguments : 
    -------------------------------------
    params :  : combinaison d'hyperparmètres à tester 
    X : np.array : jeu d'entrainement
    y : np.array : cibles à prédire
    transformers : list of dict : liste des transformations à appliquer sur des variables spécifiques
    sampler : list : liste des transformations à appliquer sur le jeu d'entrainement entier

    Optional arguments : 
    -------------------------------------
    random_state : int : si renseigné résultats reproductibles à chaque appel
    cv : int : nombre de folds dans la validation croisée
    
    Returns : 
    -------------------------------------
    score : float : -score métier moyen
    """
    clf = LGBMClassifier(random_state=random_state, **params)
    model = build_model(clf, transformers, sampler)

    custom_score = make_scorer(custom_cost_score,
                               needs_proba=True,
                               greater_is_better=False)

    score = -cross_val_score(model, X, y, cv=cv,
                             scoring=custom_score, n_jobs=-1).mean()

    return score