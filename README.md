# OC DS Projet 7 : Implémentez un modèle de scoring
Formation OpenClassrooms - Parcours data scientist - Projet n°7 - Implementez un modele de scoring

## Missions : 
 * Elaborer un modèle de classification qui calcule la probabilité qu'un client ne rembourse pas son crédit et classifie la demande en "crédit accordé" ou "crédit refusé".
 * Construire un dashboard interactif pour interpréter les prédictions du modèle et explorer les informations personnelles des clients.
 * Mettre en production le modèle de scoring, via une API, ainsi que le dashboard interactif qui appelle l'API.
 
 ## Spécifications Dashboard : 
Le dashboard interactif devra contenir au minimum les fonctionnalités suivantes :
 - Permettre de visualiser le score et l’interprétation de ce score pour chaque client.
 - Permettre de visualiser des informations descriptives relatives à un client.
 - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

## Compétences évaluées :  
 * Définir la stratégir d'élaboration d'un modèle d'apprentissage supervisé
 * Présenter son travail de modélisation à l'oral
 * Réaliser un dashboard pour présenter son travail de modélisation
 * Rédiger une note méthodologique afin de communiquer sa démarche de modélisation
 * Utiliser un logiciel de version de code pour assurer l’intégration du modèle
 * Déployer un modèle via une API dans le Web

 ## Découpage des dossiers :
 * api :
     * dossier "app" (code permettant de déployer l'API),
     * dossier "data" (infos clients),
     * dossier "models" (modèle de scoring sérialisé)
 * dashboard :
     * fichier "dashboard" (code générant le dashboard),
     * fichier "functions" (fonctions appelées par le dashboard)
 * notebook : notebook, fonctions appelées par le notebook, tableau HTML d'analyse de data drift
 * note_technique : note méthodologique
 * tests : tests unitaires
 * htmlcov : tableau HTML de couverture de test

## Swagger API de Prédiction : 

* Azure Web App (non maintenue) : https://projet7api.azurewebsites.net/docs

## Dashboard client : 

* Azure Web application (non maintenue) : https://ocprojet7.azurewebsites.net

## Données sources : 
 
 https://www.kaggle.com/c/home-credit-default-risk/data

## Note technique :

 https://github.com/J28u/oc-projet7/tree/main/note_technique/NoteTechnique.pdf

## Data Drift Report :

 https://github.com/J28u/oc-projet7/tree/main/notebook

## Notebook d'élaboration du modèle :

https://github.com/J28u/oc-projet7/blob/main/notebook/Projet7.ipynb

 
