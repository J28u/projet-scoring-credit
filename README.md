# OC/DS Projet 7 : Implémentez un modèle de scoring
Formation OpenClassrooms - Parcours data scientist - Projet Professionnalisant (Mai - Juillet 2023)

## Secteur : 
Banque - Finance - Assurance

## Technologies utilisées : 
  * Jupyter Notebook
  * Python
    - Scikit-learn
    - HyperOpt
    - Streamlit
    - FastAPI
    - Evidently
  * Docker
  * Azure
    - App Service
    - Container Registry

## Mots-clés : 
Dashboard, Régression supervisée, Cloud, Tests unitaires

## Le contexte : 
Le client est une société financière qui propose des crédits à la consommation pour des personnes ayant peu ou pas d’historique de prêt.
Il a besoin d’un outil de scoring qui calcule la probabilité qu’un client rembourse ou non son crédit et classifie sa demande de crédit en «accordée» ou «refusée».

## La mission : 
* Élaborer ce modèle de scoring de prédiction et le mettre en production à l’aide d’une API déployée dans le cloud.
* Construire et mettre en production un dashboard interactif, que les chargés de relation client puissent facilement utiliser pour expliquer de manière transparente les décisions d’octroi de crédit basées sur les prédictions du modèle.

## Note technique (détaillant la démarche de modélisation) :
 https://github.com/J28u/oc-projet7/tree/main/note_technique/NoteTechnique.pdf
 
## Découpage des dossiers :
 * api :
   - dossier "app" : code permettant de déployer l'API,
   - dossier "data" : données clients,
   - dossier "models" (modèle de scoring sérialisé)
 * dashboard :
   - fichier "dashboard" (code générant le dashboard),
   - fichier "functions" (fonctions appelées par le dashboard)
 * notebook :
   - notebook,
   - fonctions appelées par le notebook,
   - tableau HTML d'analyse de data drift
 * note_technique : note méthodologique
 * tests : tests unitaires
 * htmlcov : tableau HTML de couverture de test
 * presentation.pdf : support de présentation pour la soutenance détaillant le travail réalisé

## Spécifications Dashboard : 
Le dashboard interactif devra contenir au minimum les fonctionnalités suivantes :
 - Permettre de visualiser le score et l’interprétation de ce score pour chaque client.
 - Permettre de visualiser des informations descriptives relatives à un client.
 - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

## Compétences évaluées :  
* Définir et mettre en oeuvre une stratégie de suivi de la performance d’un modèle
* Évaluer les performances des modèles d’apprentissage supervisé
* Utiliser un logiciel de version de code pour assurer l’intégration du modèle
* Définir la stratégie d’élaboration d’un modèle d’apprentissage supervisé
* Réaliser un dashboard pour présenter mon travail de modélisation
* Rédiger une note méthodologique afin de communiquer ma démarche de modélisation
* Déployer un modèle via une API dans le Web
* Définir et mettre en œuvre un pipeline de déploiement des modèles

## Data Source : 
 https://www.kaggle.com/c/home-credit-default-risk/data

## Data Drift Report :
https://github.com/J28u/oc-projet7/tree/main/notebook

## Notebook d'élaboration du modèle :
https://github.com/J28u/oc-projet7/blob/main/notebook/Projet7.ipynb
