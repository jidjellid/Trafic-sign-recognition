Dépendances pip à installer :

python3 -m pip install torch
python3 -m pip install torchvision
python3 -m pip install natsort
python3 -m pip install scikit-image

Rapport : main.ipynb, main.pdf
Script d'entrainement : train.py
Script de prédiction : predict.py

Il est conseillé d'aller voir la dernière partie "Observation" du rapport sur le PDF afin d'avoir les images

Pour entrainer un nouveau modèle a l'aide de train.py :

    python3 train.py train monDossier

    Exemple : python3 train.py panneaux_route/Train

L'entraînement sauvegardera ses poids sous un fichier "modelWeights"


Pour effectuer des prédictions a l'aide de predict.py :

    python3 predict.py cheminAPredire

    Exemple : python3 predict.py panneaux_route/Test
              python3 predict.py panneau.jpg
    

Si le chemin de la prédiction est une image, le script affichera son résultat.

Si le chemin de la prédiction est un dossier, le script va produire un fichier Results.csv contenant l'ensemble des prédictions.
Si un fichier cheminAPredire.csv existe, le script affichera en plus le taux de réussite des prédictions.