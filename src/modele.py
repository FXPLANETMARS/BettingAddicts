# Fichier : src/modele.py

import joblib, os   ##joblib -> sauvegarde des objets python lourds dans des fichiers
from sklearn.linear_model import LogisticRegression ## L'algorithme de regression
from sklearn.metrics import accuracy_score  ## Calcule le poucentage de bonnes reponses de l'IA
from donnees import JeuDeDonnees, VariableNumerique

class EntraineurModele:
    def __init__(self):
        self.algorithme = LogisticRegression(max_iter=1000)## donner plus de temps, autorise 1000 tours de calculs pour
                                                           ## trouver sa logique

    def entrainer(self, x_train, y_train):
        self.algorithme.fit(x_train, y_train)   ## .fit() mets les donnees d'entrainement x et les reponses y dans l'algo

    def evaluer(self, x_test, y_test):
        predictions = self.algorithme.predict(x_test)   ## .predict(..) demande a l'ia de predire les paquets de donnees
                                                        ## qu'elle a jamais vu
        score = accuracy_score(y_test, predictions) ## compare les donnees reelles des donnees predites par l'ia.
        return score

    def sauvegarder(self, chemin_fichier):
        os.makedirs("modeles", exist_ok=True)   ## S'assure que le dossier de destination existe pour pas se prendre
                                                      ## pour pas avoir de bugs de dossier introuvable
        joblib.dump(self.algorithme, chemin_fichier)    ## Compresse et ecrit le modele entraine dans un fichier .joblib
                                                        ## **Le fichier .joblib permet au programme de ne pas reentrainer
                                                        ## a chaque fois l'IA pour faire les predictions. Comme cela,
                                                        ## il est plus rapide de reutiliser les donnees deja existantes
                                                        ## que d'en reecrire de nouvelles.

if __name__ == '__main__':

    variables = [
        VariableNumerique("Home_Team_Odds"),
        VariableNumerique("Away_Team_Odds"),
    ]

    dataset2 = JeuDeDonnees("donnees/sports_betting_predictive_analysis.csv", variables)
    dataset2.nettoyer()
    dataset2.valider()
    x_train, x_test, y_train, y_test = dataset2.separer_train_test("Resultat_Simplifie")
    mon_modele = EntraineurModele()
    mon_modele.entrainer(x_train, y_train)
    evaluation = mon_modele.evaluer(x_test, y_test)
    print(f"Rendement brut : {evaluation}")
    print(f"Rendement arrondi : {round(evaluation, 2) * 100} %")
    mon_modele.sauvegarder("modeles/modeles_paris.joblib")