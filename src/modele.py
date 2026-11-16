# Fichier : src/modele.py

# import joblib, os   ##joblib -> sauvegarde des objets python lourds dans des fichiers
# from sklearn.linear_model import LogisticRegression ## L'algorithme de regression
# from sklearn.metrics import accuracy_score  ## Calcule le poucentage de bonnes reponses de l'IA
# from donnees import JeuDeDonnees, VariableNumerique
import joblib, os, time, json
from sklearn.linear_model import LogisticRegression
from donnees import JeuDeDonnees, VariableNumerique, VariableCategorielle, creer_variable
from exceptions import DonneeInvalideError, ModeleIntrouvableError

def chronometrer(fonction):
    def nouvelle(modele, x_train, y_train):
        debut = time.time()
        resultat = fonction(modele, x_train, y_train)
        fin = time.time()
        print("Duree de l'entrainement :", round(fin - debut, 3), "s")
        return resultat
    return nouvelle

class EntraineurModele:
    def __init__(self, variables):
        self.variables = variables
        self.algorithme = LogisticRegression(max_iter=1000)## donner plus de temps, autorise 1000 tours de calculs pour
                                                           ## trouver sa logique
    @chronometrer
    def entrainer(self, x_train, y_train):
        self.algorithme.fit(x_train, y_train)   ## .fit() mets les donnees d'entrainement x et les reponses y dans l'algo

    @classmethod
    def charger(cls, chemin_modele, chemin_metadonnees):
        if not os.path.exists(chemin_modele) or not os.path.exists(chemin_metadonnees):
            raise ModeleIntrouvableError("Modele introuvable. Lancez d'abord modele.py pour l'entrainer.")

        with open(chemin_metadonnees, encoding="utf-8") as fichier:
            metadonnees = json.load(fichier)

        variables = [creer_variable(description) for description in metadonnees["variables"]]
        entraineur = cls(variables)
        entraineur.algorithme = joblib.load(chemin_modele)
        return entraineur

    def evaluer(self, x_test, y_test):
        predictions = self.algorithme.predict(x_test)
        score = float((y_test == predictions).mean())
        return score

    def calculer_reference(self, y_train, y_test):
        classe_majoritaire = y_train.value_counts().idxmax()  ##La classe la plus frequente a l'entrainement
        score = float((y_test == classe_majoritaire).mean())  ##Exactitude si on predit toujours celle-la
        return classe_majoritaire, score

    def predire(self, saisies):
        """Recoit un dictionnaire {nom_variable: valeur saisie}. Retourne (classe predite, probabilites)."""
        ligne = []
        for variable in self.variables:
            brut = saisies[variable.nom]
            if brut is None or str(brut).strip() == "":
                raise DonneeInvalideError(f"Le champ '{variable.nom}' est vide.")
            valeur = variable.valider_saisie(brut)
            ligne = ligne + variable.encoder([valeur])[0]

        classes = self.algorithme.classes_
        probas = self.algorithme.predict_proba([ligne])[0]
        probabilites = {}
        for i in range(len(classes)):
            probabilites[str(classes[i])] = float(probas[i])

        classe_predite = str(self.algorithme.predict([ligne])[0])
        return classe_predite, probabilites

    def sauvegarder_json(self, chemin_fichier, description_variables, exactitude, reference):
        metadonnees = {
            "colonnes": [variable.nom for variable in self.variables],
            "variables": description_variables,
            "classes": [str(c) for c in self.algorithme.classes_],
            "exactitude": exactitude,
            "reference": reference,
        }
        os.makedirs(os.path.dirname(chemin_fichier), exist_ok=True)
        with open(chemin_fichier, "w", encoding="utf-8") as fichier:
            json.dump(metadonnees, fichier, indent=2, ensure_ascii=False)

    def sauvegarder(self, chemin_fichier):
        os.makedirs(os.path.dirname(chemin_fichier), exist_ok=True)   ## S'assure que le dossier de destination existe pour pas se prendre
                                                      ## pour pas avoir de bugs de dossier introuvable
        joblib.dump(self.algorithme, chemin_fichier)    ## Compresse et ecrit le modele entraine dans un fichier .joblib
                                                        ## **Le fichier .joblib permet au programme de ne pas reentrainer
                                                        ## a chaque fois l'IA pour faire les predictions. Comme cela,
                                                        ## il est plus rapide de reutiliser les donnees deja existantes
                                                        ## que d'en reecrire de nouvelles.

if __name__ == '__main__':

    variables = [
        VariableCategorielle("Sport", ["Baseball", "Basketball", "Football", "Hockey", "Tennis"]),
        VariableNumerique("Home_Team_Odds"),
        VariableNumerique("Away_Team_Odds"),
        VariableNumerique("Draw_Odds"),
        VariableNumerique("Mois"),
    ]

    dataset2 = JeuDeDonnees("donnees/sports_betting_predictive_analysis.csv", variables)
    dataset2.nettoyer()
    dataset2.valider()
    x_train, x_test, y_train, y_test = dataset2.separer_train_test("Resultat_Simplifie")

    mon_modele = EntraineurModele(variables)
    mon_modele.entrainer(x_train, y_train)
    evaluation = mon_modele.evaluer(x_test, y_test)
    majoritaire, reference = mon_modele.calculer_reference(y_train, y_test)
    print(f"Exactitude du modele : {evaluation * 100:.1f} %")
    print(f"Reference (toujours {majoritaire}) : {reference * 100:.1f} %")

    classe, probas = mon_modele.predire({"Sport": "Tennis", "Home_Team_Odds": "1.5",
                                         "Away_Team_Odds": "3.0", "Draw_Odds": "0", "Mois": "6"})
    print(classe, probas)

    mon_modele.sauvegarder("modeles/modeles_paris.joblib")
    description = dataset2.decrire_variables()
    mon_modele.sauvegarder_json("modeles/metadonnees.json", description, evaluation, reference)


    modele_recharge = EntraineurModele.charger("modeles/modeles_paris.joblib", "modeles/metadonnees.json")
    saisie = {"Sport": "Tennis", "Home_Team_Odds": "999", "Away_Team_Odds": "3.0", "Draw_Odds": "0", "Mois": "6"}
    try:
        modele_recharge.predire(saisie)
    except DonneeInvalideError as erreur:
        print("Erreur attendue :", erreur)