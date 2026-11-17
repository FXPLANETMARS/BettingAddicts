# Fichier : src/modele.py

import joblib, os, time, json  # joblib sauvegarde le modèle, os gère les dossiers, time mesure la durée, json lit et écrit le fichier de métadonnées
from sklearn.linear_model import LogisticRegression  # L'algorithme de régression logistique (accuracy_score n'est plus importé car on calcule l'exactitude nous-mêmes)
from donnees import JeuDeDonnees, VariableNumerique, VariableCategorielle, creer_variable  # Importe nos classes et notre fonction
from exceptions import DonneeInvalideError, ModeleIntrouvableError  # Importe nos erreurs


def chronometrer(fonction):  # Le décorateur reçoit la fonction à décorer
    def nouvelle(modele, x_train, y_train):  # La nouvelle version a les mêmes paramètres que entrainer
        debut = time.time()  # L'heure avant, en secondes
        resultat = fonction(modele, x_train, y_train)  # Exécute la vraie fonction
        fin = time.time()  # L'heure après
        print("Duree de l'entrainement :", round(fin - debut, 3), "s")  # Affiche la durée arrondie à 3 décimales
        return resultat  # Retourne ce que la fonction retournait
    return nouvelle  # Retourne la nouvelle fonction, sans parenthèses donc sans l'exécuter


class EntraineurModele:
    def __init__(self, variables):  # Le constructeur reçoit la liste des variables
        self.variables = variables  # La mémorise (composition : l'objet contient des variables)
        self.algorithme = LogisticRegression(max_iter=1000)  # Crée le modèle, avec droit à 1000 tours de calcul

    @chronometrer  # Au-dessus d'entrainer : équivaut à entrainer = chronometrer(entrainer)
    def entrainer(self, x_train, y_train):
        self.algorithme.fit(x_train, y_train)  # Montre au modèle les matchs d'entraînement et leurs résultats pour qu'il apprenne

    @classmethod  # La méthode s'appelle sur la classe (EntraineurModele.charger(...)), et cls désigne la classe
    def charger(cls, chemin_modele, chemin_metadonnees):
        if not os.path.exists(chemin_modele) or not os.path.exists(chemin_metadonnees):  # Vrai si l'un des deux fichiers manque
            raise ModeleIntrouvableError("Modele introuvable. Lancez d'abord modele.py pour l'entrainer.")  # Lève l'erreur avec la marche à suivre

        with open(chemin_metadonnees, encoding="utf-8") as fichier:  # Ouvre le JSON en lecture
            metadonnees = json.load(fichier)  # Le convertit en dictionnaire

        variables = [creer_variable(description) for description in metadonnees["variables"]]  # Recrée chaque variable à partir de sa fiche
        entraineur = cls(variables)  # Crée un entraîneur, avec un modèle encore vide
        entraineur.algorithme = joblib.load(chemin_modele)  # Remplace le modèle vide par celui déjà entraîné
        return entraineur  # Retourne l'objet prêt à prédire

    def evaluer(self, x_test, y_test):
        predictions = self.algorithme.predict(x_test)  # Le modèle devine le résultat des matchs de test
        # == compare chaque vrai résultat à sa devinette (vrai ou faux). .mean() donne la proportion de vrai (l'exactitude). float() convertit en nombre ordinaire.
        score = float((y_test == predictions).mean())
        return score

    def calculer_reference(self, y_train, y_test):
        classe_majoritaire = y_train.value_counts().idxmax()  # value_counts() compte chaque classe à l'entraînement et idxmax() donne le nom de la plus fréquente
        score = float((y_test == classe_majoritaire).mean())  # Proportion de matchs de test où répondre toujours cette classe aurait raison
        return classe_majoritaire, score

    def predire(self, saisies):
        """Recoit un dictionnaire {nom_variable: valeur saisie}. Retourne (classe predite, probabilites)."""
        ligne = []  # Liste qui contiendra les 9 nombres du match
        for variable in self.variables:  # Passe les 5 variables dans l'ordre
            brut = saisies[variable.nom]  # Lit dans le dictionnaire la valeur saisie pour cette variable
            if brut is None or str(brut).strip() == "":  # Vrai si la saisie est absente ou ne contient que des espaces
                raise DonneeInvalideError(f"Le champ '{variable.nom}' est vide.")  # Lève l'erreur « champ vide »
            valeur = variable.valider_saisie(brut)  # Valide et convertit la saisie (lève DonneeInvalideError si elle est invalide)
            # Encode la valeur (encoder reçoit une liste et retourne une liste de lignes), [0] prend l'unique ligne, et + la colle à la suite
            ligne = ligne + variable.encoder([valeur])[0]

        classes = self.algorithme.classes_  # Les classes connues du modèle, en ordre alphabétique (Away, Draw, Home)
        # Les probabilités de chaque classe pour ce match ([ligne] car la méthode attend une liste de matchs, [0] pour notre seul match)
        probas = self.algorithme.predict_proba([ligne])[0]
        probabilites = {}  # Dictionnaire vide
        for i in range(len(classes)):
            probabilites[str(classes[i])] = float(probas[i])  # Associe chaque nom de classe à sa probabilité

        classe_predite = str(self.algorithme.predict([ligne])[0])
        return classe_predite, probabilites  # Retourne la classe prédite et le dictionnaire des probabilités

    def sauvegarder_json(self, chemin_fichier, description_variables, exactitude, reference):
        # Dictionnaire des informations pour l'interface : "colonnes" (noms des variables), "variables" (les fiches), "classes", "exactitude" et "reference"
        metadonnees = {
            "colonnes": [variable.nom for variable in self.variables],  # Liste construite en une ligne, avec le nom de chaque variable
            "variables": description_variables,
            "classes": [str(c) for c in self.algorithme.classes_],
            "exactitude": exactitude,
            "reference": reference,
        }
        os.makedirs(os.path.dirname(chemin_fichier), exist_ok=True)  # Crée le dossier si besoin
        with open(chemin_fichier, "w", encoding="utf-8") as fichier:  # Ouvre le fichier en écriture et le ferme automatiquement à la fin du bloc
            json.dump(metadonnees, fichier, indent=2, ensure_ascii=False)  # Écrit le dictionnaire en JSON (indent=2 le rend lisible, ensure_ascii=False garde les accents)

    def sauvegarder(self, chemin_fichier):
        os.makedirs(os.path.dirname(chemin_fichier), exist_ok=True)  # dirname extrait le dossier du chemin ; makedirs le crée s'il n'existe pas, sans erreur s'il existe déjà
        joblib.dump(self.algorithme, chemin_fichier)  # Écrit le modèle entraîné dans un fichier .joblib (évite de réentraîner l'IA à chaque lancement)


if __name__ == '__main__':
    variables = [  # Les 5 variables
        VariableCategorielle("Sport", ["Baseball", "Basketball", "Football", "Hockey", "Tennis"]),
        VariableNumerique("Home_Team_Odds"),
        VariableNumerique("Away_Team_Odds"),
        VariableNumerique("Draw_Odds"),
        VariableNumerique("Mois"),
    ]

    dataset2 = JeuDeDonnees("donnees/sports_betting_predictive_analysis.csv", variables)
    dataset2.nettoyer()  # Chargement et nettoyage
    dataset2.valider()  # Conversion et validation
    x_train, x_test, y_train, y_test = dataset2.separer_train_test("Resultat_Simplifie")  # Coupe les données

    mon_modele = EntraineurModele(variables)  # Crée l'entraîneur
    mon_modele.entrainer(x_train, y_train)  # Entraîne (affiche la durée)
    evaluation = mon_modele.evaluer(x_test, y_test)  # Calcule l'exactitude
    majoritaire, reference = mon_modele.calculer_reference(y_train, y_test)  # Calcule la classe majoritaire et le score de référence
    print(f"Exactitude du modele : {evaluation * 100:.1f} %")  # Affiche l'exactitude en pourcentage avec 1 décimale
    print(f"Reference (toujours {majoritaire}) : {reference * 100:.1f} %")  # Même chose pour la référence

    # Prédit un match inventé, passé en dictionnaire
    classe, probas = mon_modele.predire({"Sport": "Tennis", "Home_Team_Odds": "1.5",
                                         "Away_Team_Odds": "3.0", "Draw_Odds": "0", "Mois": "6"})
    print(classe, probas)  # Affiche la classe et les probabilités

    mon_modele.sauvegarder("modeles/modeles_paris.joblib")  # Écrit le modèle
    description = dataset2.decrire_variables()  # Récupère les 5 fiches
    mon_modele.sauvegarder_json("modeles/metadonnees.json", description, evaluation, reference)  # Écrit le JSON

    # Recharge le modèle depuis les fichiers (test temporaire)
    modele_recharge = EntraineurModele.charger("modeles/modeles_paris.joblib", "modeles/metadonnees.json")
    saisie = {"Sport": "Tennis", "Home_Team_Odds": "999", "Away_Team_Odds": "3.0", "Draw_Odds": "0", "Mois": "6"}
    try:
        modele_recharge.predire(saisie)  # Essaie une cote de 999
    except DonneeInvalideError as erreur:
        print("Erreur attendue :", erreur)  # Affiche l'erreur attendue (test temporaire)