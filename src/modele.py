"""Entraînement, évaluation, prédiction et sauvegarde du modèle."""
import json
import os
import joblib  # Sauvegarde le modèle dans un fichier
import pandas as pd
from sklearn.linear_model import LogisticRegression

from donnees import creer_variable, Variable
from exceptions import DonneeInvalideError, ModeleIntrouvableError



class EntraineurModele:
    """Entraîne, évalue, sauvegarde et utilise le modèle. Contient la liste des variables (composition)."""

    def __init__(self, variables: list[Variable]) -> None:
        self._variables = variables
        self._algorithme = LogisticRegression(max_iter=1000)  # Droit à 1000 tours de calcul

    @property
    def variables(self) -> list[Variable]:
        """Variables d'entrée du modèle (lecture seule)."""
        return self._variables

    @property
    def algorithme(self) -> LogisticRegression:
        """Modèle scikit-learn sous-jacent (lecture seule)."""
        return self._algorithme

    def entrainer(self, x_train: list, y_train: pd.Series) -> None:
        """Montre au modèle les matchs d'entraînement et leurs résultats pour qu'il apprenne."""
        self._algorithme.fit(x_train, y_train)

    @classmethod  # S'appelle sur la classe : EntraineurModele.charger(...)
    def charger(cls, chemin_modele: str, chemin_metadonnees: str) -> "EntraineurModele":
        """Recharge un modèle déjà entraîné, sans le ré-entraîner.
        Lève ModeleIntrouvableError si l'un des deux fichiers est absent."""
        if not os.path.exists(chemin_modele) or not os.path.exists(chemin_metadonnees):
            raise ModeleIntrouvableError("Modele introuvable. Lancez d'abord main.py pour l'entrainer.")

        with open(chemin_metadonnees, encoding="utf-8") as fichier:
            metadonnees = json.load(fichier)  # Le JSON redevient un dictionnaire

        variables = [creer_variable(description) for description in metadonnees["variables"]]
        entraineur = cls(variables)  # Crée un entraîneur avec un modèle encore vide
        entraineur._algorithme = joblib.load(chemin_modele)  # Le remplace par le modèle déjà entraîné
        return entraineur

    def evaluer(self, x_test: list, y_test: pd.Series) -> float:
        """Exactitude du modèle sur les matchs de test (proportion de bonnes réponses, entre 0 et 1)."""
        predictions = self._algorithme.predict(x_test)  # Le modèle devine les résultats des matchs de test
        # == compare chaque vrai résultat à sa devinette ; .mean() donne la proportion de vrai
        score = float((y_test == predictions).mean())
        return score

    def calculer_reference(self, y_train: pd.Series, y_test: pd.Series) -> tuple[str, float]:
        """Modèle trivial de comparaison : répondre toujours la classe la plus fréquente à l'entraînement.
        Retourne (cette classe, son exactitude sur le test)."""
        classe_majoritaire = y_train.value_counts().idxmax()  # Nom de la classe la plus fréquente
        score = float((y_test == classe_majoritaire).mean())
        return classe_majoritaire, score

    def predire(self, saisies: dict[str, str]) -> tuple[str, dict[str, float]]:
        """Reçoit {nom_variable: valeur saisie}. Retourne (classe prédite, probabilités de chaque classe).
        Lève DonneeInvalideError si une saisie est vide, non numérique ou hors intervalle."""
        ligne = []  # Contiendra les 9 nombres du match
        for variable in self._variables:
            brut = saisies[variable.nom]  # Valeur saisie pour cette variable
            if brut is None or str(brut).strip() == "":
                raise DonneeInvalideError(f"Le champ '{variable.nom}' est vide.")
            valeur = variable.valider_saisie(brut)  # Valide et convertit
            ligne = ligne + variable.encoder([valeur])[0]  # Même encodage qu'à l'entraînement

        classes = self._algorithme.classes_  # Classes connues, en ordre alphabétique (Away, Draw, Home)
        probas = self._algorithme.predict_proba([ligne])[0]  # [ligne] : la méthode attend une liste de matchs
        probabilites = {}
        for i in range(len(classes)):
            probabilites[str(classes[i])] = float(probas[i])

        classe_predite = str(self._algorithme.predict([ligne])[0])
        return classe_predite, probabilites

    def sauvegarder_json(self, chemin_fichier: str, description_variables: list[dict],
                         exactitude: float, reference: float) -> None:
        """Écrit les métadonnées du modèle (variables, classes, scores) dans un fichier JSON."""
        metadonnees = {
            "colonnes": [variable.nom for variable in self._variables],
            "variables": description_variables,
            "classes": [str(c) for c in self._algorithme.classes_],
            "exactitude": exactitude,
            "reference": reference,
        }
        os.makedirs(os.path.dirname(chemin_fichier), exist_ok=True)  # Crée le dossier si besoin
        with open(chemin_fichier, "w", encoding="utf-8") as fichier:  # with ferme le fichier automatiquement
            json.dump(metadonnees, fichier, indent=2, ensure_ascii=False)

    def sauvegarder(self, chemin_fichier: str) -> None:
        """Écrit le modèle entraîné dans un fichier .joblib (évite de ré-entraîner à chaque lancement)."""
        os.makedirs(os.path.dirname(chemin_fichier), exist_ok=True)
        joblib.dump(self._algorithme, chemin_fichier)

    def __str__(self) -> str:
        return f"EntraineurModele({len(self._variables)} variables)"
