"""Lecture, nettoyage et préparation du jeu de données des paris sportifs."""
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
import matplotlib.pyplot as plt  # Outil de graphiques (pour tracer_resultats)
import pandas as pd
from sklearn.model_selection import train_test_split
from exceptions import DonneeInvalideError  # Notre exception personnalisée


class Variable(ABC):
    """Classe parente abstraite : une variable d'entrée du modèle (une colonne du jeu de données).

    Chaque classe enfant doit redéfinir valider_saisie, encoder et decrire (polymorphisme).
    """

    def __init__(self, nom: str, type_donnee: type = str) -> None:
        self._nom = nom  # Attribut protégé : on le lit avec la propriété nom
        self.type_donnee = type_donnee

    @property
    def nom(self) -> str:
        """Nom de la variable (lecture seule)."""
        return self._nom

    @abstractmethod
    def valider_saisie(self, valeur: object) -> object:
        """Valide une valeur et la retourne convertie. Lève DonneeInvalideError si elle est invalide."""

    @abstractmethod
    def encoder(self, valeurs: Iterable) -> list[list[float]]:
        """Transforme une colonne de valeurs en une liste de lignes de nombres."""

    @abstractmethod
    def decrire(self, colonne: pd.Series) -> dict:
        """Retourne la fiche de la variable (dictionnaire écrit dans le JSON de métadonnées)."""

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._nom})"  # Exemple : VariableNumerique(Mois)


class VariableNumerique(Variable):
    """Variable numérique, avec un intervalle permis optionnel."""

    def __init__(self, nom: str, minimum: float | None = None, maximum: float | None = None) -> None:
        super().__init__(nom, float)
        self._minimum = minimum  # Borne minimale de l'intervalle (None = aucune)
        self._maximum = maximum  # Borne maximale de l'intervalle (None = aucune)

    @property
    def minimum(self) -> float | None:
        """Borne minimale permise (lecture seule)."""
        return self._minimum

    @property
    def maximum(self) -> float | None:
        """Borne maximale permise (lecture seule)."""
        return self._maximum

    def valider_saisie(self, valeur: object) -> float | None:
        """Convertit en nombre. Retourne None si la case est vide.
        Lève DonneeInvalideError si ce n'est pas un nombre ou s'il est hors de l'intervalle."""
        if pd.isna(valeur):
            return None

        try:
            nombre = float(valeur)  # Essaie de convertir en nombre
        except (TypeError, ValueError):  # Attrape seulement ces deux erreurs de Python
            raise DonneeInvalideError(f"La valeur '{valeur}' n'est pas numérique pour '{self._nom}'.")

        # Si un intervalle existe et que le nombre est hors de celui-ci (bornes incluses), on lève l'erreur
        # (nan est aussi refusé, car aucune comparaison avec nan n'est vraie)
        if self._minimum is not None and not self._minimum <= nombre <= self._maximum:
            raise DonneeInvalideError(f"'{self._nom}' doit être entre {self._minimum} et {self._maximum}.")

        return nombre

    def decrire(self, colonne: pd.Series) -> dict:
        """Fiche de la variable : nom, type, plus petite et plus grande valeur de la colonne."""
        return {
            "nom": self._nom,
            "type": "numerique",
            "minimum": float(colonne.min()),  # float() pour que le JSON accepte la valeur
            "maximum": float(colonne.max()),
        }

    def encoder(self, valeurs: Iterable) -> list[list[float]]:
        """Chaque valeur devient une mini-liste d'un nombre : 3.34 devient [3.34]."""
        resultat = []
        for valeur in valeurs:
            resultat.append([float(valeur)])
        return resultat


class VariableCategorielle(Variable):
    """Variable catégorielle (ex : le sport), avec la liste des valeurs permises."""

    def __init__(self, nom: str, valeurs_permises: list[str]) -> None:
        super().__init__(nom, str)
        self._valeurs_permises = valeurs_permises  # Ex : ["Tennis", "Hockey"]

    @property
    def valeurs_permises(self) -> list[str]:
        """Valeurs acceptées (lecture seule)."""
        return self._valeurs_permises

    def valider_saisie(self, valeur: object) -> str | None:
        """Retourne la valeur si elle est permise, None si la case est vide.
        Lève DonneeInvalideError si la valeur n'est pas dans la liste."""
        if pd.isna(valeur):
            return None

        if valeur not in self._valeurs_permises:
            raise DonneeInvalideError(f"La valeur '{valeur}' n'est pas permise pour '{self._nom}'.")

        return valeur

    def decrire(self, colonne: pd.Series) -> dict:
        """Fiche de la variable : nom, type et liste des valeurs permises."""
        return {"nom": self._nom, "type": "categorielle", "valeurs": self._valeurs_permises}

    def encoder(self, valeurs: Iterable) -> list[list[float]]:
        """Une case par valeur permise : 1.0 si c'est cette valeur, 0.0 sinon.
        Exemple avec ["Hockey", "Tennis"] : « Tennis » devient [0.0, 1.0]."""
        resultat = []
        for valeur in valeurs:  # Passe chaque match
            ligne = []
            for permise in self._valeurs_permises:  # Passe chaque valeur permise
                if valeur == permise:
                    ligne.append(1.0)
                else:
                    ligne.append(0.0)
            resultat.append(ligne)
        return resultat


def creer_variable(description: dict) -> Variable:
    """Recrée une variable à partir de sa fiche (l'inverse de decrire)."""
    if description["type"] == "numerique":
        return VariableNumerique(description["nom"], description["minimum"], description["maximum"])
    return VariableCategorielle(description["nom"], description["valeurs"])


class JeuDeDonnees:
    """Charge, nettoie et prépare le jeu de données (fichier CSV)."""

    def __init__(self, chemin_fichier: str, variables: list[Variable] | None = None) -> None:
        self._chemin_fichier = chemin_fichier
        self._variables = variables or []
        self._df = self.charger()  # Dès que l'objet est créé, il charge le fichier dans un DataFrame (df)

    @property
    def df(self) -> pd.DataFrame:
        """Tableau de données courant (lecture seule)."""
        return self._df

    def charger(self) -> pd.DataFrame:
        """Cherche le CSV à plusieurs endroits possibles et le lit.
        Lève FileNotFoundError si le fichier n'existe nulle part."""
        chemins = []

        if os.path.isabs(self._chemin_fichier):
            chemins.append(self._chemin_fichier)
        else:
            chemins.extend([
                os.path.join(os.getcwd(), self._chemin_fichier),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), self._chemin_fichier),
                os.path.join(os.path.dirname(__file__), self._chemin_fichier),
            ])

        for chemin in chemins:
            if os.path.exists(chemin):
                self._chemin_fichier = chemin
                with open(chemin, encoding="utf-8") as fichier:  # with ferme le fichier automatiquement
                    return pd.read_csv(fichier)  # Lit le CSV et le transforme en tableau

        raise FileNotFoundError(f"Le fichier n'existe pas : {self._chemin_fichier}")

    def nettoyer(self) -> pd.DataFrame:
        """Nettoie le tableau et ajoute les colonnes Mois et Resultat_Simplifie. Retourne le tableau."""
        self._df = self._df.copy()  # Évite de modifier l'original par accident

        for colonne in self._df.columns:
            self._df[colonne] = self._df[colonne].map(lambda x: str(x).strip() if pd.notna(x) else x)

        # Remplace les mots vides par le format officiel de pandas pour les données manquantes
        self._df = self._df.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA})

        # Supprime les lignes où l'une de ces 3 colonnes est vide, et ignore les autres
        # (un dropna() sur tout le tableau ne laissait que 443 lignes, car Draw_Odds est souvent vide)
        self._df = self._df.dropna(subset=["Home_Team_Odds", "Away_Team_Odds", "Sport"])

        # Cote du nul vide = pas de nul possible : "0.0" (en texte, car la colonne est encore du texte)
        self._df["Draw_Odds"] = self._df["Draw_Odds"].fillna("0.0")

        # Convertit Date en vraie date et extrait le numéro du mois (1 à 12) dans une nouvelle colonne
        self._df["Mois"] = pd.to_datetime(self._df["Date"]).dt.month

        # Colonne réponse : apply appelle la fonction une fois par ligne (axis=1)
        # "Home" si le gagnant est l'équipe à domicile, "Away" si c'est la visiteuse, sinon "Draw"
        self._df["Resultat_Simplifie"] = self._df.apply(
            lambda ligne: "Home" if (ligne["Actual_Winner"] == ligne["Home_Team"])
            else ("Away" if (ligne["Actual_Winner"] == ligne["Away_Team"]) else ("Draw")),
            axis=1
        )

        return self._df

    def valider(self) -> pd.DataFrame:
        """Convertit chaque colonne d'entrée avec valider_saisie de sa variable.
        Lève DonneeInvalideError si une colonne est absente ou contient une valeur invalide."""
        if not self._variables:  # Aucune variable à surveiller : on ne modifie rien
            return self._df

        for variable in self._variables:
            if variable.nom not in self._df.columns:
                raise DonneeInvalideError(f"La colonne '{variable.nom}' est absente du dataset.")

            self._df[variable.nom] = self._df[variable.nom].map(variable.valider_saisie)  # Chaque case passe dans valider_saisie

        return self._df

    def separer_train_test(self, cible: str, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """Encode les variables puis sépare en (x_train, x_test, y_train, y_test).
        Lève KeyError si la cible n'existe pas."""
        if cible not in self._df.columns:
            raise KeyError(f"La cible '{cible}' n'existe pas dans le dataset.")

        # Pour chaque variable, encode toute sa colonne
        colonnes_encodees = [var.encoder(self._df[var.nom]) for var in self._variables]
        X = []  # Contiendra tous les matchs encodés
        for i in range(len(self._df)):  # i = numéro du match
            ligne = []
            for colonne in colonnes_encodees:
                ligne = ligne + colonne[i]  # Colle à la suite le morceau de ce match (+ met bout à bout deux listes)
            X.append(ligne)  # Le match devient une ligne de 9 nombres

        y = self._df[cible]  # Colonne réponse

        # random_state fixe la coupe pour qu'elle soit identique à chaque lancement
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def decrire_variables(self) -> list[dict]:
        """Liste des fiches de toutes les variables (à appeler après valider())."""
        description = []
        for variable in self._variables:
            description.append(variable.decrire(self._df[variable.nom]))
        return description

    def __str__(self) -> str:
        return f"JeuDeDonnees({self._chemin_fichier}, {len(self._df)} lignes)"


def tracer_resultats(df: pd.DataFrame, chemin_image: str) -> None:
    """Graphique d'exploration : nombre de matchs par résultat, enregistré dans un fichier image."""
    comptes = df["Resultat_Simplifie"].value_counts()  # Combien de fois chaque résultat apparaît
    plt.bar(comptes.index, comptes.values)  # .index = noms, .values = hauteurs
    plt.title("Nombre de matchs selon le résultat")
    plt.xlabel("Résultat")
    plt.ylabel("Nombre de matchs")
    plt.savefig(chemin_image, dpi=150)  # dpi = résolution de l'image
    plt.close()  # Ferme la figure