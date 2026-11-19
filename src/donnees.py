"""Classes et fonctions pour charger, nettoyer et préparer les données sportives."""

import os
from typing import Any, Optional
import math
import pandas as pd
import matplotlib.pyplot as plt  # Importe l'outil de graphiques (pour tracer_resultats)
from exceptions import DonneeInvalideError  # Importe notre exception personnalisée depuis exceptions.py
from sklearn.model_selection import train_test_split


class Variable:
    def __init__(self, nom: str, type_donnee: type = str) -> None:
        """Initialise une variable avec son nom et le type de données attendu."""
        self.nom = nom
        self.type_donnee = type_donnee

    def valider_saisie(self, valeur: Any) -> Any:
        """Valide une valeur et la convertit dans le type attendu.

        Args:
            valeur: Valeur à vérifier.

        Returns:
            La valeur convertie, ou None si elle est manquante.

        Raises:
            DonneeInvalideError: Si la conversion est impossible.
        """
        # Vérifie si une cellule du dataset contient une valeur valide
        if pd.isna(valeur):  # Fonction de pandas qui permet de savoir si une cellule est vide
            return None

        try:  # Essaie de convertir la valeur dans le type attendu
            return self.type_donnee(valeur)
        except (TypeError, ValueError):
            raise DonneeInvalideError(
                f"La valeur '{valeur}' n'est pas compatible avec la variable '{self.nom}'."
            )


class VariableNumerique(Variable):
    def __init__(
        self,
        nom: str,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None
    ) -> None:
        """Initialise une variable numérique et ses bornes facultatives."""
        # Le constructeur accepte un intervalle optionnel (None = aucun)
        super().__init__(nom, float)
        self.minimum = minimum  # Mémorise la borne minimale de l'intervalle
        self.maximum = maximum  # Mémorise la borne maximale de l'intervalle

    def valider_saisie(self, valeur: Any) -> Optional[float]:
        """Convertit une valeur en flottant et vérifie ses bornes.

        Args:
            valeur: Valeur numérique à valider.

        Returns:
            Le nombre converti, ou None si la valeur est manquante.

        Raises:
            DonneeInvalideError: Si la valeur n'est pas numérique ou hors limites.
        """
        if pd.isna(valeur):
            return None

        try:
            nombre = float(valeur)
        except (TypeError, ValueError):
            raise DonneeInvalideError(
                f"La valeur '{valeur}' n'est pas numérique pour '{self.nom}'."
            )

        if not math.isfinite(nombre):
            raise DonneeInvalideError(
                f"La valeur '{valeur}' n'est pas un nombre fini pour '{self.nom}'."
            )

        # Vérifie indépendamment la borne minimale
        if self.minimum is not None and nombre < self.minimum:
            raise DonneeInvalideError(
                f"'{self.nom}' doit être supérieur ou égal à {self.minimum}."
            )

        # Vérifie indépendamment la borne maximale
        if self.maximum is not None and nombre > self.maximum:
            raise DonneeInvalideError(
                f"'{self.nom}' doit être inférieur ou égal à {self.maximum}."
            )

        return nombre  # Retourne le nombre, qui est valide

    def decrire(self, colonne: pd.Series) -> dict[str, Any]:
        """Produit les métadonnées de la variable numérique à partir d'une colonne."""
        # Même nom et mêmes paramètres dans les deux classes de variables
        # Retourne un dictionnaire (une fiche). .min() et .max() donnent la plus petite et la plus grande valeur.
        # float() convertit en flottant Python standard pour que le JSON accepte la valeur.
        return {
            "nom": self.nom,
            "type": "numerique",
            "minimum": float(colonne.min()),
            "maximum": float(colonne.max()),
        }

    def encoder(self, valeurs: Any) -> list[list[float]]:
        """Encode chaque valeur numérique sous forme d'une liste contenant un flottant."""
        # Reçoit une colonne de valeurs
        resultat = []  # Crée la liste de résultats, vide au départ
        for valeur in valeurs:  # Passe chaque valeur
            resultat.append([float(valeur)])  # Convertit en nombre et l'ajoute dans une mini-liste : 3.34 devient [3.34]
        return resultat  # Retourne la liste de mini-listes


class VariableCategorielle(Variable):
    # Crée une classe enfant de Variable
    def __init__(self, nom: str, valeurs_permises: list[str]) -> None:
        """Initialise une variable catégorielle et ses valeurs autorisées."""
        # Le constructeur reçoit le nom et la liste des valeurs acceptées
        super().__init__(nom, str)  # Demande au parent de mémoriser le nom et le type texte
        self.valeurs_permises = valeurs_permises  # Mémorise la liste des valeurs acceptées

    def valider_saisie(self, valeur: Any) -> Optional[str]:
        """Valide une valeur catégorielle selon les catégories autorisées.

        Args:
            valeur: Valeur à vérifier.

        Returns:
            La catégorie validée, ou None si la valeur est manquante.

        Raises:
            DonneeInvalideError: Si la catégorie n'est pas autorisée.
        """
        # Redéfinit la méthode du parent
        if pd.isna(valeur):  # Une case vide retourne None
            return None

        if valeur not in self.valeurs_permises:  # Vrai si la valeur n'est pas dans la liste
            raise DonneeInvalideError(
                f"La valeur '{valeur}' n'est pas permise pour '{self.nom}'."
            )  # Lève l'erreur avec un message

        return valeur  # Retourne la valeur, qui est valide

    def decrire(self, colonne: pd.Series) -> dict[str, Any]:
        """Produit les métadonnées de la variable catégorielle à partir d'une colonne."""
        # Même idée que VariableNumerique, avec la liste des valeurs permises
        return {
            "nom": self.nom,
            "type": "categorielle",
            "valeurs": self.valeurs_permises
        }

    def encoder(self, valeurs: Any) -> list[list[float]]:
        """Encode les catégories en vecteurs binaires (un indicateur par catégorie)."""
        resultat = []  # Liste de résultats
        for valeur in valeurs:  # Passe chaque match
            ligne = []  # Liste de cases pour ce match
            for permise in self.valeurs_permises:  # Passe chaque valeur permise (chaque sport)
                if valeur == permise:
                    ligne.append(1.0)  # Met 1.0 si c'est ce sport
                else:
                    ligne.append(0.0)  # Met 0.0 sinon
            resultat.append(ligne)  # Ajoute la ligne de cases pour ce match
        return resultat  # Retourne la liste de lignes. Exemple avec ["Hockey", "Tennis"] : « Tennis » devient [0.0, 1.0]


def creer_variable(description: dict[str, Any]) -> Variable:
    """Reconstruit une variable à partir de sa description enregistrée."""
    # Reçoit une fiche (un dictionnaire)
    if description["type"] == "numerique":  # Si la fiche dit numérique
        return VariableNumerique(
            description["nom"],
            description["minimum"],
            description["maximum"]
        )  # Crée une VariableNumerique avec son intervalle

    return VariableCategorielle(
        description["nom"],
        description["valeurs"]
    )  # Sinon crée une VariableCategorielle avec ses valeurs


class JeuDeDonnees:  # Classe qui gère le dataset CSV
    def __init__(
        self,
        chemin_fichier: str,
        variables: Optional[list[Variable]] = None
    ) -> None:
        """Initialise le jeu de données et charge le fichier CSV."""
        self.chemin_fichier = chemin_fichier
        self.variables = variables or []
        self.df = self.charger()  # Dès que l'objet est créé, il charge le fichier et stocke le tableau dans DataFrame (df)

    def charger(self) -> pd.DataFrame:
        """Recherche puis charge le fichier CSV dans un DataFrame.

        Raises:
            FileNotFoundError: Si le fichier est introuvable dans les chemins vérifiés.
        """
        chemins = []

        if os.path.isabs(self.chemin_fichier):
            chemins.append(self.chemin_fichier)
        else:
            chemins.extend([
                os.path.join(os.getcwd(), self.chemin_fichier),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), self.chemin_fichier),
                os.path.join(os.path.dirname(__file__), self.chemin_fichier),
            ])

        for chemin in chemins:
            if os.path.exists(chemin):
                self.chemin_fichier = chemin
                return pd.read_csv(chemin)  # Méthode de pandas qui lit le dataset CSV et le transforme en tableau

        raise FileNotFoundError(f"Le fichier n'existe pas : {self.chemin_fichier}")

    def nettoyer(self) -> pd.DataFrame:
        """Nettoie les colonnes, traite certaines valeurs manquantes et crée les variables dérivées."""
        self.df = self.df.copy()  # Évite de modifier l'original par accident

        for colonne in self.df.columns:
            self.df[colonne] = self.df[colonne].map(
                lambda x: str(x).strip() if pd.notna(x) else x
            )

        # Remplace les mots vides par le format officiel de pandas pour les données manquantes
        self.df = self.df.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA})

        # Supprime les lignes où l'une de ces 3 colonnes est vide, et ignore les autres
        # (L'ancien dropna() supprimait toute ligne avec une case vide et ne laissait que 443 lignes)
        self.df = self.df.dropna(subset=["Home_Team_Odds", "Away_Team_Odds", "Sport"])

        # Remplace chaque case vide de Draw_Odds par "0.0" (= pas de nul possible), en texte car la colonne est encore du texte
        self.df["Draw_Odds"] = self.df["Draw_Odds"].fillna("0.0")

        # Convertit Date en vraie date, extrait le numéro du mois (1 à 12) et le range dans une nouvelle colonne Mois
        self.df["Mois"] = pd.to_datetime(self.df["Date"]).dt.month

        # Crée la colonne réponse : apply appelle la fonction donnée une fois par ligne (axis=1 = ligne par ligne)
        # lambda ligne : fonction sans nom où ligne est le match en cours
        # "Home" si le gagnant est l'équipe à domicile, "Away" si c'est l'équipe visiteuse, sinon "Draw"
        self.df["Resultat_Simplifie"] = self.df.apply(
            lambda ligne: "Home" if (ligne["Actual_Winner"] == ligne["Home_Team"])
            else ("Away" if (ligne["Actual_Winner"] == ligne["Away_Team"]) else ("Draw")),
            axis=1
        )

        return self.df

    def valider(self) -> pd.DataFrame:
        """Valide et convertit les colonnes associées aux variables déclarées.

        Raises:
            DonneeInvalideError: Si une colonne manque ou contient une valeur invalide.
        """
        if not self.variables:  # Si aucune variable à surveiller, retourne le tableau df sans rien modifier
            return self.df

        for variable in self.variables:
            if variable.nom not in self.df.columns:
                raise DonneeInvalideError(
                    f"La colonne '{variable.nom}' est absente du dataset."
                )

            # .map(...) prend chaque case et l'envoie à .valider_saisie()
            self.df[variable.nom] = self.df[variable.nom].map(variable.valider_saisie)

        return self.df  # Retourne le tableau nettoyé et validé

    def separer_train_test(
        self,
        cible: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> tuple[Any, Any, Any, Any]:
        """Encode les caractéristiques puis sépare les données en ensembles d'entraînement et de test.

        Args:
            cible: Nom de la colonne à prédire.
            test_size: Proportion réservée aux données de test.
            random_state: Graine assurant une séparation reproductible.

        Returns:
            Les quatre ensembles X_train, X_test, y_train et y_test.

        Raises:
            KeyError: Si la colonne cible n'existe pas.
        """
        if cible not in self.df.columns:
            raise KeyError(f"La cible '{cible}' n'existe pas dans le dataset.")

        # Pour chaque variable, encode toute sa colonne : on obtient une liste de valeurs
        colonnes_encodees = [var.encoder(self.df[var.nom]) for var in self.variables]

        X = []  # Liste qui contiendra tous les matchs encodés
        for i in range(len(self.df)):  # i prend tous les numéros de match (de 0 au dernier)
            ligne = []  # Ligne vide pour le match i
            for colonne in colonnes_encodees:  # Passe chaque variable encodée
                ligne = ligne + colonne[i]  # Colle à la suite le morceau de ce match pour cette variable
            X.append(ligne)  # Ajoute le match à X

        y = self.df[cible]  # Prend la colonne réponse

        # Coupe en x_train, x_test, y_train, y_test, avec 20 % pour le test
        # random_state fixe la coupe pour qu'elle soit identique à chaque lancement
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

    def decrire_variables(self) -> list[dict[str, Any]]:
        """Retourne les descriptions des variables pour leur sauvegarde en métadonnées."""
        description = []  # Liste vide
        for variable in self.variables:  # Passe chaque variable
            description.append(variable.decrire(self.df[variable.nom]))  # Ajoute la fiche de la variable, en lui donnant sa colonne
        return description  # Retourne la liste des fiches


def tracer_resultats(df: pd.DataFrame, chemin_image: str) -> None:
    """Crée et enregistre un graphique du nombre de matchs par résultat.

    Args:
        df: Tableau contenant la colonne Resultat_Simplifie.
        chemin_image: Chemin du fichier image à créer.
    """
    comptes = df["Resultat_Simplifie"].value_counts()  # Compte combien de fois chaque résultat apparaît
    plt.bar(comptes.index, comptes.values)  # Dessine des barres (.index donne les noms, .values donne les hauteurs)
    plt.title("Nombre de matchs selon le résultat")  # Titre du graphique
    plt.xlabel("Résultat")  # Étiquette de l'axe X
    plt.ylabel("Nombre de matchs")  # Étiquette de l'axe Y
    plt.savefig(chemin_image, dpi=150)  # Enregistre l'image dans un fichier, en résolution 150
    plt.show()


if __name__ == "__main__":
    dataset = JeuDeDonnees("donnees/sports_betting_predictive_analysis.csv")
    print(dataset.df.columns.tolist())
    print(dataset.df.head())

    variables = [  # Déclare les 5 caractéristiques
        VariableCategorielle("Sport", ["Baseball", "Basketball", "Football", "Hockey", "Tennis"]),
        VariableNumerique("Home_Team_Odds"),
        VariableNumerique("Away_Team_Odds"),
        VariableNumerique("Draw_Odds"),
        VariableNumerique("Mois"),
    ]

    dataset2 = JeuDeDonnees(
        "donnees/sports_betting_predictive_analysis.csv",
        variables
    )  # Crée l'objet, qui charge le CSV

    dataset2.nettoyer()  # Prépare le tableau
    tracer_resultats(dataset2.df, "modeles/exploration.png")
    print(len(dataset2.df))  # Affiche le nombre de lignes (1223 attendu)
    print(dataset2.df["Resultat_Simplifie"].value_counts())  # Affiche le compte de Home, Away et Draw
    dataset2.valider()  # Convertit chaque colonne d'entrée avec valider_saisie
    x_train, x_test, y_train, y_test = dataset2.separer_train_test("Resultat_Simplifie")
    print(len(x_train), len(x_test))  # Affiche 978 245